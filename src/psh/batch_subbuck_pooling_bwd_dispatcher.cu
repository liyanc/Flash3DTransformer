/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 10/25/24
 */


#include <c10/cuda/CUDAStream.h>
#include <ATen/ATen.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "common/runtime.h"
#include "common/fp_types.cuh"
#include "common/device_manager.cuh"

#include "common/kernel_dispatcher.cuh"

#include "psh/batch_psh_subbuck_reduce_bwd.cuh"

namespace f3d {

extern void
batch_subbuck_reduce_backward(
  at::Tensor &grad_red_feat, at::Tensor &input_feat, at::Tensor &unpool_ind, at::Tensor &grad_input_feat,
  uint32_t subbuck_size, uint16_t reduction_op) {
  constexpr uint32_t warps_per_cta = 32u;
  using CounterT = uint32_t;
  using FeatT = bf16;
  uint32_t reduced_N = grad_red_feat.size(0);

  assert_contiguous({grad_red_feat, input_feat, unpool_ind, grad_input_feat});
  assert_same_device({grad_red_feat, input_feat, unpool_ind, grad_input_feat});
  assert_dtype({grad_red_feat, input_feat, grad_input_feat}, at::ScalarType::BFloat16);
  assert_dtype({unpool_ind}, at::ScalarType::UInt32);

  if (grad_red_feat.size(1) != input_feat.size(1) || grad_red_feat.size(1) != grad_input_feat.size(1)) {
    throw_format_error(
      "Grad_reduced feat_dim(%d), input feat_dim(%d), Grad_input feat_dim(%d) mismatch!",
      grad_red_feat.size(1), input_feat.size(1), grad_input_feat.size(1));
  }
  if (unpool_ind.size(0) > grad_red_feat.size(0)) {
    throw_format_error(
      "grad_red_feat length(%d) is incompatible to unpool_ind length(%d).%s",
      grad_red_feat.size(0), unpool_ind.size(0),
      "We expect grad_red_feat length >= Unpool_ind length"
      );
  }
  if (unpool_ind.size(1) != subbuck_size) {
    throw_format_error(
      "unpool_ind dim(%d) mismatch subbuck_size=%d",
      unpool_ind.size(1), subbuck_size);
  }

  auto stream = c10::cuda::getCurrentCUDAStream(grad_red_feat.get_device());
  auto & dev_mgr = DeviceManagerSingleton::instance();
  dev_mgr.set_device(grad_red_feat.get_device());

  if (dev_mgr.get_sync_stream()) {
    cudaCheckErr(cudaStreamSynchronize(stream));
  }

  auto red_bwd_ker = reduce_feat_backward_ker<CounterT, FeatT>;
  cudaFuncSetCacheConfig(red_bwd_ker, cudaFuncCachePreferL1);
  cudaCheckLastErr();

  auto bwd_grid = dim3{cdiv(reduced_N, warps_per_cta)};

  red_bwd_ker<<<bwd_grid, warps_per_cta * WarpSize, 0, stream>>>(
    (FeatT *) grad_red_feat.data_ptr(), (FeatT *) input_feat.data_ptr(), (CounterT *) unpool_ind.data_ptr(),
    (FeatT *) grad_input_feat.data_ptr(),
    grad_input_feat.size(0), grad_red_feat.size(0),
    grad_input_feat.size(1),
    subbuck_size, REDUCE_OP(reduction_op),
    grad_red_feat.stride(0), unpool_ind.stride(0)
    );
  cudaCheckLastErr();

  if (dev_mgr.get_sync_stream()) {
    cudaCheckErr(cudaStreamSynchronize(stream));
  }
}

} // end of ::f3d