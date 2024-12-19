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

#include "psh/additive_unpool.cuh"


namespace f3d {

extern void
additive_unpool_fwd(at::Tensor &res_feat, at::Tensor &down_feat, at::Tensor &up_add_feat,
                    at::Tensor &unpool_ind, uint32_t subbuck_size) {
  constexpr uint32_t warps_per_cta = 32u;
  assert_contiguous({res_feat, down_feat, up_add_feat, unpool_ind});
  assert_same_device({res_feat, down_feat, up_add_feat, unpool_ind});
  assert_dtype({res_feat, down_feat, up_add_feat}, at::ScalarType::BFloat16);

  if (res_feat.size(1) != down_feat.size(1) || res_feat.size(1) != up_add_feat.size(1)) {
    throw_format_error(
      "Residual feat_dim(%d), down path feat_dim(%d), up path feat_dim(%d) mismatch!",
      res_feat.size(1), down_feat.size(1), up_add_feat.size(1));
  }
  if (res_feat.size(0) != up_add_feat.size(0)) {
    throw_format_error(
      "Residual size(%d) and up path size(%d) mismatch!",
      res_feat.size(0), up_add_feat.size(0));
  }
  if (unpool_ind.size(0) > down_feat.size(0) || unpool_ind.size(1) != subbuck_size) {
    throw_format_error(
      "Unpool_ind shape[%d,%d] is incompatible to down_feat length(%d) and subbuck_size(%d).%s",
      unpool_ind.size(0), unpool_ind.size(1), down_feat.size(0), subbuck_size,
      "We expect down_feat length >= Unpool_ind length");
  }

  using CounterT = uint32_t;
  using FeatT = bf16;
  auto stream = c10::cuda::getCurrentCUDAStream(res_feat.get_device());
  auto reduced_N = uint32_t(unpool_ind.size(0));
  auto & dev_mgr = DeviceManagerSingleton::instance();
  dev_mgr.set_device(res_feat.get_device());

  auto add_unpool_fwd_ker = additive_unpool_fwd_ker<CounterT, FeatT>;
  cudaFuncSetCacheConfig(add_unpool_fwd_ker, cudaFuncCachePreferL1);
  cudaCheckLastErr();

  if (dev_mgr.get_sync_stream()) {
    cudaCheckErr(cudaStreamSynchronize(stream));
  }

  auto unpool_grid = dim3{cdiv(reduced_N, warps_per_cta)};
  add_unpool_fwd_ker<<<unpool_grid, warps_per_cta * WarpSize, 0, stream>>>(
    (FeatT *) res_feat.data_ptr(), (FeatT *) down_feat.data_ptr(), (FeatT *) up_add_feat.data_ptr(),
    (CounterT *) unpool_ind.data_ptr(),
    res_feat.size(0), down_feat.size(0),
    res_feat.size(1), subbuck_size,
    res_feat.stride(0), unpool_ind.stride(0)
    );
  cudaCheckLastErr();

  if (dev_mgr.get_sync_stream()) {
    cudaCheckErr(cudaStreamSynchronize(stream));
  }
}

extern void
additive_unpool_bwd(at::Tensor &grad_res, at::Tensor &grad_down, at::Tensor &grad_up_added,
                    at::Tensor &unpool_ind, uint32_t subbuck_size) {
  constexpr uint32_t warps_per_cta = 32u;
  assert_contiguous({grad_res, grad_down, grad_up_added, unpool_ind});
  assert_same_device({grad_res, grad_down, grad_up_added, unpool_ind});
  assert_dtype({grad_res, grad_down, grad_up_added}, at::ScalarType::BFloat16);

  if (grad_res.size(1) != grad_down.size(1) || grad_res.size(1) != grad_up_added.size(1)) {
    throw_format_error(
      "Residual feat_dim(%d), down path feat_dim(%d), up path feat_dim(%d) mismatch!",
      grad_res.size(1), grad_down.size(1), grad_up_added.size(1));
  }
  if (grad_res.size(0) != grad_up_added.size(0)) {
    throw_format_error(
      "Residual size(%d) and up path size(%d) mismatch!",
      grad_res.size(0), grad_up_added.size(0));
  }
  if (unpool_ind.size(0) > grad_down.size(0) || unpool_ind.size(1) != subbuck_size) {
    throw_format_error(
      "Unpool_ind shape[%d,%d] is incompatible to grad_down length(%d) and subbuck_size(%d).%s",
      unpool_ind.size(0), unpool_ind.size(1), grad_down.size(0), subbuck_size,
      "We expect grad_down length >= Unpool_ind length");
  }

  using CounterT = uint32_t;
  using FeatT = bf16;
  auto stream = c10::cuda::getCurrentCUDAStream(grad_res.get_device());
  auto reduced_N = uint32_t(unpool_ind.size(0));
  auto & dev_mgr = DeviceManagerSingleton::instance();
  dev_mgr.set_device(grad_res.get_device());

  auto add_unpool_bwd_ker = additive_unpool_bwd_ker<CounterT, FeatT>;
  cudaFuncSetCacheConfig(add_unpool_bwd_ker, cudaFuncCachePreferL1);
  cudaCheckLastErr();

  if (dev_mgr.get_sync_stream()) {
    cudaCheckErr(cudaStreamSynchronize(stream));
  }

  auto unpool_grid = dim3{cdiv(reduced_N, warps_per_cta)};
  add_unpool_bwd_ker<<<unpool_grid, warps_per_cta * WarpSize, 0, stream>>>(
    (FeatT *) grad_res.data_ptr(), (FeatT *) grad_down.data_ptr(), (FeatT *) grad_up_added.data_ptr(),
    (CounterT *) unpool_ind.data_ptr(),
    grad_res.size(0), grad_down.size(0),
    grad_res.size(1), subbuck_size,
    grad_res.stride(0), unpool_ind.stride(0)
    );
  cudaCheckLastErr();

  if (dev_mgr.get_sync_stream()) {
    cudaCheckErr(cudaStreamSynchronize(stream));
  }
}

} // end of ::f3d