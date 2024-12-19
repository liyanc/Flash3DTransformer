/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 10/6/24
 */


#include <chrono>
#include <stdexcept>
#include <glog/logging.h>

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/ATen.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <kittens.cuh>
#include "common/runtime.h"
#include "common/arithmetic.cuh"
#include "common/fp_types.cuh"
#include "common/device_manager.cuh"

#include "common/kernel_dispatcher.cuh"

#include "psh/batch_psh_subbuck_reduce.cuh"
#include "psh/additive_unpool.cuh"


namespace f3d {

extern void
batch_subbuck_reduce(
  at::Tensor &scattered_coord, at::Tensor &scattered_feat, at::Tensor &batch_sep,
  uint32_t instance_max_N, uint32_t subbuck_size, uint32_t num_vox, at::Tensor &bbox_min, at::Tensor &bbox_max,
  at::Tensor &subbuck_id, at::Tensor &subbuck_off, at::Tensor &reduced_batch_sep,
  at::Tensor &reduced_coord, at::Tensor &reduced_feat, at::Tensor &unpool_ind, uint16_t hash_type,
  uint16_t reduction_op
  ) {
  auto total_N = scattered_coord.size(0);
  constexpr uint32_t warps_per_block = 32u;
  using CoordT = fp16;
  using CounterT = uint32_t;
  using HashT = uint16_t;
  using FeatT = bf16;

  assert_contiguous({scattered_coord, scattered_feat, batch_sep, subbuck_id, subbuck_off,
                     reduced_batch_sep, reduced_coord, reduced_feat});
  assert_same_device({scattered_coord, scattered_feat, batch_sep, subbuck_id, subbuck_off,
                      reduced_batch_sep, reduced_coord, reduced_feat});
  assert_dtype({scattered_feat, reduced_feat}, at::ScalarType::BFloat16);
  assert_dtype({scattered_coord, reduced_coord, bbox_min, bbox_max}, at::ScalarType::Half);
  assert_dtype({batch_sep, subbuck_id, reduced_batch_sep}, at::ScalarType::UInt32);
  assert_dtype({subbuck_off}, at::ScalarType::UInt16);

  if (scattered_feat.size(1) != reduced_feat.size(1)) {
    throw_format_error(
      "Input feature dim(%d) != output feature dim(%d).",
      scattered_feat.size(1), reduced_feat.size(1));
  }
  if (reduced_coord.size(0) != reduced_feat.size(0)) {
    throw_format_error(
      "Reduced coordinate length(%d) != reduced feature length(%d).",
      reduced_coord.size(0), reduced_feat.size(0));
  }
  if (scattered_coord.size(1) != 3 || reduced_coord.size(1) != 3) {
    throw_format_error(
      "Input coordinates dim(%d) or reduced coordinates dim(%d) are NOT 3D. %s",
      scattered_coord.size(1), reduced_coord.size(1),
      "Contact us if you want support for other dimensions."
      );
  }
  if (reduction_op < 0 || reduction_op > 3){
    throw_format_error(
      "Unsupported reduction operation id-%d", reduction_op);
  }
  if (unpool_ind.size(1) != subbuck_size) {
    throw_format_error(
      "Unpool_ind last dimension(%d) != subbucket_size(%d)", unpool_ind.size(1), subbuck_size);
  }
  if (reduced_feat.size(0) != reduced_coord.size(0) || reduced_coord.size(1) != scattered_coord.size(1)) {
    throw_format_error(
      "reduced_coord size[%d,%d] unmatch the expected[%d,%d]",
      reduced_coord.size(0), reduced_coord.size(1),
      reduced_coord.size(0), scattered_coord.size(1));
  }
  if (reduction_op >= 4) {
    throw_format_error("Unsupported reduction_op=%d", reduction_op);
  }

  auto bsize = uint32_t(batch_sep.size(0));
  auto reduced_N = uint32_t(unpool_ind.size(0));
  auto stream = c10::cuda::getCurrentCUDAStream(scattered_coord.get_device());
  auto shmem_size = tk::MAX_SHARED_MEMORY;

  auto & dev_mgr = DeviceManagerSingleton::instance();
  auto rand_states = dev_mgr.get_states_dev(scattered_coord.get_device());
  dev_mgr.set_device(scattered_coord.get_device());
  auto subbuck_cnt = torch::zeros_like(reduced_batch_sep);

  if (dev_mgr.get_sync_stream()) {
    cudaCheckErr(cudaStreamSynchronize(stream));
  }

  auto subbuck_ker = local_subbuck_ker<
    CoordT, CounterT, HashT, 3, 1024, H_ZORDER_DIV, H_XORSUM_DIV, S_GLOBAL>;
  auto feat_red_ker = reduce_feat_ker<CounterT, FeatT>;
  cudaFuncSetAttribute(subbuck_ker, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
  cudaCheckLastErr();
  cudaFuncSetCacheConfig(feat_red_ker, cudaFuncCachePreferL1);
  cudaCheckLastErr();

  auto subbuck_grid = dim3{cdiv(instance_max_N, warps_per_block * WarpSize), bsize, 1};
  auto feat_grid = dim3{cdiv(reduced_N, warps_per_block)};

  subbuck_ker<<<subbuck_grid, warps_per_block * WarpSize, shmem_size, stream>>>(
    (CoordT *) scattered_coord.data_ptr(), (CounterT *) subbuck_id.data_ptr(), (HashT *) subbuck_off.data_ptr(),
    (CounterT *) subbuck_cnt.data_ptr(), (CounterT *) reduced_batch_sep.data_ptr(),
    (CounterT *) batch_sep.data_ptr(), (CoordT *) reduced_coord.data_ptr(), (CounterT *) unpool_ind.data_ptr(),
    bsize, total_N, subbuck_size, num_vox,
    (CoordT *) bbox_min.data_ptr(), (CoordT *) bbox_max.data_ptr(),
    scattered_coord.stride(0),
    unpool_ind.stride(0),
    rand_states
    );
  cudaCheckLastErr();

  feat_red_ker<<<feat_grid, warps_per_block * WarpSize, 0, stream>>>(
    (FeatT *) scattered_feat.data_ptr(), (FeatT *) reduced_feat.data_ptr(),
    (CounterT *) unpool_ind.data_ptr(),
    total_N, reduced_N,
    reduced_feat.size(1),
    subbuck_size, REDUCE_OP(reduction_op),
    reduced_feat.stride(0), unpool_ind.stride(0)
    );
  cudaCheckLastErr();

  if (dev_mgr.get_sync_stream()) {
    cudaCheckErr(cudaStreamSynchronize(stream));
  }
}

} // end of ::f3d