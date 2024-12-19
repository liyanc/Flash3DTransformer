/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 10/25/24
 */


#ifndef FLASH3DPSHATTN_ADDTIVE_UNPOOL_H
#define FLASH3DPSHATTN_ADDTIVE_UNPOOL_H

#include <stdexcept>
#include <glog/logging.h>

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda/pipeline>
#include <cuda/barrier>

#include <kittens.cuh>
#include "common/runtime.h"
#include "common/arithmetic.cuh"
#include "common/fp_types.cuh"
#include "common/rand.cuh"
#include "common/load_store.cuh"
#include "common/load_store_async.cuh"
#include "common/device_manager.cuh"
#include "common/block_minmax.cuh"
#include "psh/hash_fns.cuh"
#include "psh/two_stage_counter.cuh"
#include "psh/shmem_cumsum_counter.cuh"

namespace tk = kittens;

namespace f3d {

template <typename CounterT, typename FeatT>
__global__ void additive_unpool_fwd_ker(
  const FeatT * __restrict__ residual_feat, const FeatT * __restrict__ down_feat, FeatT * __restrict__ up_add_feat,
  const CounterT * __restrict__ unpool_ind,
  uint32_t up_N, uint32_t down_N,
  uint32_t feat_dim, uint32_t subbuck_size,
  uint32_t stride_featN, uint32_t stride_unpoolN
  ) {
  // One warp per reduced row
  auto warps_per_block = blockDim.x / WarpSize;
  auto block_start = warps_per_block * blockIdx.x;
  auto warpid = tk::warpid();
  auto laneid = tk::laneid();

  auto global_down_row = block_start + warpid;
  // Mask out warps that are beyond reduced_N
  if (global_down_row >= down_N) return;

  float feat_acc_reg = .0f;

  auto num_loops = cdiv_dev(feat_dim, WarpSize);
  for (auto feat_iter = 0; feat_iter < num_loops; ++feat_iter) {
    // Start a feature stride for all subbucks. clear feature accumulators
    auto dim_feat_elem = feat_iter * WarpSize + laneid;
    feat_acc_reg = .0f;

    // Only unpool if lane_dim in [0, feat_dim]
    if (dim_feat_elem < feat_dim) {
      // Start of all subbuck unpool projection
      feat_acc_reg = float(down_feat[global_down_row * stride_featN + dim_feat_elem]);

      for (auto sb_iter = 0; sb_iter < subbuck_size; ++sb_iter) {
        auto up_row = unpool_ind[global_down_row * stride_unpoolN + sb_iter];

        // Only unpool to subbuck slot that was assigned
        if (up_row < up_N) {
          auto & up_feat_elem = residual_feat[up_row * stride_featN + dim_feat_elem];

          float added_feat = feat_acc_reg + float(up_feat_elem);
          up_add_feat[up_row * stride_featN + dim_feat_elem] = FeatT(added_feat);
        }
      }
    }
  }
}


template <typename CounterT, typename FeatT>
__global__ void additive_unpool_bwd_ker(
  FeatT * __restrict__ grad_residual, FeatT * __restrict__ grad_down, const FeatT * __restrict__ grad_up_added,
  const CounterT * __restrict__ unpool_ind,
  uint32_t up_N, uint32_t down_N,
  uint32_t feat_dim, uint32_t subbuck_size,
  uint32_t stride_featN, uint32_t stride_unpoolN
  ) {
  // One warp per reduced row
  auto warps_per_block = blockDim.x / WarpSize;
  auto block_start = warps_per_block * blockIdx.x;
  auto warpid = tk::warpid();
  auto laneid = tk::laneid();

  auto global_down_row = block_start + warpid;
  // Mask out warps that are beyond reduced_N
  if (global_down_row >= down_N) return;

  float grad_acc_reg = .0f;

  auto num_loops = cdiv_dev(feat_dim, WarpSize);
  for (auto feat_iter = 0; feat_iter < num_loops; ++feat_iter) {
    // Start a feature stride for all subbucks. clear feature accumulators
    auto dim_feat_elem = feat_iter * WarpSize + laneid;
    grad_acc_reg = .0f;

    // Only unpool if lane_dim in [0, feat_dim]
    if (dim_feat_elem < feat_dim) {
      // Start of all subbuck grad accumulation
      for (auto sb_iter = 0; sb_iter < subbuck_size; ++sb_iter) {
        auto up_row = unpool_ind[global_down_row * stride_unpoolN + sb_iter];

        // Only aggregate and pass down gradients to subbuck slots that was assigned
        if (up_row < up_N) {
          auto & up_grad_elem = grad_up_added[up_row * stride_featN + dim_feat_elem];

          // Residual gradients are identity
          grad_residual[up_row * stride_featN + dim_feat_elem] = up_grad_elem;
          // Accumulate gradients to the down-sampled stream
          grad_acc_reg += float(up_grad_elem);
        }
      }

      // One gradient output per subbuck to the down-sampled stream, so outside the sub_iter loop
      grad_down[global_down_row * stride_featN + dim_feat_elem] = FeatT(grad_acc_reg);
    }
  }
}

} // end of ::f3d

#endif //FLASH3DPSHATTN_ADDTIVE_UNPOOL_H
