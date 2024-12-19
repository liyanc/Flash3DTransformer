/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 10/23/24
 */

#ifndef FLASH3DXFMR_BATCH_PSH_SUBBUCK_REDUCE_BWD_H
#define FLASH3DXFMR_BATCH_PSH_SUBBUCK_REDUCE_BWD_H

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
#include "psh/batch_psh_subbuck_reduce.cuh"

namespace f3d {

template <typename CounterT, typename FeatT>
__global__ void reduce_feat_backward_ker(
  const FeatT * __restrict__ grad_red_feat,    // [reduced_N, feat_dim] gradient from forward output
  const FeatT * __restrict__ input_feat,      // [total_N, feat_dim] original input features
  const CounterT * __restrict__ unpool_ind,   // [reduced_N, subbuck_size] unpool indices from forward
  FeatT * __restrict__ grad_input_feat,          // [total_N, feat_dim] output input gradients
  uint32_t total_N, uint32_t reduced_N,
  uint32_t feat_dim,
  uint32_t subbuck_size, REDUCE_OP red_op,
  uint32_t stride_featN, uint32_t stride_unpoolN
) {
  // Each warp in a block processes one reduced row.
  // Global reduced row index: global_reduced_row = warps_per_block * blockIdx.x + tk::warpid()
  auto warps_per_block = blockDim.x / WarpSize;
  auto block_start = warps_per_block * blockIdx.x;
  auto warpid = tk::warpid();
  auto laneid = tk::laneid();

  auto global_reduced_row = block_start + warpid;
  if (global_reduced_row >= reduced_N) return;

  // Loop over feature dimensions; each lane processes a subset of feat_dim.
  auto num_loops = cdiv_dev(feat_dim, WarpSize);
  for (uint32_t feat_iter = 0; feat_iter < num_loops; ++feat_iter) {
    uint32_t dim_feat_elem = feat_iter * WarpSize + laneid;
    uint32_t cnt = 0;
    uint32_t input_row = total_N + 3;
    uint32_t subbuck_activated = 0u;

    if (dim_feat_elem < feat_dim) {
      // For O_SUM and O_MEAN, distribute the gradient to each valid input.
      if (O_SUM == red_op || O_MEAN == red_op) {
        // Count valid indices in the subbucket.
        cnt = 0;
        for (uint32_t sb_iter = 0; sb_iter < subbuck_size; ++sb_iter) {
          // Given unpool_ind covers [0, total_N-1] with no duplicates.
          input_row = unpool_ind[global_reduced_row * stride_unpoolN + sb_iter];
          if (input_row < total_N) {
            ++cnt;
          }
        }
        // Retrieve the gradient for this feature dimension.
        float grad = grad_red_feat[global_reduced_row * stride_featN + dim_feat_elem];
        // For O_MEAN, scale the gradient by the count.
        if (O_MEAN == red_op) {
          grad = grad / float(max(cnt, 1));
        }
        // Assign the gradient to each corresponding input feature.
        for (uint32_t sb_iter = 0; sb_iter < subbuck_size; ++sb_iter) {
          input_row = unpool_ind[global_reduced_row * stride_unpoolN + sb_iter];
          if (input_row < total_N){
            grad_input_feat[input_row * stride_featN + dim_feat_elem] = FeatT(grad);
          }
        }
      }

      if (O_MIN == red_op || O_MAX == red_op) {
        // For O_MIN/O_MAX, recompute the candidate value from input features.
        float candidate = (O_MIN == red_op) ? INFINITY : -INFINITY;
        // First pass: compute the candidate value.
        for (uint32_t sb_iter = 0; sb_iter < subbuck_size; ++sb_iter) {
          input_row = unpool_ind[global_reduced_row * stride_unpoolN + sb_iter];
          if (input_row < total_N) {
            float val = float(input_feat[input_row * stride_featN + dim_feat_elem]);
            bool is_winner = (O_MAX == red_op) ? (val > candidate) : (val < candidate);
            if (is_winner) {
              candidate = val;
              subbuck_activated = sb_iter;
            }
          }
        }
        // Retrieve the gradient from the reduced feature.
        auto & grad = grad_red_feat[global_reduced_row * stride_featN + dim_feat_elem];
        // Second pass: assign the gradient to the first matching input.
        for (uint32_t sb_iter = 0; sb_iter < subbuck_size; ++sb_iter) {
          input_row = unpool_ind[global_reduced_row * stride_unpoolN + sb_iter];
          if (input_row < total_N) {
            auto & val = input_feat[input_row * stride_featN + dim_feat_elem];
            if (sb_iter == subbuck_activated) {
              grad_input_feat[input_row * stride_featN + dim_feat_elem] = grad;
            } else {
              grad_input_feat[input_row * stride_featN + dim_feat_elem] = FeatT(.0f);
            }
          }
        }
      }
    }
  }
}

} // end of ::f3d

#endif //FLASH3DXFMR_BATCH_PSH_SUBBUCK_REDUCE_BWD_H
