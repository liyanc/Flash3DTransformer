/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 10/18/24
 */


#ifndef FLASH3DPSHATTN_BATCH_PSH_PAIR_H
#define FLASH3DPSHATTN_BATCH_PSH_PAIR_H

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

enum SUBBUCK_CNT_SCOPE {
  S_LOCAL = 0,
  S_GLOBAL = 1
};

enum REDUCE_OP {
  O_MEAN = 0,
  O_SUM = 1,
  O_MIN = 2,
  O_MAX = 3
};

template <typename CoordT>
__device__ __forceinline__ void rand_quat(float * quat_shm, LocalRandGen & rand_gen) {
  auto tid = threadIdx.x;
  float q = .0f;
  // Init shmem
  if (tid < 5) {
    quat_shm[tid] = 0.f;
  }
  __syncthreads();

  // Init quaternion and normalizer
  if (tid < 4) {
    q = rand_gen.uniform_fp(0.1f, 1.0f);
    quat_shm[tid] = q;
    // Use quat_shm[4] as accumulator / normalizer
    atomicAdd_block(quat_shm + 4, q * q);
  }
  __syncthreads();

  // Normalize quaternion
  if (tid < 4) {
    quat_shm[tid] = quat_shm[tid] / sqrt(quat_shm[4]);
  }

  __syncthreads();
}

template <typename CoordT>
__device__ __forceinline__ void apply_quat_3d(float * quat_shm, CoordT (&in)[3], CoordT (&out)[3]) {
  float w = quat_shm[0], x = quat_shm[1], y = quat_shm[2], z = quat_shm[3];
  float px = in[0], py = in[1], pz = in[2];

  float num = x * 2.0f;
  float num2 = y * 2.0f;
  float num3 = z * 2.0f;
  float num4 = x * num;
  float num5 = y * num2;
  float num6 = z * num3;
  float num7 = x * num2;
  float num8 = x * num3;
  float num9 = y * num3;
  float num10 = w * num;
  float num11 = w * num2;
  float num12 = w * num3;

  out[0] = ((1.0f - (num5 + num6)) * px) + ((num7 - num12) * py) + ((num8 + num11) * pz);
  out[1] = ((num7 + num12) * px) + ((1.0f - (num4 + num6)) * py) + ((num9 - num10) * pz);
  out[2] = ((num8 - num11) * px) + ((num9 + num10) * py) + ((1.0f - (num4 + num5)) * pz);
}

template <typename CoordT, typename CounterT, typename HashT, uint16_t D,
  uint16_t BLOCK_SIZE_N, HASHTYPE FIRST_HASH, HASHTYPE SECOND_HASH, SUBBUCK_CNT_SCOPE CNT_SCOPE>
__global__ void local_subbuck_ker(
  const CoordT * __restrict__ coord, CounterT * __restrict__ subbuck_id, HashT * __restrict__ subbuck_offset,
  CounterT * __restrict__ subbuck_cnt, CounterT * __restrict__ reduced_batch_sep,
  const CounterT * __restrict__ batch_sep, CoordT * __restrict__ reduced_coord, CounterT * __restrict__ unpool_ind,
  uint32_t batch_size, uint32_t total_N, uint32_t subbuck_size, uint32_t num_vox,
  const CoordT * __restrict__ bbox_min, const CoordT * __restrict__ bbox_max,
  uint32_t stride_coordN,
  uint32_t stride_unpoolN,
  curandState * global_rand_base
  ) {
  /*
   * Tiny buckets for cluster-based pooling layer
   * coords: [N, D], fp16
   * bucket_id: [N], uint16
   * subbuck_id_cnt: [B], uint32, sub-bucket id counter, one global per instance
   * bucket_offset: [N], uint32
   * batch_offset: [B], uint32
   * unpool_ind: [RedN, Subbuck], uint32, inverse map for pooled rows, [rc, s] -> original row.
   * unpool_ind: Values > N indicate empty slots with no rows mapped
   *
   * Design heuristics: N = [100k, 400k], hash = [0, 65535], Bucket_size = [128, 4096], NB = [24, 3125]
   */
  auto threadid = threadIdx.x;

  constexpr uint16_t CSCSetSize = 8192;
  constexpr uint16_t CSCSegDiv = cdiv_dev(65536, CSCSetSize);

  auto block_start = blockIdx.x * blockDim.x;
  auto batch_ind = blockIdx.y;
  auto instance_row_ind = block_start + threadIdx.x;
  auto batch_start = 0;

  if (batch_ind >= batch_size)
    return;

  // Compute input instance starting row
  if (batch_ind > 0)
    batch_start = batch_sep[batch_ind - 1];
  auto N = batch_sep[batch_ind] - batch_start;

  // Compute reduced instance starting row
  auto red_batch_start = 0;
  if (batch_ind > 0)
    red_batch_start = reduced_batch_sep[batch_ind - 1];

  auto reduced_N = reduced_batch_sep[batch_ind] - red_batch_start;

  if (block_start >= N)
    return ;

  // If a block is completely out of bounds, exist all together
  // So block_size is non-negative
  auto block_size = min(BLOCK_SIZE_N, N - block_start);
  auto final_num_subbuck = cdiv_dev(block_size, subbuck_size);

  bool row_mask = (instance_row_ind < N && batch_ind < batch_size && block_start < N);
  //if (col_ind >= N || col_ind >= total_N)
  if ((instance_row_ind >= N || instance_row_ind >= total_N) && threadid >= CSCSetSize)
    return;

  extern __shared__ tk::alignment_dummy __shm[];
  tk::shared_allocator al((int *) &__shm[0]);
  auto reusable_shmem_ptr = al.ptr;

  CoordT coord_reg[D], rot_coord_reg[D], block_min[D], block_max[D];
  HashT vox_reg[D], subbuck_id_reg[1];
  CounterT subbuck_off_reg[1];
  // Seed choice: https://arxiv.org/abs/2109.08203
  LocalRandGen randgen(global_rand_base);

  auto min_bb = al.allocate<CoordT, D>();
  auto max_bb = al.allocate<CoordT, D>();

  if (threadid < D) {
    min_bb[threadid] = bbox_min[threadid];
    max_bb[threadid] = bbox_max[threadid];
  }

  using CoordBlockLoad = cub::BlockLoad<CoordT, BLOCK_SIZE_N, D, cub::BLOCK_LOAD_VECTORIZE>;
  using CoordBlockLoadTemp = CoordBlockLoad::TempStorage;
  auto coord_block_load_temp = al.allocate<CoordBlockLoadTemp>();
  // Block coord_base starts with batch+block
  auto coord_base = coord + (block_start + batch_start) * stride_coordN;
  CoordBlockLoad(coord_block_load_temp).Load(coord_base, coord_reg);

  __syncthreads();
  // Random rotation to coordinates
  float * quat_shm = al.allocate<float, 5>();
  rand_quat<CoordT>(quat_shm, randgen);
  apply_quat_3d(quat_shm, coord_reg, rot_coord_reg);

  // Rebound the bounding box for this block
  nd_reduce<CoordT, D, BLOCK_SIZE_N>(rot_coord_reg, block_min, block_max, &al);
  //float mean_bb = mean_bbox<CoordT, D>(block_min, block_max);
  vox_coord<CoordT, HashT, D>(rot_coord_reg, vox_reg, block_min, block_max, num_vox);

  if (FIRST_HASH == H_ZORDER_DIV)
    h1_zorder_div_3d<HashT>(vox_reg, subbuck_id_reg[0], CSCSegDiv);
  if (FIRST_HASH == H_ZORDER_MOD)
    h1_zorder_mod_3d(vox_reg, subbuck_id_reg[0], CSCSetSize);
  if (FIRST_HASH == H_XORSUM_DIV)
    h1_xorsum_div<HashT, D>(vox_reg, subbuck_id_reg[0], CSCSegDiv);
  if (FIRST_HASH == H_XORSUM_MOD)
    h1_xorsum_mod<HashT, D>(vox_reg, subbuck_id_reg[0], CSCSetSize);

  // Begin subbuck construction
  al.ptr = reusable_shmem_ptr;
  __syncthreads();

  CumsumCounter<CoordT, CounterT, D, CSCSetSize> csc(&al);

  if (row_mask) {
    subbuck_off_reg[0] = csc.atomic_add_subbuck(subbuck_id_reg[0]);
  }

  // Coalesce subbuckets
  __syncthreads();
  csc.cumsum_base_sync();
  if (!row_mask) {
    subbuck_id_reg[0] = 0;
  }
  uint32_t pt_ind = csc.get_subbuck_base(subbuck_id_reg[0]) + subbuck_off_reg[0];
  uint32_t new_buck_local = pt_ind / subbuck_size;
  uint32_t new_off = pt_ind % subbuck_size;

  csc.sync_reset_newsubbck(reusable_shmem_ptr, final_num_subbuck);
  if (row_mask) {
    csc.atomic_add_subbuck(new_buck_local);
  }

  __syncthreads();
  // Get subbuck base counter to locate the output rows
  auto subbuck_base = csc.commit_subbuck_cnt(subbuck_cnt + batch_ind, final_num_subbuck);

  __syncthreads();
  // Output subbuck id & offs for original input array, so no remapping
  auto global_row_ind = batch_start + instance_row_ind;
  uint32_t reduced_global_row = 0;
  if (row_mask) {
    if (CNT_SCOPE == S_GLOBAL) {
      subbuck_id[global_row_ind] = new_buck_local + subbuck_base;
    }
    if (CNT_SCOPE == S_LOCAL) {
      subbuck_id[global_row_ind] = new_buck_local;
    }
    subbuck_offset[global_row_ind] = new_off;

    // Output unpool inverse map
    reduced_global_row = red_batch_start + new_buck_local + subbuck_base;
    unpool_ind[reduced_global_row * stride_unpoolN + new_off] = global_row_ind;
  }

  // Reduce, zero-initialize
  //al.ptr = reusable_shmem_ptr;
  __syncthreads();
  constexpr uint16_t RedSlots = 512;
  auto (&red_coord_shm)[D][RedSlots] = al.allocate<float, D, RedSlots>();
  if (threadid < RedSlots) {
    for (uint16_t d = 0u; d < D; ++d) {
      red_coord_shm[d][threadid] = .0f;
    }
  }

  // Reduce, accumulate
  __syncthreads();
  if (row_mask) {
    for (uint16_t d = 0; d < D; ++d) {
      atomicAdd_block(&(red_coord_shm[d][new_buck_local]), coord_reg[d]);
    }
  }
  __syncwarp();

  // Reduce, normalize
  __syncthreads();
  if (threadid < RedSlots) {
    for (uint16_t d = 0; d < D; ++d) {
      red_coord_shm[d][threadid] /= float(max(csc.get_subbuck_cnt(threadid), 1));
    }
  }

  // Reduce, output
  __syncthreads();
  // subbuck_base is a batch-wise global variable
  uint32_t local_red_row = threadid + subbuck_base;
  uint32_t global_red_row = red_batch_start + local_red_row;
  uint32_t global_red_max = red_batch_start + final_num_subbuck + subbuck_base;
  if (threadid < RedSlots && local_red_row < reduced_N &&
      global_red_row < global_red_max && global_red_row >= red_batch_start) {
    for (uint16_t d = 0; d < D; ++d) {
      reduced_coord[global_red_row * stride_coordN + d] = CoordT(red_coord_shm[d][threadid]);
    }
  }
}

template <typename CounterT, typename FeatT>
__global__ void reduce_feat_ker(
  const FeatT * __restrict__ input_feat, FeatT * __restrict__ reduce_feat,
  const CounterT * __restrict__ unpool_ind,
  uint32_t total_N, uint32_t reduced_N,
  uint32_t feat_dim,
  uint32_t subbuck_size, REDUCE_OP red_op,
  uint32_t stride_featN, uint32_t stride_unpoolN
  ) {
  // One warp per reduced row
  auto warps_per_block = blockDim.x / WarpSize;
  auto block_start = warps_per_block * blockIdx.x;
  auto warpid = tk::warpid();
  auto laneid = tk::laneid();

  auto global_reduced_row = block_start + warpid;
  // Mask out warps that are beyond reduced_N
  if (global_reduced_row >= reduced_N) return;

  uint32_t cnt = 0u;
  float feat_acc = .0f;

  auto num_loops = cdiv_dev(feat_dim, WarpSize);
  for (auto feat_iter = 0; feat_iter < num_loops; ++feat_iter) {
    // Start a feature stride for all subbucks. clear subbuck counters
    auto dim_feat_elem = feat_iter * WarpSize + laneid;

    // Only reduce if lane_dim in [0, feat_dim]
    if (dim_feat_elem < feat_dim) {
      // Start of all subbuck reducing, clear counters and accumulators
      cnt = 0u;
      // Reset accumulators based on red_op
      if (O_SUM   == red_op) feat_acc = .0f;
      if (O_MEAN  == red_op) feat_acc = .0f;
      if (O_MIN   == red_op) feat_acc = INFINITY;
      if (O_MAX   == red_op) feat_acc = -INFINITY;

      for (auto sb_iter = 0; sb_iter < subbuck_size; ++sb_iter) {
        auto input_row = unpool_ind[global_reduced_row * stride_unpoolN + sb_iter];

        // Accumulate if the subbuck slot was assigned
        if (input_row < total_N) {
          ++cnt;
          auto & input_feat_elem = input_feat[input_row * stride_featN + dim_feat_elem];

          if (O_SUM   == red_op) { feat_acc += float(input_feat_elem);}
          if (O_MEAN  == red_op) { feat_acc += float(input_feat_elem);}
          if (O_MIN   == red_op) { feat_acc = min(feat_acc, float(input_feat_elem));}
          if (O_MAX   == red_op) { feat_acc = max(feat_acc, float(input_feat_elem));}
        }
      }

      // Finish all subbuck slots, final adjustment and output
      // One output per subbuck, so outside the sub_iter loop
      if (O_MEAN == red_op) {feat_acc /= float(cnt);}
      if (0 == cnt) {feat_acc = .0f;}
      reduce_feat[global_reduced_row * stride_featN + dim_feat_elem] = FeatT(feat_acc);
    }
  }
}
} // end of ::f3d

#endif //FLASH3DPSHATTN_BATCH_PSH_PAIR_H
