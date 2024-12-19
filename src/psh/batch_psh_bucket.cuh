/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 05/15/24
 */


#ifndef FLASH3DPSHATTN_BATCH_PSH_BUCKET_H
#define FLASH3DPSHATTN_BATCH_PSH_BUCKET_H

#include <stdexcept>
#include <glog/logging.h>

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <kittens.cuh>
#include "common/runtime.h"
#include "common/arithmetic.cuh"
#include "common/fp_types.cuh"
#include "common/rand.cuh"
#include "common/load_store.cuh"
#include "common/device_manager.cuh"
#include "psh/hash_fns.cuh"
#include "psh/two_stage_counter.cuh"

namespace tk = kittens;

namespace f3d {

const uint16_t num_warps = 32;
const uint16_t num_probs = 128;


// Template for forward-compatibility
template <typename CoordT, typename CounterT, typename HashT, uint16_t D,
  uint16_t NUM_BUCK, uint16_t TWO_STAGE_SEG, uint16_t RECYCLE_BUCK,
  uint16_t BLOCK_SIZE_N, BUCKETSTAGE BUCK_STAGE, HASHTYPE FIRST_HASH>
__global__ void batch_psh_bucketing_balancing_ker(
  const CoordT * __restrict__ coord, HashT * __restrict__ bucket_id, CounterT * __restrict__ bucket_counter,
  CounterT * __restrict__ bucket_offset, const CounterT * __restrict__ batch_sep, uint32_t batch_size,
  uint32_t total_N, uint32_t bucket_seg_divisor, uint32_t bucket_size, uint32_t num_vox,
  const CoordT * __restrict__ bbox_min, const CoordT * __restrict__ bbox_max, const int16_t * __restrict__ probe_offset,
  uint16_t stride_coordD, uint16_t stride_coordN,
  uint16_t stride_bucketid_N, uint16_t stride_bucket_cntB, uint16_t stride_bucket_cntN,
  uint16_t stride_bucketoff_N,
  curandState * global_rand_base
  ) {
  /*
   * coords: [N, D], fp16
   * bucket_id: [N], uint16
   * bucket_counter: [B, NB], uint32
   * bucket_offset: [N], uint32
   * batch_offset: [B], uint32
   * probe_offset: [32, D], int16, row-major
   *
   * Design heuristics: N = [100k, 400k], hash = [0, 65535], Bucket_size = [128, 4096], NB = [24, 3125]
   */
  auto threadid = threadIdx.x;
  auto warpid = tk::warpid();
  auto laneid = tk::laneid();

  auto block_start = blockIdx.x * blockDim.x;
  auto batch_ind = blockIdx.y;
  auto col_ind = block_start + threadid;
  auto batch_start = 0;

  if (batch_ind >= batch_size)
    return;

  if (batch_ind > 0)
    batch_start = batch_sep[batch_ind - 1];
  auto N = batch_sep[batch_ind] - batch_start;

  bool col_mask = (col_ind < N);
  //if (col_ind >= N || col_ind >= total_N)
  if ((col_ind >= N || col_ind >= total_N) && threadid >= NUM_BUCK)
    return;

  extern __shared__ tk::alignment_dummy __shm[];
  tk::shared_allocator al((int*) &__shm[0]);
  auto reusable_shmem_ptr = al.ptr;

  CoordT coord_reg[D];
  HashT vox_reg[D], bucketid_reg[1], newvox_reg[D];
  CounterT bucket_offset_reg[1];

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

  vox_coord<CoordT, HashT, D>(coord_reg, vox_reg, min_bb, max_bb, num_vox);

  if (FIRST_HASH == H_ZORDER_DIV)
    h1_zorder_div_3d<HashT>(vox_reg, bucketid_reg[0], bucket_seg_divisor);
  if (FIRST_HASH == H_ZORDER_MOD)
    h1_zorder_mod_3d(vox_reg, bucketid_reg[0], NUM_BUCK);
  if (FIRST_HASH == H_XORSUM_DIV)
    h1_xorsum_div<HashT, D>(vox_reg, bucketid_reg[0], bucket_seg_divisor);
  if (FIRST_HASH == H_XORSUM_MOD)
    h1_xorsum_mod<HashT, D>(vox_reg, bucketid_reg[0], NUM_BUCK);

  // bucket_counter are separate across the batch
  // bucket_counter base is bucket_counter + batch_ind * stride_bucket_cntB
  // For TwoStageCounter, the bucket_counter base is bucket_counter + batch_ind * stride_bucket_cntB
  int32_t assigned_bucket_id = -1;
  auto init_bucket_id = bucketid_reg[0];
  auto bucket_counter_lane = bucket_counter + batch_ind * stride_bucket_cntB + bucketid_reg[0];
  if (TWO_STAGE_SEG <= 1) {
    //val = 0xFFFFFFFF to bypass thresholding
    if (col_mask) {
      bucket_offset_reg[0] = atomicInc(bucket_counter_lane, 0xFFFFFFFF);
    }
  } else {
    CounterT * counter_shm = al.allocate<CounterT, NUM_BUCK>();
    TwoStageCounter<CounterT, NUM_BUCK, BLOCK_SIZE_N> two_stage{
      counter_shm, bucket_counter + batch_ind * stride_bucket_cntB, &al};
    bucket_offset_reg[0] = two_stage.atomic_inc_at(bucketid_reg[0], 0xFFFFFFFF, col_mask);
    two_stage.block_commit_and_revise(bucketid_reg[0], bucket_offset_reg, col_mask);
  }

  // memory barrier
  __syncthreads();

  // In batch mode, block_start is (block_start + batch_start)
  if (BUCK_STAGE == BUCKETSTAGE::S_INIT) {
    if (col_mask) {
      //bucketid_reg[0] = assigned_bucket_id; //only for debug
      bulk_store_bucket_id_off<HashT, CounterT, BLOCK_SIZE_N>(
        block_start + batch_start, bucket_id, bucket_offset, stride_bucketid_N, stride_bucketoff_N,
        bucketid_reg, bucket_offset_reg, &al);
    }
    return ;
  }

  // reuse shmem pointer
  al.ptr = reusable_shmem_ptr;

  // Prepare to rebalance
  // Seed choice: https://arxiv.org/abs/2109.08203
  LocalRandGen randgen(global_rand_base);
  // probe_offsets_shm: [128, D], Int16, row-major
  int16_t * probe_offsets_shm = al.allocate<16, int16_t, num_probs * D>();
  load_tile_coalesced<num_probs, D, BLOCK_SIZE_N>(
    probe_offset, probe_offsets_shm, 0u, D);

  // For !col_mask, assigned_bucket_id is still -1
  if (col_mask) {
    if ((bucket_offset_reg[0] < bucket_size) || bucketid_reg[0] == RECYCLE_BUCK) {
      assigned_bucket_id = bucketid_reg[0];
    } else {
      // Remove point from original bucket
      atomicDec(bucket_counter_lane, 0xFFFFFFFF);
    }
  }

  // memory barrier
  __syncthreads();

  CounterT * counter_shm = al.allocate<CounterT, NUM_BUCK>();
  TwoStageCounter<CounterT, NUM_BUCK, BLOCK_SIZE_N> two_stage{
    counter_shm, bucket_counter + batch_ind * stride_bucket_cntB, &al};
  bool is_attempted_replacement = false;
  constexpr auto SEG_SIZE = num_probs / TWO_STAGE_SEG;

  #pragma unroll
  for (auto i = 0; i < num_probs; ++i) {
    // If needs replacement and not attempted replacing yet
    if (col_mask && !is_attempted_replacement && assigned_bucket_id < 0 ) {
      randgen.perturb_vox<HashT, D>(
        2.56f, 5.12f, num_vox, bucket_seg_divisor, probe_offsets_shm, i, vox_reg, newvox_reg);

      if (FIRST_HASH == H_ZORDER_DIV)
        h1_zorder_div_3d<HashT>(newvox_reg, bucketid_reg[0], bucket_seg_divisor);
      if (FIRST_HASH == H_ZORDER_MOD)
        h1_zorder_mod_3d(newvox_reg, bucketid_reg[0], NUM_BUCK);
      if (FIRST_HASH == H_XORSUM_DIV)
        h1_xorsum_div<HashT, D>(newvox_reg, bucketid_reg[0], bucket_seg_divisor);
      if (FIRST_HASH == H_XORSUM_MOD)
        h1_xorsum_mod<HashT, D>(newvox_reg, bucketid_reg[0], NUM_BUCK);
    }

    bucket_counter_lane = bucket_counter + batch_ind * stride_bucket_cntB + bucketid_reg[0];

    bool optimistic_racing =
      col_mask &&
      !is_attempted_replacement &&
      assigned_bucket_id < 0 &&
      bucketid_reg[0] != init_bucket_id &&
      bucketid_reg[0] != RECYCLE_BUCK &&
      (*bucket_counter_lane) < bucket_size;

    if (optimistic_racing) {
      // Partial offset_reg, to be updated
      bucket_offset_reg[0] = two_stage.atomic_inc_at(bucketid_reg[0], 0xFFFFFFFF, col_mask);
      is_attempted_replacement = true;
    }

    // Bulk committing stage
    if (i + 1 % SEG_SIZE == 0 || i + 1 == num_probs) {
      // Get revised offset
      two_stage.block_commit_and_revise(bucketid_reg[0], &bucket_offset_reg[0], is_attempted_replacement);

      // Check bucket replacement succeeds if attempted replacement in the first place
      if (is_attempted_replacement && bucket_offset_reg[0] < bucket_size) {
        assigned_bucket_id = bucketid_reg[0];
      }

      // When optimistically raced and failed to find a new bucket, revert the transaction
      if (is_attempted_replacement && bucket_offset_reg[0] >= bucket_size) {
        // Backoff from optimistic racing
        atomicDec(bucket_counter_lane, 0xFFFFFFFF);
      }

      two_stage.flush_shmem_zero();
      is_attempted_replacement = false;
    }
  }

  // memory barrier
  __syncthreads();

  // Still failed :<
  if (assigned_bucket_id < 0 && col_mask) {
    assigned_bucket_id = RECYCLE_BUCK;
    bucket_counter_lane = bucket_counter + batch_ind * stride_bucket_cntB + assigned_bucket_id;
    bucket_offset_reg[0] = atomicInc(bucket_counter_lane, 0xFFFFFFFF);
  }

  if (BUCK_STAGE == BUCKETSTAGE::S_FINAL) {
    bucketid_reg[0] = assigned_bucket_id;

    if (col_mask) {
      bulk_store_bucket_id_off<HashT, CounterT, BLOCK_SIZE_N>(
        block_start + batch_start, bucket_id, bucket_offset, stride_bucketid_N, stride_bucketoff_N,
        bucketid_reg, bucket_offset_reg, &al);
    }
  }
}

template <typename CoordT, typename CounterT, typename HashT, uint16_t D,
  uint16_t NUM_BUCK, uint16_t RECYCLE_BUCK,
  uint16_t BLOCK_SIZE_N, HASHTYPE FIRST_HASH>
__global__ void batch_psh_distribute_recycle_ker(
  const CoordT * __restrict__ coord, HashT * __restrict__ bucket_id, CounterT * __restrict__ bucket_counter,
  CounterT * __restrict__ bucket_offset, const CounterT * __restrict__ batch_sep, uint32_t batch_size,
  uint32_t total_N, uint32_t bucket_seg_divisor, uint32_t bucket_size, uint32_t num_vox,
  const CoordT * __restrict__ bbox_min, const CoordT * __restrict__ bbox_max, const int16_t * __restrict__ probe_offset,
  uint16_t stride_coordD, uint16_t stride_coordN,
  uint16_t stride_bucket_cntB, uint16_t stride_bucket_cntN,
  curandState * global_rand_base
  ) {
  /*
   * coords: [N, D], fp16
   * bucket_id: [N], uint16
   * bucket_counter: [B, NB], uint32
   * bucket_offset: [N], uint32
   * probe_offset: [32, D], int16, row-major
   *
   * Design heuristics: N = [100k, 400k], hash = [0, 65535], Bucket_size = [128, 4096], NB = [24, 3125]
   */
  auto threadid = threadIdx.x;
  auto warpid = tk::warpid();
  auto laneid = tk::laneid();

  auto block_start = blockIdx.x * blockDim.x;
  auto batch_ind = blockIdx.y;
  auto col_ind = block_start + threadid;
  auto batch_start = 0;

  if (batch_ind >= batch_size)
    return;

  if (batch_ind > 0)
    batch_start = batch_sep[batch_ind - 1];
  auto N = batch_sep[batch_ind] - batch_start;
  auto global_col = batch_start + col_ind;

  bool col_mask = (col_ind < N);
  //if ((col_ind >= N || col_ind >= total_N) && threadid >= NUM_BUCK)
  if (col_ind >= N || global_col >= total_N)
    return;

  extern __shared__ tk::alignment_dummy __shm[];
  tk::shared_allocator al((int*) &__shm[0]);

  auto min_bb = al.allocate<CoordT, D>();
  auto max_bb = al.allocate<CoordT, D>();

  if (threadid < D) {
    min_bb[threadid] = bbox_min[threadid];
    max_bb[threadid] = bbox_max[threadid];
  }

  CoordT coord_reg[D];
  HashT vox_reg[D], bucketid_reg[1], newvox_reg[D], replace_bucketid_reg[1];
  CounterT bucket_offset_reg[1];

  using BucketidBlockLoad = cub::BlockLoad<HashT, BLOCK_SIZE_N, 1, cub::BLOCK_LOAD_VECTORIZE>;
  using BucketidBlockLoadTemp = BucketidBlockLoad::TempStorage;
  auto bucketid_block_load_temp = al.allocate<BucketidBlockLoadTemp>();
  auto bucketid_base = bucket_id + batch_start + block_start;
  BucketidBlockLoad(bucketid_block_load_temp).Load(bucketid_base, bucketid_reg);

  if (bucketid_reg[0] != RECYCLE_BUCK)
    return;

  using BucketoffBlockLoad = cub::BlockLoad<CounterT, BLOCK_SIZE_N, 1, cub::BLOCK_LOAD_VECTORIZE>;
  auto bucketoff_base = bucket_offset + batch_start + block_start;
  BucketoffBlockLoad().Load(bucketoff_base, bucket_offset_reg);
  //bucket_offset_reg[0] = *(bucket_offset + global_col);

  if (bucket_offset_reg[0] < bucket_size)
    return;

  // load of FP coords into reg and sync
  using CoordBlockLoad = cub::BlockLoad<CoordT, BLOCK_SIZE_N, D, cub::BLOCK_LOAD_VECTORIZE>;
  auto coord_base = coord + (batch_start + block_start) * stride_coordN;
  CoordBlockLoad().Load(coord_base, coord_reg);

  __syncthreads();

  auto init_bucket_id = bucketid_reg[0];
  vox_coord<CoordT, HashT, D>(coord_reg, vox_reg, min_bb, max_bb, num_vox);

  // Seed choice: https://arxiv.org/abs/2109.08203
  LocalRandGen randgen(global_rand_base);
  auto probe_ind = randgen.uniform_int(0, num_probs - 1);
  randgen.perturb_vox<HashT, D>(
    2.56f, 5.12f, num_vox, bucket_seg_divisor, probe_offset, probe_ind, vox_reg, newvox_reg);

  if (FIRST_HASH == H_ZORDER_DIV)
    h1_zorder_div_3d<HashT>(newvox_reg, bucketid_reg[0], bucket_seg_divisor);
  if (FIRST_HASH == H_ZORDER_MOD)
    h1_zorder_mod_3d(newvox_reg, bucketid_reg[0], NUM_BUCK);
  if (FIRST_HASH == H_XORSUM_DIV)
    h1_xorsum_div<HashT, D>(newvox_reg, bucketid_reg[0], bucket_seg_divisor);
  if (FIRST_HASH == H_XORSUM_MOD)
    h1_xorsum_mod<HashT, D>(newvox_reg, bucketid_reg[0], NUM_BUCK);

  if (init_bucket_id == RECYCLE_BUCK &&
      bucket_offset_reg[0] > bucket_size) {
    //val = 0xFFFFFFFF to bypass thresholding
    atomicDec(bucket_counter + batch_ind * stride_bucket_cntB + RECYCLE_BUCK, 0xFFFFFFFF);
    if (bucketid_reg[0] == RECYCLE_BUCK) {
      auto rand_buck = randgen.uniform_int(0, NUM_BUCK - 1);
      if (rand_buck == RECYCLE_BUCK && RECYCLE_BUCK < NUM_BUCK - 1) {
        rand_buck++;
      }
      if (rand_buck == RECYCLE_BUCK && RECYCLE_BUCK == NUM_BUCK - 1) {
        rand_buck--;
      }
      bucketid_reg[0] = rand_buck;
    }

    auto bucket_counter_lane = bucket_counter + batch_ind * stride_bucket_cntB + bucketid_reg[0];
    bucket_offset_reg[0] = atomicInc(bucket_counter_lane, 0xFFFFFFFF);

    using BucketBlockStore = cub::BlockStore<HashT, BLOCK_SIZE_N, 1, cub::BLOCK_STORE_VECTORIZE>;
    BucketBlockStore().Store(bucketid_base, bucketid_reg);
    using BucketoffBlockStore = cub::BlockStore<CounterT, BLOCK_SIZE_N, 1, cub::BLOCK_STORE_VECTORIZE>;
    BucketoffBlockStore().Store(bucketoff_base, bucket_offset_reg);
  }
}

template <typename CoordT, typename CounterT, typename HashT, uint16_t D, uint16_t N_bucket, uint16_t BLOCK_SIZE_N>
__global__ void batch_psh_cumsum_scatter_ker(
  const CoordT * __restrict__ coord, HashT * __restrict__ bucket_id, CounterT * __restrict__ bucket_counter,
  CounterT * __restrict__ bucket_offset, CounterT * __restrict__ cumsum_counter,
  const CounterT * __restrict__ batch_sep, uint32_t batch_size,
  CoordT * __restrict__ scattered_coord,
  uint32_t total_N, uint32_t bucket_seg_divisor, uint32_t bucket_size, uint32_t num_vox,
  uint16_t stride_coordD, uint16_t stride_coordN,
  uint16_t stride_bucket_cntB, uint16_t stride_bucket_cntN,
  uint16_t stride_cumsum_cntB, uint16_t stride_cumsum_cntN
  ) {
  /*
   * coords: [N, D], fp16
   * bucket_id: [N], uint16
   * bucket_counter: [B, NB], uint32
   * cumsum_counter: [B, NB], uint32
   * bucket_offset: [N], uint32
   * probe_offset: [32, D], int16, row-major
   *
   * Design heuristics: N = [100k, 400k], hash = [0, 65535], Bucket_size = [128, 4096], NB = [24, 3125]
   */
  auto threadid = threadIdx.x;
  auto warpid = tk::warpid();
  auto laneid = tk::laneid();

  auto block_start = blockIdx.x * blockDim.x;
  auto batch_ind = blockIdx.y;
  auto col_ind = block_start + threadid;
  auto batch_start = 0;

  constexpr long num_thread_cumsum = 32;
  constexpr long stride_cumsum = N_bucket / num_thread_cumsum;

  if (batch_ind >= batch_size)
    return;

  if (batch_ind > 0)
    batch_start = batch_sep[batch_ind - 1];
  auto N = batch_sep[batch_ind] - batch_start;
  auto global_col = batch_start + col_ind;

  bool col_mask = (col_ind < N);
  //if ((col_ind >= N || col_ind >= total_N) && threadid >= NUM_BUCK)
  if ((col_ind >= N || global_col >= total_N) && threadid >= num_thread_cumsum)
    return;

  extern __shared__ tk::alignment_dummy __shm[];
  tk::shared_allocator al((int*) &__shm[0]);

  CounterT (&cumsum_shm)[N_bucket] = al.allocate<16, CounterT, N_bucket>();
  auto reusable_shm_ptr = al.ptr;

  using block_load = cub::BlockLoad<
    CounterT, num_thread_cumsum, stride_cumsum, cub::BlockLoadAlgorithm::BLOCK_LOAD_VECTORIZE>;
  auto block_load_temp = al.allocate<typename block_load::TempStorage>();
  using block_cumsum = cub::BlockScan<
    CounterT, num_thread_cumsum, cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING_MEMOIZE>;
  auto block_cumsum_temp = al.allocate<typename block_cumsum::TempStorage>();
  using block_store = cub::BlockStore<
    CounterT, num_thread_cumsum, stride_cumsum, cub::BlockStoreAlgorithm::BLOCK_STORE_VECTORIZE>;
  auto block_store_temp = al.allocate<typename block_store::TempStorage>();

  auto bucket_start = bucket_counter + batch_ind * stride_bucket_cntB;
  auto cumsum_start = cumsum_counter + batch_ind * stride_cumsum_cntB;
  // Load counters from DRAM to Registers
  CounterT cumsum_reg[stride_cumsum];
  if (threadid < num_thread_cumsum) {
    block_load(block_load_temp).Load(bucket_start, cumsum_reg);
    block_cumsum(block_cumsum_temp).InclusiveSum(cumsum_reg, cumsum_reg);
    // Store cumsum base from Reg to shmem
    block_store(block_store_temp).Store(cumsum_shm, cumsum_reg);
    // Only for debug purpose
    block_store().Store(cumsum_start, cumsum_reg);
  }

  // Reclaim shmem
  __syncthreads();
  al.ptr = reusable_shm_ptr;

  CoordT coord_reg[D];
  HashT bucketid_reg[1];
  CounterT in_bucket_offset[1];

  if (col_mask) {
    using BucketOffsetBlockLoad = cub::BlockLoad<CounterT, BLOCK_SIZE_N, 1, cub::BLOCK_LOAD_VECTORIZE>;
    auto bucketoff_base = bucket_offset + (batch_start + block_start);
    BucketOffsetBlockLoad().Load(bucketoff_base, in_bucket_offset);

    // Block-wise load of FP coords into reg and sync
    using CoordBlockLoad = cub::BlockLoad<CoordT, BLOCK_SIZE_N, D, cub::BLOCK_LOAD_VECTORIZE>;
    auto coord_base = coord + (batch_start + block_start) * stride_coordN;
    CoordBlockLoad().Load(coord_base, coord_reg);

    using BucketidBlockLoad = cub::BlockLoad<HashT, BLOCK_SIZE_N, 1, cub::BLOCK_LOAD_VECTORIZE>;
    auto bucketid_base = bucket_id + (batch_start + block_start);
    BucketidBlockLoad().Load(bucketid_base, bucketid_reg);

    CounterT in_bucket_base_col = (bucketid_reg[0] == 0) ? 0u: cumsum_shm[bucketid_reg[0] - 1];
    CounterT instance_col = in_bucket_base_col + in_bucket_offset[0];
    auto col_base = scattered_coord + (batch_start + instance_col) * stride_coordN;

    #pragma unroll
    for (auto d = 0; d < D; ++d) {
      *(col_base + d * stride_coordD) = coord_reg[d];
    }
  }
}


template <typename CoordT, typename CounterT, typename HashT, uint16_t D, uint16_t N_bucket, uint16_t BLOCK_SIZE_N>
__global__ void batch_psh_cumsum_scatter_postpad_ker(
  const CoordT * __restrict__ coord, HashT * __restrict__ bucket_id, CounterT * __restrict__ bucket_counter,
  CounterT * __restrict__ bucket_offset, CounterT * __restrict__ cumsum_counter,
  const CounterT * __restrict__ batch_sep, uint32_t batch_size,
  CoordT * __restrict__ scattered_coord,
  uint32_t total_N, uint32_t bucket_seg_divisor, uint32_t bucket_size, uint32_t num_vox,
  uint16_t stride_coordD, uint16_t stride_coordN,
  uint16_t stride_bucket_cntB, uint16_t stride_bucket_cntN,
  uint16_t stride_cumsum_cntB, uint16_t stride_cumsum_cntN,
  uint32_t pad_to_N, CoordT pad_coord
  ) {
  /*
   * Launch **one more batch dimension** of blocks for padding the tailing columns!
   * coords: [N, D], fp16
   * bucket_id: [N], uint16
   * bucket_counter: [B, NB], uint32
   * cumsum_counter: [B, NB], uint32
   * bucket_offset: [N], uint32
   * probe_offset: [32, D], int16, row-major
   *
   * Design heuristics: N = [100k, 400k], hash = [0, 65535], Bucket_size = [128, 4096], NB = [24, 3125]
   */
  auto threadid = threadIdx.x;
  auto warpid = tk::warpid();
  auto laneid = tk::laneid();

  auto block_start = blockIdx.x * blockDim.x;
  auto batch_ind = blockIdx.y;
  auto col_ind = block_start + threadid;
  auto batch_start = 0;

  constexpr long num_thread_cumsum = 32;
  constexpr long stride_cumsum = N_bucket / num_thread_cumsum;

  // Need batch+1 dimension of blocks for tail padding
  // legitimate number of batches = batch_size + 1
  if (batch_ind > batch_size)
    return;

  if (batch_ind > 0)
    batch_start = batch_sep[batch_ind - 1];
  auto N = batch_sep[batch_ind] - batch_start;
  auto global_col = batch_start + col_ind;

  if (batch_ind == batch_size) {
    if (global_col < pad_to_N) {
      auto col_base = scattered_coord + global_col * stride_coordN;
      #pragma unroll
      for (auto d = 0; d < D; ++d) {
        col_base[d * stride_coordD] = pad_coord;
      }
    }
    return;
  }

  bool col_mask = (col_ind < N);
  //if ((col_ind >= N || col_ind >= total_N) && threadid >= NUM_BUCK)
  if ((col_ind >= N || global_col >= pad_to_N) && threadid >= num_thread_cumsum)
    return;

  extern __shared__ tk::alignment_dummy __shm[];
  tk::shared_allocator al((int*) &__shm[0]);

  CounterT (&cumsum_shm)[N_bucket] = al.allocate<16, CounterT, N_bucket>();
  auto reusable_shm_ptr = al.ptr;

  using block_load = cub::BlockLoad<
    CounterT, num_thread_cumsum, stride_cumsum, cub::BlockLoadAlgorithm::BLOCK_LOAD_VECTORIZE>;
  auto block_load_temp = al.allocate<typename block_load::TempStorage>();
  using block_cumsum = cub::BlockScan<
    CounterT, num_thread_cumsum, cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING_MEMOIZE>;
  auto block_cumsum_temp = al.allocate<typename block_cumsum::TempStorage>();
  using block_store = cub::BlockStore<
    CounterT, num_thread_cumsum, stride_cumsum, cub::BlockStoreAlgorithm::BLOCK_STORE_VECTORIZE>;
  auto block_store_temp = al.allocate<typename block_store::TempStorage>();

  auto bucket_start = bucket_counter + batch_ind * stride_bucket_cntB;
  auto cumsum_start = cumsum_counter + batch_ind * stride_cumsum_cntB;
  // Load counters from DRAM to Registers
  CounterT cumsum_reg[stride_cumsum];
  if (threadid < num_thread_cumsum) {
    block_load(block_load_temp).Load(bucket_start, cumsum_reg);
    block_cumsum(block_cumsum_temp).InclusiveSum(cumsum_reg, cumsum_reg);
    // Store cumsum base from Reg to shmem
    block_store(block_store_temp).Store(cumsum_shm, cumsum_reg);
    // Only for debug purpose
    block_store().Store(cumsum_start, cumsum_reg);
  }

  // Reclaim shmem
  __syncthreads();
  al.ptr = reusable_shm_ptr;

  CoordT coord_reg[D];
  HashT bucketid_reg[1];
  CounterT in_bucket_offset[1];

  if (col_mask) {
    using BucketOffsetBlockLoad = cub::BlockLoad<CounterT, BLOCK_SIZE_N, 1, cub::BLOCK_LOAD_VECTORIZE>;
    auto bucketoff_base = bucket_offset + (batch_start + block_start);
    BucketOffsetBlockLoad().Load(bucketoff_base, in_bucket_offset);

    // Block-wise load of FP coords into reg and sync
    using CoordBlockLoad = cub::BlockLoad<CoordT, BLOCK_SIZE_N, D, cub::BLOCK_LOAD_VECTORIZE>;
    auto coord_base = coord + (batch_start + block_start) * stride_coordN;
    CoordBlockLoad().Load(coord_base, coord_reg);

    using BucketidBlockLoad = cub::BlockLoad<HashT, BLOCK_SIZE_N, 1, cub::BLOCK_LOAD_VECTORIZE>;
    auto bucketid_base = bucket_id + (batch_start + block_start);
    BucketidBlockLoad().Load(bucketid_base, bucketid_reg);

    CounterT in_bucket_base_col = (bucketid_reg[0] == 0) ? 0u: cumsum_shm[bucketid_reg[0] - 1];
    CounterT instance_col = in_bucket_base_col + in_bucket_offset[0];
    auto col_base = scattered_coord + (batch_start + instance_col) * stride_coordN;

    #pragma unroll
    for (auto d = 0; d < D; ++d) {
      *(col_base + d * stride_coordD) = coord_reg[d];
    }
  }
}


template <typename CoordT, typename CounterT, typename HashT, typename FeatT,
uint16_t D, uint16_t N_bucket, uint16_t BLOCK_SIZE_N>
__global__ void batch_psh_cumsum_scatter_feat_index_postpad_ker(
  const CoordT * __restrict__ coord, HashT * __restrict__ bucket_id, CounterT * __restrict__ bucket_counter,
  CounterT * __restrict__ bucket_offset,
  const CounterT * __restrict__ batch_sep, uint32_t batch_size,
  const FeatT * __restrict__ feat,
  CoordT * __restrict__ scattered_coord,
  CounterT * __restrict__ scatter_idx_out2in,
  FeatT * __restrict__ scattered_feat,
  uint32_t total_N, uint32_t feat_dim,
  uint16_t stride_coordD, uint16_t stride_coordN,
  uint16_t stride_bucket_cntB, uint16_t stride_bucket_cntN,
  uint32_t stride_feat,
  uint32_t pad_to_N, CoordT pad_coord
  ) {
  /*
   * Launch **one more batch dimension** of blocks for padding the tailing columns!
   * coords: [N, D], fp16
   * feat: [N, F], fp16/bf16/fp32
   * bucket_id: [N], uint16
   * bucket_counter: [B, NB], uint32
   * cumsum_counter: [B, NB], uint32
   * bucket_offset: [N], uint32
   * scatter_idx_out2in: [N], uint32  (out-to-in index)
   * scattered_feat: [N, F], fp16/bf16/fp32
   *
   * Design heuristics: N = [100k, 400k], hash = [0, 65535], Bucket_size = [128, 4096], NB = [24, 3125]
   */
  auto threadid = threadIdx.x;
  auto warpid = tk::warpid();
  auto laneid = tk::laneid();

  auto block_start = blockIdx.x * blockDim.x;
  auto batch_ind = blockIdx.y;
  auto row_ind = block_start + threadid;
  auto batch_start = 0;

  constexpr long num_thread_cumsum = WarpSize;
  constexpr long stride_cumsum = N_bucket / num_thread_cumsum;

  // Need batch+1 dimension of blocks for tail padding
  // legitimate number of batches = batch_size + 1
  if (batch_ind > batch_size)
    return;

  if (batch_ind > 0)
    batch_start = batch_sep[batch_ind - 1];
  auto N = batch_sep[batch_ind] - batch_start;
  auto global_row = batch_start + row_ind;

  if (batch_ind == batch_size) {
    if (global_row < pad_to_N) {
      auto row_base = scattered_coord + global_row * stride_coordN;
      #pragma unroll
      for (auto d = 0; d < D; ++d) {
        row_base[d * stride_coordD] = pad_coord;
      }

      // Mask out the scatter index padding by setting padded index = total_N + 3
      scatter_idx_out2in[global_row] = total_N + 3;
    }
    return;
  }

  bool row_mask = (row_ind < N);
  //if ((col_ind >= N || col_ind >= total_N) && threadid >= NUM_BUCK)
  if ((row_ind >= N || global_row >= pad_to_N) && threadid >= num_thread_cumsum)
    return;

  extern __shared__ tk::alignment_dummy __shm[];
  tk::shared_allocator al((int*) &__shm[0]);

  CounterT (&cumsum_shm)[N_bucket] = al.allocate<16, CounterT, N_bucket>();
  auto reusable_shm_ptr = al.ptr;

  using block_load = cub::BlockLoad<
    CounterT, num_thread_cumsum, stride_cumsum, cub::BlockLoadAlgorithm::BLOCK_LOAD_VECTORIZE>;
  auto block_load_temp = al.allocate<typename block_load::TempStorage>();
  using block_cumsum = cub::BlockScan<
    CounterT, num_thread_cumsum, cub::BlockScanAlgorithm::BLOCK_SCAN_RAKING_MEMOIZE>;

  auto block_cumsum_temp = al.allocate<typename block_cumsum::TempStorage>();

  auto bucket_start = bucket_counter + batch_ind * stride_bucket_cntB;
  // Load counters from DRAM to Registers
  CounterT cumsum_reg[stride_cumsum];
  if (threadid < num_thread_cumsum) {
    block_load(block_load_temp).Load(bucket_start, cumsum_reg);
    block_cumsum(block_cumsum_temp).InclusiveSum(cumsum_reg, cumsum_reg);
  }

  // Reclaim shmem
  __syncthreads();
  al.ptr = reusable_shm_ptr;

  CoordT coord_reg[D];
  HashT bucketid_reg[1];
  CounterT in_bucket_offset[1];
  auto & tid2dst_global_shm = al.allocate<CounterT, BLOCK_SIZE_N>();

  if (row_mask) {
    using BucketOffsetBlockLoad = cub::BlockLoad<CounterT, BLOCK_SIZE_N, 1, cub::BLOCK_LOAD_VECTORIZE>;
    auto bucketoff_base = bucket_offset + (batch_start + block_start);
    BucketOffsetBlockLoad().Load(bucketoff_base, in_bucket_offset);

    // Block-wise load of FP coords into reg and sync
    using CoordBlockLoad = cub::BlockLoad<CoordT, BLOCK_SIZE_N, D, cub::BLOCK_LOAD_VECTORIZE>;
    auto coord_base = coord + (batch_start + block_start) * stride_coordN;
    CoordBlockLoad().Load(coord_base, coord_reg);

    using BucketidBlockLoad = cub::BlockLoad<HashT, BLOCK_SIZE_N, 1, cub::BLOCK_LOAD_VECTORIZE>;
    auto bucketid_base = bucket_id + (batch_start + block_start);
    BucketidBlockLoad().Load(bucketid_base, bucketid_reg);

    CounterT in_bucket_base_row = (bucketid_reg[0] == 0) ? 0u : cumsum_shm[bucketid_reg[0] - 1];
    CounterT instance_row = in_bucket_base_row + in_bucket_offset[0];
    auto dst_global_row = batch_start + instance_row;
    auto row_base = scattered_coord + dst_global_row * stride_coordN;

    // Write scattering index by setting the scatter_idx_out2in[batch_start + instance_col] = global_col
    scatter_idx_out2in[dst_global_row] = global_row;
    tid2dst_global_shm[threadid] = dst_global_row;

    #pragma unroll
    for (auto d = 0; d < D; ++d) {
      row_base[d * stride_coordD] = coord_reg[d];
    }
  }
  else {
    tid2dst_global_shm[threadid] = total_N + 3;
  }

  __syncthreads();


}


} // end of ::f3d

#endif //FLASH3DPSHATTN_BATCH_PSH_BUCKET_H