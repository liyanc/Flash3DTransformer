/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 05/15/24
 */


#include <chrono>
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
__global__ void psh_bucketing_balancing_ker(
  // coords [N, D]
  const CoordT * __restrict__ coord, HashT * __restrict__ bucket_id, CounterT * __restrict__ bucket_counter,
  CounterT * __restrict__ bucket_offset,
  uint32_t N, uint32_t bucket_seg_divisor, uint32_t bucket_size, uint32_t num_vox,
  const CoordT * __restrict__ bbox_min, const CoordT * __restrict__ bbox_max, const int16_t * __restrict__ probe_offset,
  uint16_t stride_coordD, uint16_t stride_coordN,
  uint16_t stride_bucketid_N, uint16_t stride_bucket_cntN,
  uint16_t stride_bucketoff_N,
  curandState * global_rand_base
  ) {
  /*
   * coords: [N, D], fp16
   * bucket_id: [N], uint16
   * bucket_counter: [NB], uint32
   * bucket_offset: [N], uint32
   * probe_offset: [32, D], int16, row-major
   *
   * Design heuristics: N = [100k, 400k], hash = [0, 65535], Bucket_size = [128, 4096], NB = [24, 3125]
   */
  auto threadid = threadIdx.x;
  auto warpid = tk::warpid();
  auto laneid = tk::laneid();
  auto block_start = blockIdx.x * blockDim.x;
  auto col_ind = block_start + threadid;

  if (col_ind >= N)
    return;

  extern __shared__ tk::alignment_dummy __shm[];
  tk::shared_allocator al((int*) &__shm[0]);
  auto reusable_shmem_ptr = al.ptr;

  using CoordBlockLoad = cub::BlockLoad<CoordT, BLOCK_SIZE_N, D, cub::BLOCK_LOAD_VECTORIZE>;
  using CoordBlockLoadTemp = CoordBlockLoad::TempStorage;
  auto coord_block_load_temp = al.allocate<CoordBlockLoadTemp>();

  CoordT coord_reg[D];
  HashT vox_reg[D], bucketid_reg[1], newvox_reg[D];
  CounterT bucket_offset_reg[1];

  auto min_bb = al.allocate<CoordT, D>();
  auto max_bb = al.allocate<CoordT, D>();

  if (threadid < D) {
    min_bb[threadid] = bbox_min[threadid];
    max_bb[threadid] = bbox_max[threadid];
  }

  auto coord_base = coord + block_start * stride_coordN;
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

  int32_t assigned_bucket_id = -1;
  auto init_bucket_id = bucketid_reg[0];
  auto bucket_counter_lane = bucket_counter + bucketid_reg[0];
  if (TWO_STAGE_SEG <= 1) {
    //val = 0xFFFFFFFF to bypass thresholding
    bucket_offset_reg[0] = atomicInc(bucket_counter_lane, 0xFFFFFFFF);
  } else {
    CounterT * counter_shm = al.allocate<CounterT, NUM_BUCK>();
    TwoStageCounter<CounterT, NUM_BUCK, BLOCK_SIZE_N> two_stage{counter_shm, bucket_counter, &al};
    bucket_offset_reg[0] = two_stage.atomic_inc_at(bucketid_reg[0], 0xFFFFFFFF, true);
    two_stage.block_commit_and_revise(bucketid_reg[0], bucket_offset_reg, true);
  }

  if (BUCK_STAGE == BUCKETSTAGE::S_INIT) {
    // memory barrier
    __syncthreads();

    bulk_store_bucket_id_off<HashT, CounterT, BLOCK_SIZE_N>(
      block_start, bucket_id, bucket_offset, stride_bucketid_N, stride_bucketoff_N,
      bucketid_reg, bucket_offset_reg, &al);
    return;
  }


  // reuse shmem pointer
  al.ptr = reusable_shmem_ptr;

  // Prepare to rebalance
  // Seed choice: https://arxiv.org/abs/2109.08203
  LocalRandGen randgen(global_rand_base, 3407);
  // probe_offsets_shm: [128, D], Int16, row-major
  auto (&probe_offsets_shm)[num_warps] = al.allocate<16, tk::sv<int16_t, 8 * D>, num_warps>();
  tk::load(probe_offsets_shm[warpid], probe_offset);

  if (bucket_offset_reg[0] < bucket_size || bucketid_reg[0] == RECYCLE_BUCK) {
    assigned_bucket_id = bucketid_reg[0];
  }
  // Begin re-balance
  else {
    // Remove point from original bucket
    atomicDec(bucket_counter_lane, 0xFFFFFFFF);
  }

  // memory barrier
  __syncthreads();

  if (assigned_bucket_id < 0) {
    #pragma  unroll
    for (auto i = 0; i < num_probs; ++i) {
      randgen.perturb_vox<HashT, D>(
        2.56f, 5.12f, num_vox, bucket_seg_divisor, probe_offsets_shm[warpid].data, i, vox_reg, newvox_reg);

      // Get new bucket candidate in `replace_bucket_id`
      h1_zorder_div_3d<HashT>(newvox_reg, bucketid_reg[0], bucket_seg_divisor);
      auto replace_bucket_id = bucketid_reg[0];
      bucket_counter_lane = bucket_counter + replace_bucket_id;

      // Optimistic atomicInc instead of atomicCAS
      bool optimistic_racing = false;
      if (replace_bucket_id != init_bucket_id &&
          (*bucket_counter_lane) < bucket_size &&
          replace_bucket_id != RECYCLE_BUCK) {
        // val = 0xFFFFFFFF to bypass thresholding
        bucket_offset_reg[0] = atomicInc(bucket_counter_lane, 0xFFFFFFFF);
        optimistic_racing = true;
      }

      // Check bucket replacement succeeds if optimistically racing in the first place
      if (optimistic_racing && bucket_offset_reg[0] < bucket_size) {
        assigned_bucket_id = replace_bucket_id;
        break;
      }

      // When optimistically raced and failed to find a new bucket, revert the transaction
      if (optimistic_racing && bucket_offset_reg[0] >= bucket_size){
        // Backoff from optimistic racing
        atomicDec(bucket_counter_lane, 0xFFFFFFFF);
      }
    }
  }

  // memory barrier
  __syncthreads();

  /*
  if (assigned_bucket_id < 0) {
    auto prob_ind = randgen.uniform_int(0, num_probs - 1);
    randgen.perturb_vox<HashT, D>(
      2.56f, 5.12f, num_vox, bucket_seg_divisor, probe_offsets_shm[warpid].data, prob_ind, vox_reg, newvox_reg);

    // Get new bucket candidate in `replace_bucket_id`
    h1_xorsum<HashT, D>(newvox_reg, bucketid_reg[0], bucket_seg_divisor);
    auto replace_bucket_id = bucketid_reg[0];
    bucket_counter_lane = bucket_counter + replace_bucket_id;

    // Optimistic atomicInc instead of atomicCAS
    bool optimistic_racing = false;
    if (replace_bucket_id != init_bucket_id &&
        (*bucket_counter_lane) < bucket_size &&
        replace_bucket_id != RECYCLE_BUCK) {
      // val = 0xFFFFFFFF to bypass thresholding
      bucket_offset_reg[0] = atomicInc(bucket_counter_lane, 0xFFFFFFFF);
      optimistic_racing = true;
    }

    // Check bucket replacement succeeds if optimistically racing in the first place
    if (optimistic_racing && bucket_offset_reg[0] < bucket_size) {
      assigned_bucket_id = replace_bucket_id;
    }

    // When optimistically raced and failed to find a new bucket, revert the transaction
    if (optimistic_racing && bucket_offset_reg[0] >= bucket_size){
      // Backoff from optimistic racing
      atomicDec(bucket_counter_lane, 0xFFFFFFFF);
    }
  }
   */

  // Still failed :<
  if (assigned_bucket_id < 0) {
    assigned_bucket_id = RECYCLE_BUCK;
    bucket_offset_reg[0] = atomicInc(bucket_counter + assigned_bucket_id, 0xFFFFFFFF);
  }


  if (BUCK_STAGE == BUCKETSTAGE::S_FINAL) {
    bucketid_reg[0] = assigned_bucket_id;

    bulk_store_bucket_id_off<HashT, CounterT, BLOCK_SIZE_N>(
      block_start, bucket_id, bucket_offset, stride_bucketid_N, stride_bucketoff_N,
      bucketid_reg, bucket_offset_reg, &al);
  }
}

template <typename CoordT, typename CounterT, typename HashT, uint16_t D,
  uint16_t NUM_BUCK, uint16_t RECYCLE_BUCK, uint16_t RECYCLE_BUCK_TWO,
  uint16_t BLOCK_SIZE_N, HASHTYPE FIRST_HASH>
__global__ void psh_distribute_recycle_ker(
  // coords [N, D]
  const CoordT * __restrict__ coord, HashT * __restrict__ bucket_id, CounterT * __restrict__ bucket_counter,
  CounterT * __restrict__ bucket_offset,
  uint32_t N, uint32_t bucket_seg_divisor, uint32_t bucket_size, uint32_t num_vox,
  const CoordT * __restrict__ bbox_min, const CoordT * __restrict__ bbox_max, const int16_t * __restrict__ probe_offset,
  uint16_t stride_coordD, uint16_t stride_coordN,
  curandState * global_rand_base
  ) {
  /*
   * coords: [N, D], fp16
   * bucket_id: [N], uint16
   * bucket_counter: [NB], uint32
   * bucket_offset: [N], uint32
   * probe_offset: [32, D], int16, row-major
   *
   * Design heuristics: N = [100k, 400k], hash = [0, 65535], Bucket_size = [128, 4096], NB = [24, 3125]
   */
  auto threadid = threadIdx.x;
  auto warpid = tk::warpid();
  auto laneid = tk::laneid();
  auto block_start = blockIdx.x * blockDim.x;
  auto col_ind = block_start + threadid;

  if (col_ind >= N)
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
  auto bucketid_base = bucket_id + block_start;
  BucketidBlockLoad(bucketid_block_load_temp).Load(bucketid_base, bucketid_reg);

  if (bucketid_reg[0] != RECYCLE_BUCK)
    return;

  bucket_offset_reg[0] = *(bucket_offset + col_ind);

  if (bucket_offset_reg[0] < bucket_size)
    return;

  // load of FP coords into reg and sync
  #pragma unroll
  for (auto i = 0; i < D; ++i) {
    coord_reg[i] = *(coord + col_ind * D + i);
  }

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
    atomicDec(bucket_counter + RECYCLE_BUCK, 0xFFFFFFFF);
    if (bucketid_reg[0] == RECYCLE_BUCK)
      bucketid_reg[0] = RECYCLE_BUCK_TWO;
    bucket_offset_reg[0] = atomicInc(bucket_counter + bucketid_reg[0], 0xFFFFFFFF);

    *(bucket_id + col_ind) = bucketid_reg[0];
    *(bucket_offset + col_ind) = bucket_offset_reg[0];
  }
}

// A new kernel to coalesce buckets to a linear array
template <typename CoordT, typename CounterT, typename HashT, uint16_t D, uint16_t N_bucket, uint16_t BLOCK_SIZE_N>
__global__ void psh_cumsum_scatter_ker(
  // coords [N, D]
  const CoordT * __restrict__ coord, HashT * __restrict__ bucket_id, CounterT * __restrict__ bucket_counter,
  CounterT * __restrict__ bucket_offset, CounterT * __restrict__ cumsum_counter,
  CoordT * __restrict__ scattered_coord,
  uint32_t N, uint32_t bucket_seg_divisor, uint32_t bucket_size, uint32_t num_vox,
  uint16_t stride_coordD, uint16_t stride_coordN,
  uint16_t stride_bucket_idN, uint16_t stride_bucket_cntN,
  uint16_t stride_bucket_offsetNB
  ) {
  /*
   * coords: [N, D], fp16
   * bucket_id: [N], uint16
   * bucket_counter: [NB], uint32
   * bucket_offset: [N], uint32
   *
   * Design heuristics: N = [100k, 400k], hash = [0, 65535], Bucket_size = [128, 4096], NB = [24, 3125]
   */
  auto threadid = threadIdx.x;
  auto warpid = tk::warpid();
  auto laneid = tk::laneid();
  auto block_start = blockIdx.x * blockDim.x;
  auto col_ind = block_start + threadid;

  if (col_ind >= N)
    return;

  extern __shared__ tk::alignment_dummy __shm[];
  tk::shared_allocator al((int*) &__shm[0]);

  constexpr long num_thread_cumsum = 32;
  constexpr long stride_cumsum = N_bucket / num_thread_cumsum;

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

  // Load counters from DRAM to Registers
  CounterT cumsum_reg[stride_cumsum];
  if (threadid < num_thread_cumsum) {
    block_load(block_load_temp).Load(bucket_counter, cumsum_reg);
    block_cumsum(block_cumsum_temp).InclusiveSum(cumsum_reg, cumsum_reg);
    // Store cumsum base from Reg to shmem
    block_store(block_store_temp).Store(cumsum_shm, cumsum_reg);
  }

  // Reclaim shmem
  __syncthreads();
  al.ptr = reusable_shm_ptr;

  CoordT coord_reg[D];
  HashT bucketid_reg[1];
  CounterT in_bucket_offset[1];

  using BucketOffsetBlockLoad = cub::BlockLoad<CounterT, BLOCK_SIZE_N, 1, cub::BLOCK_LOAD_VECTORIZE>;
  using BucketOffsetBlockLoadTemp = BucketOffsetBlockLoad::TempStorage;
  auto bucketoffset_block_load_temp = al.allocate<BucketOffsetBlockLoadTemp>();
  auto bucketoff_base = bucket_offset + block_start * stride_bucket_offsetNB;
  BucketOffsetBlockLoad(bucketoffset_block_load_temp).Load(bucketoff_base, in_bucket_offset);

  // Block-wise load of FP coords into reg and sync
  using CoordBlockLoad = cub::BlockLoad<CoordT, BLOCK_SIZE_N, D, cub::BLOCK_LOAD_VECTORIZE>;
  using CoordBlockLoadTemp = CoordBlockLoad::TempStorage;
  auto coord_block_load_temp = al.allocate<CoordBlockLoadTemp>();
  auto coord_base = coord + block_start * stride_coordN;
  CoordBlockLoad(coord_block_load_temp).Load(coord_base, coord_reg);

  using BucketidBlockLoad = cub::BlockLoad<HashT, BLOCK_SIZE_N, 1, cub::BLOCK_LOAD_VECTORIZE>;
  using BucketidBlockLoadTemp = BucketidBlockLoad::TempStorage;
  auto bucketid_block_load_temp = al.allocate<BucketidBlockLoadTemp>();
  auto bucketid_base = bucket_id + block_start * stride_bucket_idN;
  BucketidBlockLoad(bucketid_block_load_temp).Load(bucketid_base, bucketid_reg);

  CounterT in_bucket_base = (bucketid_reg[0] == 0) ? 0u: cumsum_shm[bucketid_reg[0] - 1];
  CounterT final_ind = in_bucket_base + in_bucket_offset[0];
  auto col_base = scattered_coord + final_ind * stride_coordN;

  #pragma unroll
  for (auto d = 0; d < D; ++d) {
    *(col_base + d * stride_coordD) = coord_reg[d];
  }
}

extern void
psh_scatter(at::Tensor &coord, at::Tensor &bucket_id, at::Tensor &bucket_counter, at::Tensor &bucket_offset,
            uint32_t num_bucket, uint32_t bucket_divisor, uint32_t bucket_size, uint32_t num_vox,
            at::Tensor &bbox_min, at::Tensor &bbox_max, at::Tensor &probe_offsets, at::Tensor &cumsum_counter,
            at::Tensor &scatter_coord, py::object hash_dtype_obj, py::object coord_dtype_obj,
            uint16_t block_size_N) {

  auto hash_dtype = torch::python::detail::py_object_to_dtype(hash_dtype_obj);
  auto coord_dtype = torch::python::detail::py_object_to_dtype(coord_dtype_obj);
  auto N = coord.size(0);
  LOG(WARNING) << "NT, NBlock " << N << " " << cdiv(N, block_size_N) << " Buck_div " << bucket_divisor;
  LOG(WARNING) << "N_Buck=" << num_bucket << " Buck_size=" << bucket_size;
  LOG(WARNING) << "Probe shape=[" << probe_offsets.size(0) << "," << probe_offsets.size(1) << "]";
  LOG(WARNING) << "Probe stride: " << probe_offsets.stride(0) << ", " << probe_offsets.stride(1);

  if (num_vox / bucket_divisor > num_bucket) {
    throw std::runtime_error("bucket divisor too small");
  }

  using CoordT = fp16;
  using CounterT = uint32_t;
  using HashT = uint16_t;
  constexpr auto BLOCK_SIZE_N = 1024u;
  auto stream = c10::cuda::getCurrentCUDAStream(coord.get_device());
  auto shmem_size = tk::MAX_SHARED_MEMORY;
  constexpr auto store_stage = S_FINAL;
  constexpr auto hash_type = H_ZORDER_MOD;
  constexpr auto special_buck = 0;
  constexpr auto recycle_buck_two = 1;
  constexpr auto two_stage_seg = 8;

  if (num_bucket == 256) {
    constexpr auto num_buck = 256;
    auto balance_ptr = psh_bucketing_balancing_ker<CoordT, CounterT, HashT, 3, num_buck, two_stage_seg, special_buck, BLOCK_SIZE_N, store_stage, hash_type>;
    auto recycle_ptr = psh_distribute_recycle_ker<CoordT, CounterT, HashT, 3, num_buck, special_buck, recycle_buck_two, BLOCK_SIZE_N, hash_type>;
    auto scatter_ptr = psh_cumsum_scatter_ker<CoordT, CounterT, HashT, 3, 256, BLOCK_SIZE_N>;

    cudaFuncSetAttribute(balance_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
    cudaCheckLastErr();
    cudaFuncSetAttribute(recycle_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
    cudaCheckLastErr();
    cudaFuncSetAttribute(scatter_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
    cudaCheckLastErr();

    GlobalRandState global_rand(BLOCK_SIZE_N, 0);
    QuickTimer timer{};

    balance_ptr<<<cdiv(N, block_size_N), block_size_N, shmem_size, stream>>>(
      (CoordT *) coord.data_ptr(), (HashT *) bucket_id.data_ptr(), (CounterT *) bucket_counter.data_ptr(),
      (CounterT *) bucket_offset.data_ptr(),
      N, bucket_divisor, bucket_size, num_vox,
      (CoordT *) bbox_min.data_ptr(), (CoordT *) bbox_max.data_ptr(), (int16_t *) probe_offsets.data_ptr(),
      coord.stride(1), coord.stride(0),
      bucket_id.stride(0), bucket_counter.stride(0),
      bucket_offset.stride(0),
      global_rand.get_states()
      );
    cudaCheckLastErr();
    recycle_ptr<<<cdiv(N, block_size_N), block_size_N, shmem_size, stream>>>(
      (CoordT *) coord.data_ptr(), (HashT *) bucket_id.data_ptr(), (CounterT *) bucket_counter.data_ptr(),
      (CounterT *) bucket_offset.data_ptr(),
      N, bucket_divisor, bucket_size, num_vox,
      (CoordT *) bbox_min.data_ptr(), (CoordT *) bbox_max.data_ptr(), (int16_t *) probe_offsets.data_ptr(),
      coord.stride(1), coord.stride(0),
      global_rand.get_states()
      );
    cudaCheckLastErr();
    scatter_ptr<<<cdiv(N, block_size_N), block_size_N, shmem_size, stream>>>(
      (CoordT *) coord.data_ptr(), (HashT *)bucket_id.data_ptr(), (CounterT *) bucket_counter.data_ptr(),
      (CounterT *)bucket_offset.data_ptr(), (CounterT *)cumsum_counter.data_ptr(), (CoordT *)scatter_coord.data_ptr(),
      N, bucket_divisor, bucket_size, num_vox,
      coord.stride(1), coord.stride(0),
      bucket_id.stride(0), bucket_counter.stride(0),
      bucket_offset.stride(0)
      );
    cudaCheckLastErr();


    cudaCheckErr(cudaStreamSynchronize(stream));
    auto [number, str] = timer.end_and_format<std::chrono::microseconds>();
    LOG(WARNING) << "Time: " << str;
  }

}


} // end of ::f3d
