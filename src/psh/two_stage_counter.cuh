/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 06/30/24
 */


#ifndef FLASH3DPSHATTN_TWOSTAGE_H
#define FLASH3DPSHATTN_TWOSTAGE_H

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <kittens.cuh>

namespace tk = kittens;

namespace f3d {

template<typename CounterT, uint16_t N_bucket, uint16_t BLOCK_SIZE_N, int tk_alignment=16>
class TwoStageCounter {
public:
  // Always assume shmem counter starts from ZERO
  // So fetched dram values can be added
  __device__ __forceinline__ TwoStageCounter(
    CounterT * counter_shm_base, CounterT * counter_dram_base, tk::shared_allocator<tk_alignment> * al_ptr):
    counter_shm(counter_shm_base),
    counter_dram(counter_dram_base),
    al(al_ptr) {
    auto reusable_shm = al->ptr;

    // Local arena starts from zero
    if (threadIdx.x < N_bucket) {
      counter_reg[0] = 0u;

      using CounterBlockStore = cub::BlockStore<CounterT, BLOCK_SIZE_N, 1, cub::BLOCK_STORE_VECTORIZE>;
      using CounterBlockStoreTemp = CounterBlockStore::TempStorage;
      auto counter_block_store_temp = al->template allocate<CounterBlockStoreTemp>();
      CounterBlockStore(counter_block_store_temp).Store(counter_shm, counter_reg);
    }

    __syncthreads();
    al->ptr = reusable_shm;
  }

  __device__ __forceinline__ auto flush_shmem_zero() {
    auto reusable_shm = al->ptr;
    // Local arena starts from zero
    if (threadIdx.x < N_bucket) {
      counter_reg[0] = 0u;

      using CounterBlockStore = cub::BlockStore<CounterT, BLOCK_SIZE_N, 1, cub::BLOCK_STORE_VECTORIZE>;
      using CounterBlockStoreTemp = CounterBlockStore::TempStorage;
      auto counter_block_store_temp = al->template allocate<CounterBlockStoreTemp>();
      CounterBlockStore(counter_block_store_temp).Store(counter_shm, counter_reg);
    }

    __syncthreads();
    al->ptr = reusable_shm;
  }

  __device__ __forceinline__ auto atomic_inc_at(uint16_t ind, uint32_t val, bool col_mask) {
    if (col_mask) {
      return atomicInc_block(counter_shm + ind, val);
    } else {
      return *(counter_shm + ind);
    }
  }

  __device__ __forceinline__ auto atomic_dec_at(uint16_t ind, uint32_t val, bool col_mask) {
    if (col_mask) {
      return atomicDec_block(counter_shm + ind, val);
    } else {
      return *(counter_shm + ind);
    }
  }

  /*
   * shmem counters shouldn't start from non-zero values
  __device__ __forceinline__ auto load_global() {
    // Pre-flush: syncthreads() for all shmem transaction commited
    __syncthreads();
    auto reusable_ptr = al->ptr;
    if (threadIdx.x < N_bucket) {
      using CounterBlockLoad = cub::BlockLoad<CounterT, N_bucket, 1, cub::BLOCK_LOAD_VECTORIZE>;
      using CounterBlockLoadTemp = CounterBlockLoad::TempStorage;
      auto counter_block_load_temp = al->template allocate<CounterBlockLoadTemp>();
      CounterBlockLoad(counter_block_load_temp).Load(counter_dram, counter_reg);

      using CounterBlockStore = cub::BlockStore<CounterT, N_bucket, 1, cub::BLOCK_STORE_VECTORIZE>;
      using CounterBlockStoreTemp = CounterBlockStore::TempStorage;
      auto counter_block_store_temp = al->template allocate<CounterBlockStoreTemp>();
      CounterBlockStore(counter_block_store_temp).Store(counter_shm, counter_reg);
    }
    __syncthreads();
    al->ptr = reusable_ptr;
  }
   */

  __device__ __forceinline__ auto block_commit_and_revise(uint16_t ind, CounterT * old_val, bool is_revising) {
    auto reusable_shm = al->ptr;
    CounterT * counter_old_shm = al->template allocate<CounterT, N_bucket>();

    // Pre-commit: syncthreads() for all shmem transaction commited
    __syncthreads();
    // Pre-commit: Local shmem shard -> reg
    // Commit: reg +atomicAdd+ to global dram
    if (threadIdx.x < N_bucket) {
      using CounterBlockLoad = cub::BlockLoad<CounterT, N_bucket, 1, cub::BLOCK_LOAD_VECTORIZE>;
      using CounterBlockLoadTemp = CounterBlockLoad::TempStorage;
      auto counter_block_load_temp = al->template allocate<CounterBlockLoadTemp>();
      CounterBlockLoad(counter_block_load_temp).Load(counter_shm, counter_reg);

      CounterT old_dram[1];
      old_dram[0] = atomicAdd(counter_dram + threadIdx.x, counter_reg[0]);
      // Commit: old dram -> copy old shmem -> adding offset

      using CounterBlockStore = cub::BlockStore<CounterT, N_bucket, 1, cub::BLOCK_STORE_VECTORIZE>;
      using CounterBlockStoreTemp = CounterBlockStore::TempStorage;
      auto counter_block_store_temp = al->template allocate<CounterBlockStoreTemp>();
      CounterBlockStore(counter_block_store_temp).Store(counter_old_shm, old_dram);
    }

    // Post-commit: add incremental revisions to old values accordingly
    __syncthreads();

    if (is_revising) {
      (*old_val) += counter_old_shm[ind];
    }

    // Reclaim shmem
    al->ptr = reusable_shm;
    __syncthreads();
  }

private:
  CounterT *counter_shm, *counter_dram;
  CounterT counter_reg[1];
  tk::shared_allocator<tk_alignment> * al;
};

} // end of ::f3d


#endif //FLASH3DPSHATTN_TWOSTAGE_H