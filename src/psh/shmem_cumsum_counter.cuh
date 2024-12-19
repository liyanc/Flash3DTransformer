/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 9/28/24
 */


#ifndef FLASH3DPSHATTN_SHMEM_CUMSUM_COUNTER_CUH
#define FLASH3DPSHATTN_SHMEM_CUMSUM_COUNTER_CUH

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>

#include <kittens.cuh>

#include "common/arithmetic.cuh"

namespace tk = kittens;

namespace f3d {
template <typename CoordT, typename CounterT, uint16_t D, uint16_t SetSize, int tk_align=16>
class CumsumCounter {
public:
  __device__ __forceinline__ CumsumCounter(tk::shared_allocator<tk_align> * al_p) :
    al_ptr{al_p} {
    constexpr uint16_t NumRounds = cdiv_dev(SetSize, 1024);

    subbuck_cnt_sram = al_p->template allocate<uint32_t, SetSize>();
    subbuck_base_sram = al_p->template allocate<uint16_t, SetSize>();

    for (uint16_t r = 0; r < NumRounds; ++r) {
      uint16_t logical_tid = r * 1024 + threadIdx.x;

      if (logical_tid < SetSize) {
        subbuck_cnt_sram[logical_tid] = 0;
        subbuck_base_sram[logical_tid] = 0;
      }
    }
    __syncthreads();
  }

  __device__ __forceinline__ void sync_reset_newsubbck(int* base_shm, uint16_t num_newsubbuck) {
    __syncthreads();
    al_ptr->ptr = base_shm;
    subbuck_cnt_sram = al_ptr->template allocate<uint32_t, 512>();
    if (threadIdx.x < num_newsubbuck) {
      subbuck_cnt_sram[threadIdx.x] = 0;
    }
    __syncthreads();
  }

  __device__ __forceinline__ uint32_t commit_subbuck_cnt(
    uint32_t * global_subbuck_cnt, uint32_t num_newsubbuck) {
    /*
     * Commit the number of sub-buckets from the current block to the global counters
     * Return the base offset for the current block to output sub-buckets
     */
    auto shm_base = al_ptr->ptr;
    auto subbuck_cnt_base = al_ptr->template allocate<uint32_t, 1>();
    if (threadIdx.x == 0) {
      *subbuck_cnt_base = atomicAdd(global_subbuck_cnt, num_newsubbuck);
      // atomicMax(global_max_inst, *subbuck_cnt_base + num_newsubbuck);
    }
    __syncthreads();
    auto res = *subbuck_cnt_base;
    al_ptr->ptr = shm_base;
    return res;
  }

  __device__ __forceinline__ uint32_t dbg_subbuck_overflow(uint16_t subbuck_size) {
    constexpr uint16_t NumRounds = cdiv_dev(SetSize, 1024);
    auto shm_base = al_ptr->ptr;
    uint32_t * over_cnt = al_ptr->template allocate<uint32_t, 1>();
    *over_cnt = 0;
    __syncthreads();

    for (uint16_t r = 0; r < NumRounds; ++r) {
      uint16_t logical_tid = r * 1024 + threadIdx.x;

      if (logical_tid < SetSize) {
        auto cnt = subbuck_cnt_sram[logical_tid];
        if (cnt >= subbuck_size) {
          atomicInc_block(over_cnt, 0xFFFFFFFF);
        }
      }
    }
    __syncthreads();
    auto res = *over_cnt;
    al_ptr->ptr = shm_base;
    return res;
  }

  __device__ __forceinline__ uint32_t atomic_add_subbuck(uint16_t subbuck_ind) {
    if (subbuck_ind >= SetSize) {
      printf("OverSubbuck=%d ", subbuck_ind);
    }
    auto subbuck_cnt_ptr = subbuck_cnt_sram + subbuck_ind;
    return atomicInc_block(subbuck_cnt_ptr, 0xFFFFFFFF);
  }

  __device__ __forceinline__ uint32_t atomic_remove_subbuck(uint16_t subbuck_ind) {
    auto subbuck_cnt_ptr = subbuck_cnt_sram + subbuck_ind;
    return atomicDec_block(subbuck_cnt_ptr, 0xFFFFFFFF);
  }


  __device__ __forceinline__ void cumsum_base_sync() {
    constexpr uint16_t WarpSize = 32;
    constexpr uint16_t NumWarps = SetSize / WarpSize;
    constexpr uint16_t NumRounds = cdiv_dev(SetSize, 1024);
    constexpr uint16_t warp_per_blk = 1024 / WarpSize;
    auto tid = threadIdx.x;
    uint16_t warp_id = tid / WarpSize;
    auto shm_base = al_ptr->ptr;

    using wrp_cumsum = cub::WarpScan<uint16_t>;
    auto (&wrp_tmp)[NumWarps] = al_ptr->template allocate<typename wrp_cumsum::TempStorage, NumWarps>();

    for (uint16_t r = 0; r < NumRounds; ++r) {
      uint16_t elem_id = r * 1024 + tid;
      uint16_t logical_warpid = r * warp_per_blk + warp_id;

      wrp_cumsum(wrp_tmp[logical_warpid]).InclusiveSum(
        subbuck_cnt_sram[elem_id], subbuck_base_sram[elem_id]);
    }

    __syncthreads();

    if (tid < WarpSize) {
      constexpr uint16_t seg_size = WarpSize;
      constexpr uint16_t num_seg = SetSize / seg_size;
      for (uint16_t sid = 1; sid < num_seg; ++sid) {
        auto base_elem = subbuck_base_sram[sid * seg_size - 1];
        auto lane_base = sid * seg_size + tid;

        subbuck_base_sram[lane_base] += base_elem;
        __syncwarp();
      }
    }
    __syncthreads();
    al_ptr->ptr = shm_base;
  }

  __device__ __forceinline__ uint32_t get_subbuck_cnt(uint16_t subbuck_ind) {
    return subbuck_cnt_sram[subbuck_ind];
  }

  __device__ __forceinline__ uint32_t get_subbuck_end(uint16_t subbuck_ind) {
    return subbuck_base_sram[subbuck_ind];
  }

  __device__ __forceinline__ uint32_t get_subbuck_base(uint16_t subbuck_ind) {
    return (subbuck_ind > 0) ? subbuck_base_sram[subbuck_ind - 1] : 0u;
  }

private:
  uint32_t * subbuck_cnt_sram;
  uint16_t * subbuck_base_sram;

  //uint16_t cumsum_reg[StrideCumsum];

  tk::shared_allocator<tk_align> * al_ptr;
};

} // end of ::f3d

#endif //FLASH3DPSHATTN_SHMEM_CUMSUM_COUNTER_CUH
