/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 9/28/24
 */


#ifndef FLASH3DPSHATTN_BITSET_H
#define FLASH3DPSHATTN_BITSET_H

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <kittens.cuh>

#include "common/arithmetic.cuh"

namespace tk = kittens;

namespace f3d {

using word_type = uint32_t;
constexpr auto word_bits = sizeof(word_type) * 8;

template<uint16_t SetSize, typename CoordT, typename CounterT, uint16_t D, int tk_align=-1>
class BitsetCounter {
  /*
   * subbuck_cnt_sram: Uint32[SetSize]
   * subbuck_center_sram: BF16/FP16/FP32[D, SetSize], stride_D = SetSize
   */
public:
  static constexpr auto WORD_SIZE = cdiv(SetSize, word_bits);

  __device__ __forceinline__ BitsetCounter(
    word_type* bitmap_shmem, CounterT* subbuck_cnt, tk::shared_allocator<tk_align> * al_p) :
    bitmap{bitmap_shmem},
    subbuck_idcnt_dram{subbuck_cnt},
    al_ptr{al_p} {

    // Force zeroing bitmap_shmem
    if (threadIdx.x < WORD_SIZE) {
      bitmap_shmem[threadIdx.x] = 0u;
    }
    reusable_shm = al_ptr->ptr;
    subbuck_idcnt_sram = &(al_ptr->template allocate<CounterT>());
    if (threadIdx.x == 0) {
      *subbuck_idcnt_sram = 0;
    }
    subbuck_cnt_sram = al_ptr->template allocate<uint32_t, SetSize>();
    subbuck_center_ptr_sram = al_ptr->template allocate<CoordT, D * SetSize>();
    available_subbuck = al_ptr->template allocate<uint16_t, SetSize>();

    // Initialize arrays
    if (threadIdx.x < SetSize) {
      subbuck_cnt_sram[threadIdx.x] = 0u;
      available_subbuck[threadIdx.x] = 0u;
    }
    __syncthreads();
  }

  __device__ __forceinline__ int32_t setbit_atomic(uint16_t set_ind) {
    if (set_ind >= SetSize) {
      return -1;
    }

    auto word_id = set_ind / word_bits;
    auto bit_id = set_ind % word_bits;
    word_type set_mask = 1 << bit_id;
    auto word_ptr = bitmap + word_id;
    atomicOr_block(word_ptr, set_mask);

    return 0;
  }

  __device__ __forceinline__ CounterT acc_size_popc_shm_sync() {
    if (threadIdx.x < WORD_SIZE) {
      auto word_cnt = __popc(bitmap[threadIdx.x]);
      atomicAdd_block(subbuck_idcnt_sram, word_cnt);
    }
    __syncthreads();
    subbuck_idcnt_stage_one = *subbuck_idcnt_sram;
    return subbuck_idcnt_stage_one;
  }

  __device__ __forceinline__ uint8_t getbit(uint16_t set_ind) {
    if (set_ind >= SetSize) {
      return 0;
    }

    auto word_id = set_ind / word_bits;
    auto bit_id = set_ind % word_bits;
    word_type get_mask = 1 << bit_id;
    return (bitmap[word_id] & get_mask)? 1u: 0u;
  }

  __device__ __forceinline__ uint32_t try_add_subbuck(uint16_t subbuck_ind) {
    auto subbuck_cnt_ptr = subbuck_cnt_sram + subbuck_ind;
    return atomicInc_block(subbuck_cnt_ptr, 0xFFFFFFFF);
  }

  __device__ __forceinline__ uint32_t regret_add_subbuck(uint16_t subbuck_ind) {
    auto subbuck_cnt_ptr = subbuck_cnt_sram + subbuck_ind;
    return atomicDec_block(subbuck_cnt_ptr, 0xFFFFFFFF);
  }

  __device__ __forceinline__ void set_subbuck_center(uint16_t subbuck_ind, CoordT (&pt_ctr)[D]) {
    for (uint16_t d = 0; d < D; ++d) {
      auto dst_ptr = subbuck_center_ptr_sram + d * SetSize + subbuck_ind;
      *dst_ptr = pt_ctr[d];
    }
  }

  __device__ __forceinline__ void sync_scan_available_subbuck() {
    __syncthreads();
    // Assumes *subbuck_idcnt_sram = bsc.acc_size_popc_shm_sync();
    auto subbuck_id_prob = uint16_t(threadIdx.x);
    if (subbuck_id_prob >= SetSize) {
      return;
    } else {
      if (!getbit(subbuck_id_prob)) {
        auto new_subbuck_id = atomicInc_block(subbuck_idcnt_sram, 0xFFFFFFFF);
        available_subbuck[new_subbuck_id - subbuck_idcnt_stage_one] = subbuck_id_prob;
      }
    }
  }

  __device__ __forceinline__ void realign_idcnt_to_stage_one_sync() {
    __syncthreads();
    if (threadIdx.x == 0) {
      *subbuck_idcnt_sram = subbuck_idcnt_stage_one;
    }
    __syncthreads();
  }

  __device__ __forceinline__ void reset_subbuck_idcnt_sync() {
    if (threadIdx.x == 0) {
      *subbuck_idcnt_sram = 0;
    }
    __syncthreads();
  }

  /*
  __device__ __forceinline__ void race_new_subbuck(CoordT (&pt_ctr)[D], int32_t &assigned_subbuck) {
    int32_t new_subbuck_id = atomicInc_block(subbuck_idcnt_sram, 0xFFFFFFFF);
    if (new_subbuck_id >= SetSize) {
      return ;
    }

    // Compute new assigned_subbuck
    int32_t skipped_subbuck = 0;
    for (uint16_t set_ind = 0; set_ind < SetSize; ++set_ind) {
      if (!getbit(set_ind)) {
        ++skipped_subbuck;
        // 1 means 1 new subbuck found
      }

      if (skipped_subbuck - 1 == new_subbuck_id - subbuck_idcnt_stage_one) {
        assigned_subbuck = set_ind;
        // Each new subbucket will have only one new point, so it's safe to commit the center
        // But we don't set the bitset to prevent from interfering other threads.

        set_subbuck_center(assigned_subbuck, pt_ctr);

        // We've set this new bucket tight
        return ;
      }
    }

    // Set new subbuck center
  }
   */

  __device__ __forceinline__ bool race_new_subbuck(
    CoordT (&pt_ctr)[D], int32_t & assigned_subbuck, CounterT & subbuck_off) {
    int32_t new_subbuck_id = atomicInc_block(subbuck_idcnt_sram, 0xFFFFFFFF);

    if (new_subbuck_id >= SetSize) {
      return false;
    } else {
      // Grab new assigned_subbuck
      assigned_subbuck = available_subbuck[new_subbuck_id - subbuck_idcnt_stage_one];
      // Commit new assigned_subbuck
      setbit_atomic(assigned_subbuck);
      try_add_subbuck(assigned_subbuck);
      // Set the center point
      set_subbuck_center(assigned_subbuck, pt_ctr);
      // Mark this is the first point of newly allocated subbuck
      subbuck_off = 0;

      // Notify the caller of success to update offset = 0
      return true;
    }
  }

  __device__ __forceinline__ float L2_distance_to_subbuck(CoordT (&pt_ctr)[D], uint16_t subbuck_ind) {
    float acc = 0.0f;
    for (uint16_t d = 0; d < D; ++d) {
      auto src_ptr = subbuck_center_ptr_sram + d * SetSize + subbuck_ind;
      auto diff = float(pt_ctr[d]) - float(*src_ptr);
      acc += diff * diff;
    }
    return sqrt(acc);
  }

  __device__ __forceinline__ uint16_t num_new_subbuck() {
    return SetSize - subbuck_idcnt_stage_one;
  }

  __device__ __forceinline__ bool try_find_subbuck(
    CoordT (&pt_ctr)[D], int32_t &assigned_subbuck, CounterT& subbuck_off_reg, uint16_t subbuck_size) {
    int32_t first_subbuck = -1;
    float first_dist = INFINITY;

    for (uint16_t subbuck_ind = 0; subbuck_ind < num_new_subbuck(); ++subbuck_ind) {
      auto subbuck = available_subbuck[subbuck_ind];
      if (subbuck_cnt_sram[subbuck] < subbuck_size) {
        auto prob_dist = L2_distance_to_subbuck(pt_ctr, subbuck);

        if (prob_dist < first_dist) {
          first_dist = prob_dist;
          first_subbuck = subbuck;
        }
      }
    }

    if (first_subbuck < 0) {
      return false;
    }
    auto subbuck_off = try_add_subbuck(first_subbuck);
    if (subbuck_off < subbuck_size) {
      subbuck_off_reg = subbuck_off;
      assigned_subbuck = first_subbuck;
      return true;
    } else {
      regret_add_subbuck(first_subbuck);
      return false;
    }
  }

  __device__ __forceinline__ void last_resort(
    int32_t &assigned_subbuck, CounterT& subbuck_off_reg, uint16_t subbuck_size) {

    for (uint16_t subbuck_ind = 0; subbuck_ind < num_new_subbuck(); ++subbuck_ind) {
      auto subbuck = available_subbuck[subbuck_ind];
      auto subbuck_cnt_ptr = subbuck_cnt_sram + subbuck;
      auto old_cnt = *subbuck_cnt_ptr;
      if (old_cnt < subbuck_size) {
        auto cas_res = atomicCAS_block(subbuck_cnt_ptr, old_cnt, old_cnt + 1);
        if (cas_res > old_cnt) {
          assigned_subbuck = subbuck;
          subbuck_off_reg = old_cnt;
          return;
        }
      }
    }
  }

  // For debug only
  __device__ __forceinline__ uint32_t count_unfound_pts(bool unfound) {
    __syncthreads();
    auto start_ptr = al_ptr->ptr;
    unfound_num_pt = al_ptr->template allocate<uint32_t, 1>();
    *unfound_num_pt = 0;
    __syncthreads();
    if (unfound) {
      atomicInc_block(unfound_num_pt, 0xFFFFFFFF);
    }
    __syncthreads();
    auto res = *unfound_num_pt;
    al_ptr->ptr = start_ptr;
    return res;
  }

  __device__ __forceinline__ uint32_t count_filled_subbuck(uint32_t subbuck_size) {
    __syncthreads();
    auto base_shm = al_ptr->ptr;
    auto fill_cnt_ptr = al_ptr->template allocate<uint32_t, 1>();
    *fill_cnt_ptr = 0;
    __syncthreads();

    if (threadIdx.x < SetSize) {
      if (subbuck_cnt_sram[threadIdx.x] >= subbuck_size) {
        atomicInc_block(fill_cnt_ptr, 0xFFFFFFFF);
      }
    }
    __syncthreads();

    auto res = *fill_cnt_ptr;
    al_ptr->ptr = base_shm;
    __syncthreads();
    return res;
  }

private:
  word_type * bitmap;
  CounterT * subbuck_idcnt_dram, * subbuck_idcnt_sram, subbuck_idcnt_stage_one;
  uint32_t * subbuck_cnt_sram;
  uint16_t * available_subbuck;
  CoordT * subbuck_center_ptr_sram;

  // For debug
  uint32_t * unfound_num_pt;

  tk::shared_allocator<tk_align> * al_ptr;
  int * reusable_shm;
};

} // end of ::f3d
#endif //FLASH3DPSHATTN_BITSET_H
