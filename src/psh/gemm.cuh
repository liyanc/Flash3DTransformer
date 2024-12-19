/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 *  found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 2/3/25
 */


#ifndef FLASH3DPSHATTN_HEMM_H
#define FLASH3DPSHATTN_HEMM_H

#include <kittens.cuh>
#include "common/runtime.h"
#include "common/arithmetic.cuh"
#include "common/fp_types.cuh"
#include "common/rand.cuh"
#include "common/device_manager.cuh"

namespace tk = kittens;

namespace f3d {

struct gemm_globals {
  tk::gl<bf16, 1, 1, -1, -1> A, B, O;
  int M, N, K;
};

template <int M_PER_BLOCK = 64, int N_PER_BLOCK = 128, int K_PER_BLOCK = 16, int NUM_WORKERS = 4>
__global__ void gemm_sm_bf16_ker(const __grid_constant__ gemm_globals g) {
  // Each block is responsible for one M_PER_BLOCK x N_PER_BLOCK tile of the output matrix C.
  const int block_row_tile_idx = blockIdx.y;
  const int block_col_tile_idx = blockIdx.x;
  const int worker_id = tk::warpid();

  extern __shared__ tk::alignment_dummy __shm[];
  tk::shared_allocator al((int*)__shm);

  auto& a_shmem = al.allocate<tk::st_bf<M_PER_BLOCK, K_PER_BLOCK>>();
  auto& b_shmem = al.allocate<tk::st_bf<K_PER_BLOCK, N_PER_BLOCK>>();

  tk::rt_bf<16, K_PER_BLOCK> a_reg;
  tk::rt_bf<K_PER_BLOCK, 16, tk::col_l> b_reg;
  tk::rt_fl<16, N_PER_BLOCK> o_reg;

  tk::zero(o_reg);

  // Assuming perfect divisibility for the happy path test.
  const int k_blocks = g.K / K_PER_BLOCK;

  for (int k_block_idx = 0; k_block_idx < k_blocks; ++k_block_idx) {
    __syncthreads();
    // --- CORRECTED LOAD LOGIC ---
    // Load tiles using TILE indices, not element offsets.
    // The `tk::load` function will internally calculate the correct memory address.
    // All warps in the block cooperate on these loads.
    tk::group<NUM_WORKERS>::load(a_shmem, g.A, {0, 0, block_row_tile_idx, k_block_idx});
    tk::group<NUM_WORKERS>::load(b_shmem, g.B, {0, 0, k_block_idx, block_col_tile_idx});
    __syncthreads();

    // Each warp loads its 16-row slice of A from shared memory.
    auto a_shmem_warp_slice = tk::subtile_inplace<16, K_PER_BLOCK>(a_shmem, {worker_id, 0});
    tk::load(a_reg, a_shmem_warp_slice);

#pragma unroll
    for (int n_chunk = 0; n_chunk < (N_PER_BLOCK / 16); ++n_chunk) {
      auto b_shmem_subtile = tk::subtile_inplace<K_PER_BLOCK, 16>(b_shmem, {0, n_chunk});
      tk::load(b_reg, b_shmem_subtile);

      auto& o_reg_subtile = reinterpret_cast<tk::rt_fl<16, 16>&>(o_reg.tiles[0][n_chunk]);
      tk::mma_AB(o_reg_subtile, a_reg, b_reg, o_reg_subtile);
    }
  }

  // --- Store Result ---
  // The final store also uses tile indices.
  tk::rt_bf<16, N_PER_BLOCK> o_reg_bf16;
  tk::copy(o_reg_bf16, o_reg);

  // Each warp stores its computed 16x128 row-slice of the output.
  const int output_row_tile_idx = block_row_tile_idx * (M_PER_BLOCK / 16) + worker_id;
  const int output_col_tile_idx = block_col_tile_idx * (N_PER_BLOCK / 128); // Placeholder if N_PER_BLOCK could change
  tk::store(g.O, o_reg_bf16, {0, 0, output_row_tile_idx, output_col_tile_idx});
}
} // end of ::f3d

#endif //FLASH3DPSHATTN_HEMM_H
