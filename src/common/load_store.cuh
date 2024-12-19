/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 07/13/24
 */


#ifndef FLASH3DPSHATTN_LOAD_STORE_H
#define FLASH3DPSHATTN_LOAD_STORE_H

#include <cub/cub.cuh>
#include <kittens.cuh>

namespace tk = kittens;

namespace f3d {

template <typename HashT, typename CounterT, uint16_t BLOCK_SIZE_N, int tk_align=-1>
__device__ __forceinline__ void bulk_store_bucket_id_off(
  uint32_t block_start, HashT * bucketid_base, CounterT * bucketoff_base,
  uint16_t bucketid_stride, uint16_t bucketoff_stride,
  HashT (&bucketid_reg)[1], CounterT (&bucketoff_reg)[1],
  tk::shared_allocator<tk_align> * al_ptr
  ) {
  auto resuable_shm = al_ptr->ptr;
  using BucketBlockStore = cub::BlockStore<HashT, BLOCK_SIZE_N, 1, cub::BLOCK_STORE_VECTORIZE>;
  using BucketBlockStoreTemp = BucketBlockStore::TempStorage;
  auto bucket_block_store_temp = al_ptr->template allocate<BucketBlockStoreTemp>();
  auto bucketid_blk_base = bucketid_base + block_start * bucketid_stride;
  BucketBlockStore(bucket_block_store_temp).Store(bucketid_blk_base, bucketid_reg);

  using BucketOffsetBlockStore = cub::BlockStore<CounterT, BLOCK_SIZE_N, 1, cub::BLOCK_STORE_VECTORIZE>;
  using BucketOffsetBlockStoreTemp = BucketOffsetBlockStore::TempStorage;
  auto bucketoffset_block_store_temp = al_ptr->template allocate<BucketOffsetBlockStoreTemp>();
  auto bucketoff_blk_base = bucketoff_base + block_start * bucketoff_stride;
  BucketOffsetBlockStore(bucketoffset_block_store_temp).Store(bucketoff_blk_base, bucketoff_reg);
  al_ptr->ptr = resuable_shm;
}

template <uint16_t num_probs, uint16_t D, uint16_t BLOCK_SIZE_N, typename ElemT>
__device__ __forceinline__ void load_tile_coalesced(
    const ElemT* gmem_base,
    ElemT* smem_base,
    uint32_t start_prob_idx,
    uint32_t global_stride_d)
{
    const ElemT* gmem_tile_base = gmem_base + uint64_t(start_prob_idx) * global_stride_d;

    const auto total_elements = num_probs * D;

    // Use a block-wide strided loop to load all elements, promoting coalescing.
    for (uint32_t i = threadIdx.x; i < total_elements; i += BLOCK_SIZE_N) {
        // `i` is the linear index within the [num_probs, D] tile.
        // Convert this linear index back to (row, col) coordinates.
        uint32_t row_in_tile = i / D;
        uint32_t col_in_tile = i % D;

        // Read from global memory using the calculated coordinates and stride.
        const ElemT & val = gmem_tile_base[(long)row_in_tile * global_stride_d + col_in_tile];

        // Write to the corresponding linear index in shared memory.
        smem_base[i] = val;
    }
}

} // end of ::f3d


#endif //FLASH3DPSHATTN_LOAD_STORE_H