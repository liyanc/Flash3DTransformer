/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 05/15/24
 */


#ifndef FLASH3DPSHATTN_HASH_FNS_H
#define FLASH3DPSHATTN_HASH_FNS_H

#include <cub/cub.cuh>
#include <kittens.cuh>

#include "common/rand.cuh"

namespace tk = kittens;

namespace f3d {

template <typename CoordT, typename HashT, uint16_t D>
__device__ __forceinline__ void vox_coord(
  CoordT (&coord_reg)[D], HashT (&vox_reg)[D], CoordT *min_bb, CoordT *max_bb, uint32_t num_vox) {
  #pragma unroll
  for (auto d = 0; d < D; ++d) {
    auto vox = clamp(floor(
      (float(coord_reg[d]) - float(min_bb[d])) * num_vox /
      (float(max_bb[d]) - float(min_bb[d]))), 0.f, float(num_vox - 1));
    vox_reg[d] = HashT(vox);
  }
}

template <typename HashT, uint16_t D>
__device__ void h1_xorsum(tk::sv<HashT, 2 * D> &vox_shm, tk::sv<HashT, 2> &hash_shm, HashT size) {
  auto laneid = tk::laneid();
  HashT acc = 0u;

  // Two-way bank conflicts
  #pragma unroll
  for (auto d = 0; d < D; ++d) {
    acc ^= vox_shm[laneid * D + d];
  }
  hash_shm[laneid] = acc / size;
}

template <typename HashT, uint16_t D>
__device__ void h1_xorsum_div(HashT (&vox_reg)[3], HashT & hash_reg, HashT size) {
  HashT acc = 0u;

  #pragma unroll
  for (auto d = 0; d < D; ++d) {
    acc ^= vox_reg[d];
  }
  hash_reg = acc / size;
}

template <typename HashT, uint16_t D>
__device__ void h1_xorsum_mod(HashT (&vox_reg)[3], HashT & hash_reg, HashT num_buck) {
  HashT acc = 0u;

  #pragma unroll
  for (auto d = 0; d < D; ++d) {
    acc ^= vox_reg[d];
  }
  hash_reg = acc % num_buck;
}

__device__ __forceinline__ uint64_t space_out_bit(uint64_t x) {
  x = (x | (x << 16)) & 0x0000FFFF0000FFFF;
  x = (x | (x << 8)) & 0x00FF00FF00FF00FF;
  x = (x | (x << 4)) & 0x0F0F0F0F0F0F0F0F;
  x = (x | (x << 2)) & 0x3333333333333333;
  x = (x | (x << 1)) & 0x5555555555555555;
  return x;
}

__device__ __forceinline__ uint64_t space_out_bit_3d(uint64_t x) {
  x = (x | x << 32) & 0x1f00000000ffff;
  x = (x | x << 16) & 0x1f0000ff0000ff;
  x = (x | x << 8) & 0x100f00f00f00f00f;
  x = (x | x << 4) & 0x10c30c30c30c30c3;
  x = (x | x << 2) & 0x1249249249249249;
  return x;
}

template <typename HashT>
__device__ void h1_zorder_div_3d(tk::sv<HashT, 6> &vox_shm, tk::sv<HashT, 2> &hash_shm, HashT size) {
  auto laneid = tk::laneid();
  auto const D = 3u;
  // Two-way bank conflicts :(
  uint64_t x = space_out_bit_3d(vox_shm[laneid * D]);
  uint64_t y = space_out_bit_3d(vox_shm[laneid * D + 1]);
  uint64_t z = space_out_bit_3d(vox_shm[laneid * D + 2]);
  HashT interleaved = (x | y << 1 | z << 2) >> 31;

  hash_shm[laneid] = interleaved / size;
}

template <typename HashT>
__device__ void h1_zorder_div_3d(HashT (&vox_reg)[3], tk::sv<HashT, 2> &hash_shm, HashT size) {
  auto laneid = tk::laneid();
  // No bank conflict, nice :)
  uint64_t x = space_out_bit_3d(vox_reg[0]);
  uint64_t y = space_out_bit_3d(vox_reg[1]);
  uint64_t z = space_out_bit_3d(vox_reg[2]);
  HashT interleaved = (x | y << 1 | z << 2) >> 31;

  hash_shm[laneid] = interleaved / size;
}

template <typename HashT>
__device__ void h1_zorder_div_3d(HashT (&vox_reg)[3], HashT & hash_reg, HashT size) {
  auto laneid = tk::laneid();
  // No bank conflict, nice :)
  uint64_t x = space_out_bit_3d(vox_reg[0]);
  uint64_t y = space_out_bit_3d(vox_reg[1]);
  uint64_t z = space_out_bit_3d(vox_reg[2]);
  HashT interleaved = (x | y << 1 | z << 2) >> 31;

  hash_reg = interleaved / size;
}

template <typename HashT>
__device__ void h1_zorder_mod_3d(HashT (&vox_reg)[3], HashT & hash_reg, HashT num_buck) {
  auto laneid = tk::laneid();
  // No bank conflict, nice :)
  uint64_t x = space_out_bit_3d(vox_reg[0]);
  uint64_t y = space_out_bit_3d(vox_reg[1]);
  uint64_t z = space_out_bit_3d(vox_reg[2]);
  HashT interleaved = (x | y << 1 | z << 2) >> 31;

  hash_reg = interleaved % num_buck;
}

template <typename HashT, uint16_t D>
__device__ void h1_sum(tk::sv<HashT, 2 * D> &vox_shm, tk::sv<HashT, 2> &hash_shm, HashT size) {
  auto laneid = tk::laneid();
  uint32_t acc = 0u;

  // Two-way bank conflicts
  #pragma unroll
  for (auto d = 0; d < D; ++d) {
    acc += vox_shm[laneid * D + d];
  }
  hash_shm[laneid] = (acc / size);

}

enum HASHTYPE {
    H_NONE = 0,
    H_ZORDER_DIV = 1,
    H_XORSUM_DIV = 2,
    H_ZORDER_MOD = 3,
    H_XORSUM_MOD = 4

  };

enum BUCKETSTAGE {
  S_INIT = 1,
  S_FIRST = 2,
  S_FINAL = 3
};
} // end of ::f3d


#endif //FLASH3DPSHATTN_HASH_FNS_H