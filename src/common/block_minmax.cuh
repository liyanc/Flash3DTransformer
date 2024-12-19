/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 10/18/24
 */


#ifndef FLASH3DPSHATTN_BLOCK_MINMAX_H
#define FLASH3DPSHATTN_BLOCK_MINMAX_H

#include <cub/cub.cuh>
#include <kittens.cuh>

namespace tk = kittens;

namespace f3d {

template <typename T>
struct DevMin {
  __forceinline__ __device__ T operator()(const T &a, const T&b) {
    return (a <= b) ? a : b;
  }
};

template <typename T>
struct DevMax {
  __forceinline__ __device__ T operator()(const T &a, const T &b) {
    return (a >= b) ? a : b;
  }
};


template <typename ScalarT, uint16_t D, uint16_t BLOCK_SIZE, int tk_align=-1>
__device__ __forceinline__ void nd_reduce(
  ScalarT (&input)[D], ScalarT(&min)[D], ScalarT(&max)[D], tk::shared_allocator<tk_align> * al_ptr) {
  auto reusable_ptr = al_ptr->ptr;
  using MinMaxRed = cub::BlockReduce<ScalarT, BLOCK_SIZE>;

  for (uint16_t d = 0; d < D; ++d) {
    auto min_red = MinMaxRed(al_ptr->template allocate<typename MinMaxRed::TempStorage>());
    auto max_red = MinMaxRed(al_ptr->template allocate<typename MinMaxRed::TempStorage>());

    min[d] = min_red.Reduce(input[d], DevMin<ScalarT>{});
    max[d] = max_red.Reduce(input[d], DevMax<ScalarT>{});
  }

  __syncthreads();
  al_ptr->ptr = reusable_ptr;
}

template <typename ScalarT, uint16_t D>
__device__ __forceinline__ ScalarT mean_bbox(ScalarT(&min)[D], ScalarT(&max)[D]) {
  ScalarT acc = 0.f;

  for (uint16_t d = 0; d < D; ++d) {
    acc += (max[d] - min[d]);
  }
  return ScalarT(float(acc) / float(D));
}

} // end of ::f3d

#endif //FLASH3DPSHATTN_BLOCK_MINMAX_H
