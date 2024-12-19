/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 06/02/24
 */


#ifndef FLASH3DPSHATTN_ARITHMETIC_H
#define FLASH3DPSHATTN_ARITHMETIC_H

constexpr uint16_t WarpSize = 32u;

template<typename IntA, typename IntB>
constexpr IntA cdiv(IntA a, IntB b) {
  return (a < 0 ? a: (a + (b - 1))) / b;
}

template<typename IntA, typename IntB>
__device__ __forceinline__ constexpr IntA cdiv_dev(IntA a, IntB b) {
  return (a < 0 ? a: (a + (b - 1))) / b;
}

template<typename FPT>
__device__ __forceinline__ FPT clamp(FPT x, FPT lo, FPT hi) {
  return max(lo, min(x, hi));
}

#ifndef F3D_UNROLL
  #if defined(__CUDACC__)
    #define F3D_UNROLL _Pragma("unroll")
  #else
    #define F3D_UNROLL
  #endif
#endif


#endif //FLASH3DPSHATTN_ARITHMETIC_H
