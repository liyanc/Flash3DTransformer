/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 06/23/24
 */


#ifndef FLASH3DPSHATTN_RAND_H
#define FLASH3DPSHATTN_RAND_H

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "common/arithmetic.cuh"
#include "common/runtime.h"

namespace f3d {
template<typename IntT>
__device__ __forceinline__ IntT uniform_int(curandState * state, IntT min, IntT max) {
  float fp_sample = curand_uniform(state);
  float rounded_sample = floorf(min + fp_sample * float(max - min));
  auto scaled_sample = rounded_sample;
  return IntT(scaled_sample);
}

template<typename FPT>
__device__ __forceinline__ FPT uniform_fp(curandState * state, FPT min, FPT max) {
  float fp_sample = curand_uniform(state);
  auto scaled_sample = min + fp_sample * (max - min);
  return scaled_sample;
}

class LocalRandGen {
  public:
    __device__ LocalRandGen(uint64_t seed, uint64_t seq) {
      curand_init(seed, seq, 0u, &state);
      dram_base = nullptr;
    }

    __device__ LocalRandGen(uint64_t seed) {
      uint64_t seq = threadIdx.x + blockDim.x * blockIdx.x;
      curand_init(seed, seq, 0u, &state);
      dram_base = nullptr;
    }

    __device__ LocalRandGen() {
      dram_base = nullptr;
    }

    __device__ LocalRandGen(curandState * global_base) {
      uint64_t seq = threadIdx.x;
      state = *(global_base + seq);
      dram_base = global_base;
    }

    __device__ LocalRandGen(curandState * global_base, uint64_t seed) {
      uint64_t seq = threadIdx.x;
      curand_init(seed, seq, 0u, &state);
      dram_base = global_base;
    }

    __device__ __forceinline__ void writeback_global() {
      uint64_t seq = threadIdx.x;
      if (dram_base != nullptr && blockIdx.x == 0)
        *(dram_base + seq) = state;
    }

    __device__ ~LocalRandGen() {
      writeback_global();
    }

    template<typename IntT>
    __device__ __forceinline__ IntT uniform_int(IntT min, IntT max) {
      return f3d::uniform_int<IntT>(&state, min, max);
    }

    template<typename FPT>
    __device__ __forceinline__ FPT uniform_fp(FPT min, FPT max) {
      return f3d::uniform_fp<FPT>(&state, min, max);
    }

    template<typename FPT, uint16_t D>
    __device__ __forceinline__ void perturb_coord(
      float scale_min, float scale_max, FPT (&coord_in)[D], FPT (&coord_out)[D]) {
      #pragma unroll
      for (uint16_t d = 0; d < D; ++d) {
        coord_out[d] = FPT(float(coord_in[d]) + uniform_fp(scale_min, scale_max));
      }
    }

    template<typename HashT, uint16_t D>
    __device__ __forceinline__ void perturb_vox(
      float scale_min, float scale_max, uint32_t num_vox, uint32_t buck_seg_div,
      const int16_t * __restrict__ probe_table_base, auto probe_ind, HashT (&vox_reg)[D], HashT (&newvox_reg)[D]) {

      auto scaling_fac = uniform_fp(scale_min, scale_max);
      int32_t vox_shift_seg = int32_t(roundf(float(num_vox) * scaling_fac / buck_seg_div));
      int32_t vox_shift_half = vox_shift_seg / 2;

      #pragma unroll
      for (auto d = 0; d < D; ++d) {
        int32_t coord_shift =
          int32_t(probe_table_base[probe_ind * D + d]) * vox_shift_seg +
          uniform_int<int32_t>(-vox_shift_half, vox_shift_half);

        int32_t shifted_vox = int32_t(vox_reg[d]) + coord_shift;
        int32_t rebound_vox = abs(shifted_vox % int32_t(num_vox));

        newvox_reg[d] = HashT(clamp<int32_t>(rebound_vox, 0, int32_t(num_vox)));
      }
    }

  private:
    curandState * dram_base;
    curandState state;
  };

__global__ inline void init_kernel(curandState * global_base, uint64_t seed) {
  uint64_t seq = threadIdx.x;
  curand_init(seed, seq, 0u, global_base + seq);
}

// TODO: add per-device controls
class GlobalRandState {
public:
  GlobalRandState(uint16_t SIZE, int device) {
    cudaCheckErr(cudaSetDevice(device));
    BLOCK_SIZE = SIZE;
    cudaMalloc((void **)&ptr, SIZE * sizeof(curandState));
    cudaCheckLastErr();
  }

  inline curandState * get_states(){
    return ptr;
  }

  void init_global(uint64_t seed, c10::cuda::CUDAStream & stream) {
    init_kernel<<<BLOCK_SIZE, 1, 0, stream>>>(ptr, seed);
    cudaCheckLastErr();
  }

  ~GlobalRandState() {
    cudaFree((void *) ptr);
    cudaCheckLastErr();
  }
private:
  uint16_t BLOCK_SIZE;
  curandState * ptr;
};

} //end of ::f3d

#endif //FLASH3DPSHATTN_RAND_H