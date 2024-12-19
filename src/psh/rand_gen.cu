/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 06/23/24
 */


#include <vector>
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

namespace tk = kittens;

namespace f3d {

template<typename IntT>
__device__ IntT uniform_int(curandState * state, IntT min, IntT max) {
  float fp_sample = curand_uniform(state);
  float rounded_sample = floorf(min + fp_sample * float(max - min));
  auto scaled_sample = rounded_sample;
  return IntT(scaled_sample);
}

__global__ void uniform_samples_ker(
  uint32_t * __restrict__ array, uint32_t N, uint32_t min, uint32_t max, uint64_t seed=3407) {
  auto ind = threadIdx.x + blockIdx.x * blockDim.x;
  if (ind >= N)
    return;

  curandState thread_state;
  curand_init(seed, ind, 0u, &thread_state);
  array[ind] = uniform_int(&thread_state, min, max);

}

extern void
uniform_samples(at::Tensor & array, uint32_t min, uint32_t max) {
  auto size = array.size(0);
  auto BLOCK_N = 1024u;

  uniform_samples_ker<<<cdiv(size, BLOCK_N), BLOCK_N>>>(
    (uint32_t*)array.data_ptr(), size, min, max);

  cudaCheckErr(cudaGetLastError());
  cudaCheckErr(cudaDeviceSynchronize());
}

} // end of ::f3d
