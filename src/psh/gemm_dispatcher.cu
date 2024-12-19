/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 2/3/25
 */


#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/ATen.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "common/runtime.h"
#include "common/arithmetic.cuh"
#include "common/fp_types.cuh"
#include "psh/gemm.cuh"

namespace f3d {

gemm_globals gemm_init(bf16* d_A, bf16* d_B, bf16* d_O, int M, int N, int K) {
  gemm_globals g = {
    .A = tk::gl<bf16, 1, 1, -1, -1>(d_A, nullptr, nullptr, M, K),
    .B = tk::gl<bf16, 1, 1, -1, -1>(d_B, nullptr, nullptr, K, N),
    .O = tk::gl<bf16, 1, 1, -1, -1>(d_O, nullptr, nullptr, M, N),
    .M = M, .N = N, .K = K
  };
  return g;
}

extern void gemm_sm_bf16(at::Tensor &A, at::Tensor &B, at::Tensor &O) {
  assert_contiguous({A, B, O});
  assert_same_device({A, B, O});

  constexpr int M_PER_BLOCK = 64;
  constexpr int N_PER_BLOCK = 128;
  constexpr int K_PER_BLOCK = 16;
  constexpr int NUM_WORKERS = 4;

  if (A.size(1) != B.size(0)) {
    throw_format_error("Unmached K dim, K_A = %d but K_B = %d",
                       A.size(1), B.size(0));
  }
  if (A.size(0) != 128) {
    throw_format_error("M must be 128 instead of %d", A.size(0));
  }

  auto stream = c10::cuda::getCurrentCUDAStream(A.get_device());
  auto shmem_size = tk::MAX_SHARED_MEMORY;

  auto gemm_ker = gemm_sm_bf16_ker<M_PER_BLOCK, N_PER_BLOCK, K_PER_BLOCK, NUM_WORKERS>;
  cudaFuncSetAttribute(gemm_ker, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
  cudaCheckLastErr();

  auto & dev_mgr = DeviceManagerSingleton::instance();
  dev_mgr.set_device(A.get_device());

  const int M = A.size(0);
  const int K = A.size(1);
  const int N = B.size(1); // N is the second dimension of B

  gemm_globals g = gemm_init(
    (bf16*)A.data_ptr(), (bf16*)B.data_ptr(), (bf16*)O.data_ptr(), M, N, K
  );

  dim3 grid((N + N_PER_BLOCK - 1) / N_PER_BLOCK, (M + M_PER_BLOCK - 1) / M_PER_BLOCK, 1);
  dim3 block(NUM_WORKERS * tk::WARP_THREADS);
  // The kernel launch now uses the grid/block dimensions from the globals struct.


  gemm_ker<<<grid, block, shmem_size, stream>>>(g);
  cudaCheckLastErr();

  cudaStreamSynchronize(stream);
  cudaCheckLastErr();
}

} // end of ::f3d