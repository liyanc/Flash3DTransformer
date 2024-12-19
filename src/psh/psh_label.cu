/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 05/15/24
 */


#include <chrono>
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
#include "common/rand.cuh"
#include "psh/hash_fns.cuh"
#include "psh/two_stage_counter.cuh"

namespace tk = kittens;

namespace f3d {

// Template for forward-compatibility
template <typename CoordT, typename CounterT, typename HashT, uint16_t D,
  uint16_t NUM_BUCKET, uint16_t BLOCK_SIZE_N, bool DO_TWO_STAGE>
__global__ void psh_partition_ker(
  // coords [N, D]
  const CoordT * __restrict__ coord, HashT * __restrict__ bucket_id, CounterT * __restrict__ bucket_counter,
  CounterT * __restrict__ bucket_offset,
  uint32_t N, uint32_t N_bucket, uint32_t bucket_seg_divisor, uint32_t num_vox,
  const CoordT * __restrict__ bbox_min, const CoordT * __restrict__ bbox_max,
  uint16_t stride_coordD, uint16_t stride_coordN,
  uint16_t stride_bucket_idN, uint16_t stride_bucket_cntN,
  uint16_t stride_bucket_offsetN
  ) {
  auto threadid = threadIdx.x;
  auto block_start = blockIdx.x * blockDim.x;
  auto col_ind = block_start + threadid;

  if (col_ind >= N)
    return;

  extern __shared__ tk::alignment_dummy __shm[];
  tk::shared_allocator al((int*) &__shm[0]);

  using CoordBlockLoad = cub::BlockLoad<CoordT, BLOCK_SIZE_N, D, cub::BLOCK_LOAD_VECTORIZE>;
  using BucketBlockStore = cub::BlockStore<HashT, BLOCK_SIZE_N, 1, cub::BLOCK_STORE_VECTORIZE>;
  using BucketOffsetStore = cub::BlockStore<CounterT, BLOCK_SIZE_N, 1, cub::BLOCK_STORE_VECTORIZE>;
  using CoordBlockLoadTemp = CoordBlockLoad::TempStorage;
  using BucketBlockStoreTemp = BucketBlockStore::TempStorage;
  auto coord_block_load_temp = al.allocate<CoordBlockLoadTemp>();
  auto bucket_block_store_temp = al.allocate<BucketBlockStoreTemp>();
  auto bucket_offset_store_temp = al.allocate<typename BucketOffsetStore::TempStorage>();

  auto min_bb = al.allocate<CoordT, D>();
  auto max_bb = al.allocate<CoordT, D>();
  auto len_bb = al.allocate<CoordT, D>();

  if (threadid < D) {
    min_bb[threadid] = bbox_min[threadid];
    max_bb[threadid] = bbox_max[threadid];
    len_bb[threadid] = max_bb[threadid] - min_bb[threadid];
  }
  CoordT coord_reg[D];
  HashT vox_reg[D], bucketid_reg[1];

  auto coord_base = coord + block_start * stride_coordN * stride_coordD;
  auto bucketid_base = bucket_id + block_start * stride_bucket_idN;
  auto bucketoff_base = bucket_offset + block_start * stride_bucket_offsetN;

  CoordBlockLoad(coord_block_load_temp).Load(coord_base, coord_reg);
  __syncthreads();

  #pragma unroll
  for (auto d = 0; d < D; ++d) {
    auto vox = clamp(floor(
      float(coord_reg[d] - min_bb[d]) * num_vox / float(len_bb[d])), 0.f, float(num_vox - 1));
    vox_reg[d] = vox;
  }

  //h1_xorsum_div<HashT, D>(vox_shm[warpid], hash_shm[warpid], bucket_seg_divisor);
  h1_zorder_div_3d<HashT>(vox_reg, bucketid_reg[0], bucket_seg_divisor);
  __syncthreads();

  CounterT bucket_offset_reg[1];

  if (DO_TWO_STAGE) {
    CounterT * counter_shm = al.allocate<CounterT, NUM_BUCKET>();
    TwoStageCounter<CounterT, NUM_BUCKET, BLOCK_SIZE_N> two_stage{counter_shm, bucket_counter, &al};
    bucket_offset_reg[0] = two_stage.atomic_inc_at(bucketid_reg[0], 0xFFFFFFFF, true);
    two_stage.block_commit_and_revise(bucketid_reg[0], bucket_offset_reg, true);
  } else {
    auto bucket_counter_lane = bucket_counter + bucketid_reg[0];
    //val = 0xFFFFFFFF to bypass thresholding
    bucket_offset_reg[0] = atomicInc(bucket_counter_lane, 0xFFFFFFFF);
  }

  __syncthreads();

  BucketBlockStore(bucket_block_store_temp).Store(bucketid_base, bucketid_reg);
  BucketOffsetStore(bucket_offset_store_temp).Store(bucketoff_base, bucket_offset_reg);
}


extern void
psh_partition(at::Tensor &coord, at::Tensor &bucket_id, at::Tensor &bucket_counter, at::Tensor &bucket_offset,
              uint32_t num_bucket, uint32_t bucket_divisor, uint32_t num_vox, at::Tensor &bbox_min,
              at::Tensor &bbox_max, py::object hash_dtype_obj, py::object coord_dtype_obj, uint16_t block_size_N) {

  auto hash_dtype = torch::python::detail::py_object_to_dtype(hash_dtype_obj);
  auto coord_dtype = torch::python::detail::py_object_to_dtype(coord_dtype_obj);
  auto N = coord.size(0);
  LOG(ERROR) << "NT, NB " << N << " " << cdiv(N, block_size_N) << " Buck_div " << bucket_divisor;

  if (num_vox / bucket_divisor > num_bucket) {
    throw std::runtime_error("bucket divisor too small");
  }

  using CoordT = fp16;
  using CounterT = uint32_t;
  using HashT = uint16_t;
  constexpr auto BLOCK_SIZE_N = 1024u;
  auto stream = c10::cuda::getCurrentCUDAStream(coord.get_device());
  auto shmem_size = tk::MAX_SHARED_MEMORY;

  QuickTimer timer{};

  cudaFuncSetAttribute(
    psh_partition_ker<fp16, uint32_t, uint16_t, 3, 256, BLOCK_SIZE_N, false>,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    shmem_size);
  cudaCheckLastErr();

  psh_partition_ker<CoordT, CounterT, HashT, 3, 256, BLOCK_SIZE_N, false>
    <<<cdiv(N, block_size_N), block_size_N, shmem_size, stream>>>(
    (CoordT*)coord.data_ptr(), (HashT*)bucket_id.data_ptr(), (CounterT*)bucket_counter.data_ptr(),
    (CounterT*)bucket_offset.data_ptr(),
    N, num_bucket, bucket_divisor, num_vox,
    (CoordT*)bbox_min.data_ptr(), (CoordT*)bbox_max.data_ptr(),
    coord.stride(1), coord.stride(0),
    bucket_id.stride(0), bucket_counter.stride(0),
    bucket_offset.stride(0)
    );
  cudaCheckLastErr();
  cudaCheckErr(cudaStreamSynchronize(stream));
  auto [number, str] = timer.end_and_format<std::chrono::microseconds>();
  LOG(WARNING) << "Time: " << str;
}


} // end of ::f3d
