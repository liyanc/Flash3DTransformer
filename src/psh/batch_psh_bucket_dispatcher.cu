/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 10/6/24
 */


#include <chrono>
#include <stdexcept>
#include <glog/logging.h>

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/ATen.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <kittens.cuh>
#include "common/runtime.h"
#include "common/arithmetic.cuh"
#include "common/fp_types.cuh"
#include "common/rand.cuh"
#include "common/load_store.cuh"
#include "common/device_manager.cuh"

#include "common/kernel_dispatcher.cuh"
#include "psh/batch_psh_bucket.cuh"
#include "psh/bucket_dispatcher.cuh"

#include "batch_psh_subbuck_reduce.cuh"


namespace f3d {

extern void
batch_psh_scatter(at::Tensor &coord, at::Tensor &bucket_id, at::Tensor &bucket_counter, at::Tensor &bucket_offset,
                  at::Tensor &batch_sep, uint32_t instance_max_N, uint32_t num_bucket, uint32_t bucket_divisor,
                  uint32_t bucket_size, uint32_t num_vox, at::Tensor &bbox_min, at::Tensor &bbox_max,
                  at::Tensor &probe_offsets, at::Tensor &cumsum_counter, at::Tensor &scatter_coord,
                  py::object hash_dtype_obj, py::object coord_dtype_obj, uint16_t block_size_N) {
  auto total_N = coord.size(0);

  assert_contiguous(
    {coord, bucket_id, bucket_counter, bucket_offset, batch_sep, probe_offsets, cumsum_counter, scatter_coord});
  assert_same_device(
    {coord, bucket_id, bucket_counter, bucket_offset, batch_sep, probe_offsets, cumsum_counter, scatter_coord});

  if (num_vox / bucket_divisor > num_bucket) {
    throw std::runtime_error("bucket divisor too small");
  }

  using CoordT = fp16;
  using CounterT = uint32_t;
  using HashT = uint16_t;
  auto bsize = uint32_t(batch_sep.size(0));
  auto stream = c10::cuda::getCurrentCUDAStream(coord.get_device());
  auto shmem_size = tk::MAX_SHARED_MEMORY;

  auto & dev_mgr = DeviceManagerSingleton::instance();
  auto rand_states = dev_mgr.get_states_dev(coord.get_device());

  auto & buck_disp = BucketingDispatcher<CoordT, CounterT, HashT>::instance();
  auto & recyc_disp = RecycleDispatcher<CoordT, CounterT, HashT>::instance();
  auto & scatter_disp = ScatterCoordDispatcher<CoordT, CounterT, HashT>::instance();

  auto balance_ptr = buck_disp.get_kernel_instance(num_bucket, H_ZORDER_MOD);
  auto recycle_ptr = recyc_disp.get_kernel_instance(num_bucket, H_ZORDER_MOD);
  auto scatter_ptr = scatter_disp.get_kernel_instance(num_bucket, H_ZORDER_MOD);

  cudaFuncSetAttribute(balance_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
  cudaCheckLastErr();
  cudaFuncSetAttribute(recycle_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
  cudaCheckLastErr();
  cudaFuncSetAttribute(scatter_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
  cudaCheckLastErr();

  if (dev_mgr.get_sync_stream()) {
    cudaCheckErr(cudaStreamSynchronize(stream));
  }
  QuickTimer timer{};

  auto grid_size = dim3{cdiv(instance_max_N, block_size_N), bsize, 1};
  balance_ptr<<<grid_size, block_size_N, shmem_size, stream>>>(
    (CoordT *) coord.data_ptr(), (HashT *) bucket_id.data_ptr(), (CounterT *) bucket_counter.data_ptr(),
    (CounterT *) bucket_offset.data_ptr(), (CounterT *) batch_sep.data_ptr(), batch_sep.size(0),
    total_N, bucket_divisor, bucket_size, num_vox,
    (CoordT *) bbox_min.data_ptr(), (CoordT *) bbox_max.data_ptr(), (int16_t *) probe_offsets.data_ptr(),
    coord.stride(1), coord.stride(0),
    bucket_id.stride(0), bucket_counter.stride(0), bucket_counter.stride(1),
    bucket_offset.stride(0),
    rand_states
    );
  cudaCheckLastErr();

  recycle_ptr<<<grid_size, block_size_N, shmem_size, stream>>>(
    (CoordT *) coord.data_ptr(), (HashT *) bucket_id.data_ptr(), (CounterT *) bucket_counter.data_ptr(),
    (CounterT *) bucket_offset.data_ptr(), (CounterT *) batch_sep.data_ptr(), batch_sep.size(0),
    total_N, bucket_divisor, bucket_size, num_vox,
    (CoordT *) bbox_min.data_ptr(), (CoordT *) bbox_max.data_ptr(), (int16_t *) probe_offsets.data_ptr(),
    coord.stride(1), coord.stride(0),
    bucket_counter.stride(0), bucket_counter.stride(1),
    rand_states
    );
  cudaCheckLastErr();

  scatter_ptr<<<grid_size, block_size_N, shmem_size, stream>>>(
    (CoordT *) coord.data_ptr(), (HashT *)bucket_id.data_ptr(), (CounterT *) bucket_counter.data_ptr(),
    (CounterT *)bucket_offset.data_ptr(), (CounterT *)cumsum_counter.data_ptr(),
    (CounterT *) batch_sep.data_ptr(), batch_sep.size(0),
    (CoordT *)scatter_coord.data_ptr(),
    total_N, bucket_divisor, bucket_size, num_vox,
    coord.stride(1), coord.stride(0),
    bucket_counter.stride(0), bucket_counter.stride(1),
    cumsum_counter.stride(0), cumsum_counter.stride(1)
    );
  cudaCheckLastErr();

  if (dev_mgr.get_sync_stream()) {
    cudaCheckErr(cudaStreamSynchronize(stream));
  }
}

extern void
batch_psh_scatter_pad(at::Tensor &coord, at::Tensor &bucket_id, at::Tensor &bucket_counter, at::Tensor &bucket_offset,
                      at::Tensor &batch_sep, uint32_t instance_max_N, uint32_t num_bucket, uint32_t bucket_divisor,
                      uint32_t bucket_size, uint32_t num_vox, at::Tensor &bbox_min, at::Tensor &bbox_max,
                      at::Tensor &probe_offsets, at::Tensor &cumsum_counter, at::Tensor &scatter_coord,
                      float pad_num) {
  auto total_N = coord.size(0);
  uint32_t pad_to_N = scatter_coord.size(0);
  uint32_t pad_size = pad_to_N - total_N;
  constexpr auto block_size_N = 1024u;

  assert_contiguous(
    {coord, bucket_id, bucket_counter, bucket_offset, batch_sep, probe_offsets, cumsum_counter, scatter_coord});
  assert_same_device(
    {coord, bucket_id, bucket_counter, bucket_offset, batch_sep, probe_offsets, cumsum_counter, scatter_coord});

  if (num_vox / bucket_divisor > num_bucket) {
    throw std::runtime_error("bucket divisor too small");
  }

  using CoordT = fp16;
  using CounterT = uint32_t;
  using HashT = uint16_t;
  auto bsize = uint32_t(batch_sep.size(0));
  auto stream = c10::cuda::getCurrentCUDAStream(coord.get_device());
  auto shmem_size = tk::MAX_SHARED_MEMORY;

  auto & dev_mgr = DeviceManagerSingleton::instance();
  auto rand_states = dev_mgr.get_states_dev(coord.get_device());

  auto & buck_disp = BucketingDispatcher<CoordT, CounterT, HashT>::instance();
  auto & recyc_disp = RecycleDispatcher<CoordT, CounterT, HashT>::instance();
  auto & scatter_disp = ScatterCoordPadDispatcher<CoordT, CounterT, HashT>::instance();

  auto balance_ptr = buck_disp.get_kernel_instance(num_bucket, H_ZORDER_MOD);
  auto recycle_ptr = recyc_disp.get_kernel_instance(num_bucket, H_ZORDER_MOD);
  auto scatter_ptr = scatter_disp.get_kernel_instance(num_bucket, H_ZORDER_MOD);

  cudaFuncSetAttribute(balance_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
  cudaCheckLastErr();
  cudaFuncSetAttribute(recycle_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
  cudaCheckLastErr();
  cudaFuncSetAttribute(scatter_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
  cudaCheckLastErr();

  if (dev_mgr.get_sync_stream()) {
    cudaCheckErr(cudaStreamSynchronize(stream));
  }
  //QuickTimer timer{};

  auto grid_size = dim3{cdiv(instance_max_N, block_size_N), bsize, 1};
  auto grid_pad = dim3{cdiv(std::max(instance_max_N, pad_size), block_size_N), bsize + 1, 1};

  balance_ptr<<<grid_size, block_size_N, shmem_size, stream>>>(
    (CoordT *) coord.data_ptr(), (HashT *) bucket_id.data_ptr(), (CounterT *) bucket_counter.data_ptr(),
    (CounterT *) bucket_offset.data_ptr(), (CounterT *) batch_sep.data_ptr(), batch_sep.size(0),
    total_N, bucket_divisor, bucket_size, num_vox,
    (CoordT *) bbox_min.data_ptr(), (CoordT *) bbox_max.data_ptr(), (int16_t *) probe_offsets.data_ptr(),
    coord.stride(1), coord.stride(0),
    bucket_id.stride(0), bucket_counter.stride(0), bucket_counter.stride(1),
    bucket_offset.stride(0),
    rand_states
    );
  cudaCheckLastErr();

  recycle_ptr<<<grid_size, block_size_N, shmem_size, stream>>>(
    (CoordT *) coord.data_ptr(), (HashT *) bucket_id.data_ptr(), (CounterT *) bucket_counter.data_ptr(),
    (CounterT *) bucket_offset.data_ptr(), (CounterT *) batch_sep.data_ptr(), batch_sep.size(0),
    total_N, bucket_divisor, bucket_size, num_vox,
    (CoordT *) bbox_min.data_ptr(), (CoordT *) bbox_max.data_ptr(), (int16_t *) probe_offsets.data_ptr(),
    coord.stride(1), coord.stride(0),
    bucket_counter.stride(0), bucket_counter.stride(1),
    rand_states
    );
  cudaCheckLastErr();

  scatter_ptr<<<grid_pad, block_size_N, shmem_size, stream>>>(
    (CoordT *) coord.data_ptr(), (HashT *)bucket_id.data_ptr(), (CounterT *) bucket_counter.data_ptr(),
    (CounterT *)bucket_offset.data_ptr(), (CounterT *)cumsum_counter.data_ptr(),
    (CounterT *) batch_sep.data_ptr(), batch_sep.size(0),
    (CoordT *)scatter_coord.data_ptr(),
    total_N, bucket_divisor, bucket_size, num_vox,
    coord.stride(1), coord.stride(0),
    bucket_counter.stride(0), bucket_counter.stride(1),
    cumsum_counter.stride(0), cumsum_counter.stride(1),
    pad_to_N, __float2half_rn(pad_num)
    );
  cudaCheckLastErr();

  if (dev_mgr.get_sync_stream()) {
    cudaCheckErr(cudaStreamSynchronize(stream));
  }
}

extern void
batch_psh_scatter_pad_hash(
  at::Tensor &coord, at::Tensor &bucket_id, at::Tensor &bucket_counter, at::Tensor &bucket_offset,
  at::Tensor &batch_sep, uint32_t instance_max_N, uint32_t num_bucket, uint32_t bucket_divisor,
  uint32_t bucket_size, uint32_t num_vox, at::Tensor &bbox_min, at::Tensor &bbox_max,
  at::Tensor &probe_offsets, at::Tensor &cumsum_counter, at::Tensor &scatter_coord,
  float pad_num, uint16_t hash_type
  ) {
  auto total_N = coord.size(0);
  uint32_t pad_to_N = scatter_coord.size(0);
  uint32_t pad_size = pad_to_N - total_N;
  constexpr auto block_size_N = 1024u;

  assert_contiguous(
    {coord, bucket_id, bucket_counter, bucket_offset, batch_sep, probe_offsets, cumsum_counter, scatter_coord});
  assert_same_device(
    {coord, bucket_id, bucket_counter, bucket_offset, batch_sep, probe_offsets, cumsum_counter, scatter_coord});

  if (num_vox / bucket_divisor > num_bucket) {
    throw std::runtime_error("bucket divisor too small");
  }
  if (scatter_coord.size(0) < coord.size(0)) {
    throw_format_error(
      "Output coord buffer length(%d) is smaller than the input length(%d).", scatter_coord.size(0), coord.size(0));
  }
  if (hash_type < 1 || hash_type > 4) {
    throw_format_error(
      "Currently only four types of hash modes are supported. Requested hash_type=%d is unsupported.",
      hash_type);
  }

  using CoordT = fp16;
  using CounterT = uint32_t;
  using HashT = uint16_t;
  auto bsize = uint32_t(batch_sep.size(0));
  auto stream = c10::cuda::getCurrentCUDAStream(coord.get_device());
  auto shmem_size = tk::MAX_SHARED_MEMORY;

  auto & dev_mgr = DeviceManagerSingleton::instance();
  auto rand_states = dev_mgr.get_states_dev(coord.get_device());

  auto & buck_disp = BucketingDispatcher<CoordT, CounterT, HashT>::instance();
  auto & recyc_disp = RecycleDispatcher<CoordT, CounterT, HashT>::instance();
  auto & scatter_disp = ScatterCoordPadDispatcher<CoordT, CounterT, HashT>::instance();

  auto balance_ptr = buck_disp.get_kernel_instance(num_bucket, static_cast<HASHTYPE>(hash_type));
  auto recycle_ptr = recyc_disp.get_kernel_instance(num_bucket, static_cast<HASHTYPE>(hash_type));
  auto scatter_ptr = scatter_disp.get_kernel_instance(num_bucket, static_cast<HASHTYPE>(hash_type));
  

  cudaFuncSetAttribute(balance_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
  cudaCheckLastErr();
  cudaFuncSetAttribute(recycle_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
  cudaCheckLastErr();
  cudaFuncSetAttribute(scatter_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);
  cudaCheckLastErr();

  if (dev_mgr.get_sync_stream()) {
    cudaCheckErr(cudaStreamSynchronize(stream));
  }

  // The last y-dim blocks are responsible of filling paddings,
  // so gridDim.x = max(longest instance, padding length) / 1024, and gridDim.y = bsize + 1
  auto grid_size = dim3{cdiv(instance_max_N, block_size_N), bsize, 1};
  auto grid_pad = dim3{cdiv(std::max(instance_max_N, pad_size), block_size_N), bsize + 1, 1};

  balance_ptr<<<grid_size, block_size_N, shmem_size, stream>>>(
    (CoordT *) coord.data_ptr(), (HashT *) bucket_id.data_ptr(), (CounterT *) bucket_counter.data_ptr(),
    (CounterT *) bucket_offset.data_ptr(), (CounterT *) batch_sep.data_ptr(), batch_sep.size(0),
    total_N, bucket_divisor, bucket_size, num_vox,
    (CoordT *) bbox_min.data_ptr(), (CoordT *) bbox_max.data_ptr(), (int16_t *) probe_offsets.data_ptr(),
    coord.stride(1), coord.stride(0),
    bucket_id.stride(0), bucket_counter.stride(0), bucket_counter.stride(1),
    bucket_offset.stride(0),
    rand_states
    );
  cudaCheckLastErr();

  recycle_ptr<<<grid_size, block_size_N, shmem_size, stream>>>(
    (CoordT *) coord.data_ptr(), (HashT *) bucket_id.data_ptr(), (CounterT *) bucket_counter.data_ptr(),
    (CounterT *) bucket_offset.data_ptr(), (CounterT *) batch_sep.data_ptr(), batch_sep.size(0),
    total_N, bucket_divisor, bucket_size, num_vox,
    (CoordT *) bbox_min.data_ptr(), (CoordT *) bbox_max.data_ptr(), (int16_t *) probe_offsets.data_ptr(),
    coord.stride(1), coord.stride(0),
    bucket_counter.stride(0), bucket_counter.stride(1),
    rand_states
    );
  cudaCheckLastErr();

  scatter_ptr<<<grid_pad, block_size_N, shmem_size, stream>>>(
    (CoordT *) coord.data_ptr(), (HashT *)bucket_id.data_ptr(), (CounterT *) bucket_counter.data_ptr(),
    (CounterT *)bucket_offset.data_ptr(), (CounterT *)cumsum_counter.data_ptr(),
    (CounterT *) batch_sep.data_ptr(), batch_sep.size(0),
    (CoordT *)scatter_coord.data_ptr(),
    total_N, bucket_divisor, bucket_size, num_vox,
    coord.stride(1), coord.stride(0),
    bucket_counter.stride(0), bucket_counter.stride(1),
    cumsum_counter.stride(0), cumsum_counter.stride(1),
    pad_to_N, __float2half_rn(pad_num)
    );
  cudaCheckLastErr();

  if (dev_mgr.get_sync_stream()) {
    cudaCheckErr(cudaStreamSynchronize(stream));
  }
}

} // end of ::f3d
