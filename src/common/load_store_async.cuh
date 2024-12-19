/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 10/25/24
 */


#ifndef FLASH3DPSHATTN_LOAD_STORE_ASYNC_CUH
#define FLASH3DPSHATTN_LOAD_STORE_ASYNC_CUH

#ifndef F3D_UNROLL
  #if defined(__CUDACC__)
    #define F3D_UNROLL _Pragma("unroll")
  #else
    #define F3D_UNROLL
  #endif
#endif

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>

#include <kittens.cuh>
#include "common/arithmetic.cuh"
#include "common/fp_types.cuh"
#include "common/rand.cuh"

namespace tk = kittens;

namespace f3d {

__device__ __forceinline__ uint _smem(const void * addr) {return __cvta_generic_to_shared(addr);}
__device__ __forceinline__ void _commit_group() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void _wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }
__device__ __forceinline__ void _wait_sync_all() { asm volatile("cp.async.wait_all;" ::: "memory"); __syncthreads();}

template <uint16_t N>
__device__ __forceinline__ void _wait_groups_except_and_sync_all()
{ asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory"); __syncthreads();}
template <uint16_t N>
__device__ __forceinline__ void _wait_groups_except_no_sync()
{ asm volatile("cp.async.wait_group %0;" :: "n"(N) : "memory");}

__device__ __forceinline__ void load_reg_stride_aligned_16B(
  void * dst, void * src) {
  *(float4 *) dst = *(float4 *) src;
}

template <tk::ducks::sv::all SV>
__device__ __forceinline__ void print_sv(SV & src, uint16_t warpid, uint32_t iter) {
  if (threadIdx.x == 0) {
    printf("Blk[%d,%d], Iter[%d], Warp[%d]\n",
           blockIdx.x, blockIdx.y, iter, warpid);
    for (auto i = 0; i < SV::length; ++i) {
      printf("%.2f ", float(src[i]));
    }
    printf("\n");
  }
}

template <typename SrcType, uint16_t Rows, uint16_t Cols>
__device__ __forceinline__ void print_stile_row(
  SrcType (&st)[Rows][Cols], uint16_t row) {
  if (threadIdx.x == 0) {
    for (uint16_t i = 0; i < Cols; ++i) {
      printf("%f ", float(st[row][i]));
    }
  }
}

template <typename SrcType, uint16_t Rows, uint16_t Cols>
__device__ __forceinline__ void print_stile_col(
  SrcType (&st)[Rows][Cols], uint16_t col) {
  if (threadIdx.x == 0) {
    for (uint16_t i = 0; i < Rows; ++i) {
      printf("%f ", float(st[i][col]));
    }
  }
}


template <typename SrcType, uint16_t Rows, uint16_t Cols>
__device__ __forceinline__ void print_stile_rec_raw(
  SrcType (&st)[Rows][Cols], uint16_t rowb, uint16_t rowe, uint16_t colb, uint16_t cole) {
  if (threadIdx.x == 0) {
    printf("Blk[%d,%d], ST[%d:%d, %d:%d]\n",
           blockIdx.x, blockIdx.y, rowb, rowe, colb, cole);
    for (uint16_t r = rowb; r < rowe; ++r) {
      for (uint16_t c = colb; c < cole; ++c) {
        printf("%.2f ", float(st[r][c]));
      }
      printf("\n");
    }
  }
}

template <tk::ducks::st::all ST>
__device__ __forceinline__ void print_stile_rec(
  ST &stile, uint16_t warpid, uint32_t iter, uint16_t rowb, uint16_t rowe, uint16_t colb, uint16_t cole) {
  if (tk::laneid() == 0) {
    printf("Blk[%d,%d], Iter[%d], Warp[%d], ST[%d:%d, %d:%d]\n",
           blockIdx.x, blockIdx.y, iter, warpid, rowb, rowe, colb, cole);
    for (uint16_t r = rowb; r < rowe; ++r) {
      for (uint16_t c = colb; c < cole; ++c) {
        printf("%.2f ", float(stile[{r, c}]));
      }
      printf("\n");
    }
  }
}

template <tk::ducks::rt::all RT>
__device__ __forceinline__ void print_rtile_rec(
  RT &rtile, uint16_t rowb, uint16_t rowe, uint16_t colb, uint16_t cole) {
  if (threadIdx.x == 0) {
    printf("Blk[%d,%d]\n", blockIdx.x, blockIdx.y);
  }
  __syncthreads();

  for (uint16_t r = rowb; r < rowe; ++r) {
    for (uint16_t c = colb; c < cole; ++c) {
      fp16_2 * d = reinterpret_cast<fp16_2 *>(rtile.tiles[r][c].data);
      printf(
        "[%d,%d,t=%d] %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n",
        r, c, threadIdx.x,
        float(d[0].x), float(d[0].y), float(d[1].x), float(d[1].y),
        float(d[2].x), float(d[2].y), float(d[3].x), float(d[3].y),
        float(d[4].x), float(d[4].y), float(d[5].x), float(d[5].y),
        float(d[6].x), float(d[6].y), float(d[7].x), float(d[7].y)
        );
    }
  }
}


template<uint16_t VecStride, uint16_t BlockSize>
__global__ void block_load_async(
  const bf16 * __restrict__ input_feat, uint32_t feat_dim,
  uint16_t stride_featN
  ) {
  if (blockIdx.x > 0 || blockIdx.y > 0) return;
  auto warpid = tk::warpid();
  auto laneid = tk::laneid();
  auto threadid = threadIdx.x;

  extern __shared__ tk::alignment_dummy __shm[];
  tk::shared_allocator al((int *) &__shm[0]);

  constexpr uint16_t TileRows = 128;
  constexpr uint16_t CPSize = 16;
  constexpr uint16_t ActWarps = BlockSize / WarpSize;
  constexpr uint16_t LoopsCol = (VecStride * sizeof(bf16)) / (WarpSize * CPSize);
  constexpr uint16_t LoopsRow = TileRows / ActWarps;
  bf16 (&shm_tile)[TileRows][VecStride] = al.allocate<bf16, TileRows, VecStride>();
  auto base_src = reinterpret_cast<const char *>(input_feat);
  auto base_dst = reinterpret_cast<char *>(shm_tile);
  auto stride_row_shm_bytes = VecStride * sizeof(bf16);
  auto stride_row_dram_bytes = stride_featN * sizeof(bf16);

  for (uint16_t ri = 0; ri < LoopsRow; ++ri) {
    for (uint16_t ci = 0; ci < LoopsCol; ++ci) {
      auto row = ri * ActWarps + warpid;
      auto col = ci * WarpSize * CPSize + laneid * CPSize;
      auto src = base_src + row * stride_row_dram_bytes + col;
      auto dst = base_dst + row * stride_row_shm_bytes + col;
      asm volatile(
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;"
        :: "r"(_smem(dst)), "l"(src), "n"(CPSize)
        : "memory"
        );
    }
  }
  _commit_group();
  _wait_all();

  __syncthreads();
}


template<tk::ducks::st::all ST>
__device__ __forceinline__ static void load_async_2B_elem(
  ST &dst, const typename ST::dtype *src, uint32_t row_stride) {
  auto laneid = tk::laneid();

  __syncwarp();
  constexpr uint16_t cp_size = 16u;
  constexpr uint16_t cp_elem = cp_size / sizeof(typename ST::dtype);
  auto memcpy_per_row = dst.cols / cp_elem;
  auto total_calls = dst.rows * dst.cols / (cp_elem * WarpSize);

  #pragma unroll
  for (int32_t i = 0; i < total_calls; ++i) {
    int32_t lane_idx = i * WarpSize + laneid;
    int32_t row = lane_idx / memcpy_per_row;
    int32_t col = (lane_idx * cp_elem) % dst.cols;

    auto dst_smem = _smem(&dst[{row, col}]);
    auto src_dram = src + row * row_stride + col;
    asm volatile(
      "cp.async.cg.shared::cta.global.L2::128B [%0], [%1], %2;"
      :: "r"(dst_smem), "l"(src_dram), "n"(cp_size)
      : "memory"
      );
  }

  _commit_group();
}

template<tk::ducks::st::all ST>
__device__ __forceinline__ static void load_async_2B_elem(
  ST &dst, const typename ST::dtype *src, uint32_t row_stride, uint32_t max_row) {
  auto laneid = tk::laneid();

  __syncwarp();
  constexpr uint16_t cp_size = 16u;
  constexpr uint16_t cp_elem = cp_size / sizeof(typename ST::dtype);
  auto memcpy_per_row = dst.cols / cp_elem;
  auto total_calls = dst.rows * dst.cols / (cp_elem * WarpSize);

  #pragma unroll
  for (int32_t i = 0; i < total_calls; ++i) {
    int32_t lane_idx = i * WarpSize + laneid;
    int32_t row = lane_idx / memcpy_per_row;
    int32_t col = (lane_idx * cp_elem) % dst.cols;
    int32_t src_size = (row > max_row)? 0: cp_size;

    auto dst_smem = _smem(&dst[{row, col}]);
    auto src_dram = src + row * row_stride + col;
    asm volatile(
      "cp.async.cg.shared::cta.global.L2::128B [%0], [%1], %2, %3;"
      :: "r"(dst_smem), "l"(src_dram), "n"(cp_size), "r"(src_size)
      : "memory"
      );
  }

  _commit_group();
}

template<tk::ducks::sv::all SV, typename DSTU>
__device__ __forceinline__ void store(DSTU *dst, const SV &src, const int elem_stride) {
  constexpr int elem_per_transfer = 1;
  constexpr int total_iters = cdiv_dev(src.length, elem_per_transfer * WarpSize);
  __syncwarp();
  #pragma unroll
  for(auto warp_iter = 0; warp_iter < total_iters; ++warp_iter) {
    auto elem_id = warp_iter * WarpSize + tk::laneid();
    if (elem_id < src.length) {
      dst[elem_id * elem_stride] = DSTU(src[elem_id]);
    }
  }
}

template<tk::ducks::sv::all SV, typename SRCU>
__device__ __forceinline__ void load(SV &dst, const SRCU *src_ptr, const int elem_stride) {
  constexpr int elem_per_transfer = 1;
  constexpr int total_iters = cdiv_dev(dst.length, elem_per_transfer * WarpSize);
  using T = typename SV::dtype;
  __syncwarp();
  #pragma unroll
  for(auto warp_iter = 0; warp_iter < total_iters; ++warp_iter) {
    auto elem_id = warp_iter * WarpSize + tk::laneid();
    if (elem_id < dst.length) {
      dst[elem_id] = T(src_ptr[elem_id * elem_stride]);
    }
  }
}

} // end of ::f3d


#endif //FLASH3DPSHATTN_LOAD_STORE_ASYNC_CUH
