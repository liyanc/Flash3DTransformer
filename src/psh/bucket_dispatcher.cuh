/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 10/10/24
 */


#ifndef FLASH3DPSHATTN_BUCKET_DISPATCHER_H
#define FLASH3DPSHATTN_BUCKET_DISPATCHER_H

#include "common/kernel_dispatcher.cuh"
#include "psh/batch_psh_bucket.cuh"

namespace f3d {

template <typename CoordT, typename CounterT, typename HashT>
struct BucketingKernel {
  using PtrType = void(*) (
    const CoordT * __restrict__ coord, HashT * __restrict__ bucket_id, CounterT * __restrict__ bucket_counter,
    CounterT * __restrict__ bucket_offset, const CounterT * __restrict__ batch_sep, uint32_t batch_size,
    uint32_t total_N, uint32_t bucket_seg_divisor, uint32_t bucket_size, uint32_t num_vox,
    const CoordT * __restrict__ bbox_min, const CoordT * __restrict__ bbox_max, const int16_t * __restrict__ probe_offset,
    uint16_t stride_coordD, uint16_t stride_coordN,
    uint16_t stride_bucketid_N, uint16_t stride_bucket_cntB, uint16_t stride_bucket_cntN,
    uint16_t stride_bucketoff_N,
    curandState * global_rand_base);
};

template <typename CoordT, typename CounterT, typename HashT>
struct RecyleKernel {
  using PtrType = void(*) (
    const CoordT * __restrict__ coord, HashT * __restrict__ bucket_id, CounterT * __restrict__ bucket_counter,
    CounterT * __restrict__ bucket_offset, const CounterT * __restrict__ batch_sep, uint32_t batch_size,
    uint32_t total_N, uint32_t bucket_seg_divisor, uint32_t bucket_size, uint32_t num_vox,
    const CoordT * __restrict__ bbox_min, const CoordT * __restrict__ bbox_max, const int16_t * __restrict__ probe_offset,
    uint16_t stride_coordD, uint16_t stride_coordN,
    uint16_t stride_bucket_cntB, uint16_t stride_bucket_cntN,
    curandState * global_rand_base);
};

template <typename CoordT, typename CounterT, typename HashT>
struct ScatterCoordKernel {
  using PtrType = void(*) (
    const CoordT * __restrict__ coord, HashT * __restrict__ bucket_id, CounterT * __restrict__ bucket_counter,
    CounterT * __restrict__ bucket_offset, CounterT * __restrict__ cumsum_counter,
    const CounterT * __restrict__ batch_sep, uint32_t batch_size,
    CoordT * __restrict__ scattered_coord,
    uint32_t total_N, uint32_t bucket_seg_divisor, uint32_t bucket_size, uint32_t num_vox,
    uint16_t stride_coordD, uint16_t stride_coordN,
    uint16_t stride_bucket_cntB, uint16_t stride_bucket_cntN,
    uint16_t stride_cumsum_cntB, uint16_t stride_cumsum_cntN);
};

template <typename CoordT, typename CounterT, typename HashT>
struct ScatterCoordPadKernel {
  using PtrType = void(*) (
    const CoordT * __restrict__ coord, HashT * __restrict__ bucket_id, CounterT * __restrict__ bucket_counter,
    CounterT * __restrict__ bucket_offset, CounterT * __restrict__ cumsum_counter,
    const CounterT * __restrict__ batch_sep, uint32_t batch_size,
    CoordT * __restrict__ scattered_coord,
    uint32_t total_N, uint32_t bucket_seg_divisor, uint32_t bucket_size, uint32_t num_vox,
    uint16_t stride_coordD, uint16_t stride_coordN,
    uint16_t stride_bucket_cntB, uint16_t stride_bucket_cntN,
    uint16_t stride_cumsum_cntB, uint16_t stride_cumsum_cntN,
    uint32_t pad_to_N, CoordT pad_num);
};


template <typename CoordT, typename CounterT, typename HashT>
class BucketingDispatcher :
  public KernelDispatcher
    <BucketingDispatcher<CoordT, CounterT, HashT>,
      CoordT, CounterT, HashT,
      BucketingKernel<CoordT, CounterT, HashT>>{
public:
  void initialize() {
    if (this->kernel_map.size() > 0)
      return;

    constexpr auto store_stage = S_FINAL;
    constexpr auto special_buck = 0;
    constexpr auto two_stage_seg = 8; //64
    constexpr auto BLOCK_SIZE_N = 1024u;

    this->kernel_map[{32, H_XORSUM_DIV}] = batch_psh_bucketing_balancing_ker
      <CoordT, CounterT, HashT, 3, 32, two_stage_seg, special_buck, BLOCK_SIZE_N, store_stage, H_XORSUM_DIV>;
    this->kernel_map[{32, H_XORSUM_MOD}] = batch_psh_bucketing_balancing_ker
      <CoordT, CounterT, HashT, 3, 32, two_stage_seg, special_buck, BLOCK_SIZE_N, store_stage, H_XORSUM_MOD>;
    this->kernel_map[{32, H_ZORDER_DIV}] = batch_psh_bucketing_balancing_ker
      <CoordT, CounterT, HashT, 3, 32, two_stage_seg, special_buck, BLOCK_SIZE_N, store_stage, H_ZORDER_DIV>;
    this->kernel_map[{32, H_ZORDER_MOD}] = batch_psh_bucketing_balancing_ker
      <CoordT, CounterT, HashT, 3, 32, two_stage_seg, special_buck, BLOCK_SIZE_N, store_stage, H_ZORDER_MOD>;

    this->kernel_map[{64, H_XORSUM_DIV}] = batch_psh_bucketing_balancing_ker
      <CoordT, CounterT, HashT, 3, 64, two_stage_seg, special_buck, BLOCK_SIZE_N, store_stage, H_XORSUM_DIV>;
    this->kernel_map[{64, H_XORSUM_MOD}] = batch_psh_bucketing_balancing_ker
      <CoordT, CounterT, HashT, 3, 64, two_stage_seg, special_buck, BLOCK_SIZE_N, store_stage, H_XORSUM_MOD>;
    this->kernel_map[{64, H_ZORDER_DIV}] = batch_psh_bucketing_balancing_ker
      <CoordT, CounterT, HashT, 3, 64, two_stage_seg, special_buck, BLOCK_SIZE_N, store_stage, H_ZORDER_DIV>;
    this->kernel_map[{64, H_ZORDER_MOD}] = batch_psh_bucketing_balancing_ker
      <CoordT, CounterT, HashT, 3, 64, two_stage_seg, special_buck, BLOCK_SIZE_N, store_stage, H_ZORDER_MOD>;

    this->kernel_map[{128, H_XORSUM_DIV}] = batch_psh_bucketing_balancing_ker
      <CoordT, CounterT, HashT, 3, 128, two_stage_seg, special_buck, BLOCK_SIZE_N, store_stage, H_XORSUM_DIV>;
    this->kernel_map[{128, H_XORSUM_MOD}] = batch_psh_bucketing_balancing_ker
      <CoordT, CounterT, HashT, 3, 128, two_stage_seg, special_buck, BLOCK_SIZE_N, store_stage, H_XORSUM_MOD>;
    this->kernel_map[{128, H_ZORDER_DIV}] = batch_psh_bucketing_balancing_ker
      <CoordT, CounterT, HashT, 3, 128, two_stage_seg, special_buck, BLOCK_SIZE_N, store_stage, H_ZORDER_DIV>;
    this->kernel_map[{128, H_ZORDER_MOD}] = batch_psh_bucketing_balancing_ker
      <CoordT, CounterT, HashT, 3, 128, two_stage_seg, special_buck, BLOCK_SIZE_N, store_stage, H_ZORDER_MOD>;

    this->kernel_map[{256, H_XORSUM_DIV}] = batch_psh_bucketing_balancing_ker
      <CoordT, CounterT, HashT, 3, 256, two_stage_seg, special_buck, BLOCK_SIZE_N, store_stage, H_XORSUM_DIV>;
    this->kernel_map[{256, H_XORSUM_MOD}] = batch_psh_bucketing_balancing_ker
      <CoordT, CounterT, HashT, 3, 256, two_stage_seg, special_buck, BLOCK_SIZE_N, store_stage, H_XORSUM_MOD>;
    this->kernel_map[{256, H_ZORDER_DIV}] = batch_psh_bucketing_balancing_ker
      <CoordT, CounterT, HashT, 3, 256, two_stage_seg, special_buck, BLOCK_SIZE_N, store_stage, H_ZORDER_DIV>;
    this->kernel_map[{256, H_ZORDER_MOD}] = batch_psh_bucketing_balancing_ker
      <CoordT, CounterT, HashT, 3, 256, two_stage_seg, special_buck, BLOCK_SIZE_N, store_stage, H_ZORDER_MOD>;

    this->kernel_map[{512, H_XORSUM_DIV}] = batch_psh_bucketing_balancing_ker
      <CoordT, CounterT, HashT, 3, 512, two_stage_seg, special_buck, BLOCK_SIZE_N, store_stage, H_XORSUM_DIV>;
    this->kernel_map[{512, H_XORSUM_MOD}] = batch_psh_bucketing_balancing_ker
      <CoordT, CounterT, HashT, 3, 512, two_stage_seg, special_buck, BLOCK_SIZE_N, store_stage, H_XORSUM_MOD>;
    this->kernel_map[{512, H_ZORDER_DIV}] = batch_psh_bucketing_balancing_ker
      <CoordT, CounterT, HashT, 3, 512, two_stage_seg, special_buck, BLOCK_SIZE_N, store_stage, H_ZORDER_DIV>;
    this->kernel_map[{512, H_ZORDER_MOD}] = batch_psh_bucketing_balancing_ker
      <CoordT, CounterT, HashT, 3, 512, two_stage_seg, special_buck, BLOCK_SIZE_N, store_stage, H_ZORDER_MOD>;
  }
};


template <typename CoordT, typename CounterT, typename HashT>
class RecycleDispatcher :
  public KernelDispatcher
    <RecycleDispatcher<CoordT, CounterT, HashT>,
      CoordT, CounterT, HashT,
      RecyleKernel<CoordT, CounterT, HashT>>{
public:
  void initialize() {
    if (this->kernel_map.size() > 0)
      return;

    constexpr auto special_buck = 0;
    constexpr auto BLOCK_SIZE_N = 1024u;

    this->kernel_map[{32, H_XORSUM_DIV}] = batch_psh_distribute_recycle_ker
      <CoordT, CounterT, HashT, 3, 32, special_buck, BLOCK_SIZE_N, H_XORSUM_DIV>;
    this->kernel_map[{32, H_XORSUM_MOD}] = batch_psh_distribute_recycle_ker
      <CoordT, CounterT, HashT, 3, 32, special_buck, BLOCK_SIZE_N, H_XORSUM_MOD>;
    this->kernel_map[{32, H_ZORDER_DIV}] = batch_psh_distribute_recycle_ker
      <CoordT, CounterT, HashT, 3, 32, special_buck, BLOCK_SIZE_N, H_ZORDER_DIV>;
    this->kernel_map[{32, H_ZORDER_MOD}] = batch_psh_distribute_recycle_ker
      <CoordT, CounterT, HashT, 3, 32, special_buck, BLOCK_SIZE_N, H_ZORDER_MOD>;

    this->kernel_map[{64, H_XORSUM_DIV}] = batch_psh_distribute_recycle_ker
      <CoordT, CounterT, HashT, 3, 64, special_buck, BLOCK_SIZE_N, H_XORSUM_DIV>;
    this->kernel_map[{64, H_XORSUM_MOD}] = batch_psh_distribute_recycle_ker
      <CoordT, CounterT, HashT, 3, 64, special_buck, BLOCK_SIZE_N, H_XORSUM_MOD>;
    this->kernel_map[{64, H_ZORDER_DIV}] = batch_psh_distribute_recycle_ker
      <CoordT, CounterT, HashT, 3, 64, special_buck, BLOCK_SIZE_N, H_ZORDER_DIV>;
    this->kernel_map[{64, H_ZORDER_MOD}] = batch_psh_distribute_recycle_ker
      <CoordT, CounterT, HashT, 3, 64, special_buck, BLOCK_SIZE_N, H_ZORDER_MOD>;

    this->kernel_map[{128, H_XORSUM_DIV}] = batch_psh_distribute_recycle_ker
      <CoordT, CounterT, HashT, 3, 128, special_buck, BLOCK_SIZE_N, H_XORSUM_DIV>;
    this->kernel_map[{128, H_XORSUM_MOD}] = batch_psh_distribute_recycle_ker
      <CoordT, CounterT, HashT, 3, 128, special_buck, BLOCK_SIZE_N, H_XORSUM_MOD>;
    this->kernel_map[{128, H_ZORDER_DIV}] = batch_psh_distribute_recycle_ker
      <CoordT, CounterT, HashT, 3, 128, special_buck, BLOCK_SIZE_N, H_ZORDER_DIV>;
    this->kernel_map[{128, H_ZORDER_MOD}] = batch_psh_distribute_recycle_ker
      <CoordT, CounterT, HashT, 3, 128, special_buck, BLOCK_SIZE_N, H_ZORDER_MOD>;

    this->kernel_map[{256, H_XORSUM_DIV}] = batch_psh_distribute_recycle_ker
      <CoordT, CounterT, HashT, 3, 256, special_buck, BLOCK_SIZE_N, H_XORSUM_DIV>;
    this->kernel_map[{256, H_XORSUM_MOD}] = batch_psh_distribute_recycle_ker
      <CoordT, CounterT, HashT, 3, 256, special_buck, BLOCK_SIZE_N, H_XORSUM_MOD>;
    this->kernel_map[{256, H_ZORDER_DIV}] = batch_psh_distribute_recycle_ker
      <CoordT, CounterT, HashT, 3, 256, special_buck, BLOCK_SIZE_N, H_ZORDER_DIV>;
    this->kernel_map[{256, H_ZORDER_MOD}] = batch_psh_distribute_recycle_ker
      <CoordT, CounterT, HashT, 3, 256, special_buck, BLOCK_SIZE_N, H_ZORDER_MOD>;

    this->kernel_map[{512, H_XORSUM_DIV}] = batch_psh_distribute_recycle_ker
      <CoordT, CounterT, HashT, 3, 512, special_buck, BLOCK_SIZE_N, H_XORSUM_DIV>;
    this->kernel_map[{512, H_XORSUM_MOD}] = batch_psh_distribute_recycle_ker
      <CoordT, CounterT, HashT, 3, 512, special_buck, BLOCK_SIZE_N, H_XORSUM_MOD>;
    this->kernel_map[{512, H_ZORDER_DIV}] = batch_psh_distribute_recycle_ker
      <CoordT, CounterT, HashT, 3, 512, special_buck, BLOCK_SIZE_N, H_ZORDER_DIV>;
    this->kernel_map[{512, H_ZORDER_MOD}] = batch_psh_distribute_recycle_ker
      <CoordT, CounterT, HashT, 3, 512, special_buck, BLOCK_SIZE_N, H_ZORDER_MOD>;
  }
};

template <typename CoordT, typename CounterT, typename HashT>
class ScatterCoordDispatcher :
  public KernelDispatcher
    <ScatterCoordDispatcher<CoordT, CounterT, HashT>,
      CoordT, CounterT, HashT,
      ScatterCoordKernel<CoordT, CounterT, HashT>> {
public:
  void initialize() {
    if (this->kernel_map.size() > 0)
      return;

    constexpr auto BLOCK_SIZE_N = 1024u;

    this->kernel_map[{32, H_XORSUM_DIV}] = batch_psh_cumsum_scatter_ker
      <CoordT, CounterT, HashT, 3, 32, BLOCK_SIZE_N>;
    this->kernel_map[{32, H_XORSUM_MOD}] = batch_psh_cumsum_scatter_ker
      <CoordT, CounterT, HashT, 3, 32, BLOCK_SIZE_N>;
    this->kernel_map[{32, H_ZORDER_MOD}] = batch_psh_cumsum_scatter_ker
      <CoordT, CounterT, HashT, 3, 32, BLOCK_SIZE_N>;
    this->kernel_map[{32, H_ZORDER_DIV}] = batch_psh_cumsum_scatter_ker
      <CoordT, CounterT, HashT, 3, 32, BLOCK_SIZE_N>;

    this->kernel_map[{64, H_XORSUM_DIV}] = batch_psh_cumsum_scatter_ker
      <CoordT, CounterT, HashT, 3, 64, BLOCK_SIZE_N>;
    this->kernel_map[{64, H_XORSUM_MOD}] = batch_psh_cumsum_scatter_ker
      <CoordT, CounterT, HashT, 3, 64, BLOCK_SIZE_N>;
    this->kernel_map[{64, H_ZORDER_MOD}] = batch_psh_cumsum_scatter_ker
      <CoordT, CounterT, HashT, 3, 64, BLOCK_SIZE_N>;
    this->kernel_map[{64, H_ZORDER_DIV}] = batch_psh_cumsum_scatter_ker
      <CoordT, CounterT, HashT, 3, 64, BLOCK_SIZE_N>;

    this->kernel_map[{128, H_XORSUM_DIV}] = batch_psh_cumsum_scatter_ker
      <CoordT, CounterT, HashT, 3, 128, BLOCK_SIZE_N>;
    this->kernel_map[{128, H_XORSUM_MOD}] = batch_psh_cumsum_scatter_ker
      <CoordT, CounterT, HashT, 3, 128, BLOCK_SIZE_N>;
    this->kernel_map[{128, H_ZORDER_MOD}] = batch_psh_cumsum_scatter_ker
      <CoordT, CounterT, HashT, 3, 128, BLOCK_SIZE_N>;
    this->kernel_map[{128, H_ZORDER_DIV}] = batch_psh_cumsum_scatter_ker
      <CoordT, CounterT, HashT, 3, 128, BLOCK_SIZE_N>;

    this->kernel_map[{256, H_XORSUM_DIV}] = batch_psh_cumsum_scatter_ker
      <CoordT, CounterT, HashT, 3, 256, BLOCK_SIZE_N>;
    this->kernel_map[{256, H_XORSUM_MOD}] = batch_psh_cumsum_scatter_ker
      <CoordT, CounterT, HashT, 3, 256, BLOCK_SIZE_N>;
    this->kernel_map[{256, H_ZORDER_MOD}] = batch_psh_cumsum_scatter_ker
      <CoordT, CounterT, HashT, 3, 256, BLOCK_SIZE_N>;
    this->kernel_map[{256, H_ZORDER_DIV}] = batch_psh_cumsum_scatter_ker
      <CoordT, CounterT, HashT, 3, 256, BLOCK_SIZE_N>;

    this->kernel_map[{512, H_XORSUM_DIV}] = batch_psh_cumsum_scatter_ker
      <CoordT, CounterT, HashT, 3, 512, BLOCK_SIZE_N>;
    this->kernel_map[{512, H_XORSUM_MOD}] = batch_psh_cumsum_scatter_ker
      <CoordT, CounterT, HashT, 3, 512, BLOCK_SIZE_N>;
    this->kernel_map[{512, H_ZORDER_MOD}] = batch_psh_cumsum_scatter_ker
      <CoordT, CounterT, HashT, 3, 512, BLOCK_SIZE_N>;
    this->kernel_map[{512, H_ZORDER_DIV}] = batch_psh_cumsum_scatter_ker
      <CoordT, CounterT, HashT, 3, 512, BLOCK_SIZE_N>;
  }
};

template <typename CoordT, typename CounterT, typename HashT>
class ScatterCoordPadDispatcher :
  public KernelDispatcher
    <ScatterCoordPadDispatcher<CoordT, CounterT, HashT>,
      CoordT, CounterT, HashT,
      ScatterCoordPadKernel<CoordT, CounterT, HashT>> {
public:
  void initialize() {
    if (this->kernel_map.size() > 0)
      return;

    constexpr auto BLOCK_SIZE_N = 1024u;

    this->kernel_map[{32, H_XORSUM_DIV}] = batch_psh_cumsum_scatter_postpad_ker
      <CoordT, CounterT, HashT, 3, 32, BLOCK_SIZE_N>;
    this->kernel_map[{32, H_XORSUM_MOD}] = batch_psh_cumsum_scatter_postpad_ker
      <CoordT, CounterT, HashT, 3, 32, BLOCK_SIZE_N>;
    this->kernel_map[{32, H_ZORDER_MOD}] = batch_psh_cumsum_scatter_postpad_ker
      <CoordT, CounterT, HashT, 3, 32, BLOCK_SIZE_N>;
    this->kernel_map[{32, H_ZORDER_DIV}] = batch_psh_cumsum_scatter_postpad_ker
      <CoordT, CounterT, HashT, 3, 32, BLOCK_SIZE_N>;

    // Group for size 64
    this->kernel_map[{64, H_XORSUM_DIV}] = batch_psh_cumsum_scatter_postpad_ker
      <CoordT, CounterT, HashT, 3, 64, BLOCK_SIZE_N>;
    this->kernel_map[{64, H_XORSUM_MOD}] = batch_psh_cumsum_scatter_postpad_ker
      <CoordT, CounterT, HashT, 3, 64, BLOCK_SIZE_N>;
    this->kernel_map[{64, H_ZORDER_MOD}] = batch_psh_cumsum_scatter_postpad_ker
      <CoordT, CounterT, HashT, 3, 64, BLOCK_SIZE_N>;
    this->kernel_map[{64, H_ZORDER_DIV}] = batch_psh_cumsum_scatter_postpad_ker
      <CoordT, CounterT, HashT, 3, 64, BLOCK_SIZE_N>;

    this->kernel_map[{128, H_XORSUM_DIV}] = batch_psh_cumsum_scatter_postpad_ker
      <CoordT, CounterT, HashT, 3, 128, BLOCK_SIZE_N>;
    this->kernel_map[{128, H_XORSUM_MOD}] = batch_psh_cumsum_scatter_postpad_ker
      <CoordT, CounterT, HashT, 3, 128, BLOCK_SIZE_N>;
    this->kernel_map[{128, H_ZORDER_MOD}] = batch_psh_cumsum_scatter_postpad_ker
      <CoordT, CounterT, HashT, 3, 128, BLOCK_SIZE_N>;
    this->kernel_map[{128, H_ZORDER_DIV}] = batch_psh_cumsum_scatter_postpad_ker
      <CoordT, CounterT, HashT, 3, 128, BLOCK_SIZE_N>;

    this->kernel_map[{256, H_XORSUM_DIV}] = batch_psh_cumsum_scatter_postpad_ker
      <CoordT, CounterT, HashT, 3, 256, BLOCK_SIZE_N>;
    this->kernel_map[{256, H_XORSUM_MOD}] = batch_psh_cumsum_scatter_postpad_ker
      <CoordT, CounterT, HashT, 3, 256, BLOCK_SIZE_N>;
    this->kernel_map[{256, H_ZORDER_MOD}] = batch_psh_cumsum_scatter_postpad_ker
      <CoordT, CounterT, HashT, 3, 256, BLOCK_SIZE_N>;
    this->kernel_map[{256, H_ZORDER_DIV}] = batch_psh_cumsum_scatter_postpad_ker
      <CoordT, CounterT, HashT, 3, 256, BLOCK_SIZE_N>;

    this->kernel_map[{512, H_XORSUM_DIV}] = batch_psh_cumsum_scatter_postpad_ker
      <CoordT, CounterT, HashT, 3, 512, BLOCK_SIZE_N>;
    this->kernel_map[{512, H_XORSUM_MOD}] = batch_psh_cumsum_scatter_postpad_ker
      <CoordT, CounterT, HashT, 3, 512, BLOCK_SIZE_N>;
    this->kernel_map[{512, H_ZORDER_MOD}] = batch_psh_cumsum_scatter_postpad_ker
      <CoordT, CounterT, HashT, 3, 512, BLOCK_SIZE_N>;
    this->kernel_map[{512, H_ZORDER_DIV}] = batch_psh_cumsum_scatter_postpad_ker
      <CoordT, CounterT, HashT, 3, 512, BLOCK_SIZE_N>;
  }
};

} // end of ::f3d


#endif //FLASH3DPSHATTN_BUCKET_DISPATCHER_H
