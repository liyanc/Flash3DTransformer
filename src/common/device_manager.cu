/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 10/5/24
 */


#include <glog/logging.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include "device_manager.cuh"
#include "common/fp_types.cuh"
#include "common/runtime.h"
#include "psh/bucket_dispatcher.cuh"
//#include "psh/bucket_swin_attn_fwd_dispatcher.cuh"
//#include "psh/bucket_swin_attn_bwd_dispatcher.cuh"

namespace f3d {

constexpr uint16_t BLOCK_SIZE = 1024u;

DeviceInfo::DeviceInfo(int dev_id) :
  gpu_id{dev_id},
  global_rand_ptr{std::make_unique<GlobalRandState>(BLOCK_SIZE, dev_id)} {
}

DeviceManagerSingleton &DeviceManagerSingleton::instance() {
  static DeviceManagerSingleton instance;
  instance.init_dispatchers();
  return instance;
}

DeviceManagerSingleton::DeviceManagerSingleton() {
  cudaCheckErr(cudaGetDeviceCount(&num_gpu));

  dev_map.reserve(num_gpu);
  is_sync_stream = false;
}

bool DeviceManagerSingleton::check_dev_exist(int dev_id) {
  int attr_val;
  auto res = cudaDeviceGetAttribute(&attr_val, cudaDeviceAttr::cudaDevAttrComputeMode, dev_id);
  bool dev_valid = (res == cudaSuccess);
  bool dev_compute = (attr_val == cudaComputeModeDefault);
  return dev_valid && dev_compute;
}

bool DeviceManagerSingleton::check_dev_inited(int dev_id) {
  auto iter = dev_map.find(dev_id);
  return iter != dev_map.end();
}

void DeviceManagerSingleton::init_dev_info(int dev_id) {
  if (!check_dev_exist(dev_id))
    return;
  if (check_dev_inited(dev_id))
    return;

  auto [iter, flag] = dev_map.emplace(dev_id, DeviceInfo{dev_id});
  auto stream = c10::cuda::getCurrentCUDAStream(int8_t(dev_id));
  iter->second.global_rand_ptr->init_global(3407, stream);
}

void DeviceManagerSingleton::set_device(int dev_id) {
  cudaCheckErr(cudaSetDevice(dev_id));
}

void DeviceManagerSingleton::init_dispatchers() {
  using CoordT = fp16;
  using CounterT = uint32_t;
  using HashT = uint16_t;

  auto & buck_disp = BucketingDispatcher<CoordT, CounterT, HashT>::instance();
  auto & recyc_disp = RecycleDispatcher<CoordT, CounterT, HashT>::instance();
  auto & scatter_disp = ScatterCoordDispatcher<CoordT, CounterT, HashT>::instance();
  auto & scatter_pad_disp = ScatterCoordPadDispatcher<CoordT, CounterT, HashT>::instance();

  /*
  auto & fwd_disp = BuckSwinAttnFwdDispatcher::instance();
  auto & bwd_delta_disp = BuckSwinAttnBwdDeltaDispatcher::instance();
  auto & bwd_attn_disp = BuckSwinAttnBwdDispatcher::instance();
   */

  buck_disp.initialize();
  recyc_disp.initialize();
  scatter_disp.initialize();
  scatter_pad_disp.initialize();

  /*
  bwd_delta_disp.initialize();
  fwd_disp.initialize();
  bwd_attn_disp.initialize();
   */
}

curandState * DeviceManagerSingleton::get_states_dev(int dev_id) {
  auto is_dev_exist = check_dev_exist(dev_id);
  if (!is_dev_exist) {
    throw_format_error("Requested dev_id=%d NOT exist for get_states_dev()", dev_id);
  }

  init_dev_info(dev_id);

  auto iter = dev_map.find(dev_id);
  return iter->second.global_rand_ptr->get_states();
}

bool DeviceManagerSingleton::set_sync_stream(bool is_syncing) {
  return is_sync_stream = is_syncing;
}

bool DeviceManagerSingleton::get_sync_stream() {
  return is_sync_stream;
}

} // end of ::f3d