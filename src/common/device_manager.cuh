/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 10/5/24
 */


#ifndef FLASH3DPSHATTN_DEVICE_MANAGER_H
#define FLASH3DPSHATTN_DEVICE_MANAGER_H

#include <memory>
#include <unordered_map>

#include "common/rand.cuh"

namespace f3d {

class DeviceInfo {
public:
  DeviceInfo() = default;

  DeviceInfo(int dev_id);

  int gpu_id;
  std::unique_ptr<GlobalRandState> global_rand_ptr;
};

class DeviceManagerSingleton {
public:
  static DeviceManagerSingleton& instance();

  bool check_dev_exist(int dev_id);

  bool check_dev_inited(int dev_id);

  void init_dev_info(int dev_id);

  void init_dispatchers();

  bool set_sync_stream(bool is_syncing);

  void set_device(int dev_id);

  bool get_sync_stream();

  curandState * get_states_dev(int dev_id);

private:
  DeviceManagerSingleton();

  DeviceManagerSingleton(const DeviceManagerSingleton&) = delete;

  DeviceManagerSingleton& operator= (const DeviceManagerSingleton&) = delete;

  int num_gpu;
  bool is_sync_stream;
  std::unordered_map<int, DeviceInfo> dev_map;
};


} // end of ::f3d

#endif //FLASH3DPSHATTN_DEVICE_MANAGER_H
