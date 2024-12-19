/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 10/6/24
 */


#ifndef FLASH3DPSHATTN_KERNEL_DISPATCHER_H
#define FLASH3DPSHATTN_KERNEL_DISPATCHER_H

#include <tuple>
#include <utility>
#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <curand_kernel.h>

#include "common/runtime.h"
#include "common/arithmetic.cuh"
#include "psh/hash_fns.cuh"

namespace f3d {
using ParamTuple = std::tuple<uint16_t, f3d::HASHTYPE>;
using AttnParamTuple = std::tuple<uint16_t, uint16_t, uint16_t>;
} // end of ::f3d

namespace std {

template<>
struct hash<f3d::ParamTuple> {
  size_t operator()(const f3d::ParamTuple & t) const {
    auto h1 = hash<uint16_t>()(std::get<0>(t));
    auto h2 = hash<size_t>()(std::get<1>(t));
    return h1 ^ h2;
  }
};

template<>
struct hash<f3d::AttnParamTuple> {
  size_t operator()(const f3d::AttnParamTuple & t) const {
    auto h1 = hash<uint16_t>()(std::get<0>(t));
    auto h2 = hash<uint16_t>()(std::get<1>(t));
    auto h3 = hash<uint16_t>()(std::get<2>(t));
    return h1 ^ h2 ^ h3;
  }
};
} // end of ::std

namespace f3d {

template <typename FuncType>
unsigned int get_max_warps_per_cta(FuncType attn_ker) {
  cudaFuncAttributes attr{};
  cudaFuncGetAttributes(&attr, attn_ker);
  cudaCheckLastErr();
  assert(attr.maxThreadsPerBlock >= WarpSize);
  return attr.maxThreadsPerBlock / WarpSize;
}

template <typename Derived, typename CoordT, typename CounterT, typename HashT, typename KernelFunc>
class KernelDispatcher {
public:
  static Derived& instance() {
    static Derived instance;
    return instance;
  }

  //virtual void initialize() = 0;

  KernelFunc::PtrType& get_kernel_instance(uint16_t num_buck, f3d::HASHTYPE first_hash) {
    auto it = kernel_map.find({num_buck, first_hash});
    if (it == kernel_map.end()) {
      throw_format_error("No kernel found, num_buck=%d,first_hash=%d", num_buck, first_hash);
    } else {
      return it->second;
    }
    // Impossible to reach
    return kernel_map.begin()->second;
  }

  KernelDispatcher(const KernelDispatcher&) = delete;
  KernelDispatcher& operator=(const KernelDispatcher&) = delete;

protected:
  std::unordered_map<ParamTuple, typename KernelFunc::PtrType> kernel_map;

  KernelDispatcher() {};
};

template <typename Derived, typename KernelFunc>
class AttnKernelDispatcher {
public:
  static Derived& instance() {
    static Derived instance;
    return instance;
  }

  //virtual void initialize() = 0;

  KernelFunc::PtrType& get_kernel_instance(uint16_t head_dim, uint16_t warp_per_cta, uint16_t pipe_stage) {
    auto it = kernel_map.find({head_dim, warp_per_cta, pipe_stage});
    if (it == kernel_map.end()) {
      throw_format_error(
        "No kernel found, head_dim=%d,warp_per_cta=%d,pipe_stage=%d",
        head_dim, warp_per_cta, pipe_stage);
    } else {
      return it->second;
    }
    // Impossible to reach
    return kernel_map.begin()->second;
  }

  AttnKernelDispatcher(const AttnKernelDispatcher&) = delete;
  AttnKernelDispatcher& operator=(const AttnKernelDispatcher&) = delete;

protected:
  std::unordered_map<AttnParamTuple, typename KernelFunc::PtrType> kernel_map;

  AttnKernelDispatcher() {};
};
} // end of ::f3d



#endif //FLASH3DPSHATTN_KERNEL_DISPATCHER_H
