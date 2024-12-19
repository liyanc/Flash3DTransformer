/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 6/9/24
 */


#ifndef FLASH3DPSHATTN_RUNTIME_H
#define FLASH3DPSHATTN_RUNTIME_H

#include <cstdio>
#include <string>
#include <chrono>
#include <tuple>
#include <glog/logging.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/ATen.h>
#include <cuda_runtime.h>

namespace f3d {

#define cudaCheckErr(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define cudaCheckLastErr() {gpuAssert((cudaGetLastError()), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    LOG(ERROR) << file << ":" << line << " CUDA error reason= " << cudaGetErrorString(code);
    if (abort) exit(code);
  }
}

__forceinline__ bool assert_contiguous(const std::vector<at::Tensor> & checking_list) {
  constexpr uint16_t MAX_BUFF = 512u;
  std::string err_str(MAX_BUFF, '\0');
  bool all_contiguous = true;
  for (auto i = 0; i < checking_list.size(); ++i) {
    const auto & t = checking_list[i];
    all_contiguous &= t.is_contiguous();
    if (!t.is_contiguous()) {
      snprintf(err_str.data(), MAX_BUFF, "%d-th tensor NOT contiguous.", i);
      LOG(ERROR) << err_str.c_str();
      throw std::runtime_error(err_str);
    }
  }
  return all_contiguous;
}

__forceinline__ bool assert_dtype(const std::vector<at::Tensor> & checking_list, at::ScalarType target_dtype) {
  constexpr uint16_t MAX_BUFF = 512u;
  std::string err_str(MAX_BUFF, '\0');
  bool all_correct = true;
  for (auto i = 0; i < checking_list.size(); ++i) {
    const auto & t = checking_list[i];
    all_correct &= (target_dtype == t.dtype());
    if (target_dtype != t.dtype()) {
      snprintf(err_str.data(), MAX_BUFF, "%d-th tensor wrong dtype=%s, expect=%s.",
              i, c10::toString(t.dtype().toScalarType()), c10::toString(target_dtype));
      LOG(ERROR) << err_str.c_str();
      throw std::runtime_error(err_str);
    }
  }
  return all_correct;
}

__forceinline__ bool assert_same_device(const std::vector<at::Tensor> & checking_list) {
  constexpr uint16_t MAX_BUFF = 512u;
  bool all_correct = true;
  std::string err_str(MAX_BUFF, '\0');
  at::Device dev = checking_list.front().device();

  for (auto i = 0; i < checking_list.size(); ++i) {
    const auto & t = checking_list[i];
    all_correct &= (dev == t.device());
    if (dev != t.device()) {
      snprintf(err_str.data(), MAX_BUFF, "%d-th tensor wrong device=%s, expect=%s.",
              i, t.device().str().c_str(), dev.str().c_str());
      LOG(ERROR) << err_str.c_str();
      throw std::runtime_error(err_str);
    }
  }
  return all_correct;
}

void throw_format_error(const char * __restrict__ fmt, ...);

class QuickTimer {
public:
  QuickTimer() :
    start{std::chrono::high_resolution_clock::now()} {}

  template<typename TimeUnit>
  auto end_and_format() {
    using namespace std::chrono;
    auto end = high_resolution_clock::now();
    auto span = duration_cast<TimeUnit>(end - start);
    std::string literal = std::to_string(span.count());
    if (TimeUnit::period::den == milliseconds::period::den)
      literal += "millisec";
    if (TimeUnit::period::den == microseconds::period::den)
      literal += "microsec";
    if (TimeUnit::period::den == nanoseconds::period::den)
      literal += "nanosec";

    return std::make_tuple(span.count(), literal);
  }

private:
  std::chrono::high_resolution_clock::time_point start;
};

} //end of ::f3d

#endif //FLASH3DPSHATTN_RUNTIME_H
