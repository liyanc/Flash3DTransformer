/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 *  found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 07/01/24
 */


#include <vector>
#include <chrono>
#include <stdexcept>
#include <cstdio>

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/ATen.h>
#include <glog/logging.h>

#include <cuda_runtime.h>
#include "common/runtime.h"


namespace f3d {

void bucket_sanitizer_cpu(at::Tensor & bucket_id, at::Tensor & bucket_counter, at::Tensor & bucket_offset,
                          at::Tensor & offset_check_table, at::Tensor & bucket_residual,
                          uint32_t max_counter, uint16_t num_bucket) {
  auto bucketid_cpu = bucket_id.cpu();
  auto bucketcnt_cpu = bucket_counter.cpu();
  auto bucketoff_cpu = bucket_offset.cpu();
  auto bucketres_cpu = bucket_residual.cpu();
  auto offset_table_cpu = offset_check_table.cpu();
  char buff[256];

  assert_contiguous({bucketid_cpu, bucketcnt_cpu, bucketoff_cpu, bucketres_cpu, offset_table_cpu});

  using CounterT = uint32_t;
  using HashT = uint16_t;

  for (auto c = 0; c < bucketid_cpu.size(0); ++c) {
    auto buckid_ptr = (HashT*)(bucketid_cpu.data_ptr()) + c;
    auto buckid_col = *buckid_ptr;
    auto buckres_ptr = (CounterT*)(bucketres_cpu.data_ptr()) + buckid_col;
    *buckres_ptr -= 1;

    auto buckoff_ptr = (CounterT*)(bucketoff_cpu.data_ptr()) + c;
    auto buckoff_col = *buckoff_ptr;
    auto check_table_base = (int8_t*)offset_table_cpu.data_ptr();
    if (buckid_col >= num_bucket || buckoff_col >= offset_table_cpu.size(1)) {
      sprintf(buff, "col=%d: buckid=%u|num_bucket=%d, buckoff=%u|offset_table_dim2=%ld, overflow ",
              c, buckid_col, num_bucket, buckoff_col, offset_table_cpu.size(1));
      throw std::runtime_error(std::string(buff));
    }
    auto check_table_ptr = check_table_base +
      buckid_col * offset_table_cpu.stride(0) + buckoff_col;
    *check_table_ptr += 1;
  }

  std::vector<bool> bucket_flags(num_bucket, true);
  sprintf(buff, "bucket_flags size=%d", bucket_flags.size());
  LOG(WARNING) << std::string(buff);

  for (auto b = 0; b < num_bucket; ++b) {
    auto buckcnt_ptr = (CounterT*)(bucketcnt_cpu.data_ptr()) + b;
    auto buckcnt_max = *buckcnt_ptr;
    for (auto s = 0; s < buckcnt_max; ++s) {
      auto check_table_ptr = (int8_t*)offset_table_cpu.data_ptr() +
        b * offset_table_cpu.stride(0) + s * offset_table_cpu.stride(1);
      bucket_flags[b] = bucket_flags[b] & (*check_table_ptr == 1);

      if (*check_table_ptr != 1) {
        sprintf(buff, "Wrong scoreboard, buck=%d,[Slot=%d|MaxCnt=%d],Score=%d",
                b, s, buckcnt_max, *check_table_ptr);
        throw std::runtime_error(std::string(buff));
      }
    }
  }

  bucket_residual.copy_(bucketres_cpu);
  offset_check_table.copy_(offset_table_cpu);
  auto res = std::all_of(bucket_flags.begin(), bucket_flags.end(), [](const auto &b){return bool(b);});
  if (res){
    LOG(WARNING) << "Scoreboard Test passed for bucket_offset!";
  } else {
    throw std::runtime_error("Unknown error!");
  }

}

void batch_bucket_sanitizer_cpu(at::Tensor & bucket_id, at::Tensor & bucket_counter, at::Tensor & bucket_offset,
                                at::Tensor & offset_check_table, at::Tensor & bucket_residual, at::Tensor & batch_sep,
                                at::Tensor & offset_maxcnt, uint32_t batch_size, uint16_t num_bucket) {
  /*
   * bucket_id: [N], HashT
   * bucket_counter: [B, NB], CounterT
   * bucket_offset: [N], CounterT
   * offset_check_table: [B, NB, MaxCnt], int8
   * bucket_residual: [B, NB], CounterT
   * batch_sep: [B], CounterT
   * offset_maxcnt: [B], CounterT
   */

  auto bucketid_cpu = bucket_id.cpu();
  auto bucketcnt_cpu = bucket_counter.cpu();
  auto bucketoff_cpu = bucket_offset.cpu();
  auto bucketres_cpu = bucket_residual.cpu();
  auto offset_table_cpu = offset_check_table.cpu();
  auto offset_maxcnt_cpu = offset_maxcnt.cpu();
  auto batchsep_cpu = batch_sep.cpu();
  char buff[256];

  assert_contiguous({bucketid_cpu, bucketcnt_cpu, bucketoff_cpu, bucketres_cpu, offset_table_cpu, batchsep_cpu});

  using CounterT = uint32_t;
  using HashT = uint16_t;
  auto batchsep_acc = batchsep_cpu.accessor<CounterT, 1>();
  auto buckcnt_acc = bucketcnt_cpu.accessor<CounterT, 2>();
  auto buckres_acc = bucketres_cpu.accessor<CounterT, 2>();
  auto buckoff_acc = bucketoff_cpu.accessor<CounterT, 1>();
  auto checktable_acc = offset_table_cpu.accessor<int8_t, 3>();
  auto offsetmaxcnt_acc = offset_maxcnt_cpu.accessor<CounterT, 1>();

  /*
   * For every instance in the batch, the instance is bid.
   * For every column in the point list, the column is c.
   * Get the bucket_id at c by bucketid_acc[c]
   * Get the bucket_off at c by bucketoff_acc[c]
   * Decrement the residual by buckres_acc[bid][buckid_col] -= 1
   * Index the check_table of this item by checktable_acc[bid][buckid_col][buckoff_col]
   * Increment checktable_acc[bid][buckid_col][buckoff_col] to mark this item
   */
  for (auto bid = 0; bid < batch_size; ++bid) {
    CounterT batch_start = 0, batch_end = batchsep_acc[bid];
    if (bid > 0)
      batch_start = batchsep_acc[bid - 1];
    auto N = batch_end - batch_start;

    for (auto c = batch_start; c < batch_end; ++c) {
      auto buckid_ptr = (HashT*)(bucketid_cpu.data_ptr()) + c;
      auto buckid_col = *buckid_ptr;
      buckres_acc[bid][buckid_col] -= 1;

      auto buckoff_col = buckoff_acc[c];
      if (buckid_col >= num_bucket || buckoff_col >= offsetmaxcnt_acc[bid]) {
        sprintf(buff, "col=%d: buckid=%u|num_bucket=%d, buckoff=%u|offset_maxcnt=%d, overflow ",
                c, buckid_col, num_bucket, buckoff_col, offsetmaxcnt_acc[bid]);
        throw std::runtime_error(std::string(buff));
      }
      checktable_acc[bid][buckid_col][buckoff_col] += 1;
    }
  }

  for (auto bid = 0; bid < batch_size; ++bid) {
    std::vector<bool> bucket_flags(num_bucket, true);

    for (auto b = 0; b < num_bucket; ++b) {
      auto buckcnt_max = buckcnt_acc[bid][b];
      for (auto s = 0; s < buckcnt_max; ++s) {
        auto check_table_entry = checktable_acc[bid][b][s];
        bucket_flags[b] = bucket_flags[b] & (check_table_entry == 1);

        if (check_table_entry != 1) {
          sprintf(buff, "Wrong scoreboard, batchid=%d buck=%d,[Slot=%d|MaxCnt=%d],Score=%d",
                  bid, b, s, buckcnt_max, check_table_entry);
          throw std::runtime_error(std::string(buff));
        }
      }
    }
  }

  bucket_residual.copy_(bucketres_cpu);
  offset_check_table.copy_(offset_table_cpu);
  LOG(WARNING) << "Scoreboard Test passed for bucket_offset!";
}

} // end of ::f3d