/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 05/05/24
 */


#include <torch/extension.h>
#include <ATen/ATen.h>

#include <pybind11/pybind11.h>

#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cuda_runtime.h>
#include "common/fp_types.cuh"
#include "common/device_manager.cuh"

namespace f3d{

extern void
psh_partition(at::Tensor &coord, at::Tensor &bucket_id, at::Tensor &bucket_counter, at::Tensor &bucket_offset,
              uint32_t num_bucket, uint32_t bucket_divisor, uint32_t num_vox, at::Tensor &bbox_min,
              at::Tensor &bbox_max, py::object hash_dtype_obj, py::object coord_dtype_obj, uint16_t block_size_N);

extern void
psh_scatter(at::Tensor &coord, at::Tensor &bucket_id, at::Tensor &bucket_counter, at::Tensor &bucket_offset,
            uint32_t num_bucket, uint32_t bucket_divisor, uint32_t bucket_size, uint32_t num_vox,
            at::Tensor &bbox_min, at::Tensor &bbox_max, at::Tensor &probe_offsets, at::Tensor &cumsum_counter,
            at::Tensor &scatter_coord, py::object hash_dtype_obj, py::object coord_dtype_obj,
            uint16_t block_size_N);

extern void
batch_psh_scatter(at::Tensor &coord, at::Tensor &bucket_id, at::Tensor &bucket_counter, at::Tensor &bucket_offset,
                  at::Tensor &batch_sep, uint32_t instance_max_N, uint32_t num_bucket, uint32_t bucket_divisor,
                  uint32_t bucket_size, uint32_t num_vox, at::Tensor &bbox_min, at::Tensor &bbox_max,
                  at::Tensor &probe_offsets, at::Tensor &cumsum_counter, at::Tensor &scatter_coord,
                  py::object hash_dtype_obj, py::object coord_dtype_obj, uint16_t block_size_N);

extern void
batch_psh_scatter_pad(at::Tensor &coord, at::Tensor &bucket_id, at::Tensor &bucket_counter, at::Tensor &bucket_offset,
                      at::Tensor &batch_sep, uint32_t instance_max_N, uint32_t num_bucket, uint32_t bucket_divisor,
                      uint32_t bucket_size, uint32_t num_vox, at::Tensor &bbox_min, at::Tensor &bbox_max,
                      at::Tensor &probe_offsets, at::Tensor &cumsum_counter, at::Tensor &scatter_coord,
                      float pad_num);

extern void
batch_psh_scatter_pad_hash(
  at::Tensor &coord, at::Tensor &bucket_id, at::Tensor &bucket_counter, at::Tensor &bucket_offset,
  at::Tensor &batch_sep, uint32_t instance_max_N, uint32_t num_bucket, uint32_t bucket_divisor,
  uint32_t bucket_size, uint32_t num_vox, at::Tensor &bbox_min, at::Tensor &bbox_max,
  at::Tensor &probe_offsets, at::Tensor &cumsum_counter, at::Tensor &scatter_coord,
  float pad_num, uint16_t hash_type);

extern void
batch_subbuck_reduce(
  at::Tensor &scattered_coord, at::Tensor &scattered_feat, at::Tensor &batch_sep,
  uint32_t instance_max_N, uint32_t subbuck_size, uint32_t num_vox, at::Tensor &bbox_min, at::Tensor &bbox_max,
  at::Tensor &subbuck_id, at::Tensor &subbuck_off, at::Tensor &reduced_batch_sep,
  at::Tensor &reduced_coord, at::Tensor &reduced_feat, at::Tensor &unpool_ind, uint16_t hash_type, uint16_t red_op
  );

extern void
batch_subbuck_reduce_backward(
  at::Tensor &grad_red_feat, at::Tensor &input_feat, at::Tensor &unpool_ind, at::Tensor &grad_input_feat,
  uint32_t subbuck_size, uint16_t reduction_op);

extern void
bucket_sanitizer_cpu(at::Tensor & bucket_id, at::Tensor & bucket_counter, at::Tensor & bucket_offset,
                     at::Tensor & offset_check_table, at::Tensor & bucket_residual,
                     uint32_t max_counter, uint16_t num_bucket);

extern void
batch_bucket_sanitizer_cpu(at::Tensor & bucket_id, at::Tensor & bucket_counter, at::Tensor & bucket_offset,
                           at::Tensor & offset_check_table, at::Tensor & bucket_residual, at::Tensor & batch_sep,
                           at::Tensor & offset_maxcnt, uint32_t batch_size, uint16_t num_bucket);

extern void
attention_forward(at::Tensor &Q, at::Tensor &K, at::Tensor &V, at::Tensor &O,
                  uint32_t N, uint16_t num_head, uint16_t num_warps_opt, uint16_t num_buckets);

extern void
gemm_sm_bf16(at::Tensor &A, at::Tensor &B, at::Tensor &O);

extern void
buck_swin_fwd(at::Tensor &Q, at::Tensor &K, at::Tensor &V, at::Tensor &O, at::Tensor &L, at::Tensor &Scope_buckets,
              uint32_t bucket_size);

extern void
buck_swin_bwd(at::Tensor &Q, at::Tensor &K, at::Tensor &V, at::Tensor &O, at::Tensor &dO, at::Tensor &LSE,
              at::Tensor &Delta,at::Tensor &dQ, at::Tensor &dK, at::Tensor &dV,
              at::Tensor &Scope_buckets, uint32_t bucket_size);

extern void
additive_unpool_fwd(at::Tensor &res_feat, at::Tensor &down_feat, at::Tensor &up_add_feat,
                    at::Tensor &unpool_ind, uint32_t subbuck_size);

extern void
additive_unpool_bwd(at::Tensor &grad_res, at::Tensor &grad_down, at::Tensor &grad_up_added,
                    at::Tensor &unpool_ind, uint32_t subbuck_size);

extern void
uniform_samples(at::Tensor & array, uint32_t min, uint32_t max);

} // end of ::f3d

PYBIND11_MODULE(pshattn, m) {
  m.doc() = "Flash3d kernels";

  py::class_<f3d::DeviceManagerSingleton>(m, "DeviceManagerSingleton")
    .def_static("instance", f3d::DeviceManagerSingleton::instance, py::return_value_policy::reference)
    .def("init_dev_info", &f3d::DeviceManagerSingleton::init_dev_info)
    .def("init_dispatchers", &f3d::DeviceManagerSingleton::init_dispatchers)
    .def("set_sync_stream", &f3d::DeviceManagerSingleton::set_sync_stream)
    .def("get_sync_stream", &f3d::DeviceManagerSingleton::get_sync_stream);

  m.def("batch_psh_scatter", f3d::batch_psh_scatter);
  m.def("batch_psh_scatter_pad", f3d::batch_psh_scatter_pad);
  m.def("batch_psh_scatter_pad_hash", f3d::batch_psh_scatter_pad_hash);
  m.def("additive_unpool_fwd", f3d::additive_unpool_fwd);
  m.def("additive_unpool_bwd", f3d::additive_unpool_bwd);

  m.def("batch_subbuck_reduce", f3d::batch_subbuck_reduce);
  m.def("batch_subbuck_reduce_backward", f3d::batch_subbuck_reduce_backward);

  //m.def("uniform_samples", f3d::uniform_samples);
  //m.def("bucket_sanitizer_cpu", f3d::bucket_sanitizer_cpu);
  m.def("batch_bucket_sanitizer_cpu", f3d::batch_bucket_sanitizer_cpu);
  //m.def("attention_forward", f3d::attention_forward);
  m.def("gemm_sm_bf16", f3d::gemm_sm_bf16);
  //m.def("buck_swin_fwd", f3d::buck_swin_fwd);
  //m.def("buck_swin_bwd", f3d::buck_swin_bwd);
}