/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 05/05/24
 */


#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace f3d {

using fp8e5m2 = __nv_fp8_e5m2;
using fp8e4m3 = __nv_fp8_e4m3;

using bf16 = nv_bfloat16;
using fp16 = nv_half;

using bf16_2 = nv_bfloat162;
using fp16_2 = nv_half2;

} // end of ::f3d