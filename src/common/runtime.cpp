/*
 *  Copyright (c) 2018-present, Cruise LLC
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * found in the LICENSE file in the root directory of this source tree.
 * You may not use this file except in compliance with the License.
 * Authored by Liyan Chen (liyanc@cs.utexas.edu) on 7/5/24
 */


#include "common/runtime.h"

namespace f3d {

void throw_format_error(const char * __restrict__ fmt, ...) {
  va_list args;
  constexpr uint16_t MAX_BUFF = 512u;
  va_start(args, fmt);
  std::string err_str(MAX_BUFF, '\0');
  vsnprintf(err_str.data(), MAX_BUFF, fmt, args);
  va_end(args);
  LOG(ERROR) << err_str.c_str();
  throw std::runtime_error(err_str);
}

} // end of ::f3d