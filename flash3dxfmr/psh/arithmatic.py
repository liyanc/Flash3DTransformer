#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
#  Authored by Liyan Chen (liyanc@cs.utexas.edu)


import math


def round_next_two_power(v):
    v -= 1
    v |= v >> 1
    v |= v >> 2
    v |= v >> 4
    v |= v >> 8
    v |= v >> 16
    return v + 1


def round_next_multiply(v, base):
    v_type = type(v)
    c = math.ceil(float(v) / base)
    return v_type(c * base)


def cdiv(a, b):
    return int(math.ceil(a / b))
