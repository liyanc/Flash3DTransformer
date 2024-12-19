#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
# Authored by Liyan Chen (liyanc@cs.utexas.edu)


import torch
import random
import logging

from ..lib import pshattn
from .arithmatic import *
from .batching import sep2sizes
from .dev_context import get_dev_manager, init_dev


def generate_probe_offsets(device, row_major=True):
    cube_offsets = []
    dim_shift = [-2, -1, 0, 1, 2]
    for i in dim_shift:
        for j in dim_shift:
            for k in dim_shift:
                if not (i, j, k) == (0, 0, 0):
                    cube_offsets.append([i, j, k])

    cube_offsets += [
        [3, 3, 3],
        [3, -3, 3],
        [-3, 3, -3],
        [-3, -3, -3]
    ]

    random.shuffle(cube_offsets)
    probe_tensor = torch.ShortTensor(cube_offsets)
    if not row_major:
        probe_tensor = probe_tensor.transpose(0, 1)
    probe_offsets = probe_tensor.contiguous().to(device).contiguous()
    return probe_offsets
