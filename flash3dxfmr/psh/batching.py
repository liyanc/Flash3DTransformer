#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
#  Authored by Liyan Chen (liyanc@cs.utexas.edu) on 9/2/24


import torch
from typing import List


def collate_coords_flat_sep(coords: List[torch.Tensor], coord_dtype=None, sep_dtype=None):
    """
    collect a list of tensors of shape [Ni, D] with i in [1, B];
    concat them into a flat tensor of shape [N, D];
    calculate batch separators of shape [B]
    :param coords: Tensors of shape [[N_1, D], ..., [N_B, D]]
    :return: coords [N, D], sep [B]
    """
    coords_batch = torch.cat(coords, dim=0)
    sizes = torch.IntTensor([c.shape[0] for c in coords])
    seps = torch.cumsum(sizes, dim=-1)
    coords_batch = coords_batch.to(coord_dtype) if coord_dtype is not None else coords_batch
    seps = seps.to(sep_dtype) if sep_dtype is not None else seps
    cont_coords = coords_batch[:, :3].contiguous()

    return cont_coords, seps


def sep2sizes(seps: torch.Tensor):
    in_dtype = seps.dtype
    seps = seps.to(torch.int64)
    shift = torch.roll(seps, 1)
    shift[0] = 0
    return (seps - shift).to(in_dtype)


def max_inslen_from_seps(seps: torch.Tensor):
    return int(sep2sizes(seps.to(torch.int64)).max())


def bulk_to(ts: List[torch.Tensor], dest):
    return [t.to(dest) for t in ts]


def cdiv_int(a: torch.Tensor, b):
    return (a + (b - 1)) / b