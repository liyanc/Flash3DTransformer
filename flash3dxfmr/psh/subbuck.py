#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
#  Authored by Liyan Chen (liyanc@cs.utexas.edu)


import torch
import random
import dataclasses as dcls

from .arithmatic import *


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


def calc_red_sep_sizes(seps: torch.Tensor, subbuck_size: int):
    assert seps.dim() == 1, f'Seps dimension is expected to be 1, got {seps.dim()} instead'
    in_dtype = seps.dtype
    up_seps = seps.to(torch.int64)
    shift = torch.roll(up_seps, 1)
    shift[0] = 0
    sizes = up_seps - shift
    # Use ceiling division so that:
    # - For an even L, (L + subbuck_size - 1) // subbuck_size == L // subbuck_size
    # - For an odd L, it equals (L + 1) // subbuck_size
    red_sizes = (sizes + subbuck_size - 1) // subbuck_size
    red_len = int(torch.sum(red_sizes))
    red_sep = torch.cumsum(red_sizes, dim=0)
    return red_len, red_sizes.to(in_dtype), red_sep.to(in_dtype)

@dcls.dataclass(slots=True)
class SubbuckBuffers:
    bbox_min: torch.Tensor
    bbox_max: torch.Tensor
    expect_sep: torch.Tensor
    subbuck_id: torch.Tensor
    subbuck_off: torch.Tensor
    reduced_sep: torch.Tensor
    reduced_coord: torch.Tensor
    reduced_feat: torch.Tensor
    unpool_ind: torch.Tensor

    @staticmethod
    def create_subbuck_buff(coords: torch.Tensor, input_feat: torch.Tensor, seps: torch.Tensor,
                            subbuck_size=2, out_alignment=1):
        """
        :param coords:
        :param input_feat:
        :param seps:
        :param subbuck_size:
        :param out_alignment: reduced_len must be multiple of out_alignment, to be compatible with subsequent buck_swin
        :return:
        """
        cnt_dtype = torch.uint32
        hash_dtype = torch.uint16
        coord_dtype = torch.float16
        feat_dtype = input_feat.dtype

        bbox_min, _ = torch.min(coords, dim=0)
        bbox_max, _ = torch.max(coords, dim=0)
        assert bbox_min.dtype == coord_dtype
        Total_N, D = coords.shape
        _, DE = input_feat.shape
        dev = coords.device
        B = seps.shape[0]
        red_len, red_sizes, red_sep = calc_red_sep_sizes(seps, subbuck_size)
        aligned_red_len = cdiv(red_len, out_alignment) * out_alignment
        subbuck_id = torch.zeros(Total_N, dtype=cnt_dtype, device=dev)
        subbuck_off = torch.empty(Total_N, dtype=hash_dtype, device=dev)
        reduced_coord = torch.empty(aligned_red_len, D, dtype=coord_dtype, device=dev)
        reduced_feat = torch.empty(aligned_red_len, DE, dtype=feat_dtype, device=dev)
        # pre-filling unpool_ind is a must: default OOB indices signify unfilled subbucket slot
        unpool_ind = torch.full((red_len, subbuck_size), fill_value=Total_N + 3, dtype=cnt_dtype, device=dev)

        subbuck_buff = SubbuckBuffers(
            bbox_min, bbox_max, red_sep, subbuck_id, subbuck_off,
            red_sep, reduced_coord, reduced_feat, unpool_ind)
        return subbuck_buff

    def remove_unused_buffers(self):
        del self.subbuck_id
        del self.subbuck_off


@dcls.dataclass(slots=True)
class UnpoolBuffers:
    up_add_feat: torch.Tensor

    @staticmethod
    def create_unpool_buff(residual_feat: torch.Tensor):
        output_feat = torch.empty_like(residual_feat)
        return UnpoolBuffers(output_feat)


@dcls.dataclass(slots=True)
class UnpoolGradients:
    grad_res: torch.Tensor
    grad_down: torch.Tensor

    @staticmethod
    def create_unpool_grad(grad_up_added: torch.Tensor, sb: SubbuckBuffers):
        feat_dim = grad_up_added.shape[-1]
        down_len = sb.reduced_feat.shape[0]
        grad_res = torch.zeros_like(grad_up_added)
        grad_down = torch.zeros(down_len, feat_dim,
                                dtype=grad_up_added.dtype, device=grad_up_added.device)

        return UnpoolGradients(grad_res, grad_down)
