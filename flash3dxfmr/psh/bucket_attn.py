#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
# Authored by Liyan Chen (liyanc@cs.utexas.edu) on 7/18/24


import math
import torch
from typing import List


def generate_stride_scopes(total_N: int, buck_size: int, scope_size: int, stride=1):
    N = int(math.ceil(total_N / buck_size))
    assert N % stride == 0, f"num_bucket={N} must be divisible by stride={stride}"
    attn_scopes = []
    chunk_size = scope_size * stride  # 每个块的大小
    for chunk_start in range(0, N, chunk_size):
        for s in range(stride):
            indices = [chunk_start + s + stride * k for k in range(scope_size)]
            indices = [idx for idx in indices if idx <= N]  # 确保索引不超过 N
            if len(indices) == scope_size:
                attn_scopes.append(indices)
    return attn_scopes


def construct_bucket_scopes(I: torch.Tensor, buck_size: int, attn_scopes: List[List[int]]):
    """
    Convert a feature tensor to bucket-attention scoped tensor with indexing provided attn_scopes
    :param I: a feature tensor to generate QKV for downstream layers. Shape: [D, L]
    :param buck_size: the size of a bucket
    :param attn_scopes: indices of assigning buckets to attention scopes
    :return: flattened bucket tensor of shape [B, D, L], where B is the batch/scope dimension
    """
    scope_list = []
    for scope_ind in attn_scopes:
        bucket_list = []
        for buck_ind in scope_ind:
            buck_start, buck_end = buck_ind * buck_size, (buck_ind + 1) * buck_size
            buck_seg = I[:, buck_start: buck_end]
            bucket_list.append(buck_seg)
        scope_seg = torch.concat(bucket_list, dim=-1)
        scope_list.append(scope_seg)
    return torch.stack(scope_list, dim=0)


def revert_bucket_scopes(I: torch.Tensor, bucket_size: int, attn_scopes: List[List[int]]):
    """
    Revert a flattened bucket tensor to a feature vector
    :param I: flattened bucket tensor of shape [B, D, L], where B is the batch/scope dimension
    :param buck_size: the size of a bucket
    :param attn_scopes: indices of assigning buckets to attention scopes
    :return: a feature tensor to generate QKV for downstream layers. Shape: [D, L]
    """
    bucket2seg = {}
    for scope_seq, scope_ind in enumerate(attn_scopes):
        for buck_seq, buck_ind in enumerate(scope_ind):
            buck_start, buck_end = buck_seq * bucket_size, (buck_seq + 1) * bucket_size
            buck_seg = I[scope_seq, :, buck_start: buck_end]
            bucket2seg[buck_ind] = buck_seg

    # list of segments of shape [D, S]
    seg_list = [bucket2seg[i] for i in range(max(bucket2seg) + 1)]
    flat = torch.concat(seg_list, dim=-1)
    return flat


def reassign_bucket_scopes(I: torch.Tensor, bucket_size: int, old_scopes: List[List[int]], new_scopes: List[List[int]]):
    """
    Revert a flattened bucket tensor to a feature vector
    :param I: flattened bucket tensor of shape [B, D, L], where B is the batch/scope dimension
    :param buck_size: the size of a bucket
    :param attn_scopes: indices of assigning buckets to attention scopes
    :return: a feature tensor to generate QKV for downstream layers. Shape: [D, L]
    """
    bucket2seg = {}
    for scope_seq, scope_ind in enumerate(old_scopes):
        for buck_seq, buck_ind in enumerate(scope_ind):
            buck_start, buck_end = buck_seq * bucket_size, (buck_seq + 1) * bucket_size
            buck_seg = I[scope_seq, :, buck_start: buck_end]
            bucket2seg[buck_ind] = buck_seg

    # list of segments of shape [D, S]
    seg_list = [bucket2seg[i] for i in range(max(bucket2seg) + 1)]

    scope_list = []
    for scope_ind in new_scopes:
        bucket_list = []
        for buck_ind in scope_ind:
            buck_seg = bucket2seg[buck_ind]
            bucket_list.append(buck_seg)
        scope_seg = torch.concat(bucket_list, dim=-1)
        scope_list.append(scope_seg)
    return torch.stack(scope_list, dim=0)