#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
#  Authored by Liyan Chen (liyanc@cs.utexas.edu) on 7/18/24


import math
import torch
from typing import List, Union
from dataclasses import dataclass


def minimal_alignment(buck_size: int, scope_size_in_buck: int):
    """
    Compute the minimal alignment based on buck_size and scope_size_in_buck.

    Parameters:
        buck_size (int): The size of each bucket.
        scope_size_in_buck (int): The scope size within a single bucket.

    Returns:
        int: The minimal alignment, calculated as buck_size.
    """
    return buck_size


def swinable_alignment(buck_size: int, scope_size_in_buck: int):
    return buck_size * scope_size_in_buck


def stridable_alignment(buck_size: int, scope_size_in_buck: int):
    """
    Compute a stridable alignment by scaling the minimal alignment.

    The sensible alignment is defined as 6 times the minimal alignment.
    The factor 6 is chosen because it is the least common multiple (LCM) of 1, 2, and 3,
    ensuring that the alignment is compatible with various common alignment requirements.

    Parameters:
        buck_size (int): The size of each bucket.
        scope_size_in_buck (int): The scope size within a single bucket.

    Returns:
        int: The sensible alignment, calculated as buck_size * scope_size_in_buck * 6.
    """
    return buck_size * scope_size_in_buck * 6


@dataclass
class SwinPlan:
    plan_name: str
    buck_size: int
    scope_size_in_buck: int
    offset_or_stride: int

    def gen_scopes_from_plan(self, total_N, device):
        assert self.plan_name in ["swin", "stride"], f"Unsupported scoping plan name: {self.plan_name}"
        min_align = minimal_alignment(self.buck_size, self.scope_size_in_buck)
        swin_align = swinable_alignment(self.buck_size, self.scope_size_in_buck)
        stride_align = stridable_alignment(self.buck_size, self.scope_size_in_buck)
        assert total_N % min_align == 0, \
            f"Input buffer length must be divisible by minimal alignment = {min_align}, but got {total_N}"

        scope_list = None
        if self.plan_name == "stride" and total_N % stride_align == 0:
            scope_list = generate_stride_scopes(total_N, self.buck_size, self.scope_size_in_buck)
        elif self.plan_name == "stride":
            scope_list = generate_single_scope(total_N, self.buck_size, self.scope_size_in_buck)
        elif self.plan_name == "swin" and total_N % swin_align == 0:
            scope_list = generate_swin_scopes(total_N, self.buck_size, self.scope_size_in_buck)
        elif self.plan_name == "swin":
            scope_list = generate_single_scope(total_N, self.buck_size, self.scope_size_in_buck)
        else:
            raise RuntimeError(f"Unknown plan error for swinplan {str(self)} and input buffer length={total_N}")

        return torch.tensor(scope_list, dtype=torch.uint32, device=device)


def generate_single_scope(total_N: int, buck_size: int, scope_size: int):
    return [list(range(total_N // buck_size))]



def generate_stride_scopes(total_N: int, buck_size: int, scope_size: int, stride=1):
    N = int(math.ceil(total_N / buck_size))
    assert N % stride == 0, f"num_bucket={N} must be divisible by stride={stride}"
    attn_scopes = []
    chunk_size = scope_size * stride
    for chunk_start in range(0, N, chunk_size):
        for s in range(stride):
            indices = [chunk_start + s + stride * k for k in range(scope_size)]
            indices = [idx for idx in indices if idx <= N]
            if len(indices) == scope_size:
                attn_scopes.append(indices)
    return attn_scopes


def generate_swin_scopes(total_N: int, buck_size: int, scope_size: int, offset: int = 0) -> List[List[int]]:
    """
    Generates disjoint sliding scopes ("swin_scopes") for attention in a circular manner.

    Each scope is a contiguous block of 'scope_size' bucket indices, and together
    the scopes form a non-overlapping partition of the indices [0, N) where
    N = ceil(total_N / buck_size). The first scope starts at 'offset', and subsequent
    scopes follow sequentially. If the scopes reach the end of the bucket range, they
    wrap around to the beginning.

    Note:
        For non-overlapping scopes, it is required that N (the number of buckets) is divisible
        by scope_size.

    Parameters:
        total_N (int): Total number of elements.
        buck_size (int): Bucket size used to compute the number of buckets.
        scope_size (int): Number of consecutive indices in each scope.
        offset (int): Starting offset for the first scope.

    Returns:
        List[List[int]]: A list of scopes, each a list of 'scope_size' indices.

    Raises:
        AssertionError: If scope_size is greater than the number of buckets or if N is not
                        divisible by scope_size.
    """
    # Compute the number of buckets.
    N = int(math.ceil(total_N / buck_size))
    assert scope_size <= N, "scope_size must be less than or equal to the number of buckets (N)"
    # For non-overlapping scopes, N must be divisible by scope_size.
    assert N % scope_size == 0, "For non-overlapping scopes, the number of buckets N must be divisible by scope_size"

    attn_scopes = []
    num_scopes = N // scope_size
    for i in range(num_scopes):
        start = (offset + i * scope_size) % N
        indices = [(start + j) % N for j in range(scope_size)]
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