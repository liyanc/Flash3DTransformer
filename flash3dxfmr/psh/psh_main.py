#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
#  Authored by Liyan Chen (liyanc@cs.utexas.edu)


import math
import torch
from typing import Tuple
from dataclasses import dataclass

from . import arithmatic as f3darith
from . import batching as f3dbatch
from . import bucket as f3dbuck
from . import dev_context as f3ddev
from ..lib import pshattn


@dataclass(slots=True)
class BucketScatterBuffers:
    bucket_id: torch.Tensor
    bucket_cntr: torch.Tensor
    bucket_offs: torch.Tensor
    bucket_cumsum: torch.Tensor
    probe_offs: torch.Tensor
    scattered_coord: torch.Tensor

    @staticmethod
    def create_buffers(coords: torch.Tensor, seps: torch.Tensor, bucket_size: int, pad_to: int):
        """
        Create and allocate all buffers required for batch bucket scatter pad.

        Args:
            coords (torch.Tensor): Input coordinates tensor (contiguous, dtype torch.float16).
            seps (torch.Tensor): Separation tensor.
            bucket_size (int): Bucket size parameter.
            pad_to (int): The padded number of coordinates (provided as input).

        Returns:
            BucketScatterBuffers: An instance of BucketScatterBuffers with allocated buffers.
        """
        coord_dtype = torch.float16
        Total_N, D = coords.shape
        dev = coords.device
        N = f3dbatch.sep2sizes(seps).to(torch.float32).mean().item()
        B = seps.shape[0]
        num_buck = f3darith.round_next_two_power(int(N / bucket_size))
        cnt_dtype = torch.uint32
        hash_dtype = torch.uint16

        bucket_id = torch.ones(Total_N, dtype=hash_dtype, device=dev)
        bucket_cntr = torch.zeros(B, num_buck, dtype=cnt_dtype, device=dev)
        bucket_offs = torch.empty(Total_N, dtype=cnt_dtype, device=dev)
        bucket_cumsum = torch.zeros_like(bucket_cntr)
        probe_offs = f3dbuck.generate_probe_offsets(dev, row_major=True)
        scattered_coord = torch.full((pad_to, D), float('nan'), dtype=coord_dtype, device=dev)

        return BucketScatterBuffers(
            bucket_id=bucket_id,
            bucket_cntr=bucket_cntr,
            bucket_offs=bucket_offs,
            bucket_cumsum=bucket_cumsum,
            probe_offs=probe_offs,
            scattered_coord=scattered_coord
        )


def batch_bucket_scatter(coords: torch.Tensor, seps: torch.Tensor, bucket_size: int, pad_to: int, hash_op: int
                         ) -> Tuple[torch.Tensor, BucketScatterBuffers]:
    """
    Perform the batch bucket scatter pad operation.

    This function computes kernel parameters, allocates all required buffers,
    and calls the scatter_pad kernel dispatcher (using the provided hash_op).
    It returns the scattered_coord output and a BucketScatterBuffers instance.

    Args:
        coords (torch.Tensor): Input coordinates tensor (contiguous, dtype torch.float16).
        seps (torch.Tensor): Separation tensor.
        bucket_size (int): Bucket size parameter.
        pad_to (int): The padded number of coordinates.
        hash_op (int): Hash operation indicator.

            Hash Type Definitions:
                H_ZORDER_DIV = 1,
                H_XORSUM_DIV = 2,
                H_ZORDER_MOD = 3,
                H_XORSUM_MOD = 4

    Returns:
        Tuple[torch.Tensor, BucketScatterBuffers]:
            - scattered_coord: The output tensor with scattered coordinates.
            - buffers: An instance of BucketScatterBuffers containing all allocated buffers.
    """
    f3ddev.init_dev(coords.get_device())

    buffers = BucketScatterBuffers.create_buffers(coords, seps, bucket_size, pad_to)
    num_vox = 0xFFFF
    N = f3dbatch.sep2sizes(seps).to(torch.float32).mean().item()
    MaxNInstance = int(f3dbatch.sep2sizes(seps.to(torch.int64)).max().item())
    num_buck = f3darith.round_next_two_power(int(N / bucket_size))
    bucket_divisor_heuristic = int(math.ceil(num_vox / num_buck))

    pshattn.batch_psh_scatter_pad_hash(
        coords,
        buffers.bucket_id,
        buffers.bucket_cntr,
        buffers.bucket_offs,
        seps,
        MaxNInstance,
        num_buck,
        bucket_divisor_heuristic,
        bucket_size,
        num_vox,
        torch.min(coords, dim=0)[0],
        torch.max(coords, dim=0)[0],
        buffers.probe_offs,
        buffers.bucket_cumsum,
        buffers.scattered_coord,
        0.0,
        hash_op
    )
    return buffers.scattered_coord, buffers
