#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
#  Authored by Liyan Chen (liyanc@cs.utexas.edu) on 10/15/24

import torch
from torch import nn
from typing import Tuple
from ..psh import psh_main as pm
from ..psh import arithmatic as f3darith
import transformer_engine.pytorch as te


class PSHScatterLayer(nn.Module):
    """
    A torch.nn.Module that wraps the batch bucket scatter pad operation.

    This layer takes as input:
        - coords: Input coordinates tensor.
        - seps: Separation tensor.
        - bucket_size: Bucket size parameter.
    It computes the pad_to value from coords and calls batch_bucket_scatter to produce
    the scattered_coord output and associated buffers.
    """

    def __init__(self, bucket_size: int = 512) -> None:
        """
        Initialize the BucketScatterLayer.

        Args:
            bucket_size (int, optional): bucket size. Default is 512.
        """
        super().__init__()
        self.bucket_size = bucket_size

    def forward(self, coords: torch.Tensor, seps: torch.Tensor, hash_op: int
                ) -> Tuple[torch.Tensor, pm.BucketScatterBuffers]:
        """
        Forward pass for batch bucket scatter pad.

        Args:
            coords (Tensor): Input coordinates tensor.
            seps (Tensor): Separation tensor.
            bucket_size (int): Bucket size parameter.

        Returns:
            Tuple[Tensor, BucketScatterBuffers]: A tuple containing the scattered_coord output tensor
            and a BucketScatterBuffers instance with all allocated buffers.
        """
        pad_to = f3darith.cdiv(coords.shape[0], self.bucket_size) * self.bucket_size
        return pm.batch_bucket_scatter(coords, seps, self.bucket_size, pad_to, hash_op)


class PSH3DCoordEmbedding(nn.Module):
    def __init__(self, emb_dim: int, bucket_size, alignment, feat_dtype=torch.bfloat16):
        super().__init__()
        self.bucket_size = bucket_size
        self.alignment = alignment
        self.feat_dtype = feat_dtype
        self.lin = te.Linear(3, emb_dim)

    def forward(self, coords: torch.Tensor, seps: torch.Tensor, hash_op):
        pad_to = f3darith.cdiv(coords.shape[0], self.alignment) * self.alignment
        conditioned_coords, _ = pm.batch_bucket_scatter(coords, seps, self.bucket_size, pad_to, hash_op)
        return self.lin(conditioned_coords.to(self.feat_dtype))
