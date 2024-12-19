#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
#  Authored by Liyan Chen (liyanc@cs.utexas.edu) on 11/10/24
#
from typing import Union, Tuple

import torch
from torch import nn
import torch.cuda.nvtx as nvtx

from ..psh import subbuck as f3dsubbuck
from ..psh import batching as f3dbatch
from ..psh import dev_context as f3ddev
from ..layers import pooling, unpool
from ..lib import pshattn


class PoolingWrapper(nn.Module):
    """
    PyTorch module for a wrapped pooling layer with input/output projection.

    This module applies an input projection, performs pooling reduction using a
    provided pooling layer, and optionally applies normalization and activation.

    The forward pass takes:
      - coords      : Tensor with coordinate data.
      - input_feat  : Tensor of shape (total_N, in_channels).
      - seps        : Tensor with separation data.

    It returns:
      A tuple (reduced_feat, buffers) where:
        - reduced_feat: Tensor of shape (reduced_N, feat_dim) that is differentiable.
        - buffers     : A SubbuckBuffers instance containing auxiliary info
                        (bbox_min, bbox_max, expect_sep, subbuck_id, subbuck_off,
                         reduced_sep, reduced_coord, unpool_ind) used for backward
                         and later layers.
    """
    NVTX_SCOPE = "PoolingWrapper"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        red_op,
        out_align,
        norm_layer: callable = None,
        act_layer: callable = None
    ):
        """
        Initializes the PoolingWrapper module.

        Parameters:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            red_op: Integer indicating the reduction_op
            out_align: Output length alignment requirement for bucket_swin
            norm_layer (callable, optional): Normalization layer constructor. Defaults to None.
            act_layer (callable, optional): Activation layer constructor. Defaults to None.
        """
        super().__init__()
        # Input projection layer to map from in_channels to out_channels.
        self.proj = nn.Linear(in_channels, out_channels)
        # Pooling layer instance.
        self.pooling = pooling.InbucketPoolingLayer(red_op, out_alignment=out_align)
        # Optional normalization layer.
        if norm_layer is not None:
            self.norm = norm_layer(out_channels)
        else:
            self.norm = None
        # Optional activation layer.
        if act_layer is not None:
            self.act = act_layer()
        else:
            self.act = None

    def forward(self, coords: torch.Tensor, input_feat: torch.Tensor, seps: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, f3dsubbuck.SubbuckBuffers]:
        """
        Forward pass for the wrapped pooling layer.

        Parameters:
            coords (torch.Tensor): Tensor with coordinate data.
            input_feat (torch.Tensor): Tensor of shape (total_N, in_channels).
            seps (torch.Tensor): Tensor with separation data.

        Returns:
          A tuple (reduced_feat, buffers) where:
            - reduced_feat: Tensor of shape (reduced_N, feat_dim) that is differentiable.
            - projected_feat: Tensor of shape (total_N, feat_dim) representing the residual/neck connection.
            - buffers     : A SubbuckBuffers instance containing auxiliary info
                            (bbox_min, bbox_max, expect_sep, subbuck_id, subbuck_off,
                             reduced_sep, reduced_coord, unpool_ind) used for backward
                             and later layers.
        """
        if f3ddev.get_nvtx():
            nvtx.range_push(self.NVTX_SCOPE)
        # Apply the input projection.
        projected = self.proj(input_feat)  # Shape: (total_N, out_channels)
        # Apply the pooling layer.
        reduced_feat, subbuffers = self.pooling(coords, projected, seps)
        # Optionally apply normalization.
        if self.norm is not None:
            reduced_feat = self.norm(reduced_feat)
        # Optionally apply activation.
        if self.act is not None:
            reduced_feat = self.act(reduced_feat)
        if f3ddev.get_nvtx():
            nvtx.range_pop()
        return reduced_feat, projected, subbuffers


class AdditiveUnpoolingWrapper(nn.Module):
    """
    PyTorch module wrapper for additive unpooling with input/output projections.

    This module performs an input projection on the downsampled features and a skip
    projection on the residual features, optionally followed by normalization and activation.
    Then, it applies additive unpooling to combine the features.

    The forward pass takes:
      - residual: A tensor containing residual features.
      - down: A tensor containing downsampled features.
      - buffers: A SubbuckBuffers instance from the pooling operation.

    It returns:
      - output: The unpooled feature tensor.
    """
    NVTX_SCOPE = "AdditiveUnpoolingWrapper"

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int,
                 subbuck_size: int = 2, norm_layer: callable = None, act_layer: callable = None):
        """
        Initializes the AdditiveUnpoolingWrapper.

        Parameters:
            in_channels (int): Number of input channels for the downsampled features.
            skip_channels (int): Number of input channels for the residual (skip) features.
            out_channels (int): Number of output channels after projection.
            subbuck_size (int): Number of slots per subbucket (must be 2).
            norm_layer (callable, optional): Normalization layer constructor. Defaults to None.
            act_layer (callable, optional): Activation layer constructor. Defaults to None.
        """
        super().__init__()
        # Projection for downsampled features.
        self.proj = nn.Sequential(nn.Linear(in_channels, out_channels))
        # Projection for skip (residual) features.
        self.proj_skip = nn.Sequential(nn.Linear(skip_channels, out_channels))

        # Add optional normalization layers.
        if norm_layer is not None:
            self.proj.add_module("norm", norm_layer(out_channels))
            self.proj_skip.add_module("norm", norm_layer(out_channels))

        # Add optional activation layers.
        if act_layer is not None:
            self.proj.add_module("act", act_layer())
            self.proj_skip.add_module("act", act_layer())

        # Additive unpooling layer.
        self.unpool = unpool.AdditiveUnpoolLayer(subbuck_size)

    def forward(self, residual: torch.Tensor, down: torch.Tensor, buffers: f3dsubbuck.SubbuckBuffers
                ) -> torch.Tensor:
        """
        Forward pass for additive unpooling.

        Parameters:
            residual (torch.Tensor): Residual features tensor.
            down (torch.Tensor): Downsampled features tensor.
            buffers: A SubbuckBuffers instance containing auxiliary unpooling data.

        Returns:
            torch.Tensor: The unpooled feature tensor.
        """
        if f3ddev.get_nvtx():
            nvtx.range_push(self.NVTX_SCOPE)
        # Project the downsampled features.
        proj_down = self.proj(down)
        # Project the residual features (skip connection).
        proj_skip = self.proj_skip(residual)
        # Apply additive unpooling using the projected skip and down features.
        output = self.unpool(proj_skip, proj_down, buffers)
        if f3ddev.get_nvtx():
            nvtx.range_pop()
        return output


class HourglassModule(nn.Module):
    """
    HourglassModule encapsulates a pooling wrapper, a submodule, and an additive unpooling wrapper.

    The module performs the following steps:
      1. Applies the pooling wrapper to the input (coords, input_feat, seps) to obtain reduced features and auxiliary buffers.
         The reduced features serve as a skip connection.
      2. Processes the reduced features through the provided submodule.
      3. Uses the additive unpooling wrapper to combine the skip connection (from pooling) with the submodule output,
         using the auxiliary buffers.

    NVTX ranges are annotated based on f3ddev.get_nvtx().
    """
    NVTX_SCOPE = "HourglassModule"

    def __init__(self,
                 # Pooling parameters
                 pooling_in_channels: int,
                 pooling_out_channels: int,
                 pooling_reduction_op,
                 pooling_length_alignment: int,
                 # Unpooling parameters
                 unpooling_in_channels: int,
                 unpooling_out_channels: int,
                 pooling_norm_layer: callable = None,
                 pooling_act_layer: callable = None,
                 unpooling_norm_layer: callable = None,
                 unpooling_act_layer: callable = None,
                 # Submodule to process the pooled features
                 submodule: nn.Module = None):
        """
        Initializes the HourglassModule.

        Parameters:
            pooling_in_channels (int): Number of input channels for the pooling wrapper.
            pooling_out_channels (int): Number of output channels for the pooling wrapper.
            pooling_reduction_op (str): Reduction operation for the pooling layer (e.g., "mean").
            pooling_norm_layer (callable, optional): Normalization layer constructor for pooling.
            pooling_act_layer (callable, optional): Activation layer constructor for pooling.
            pooling_length_alignment (int): pooled buffer length (#tokens) alignment requirements for buck_swin.
            pooling_subbuck_size (int, optional): Subbucket size for pooling (currently fixed to 2).

            unpooling_in_channels (int): Input channels for the unpooling wrapper (from downsampled features).
            unpooling_out_channels (int): Output channels for the unpooling wrapper.
            unpooling_norm_layer (callable, optional): Normalization layer constructor for unpooling.
            unpooling_act_layer (callable, optional): Activation layer constructor for unpooling.
            unpooling_subbuck_size (int, optional): Subbucket size for unpooling (currently fixed to 2).

            submodule (nn.Module): A submodule to process the pooled features. This module must accept
                                   inputs as (buffers.reduced_coord, reduced_feat, buffers.reduced_sep).
        """
        super().__init__()

        # Create the pooling wrapper with input projection.
        self.pooling = PoolingWrapper(
            in_channels=pooling_in_channels,
            out_channels=pooling_out_channels,
            red_op=pooling_reduction_op,
            out_align=pooling_length_alignment,
            norm_layer=pooling_norm_layer,
            act_layer=pooling_act_layer
        )

        # Create the additive unpooling wrapper with input and skip projections.
        self.unpooling = AdditiveUnpoolingWrapper(
            in_channels=unpooling_in_channels,
            skip_channels=pooling_out_channels,
            out_channels=unpooling_out_channels,
            norm_layer=unpooling_norm_layer,
            act_layer=unpooling_act_layer
        )

        # The submodule that processes the pooled features.
        if submodule is None:
            raise ValueError("A submodule must be provided for HourglassModule.")
        self.submodule = submodule

    def forward(self, coords: torch.Tensor, input_feat: torch.Tensor, seps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the HourglassModule.

        Parameters:
            coords (torch.Tensor): Tensor with coordinate data.
            input_feat (torch.Tensor): Tensor of shape (total_N, pooling_in_channels).
            seps (torch.Tensor): Tensor with separation data.

        Returns:
            torch.Tensor: The final output tensor after combining the pooling skip connection and the submodule output.
        """
        if f3ddev.get_nvtx():
            nvtx.range_push(self.NVTX_SCOPE)

        # Step 1: Pooling - obtain reduced, residual, and auxiliary buffers.
        reduced_feat, residual_feat, buffers = self.pooling(coords, input_feat, seps)

        # Step 2: Process the reduced features through the submodule.
        sub_out = self.submodule(buffers.reduced_coord, reduced_feat, buffers.reduced_sep)

        # Step 3: Unpooling - combine the skip connection (reduced_feat) with the submodule output using the buffers.
        output = self.unpooling(residual_feat, sub_out, buffers)

        if f3ddev.get_nvtx():
            nvtx.range_pop()
        return output