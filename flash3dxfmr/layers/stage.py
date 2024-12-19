#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
#  Authored by Liyan Chen (liyanc@cs.utexas.edu) on 11/10/24
#

import torch
from torch import nn
import torch.cuda.nvtx as nvtx
from typing import Union, Tuple, List
from dataclasses import dataclass

from ..psh import bucket_scope
from ..psh import dev_context as f3ddev
from ..layers import pooling, unpool, hourglass, elem_wise, buck_swin
from ..lib import pshattn


@dataclass(slots=True)
class XFMRSpecs:
    channels: int
    hid_channels: int
    num_heads: int
    qkv_bias: bool
    swin_plan: bucket_scope.SwinPlan
    norm_layer: callable


class XFMR(nn.Module):
    """
    XFMR module that processes pooled features.

    The module follows the sequence:
      1. norm1: Normalizes the input reduced features.
      2. bwa: Applies BucketSwinAttentionModule using the reduced coordinates as scope buckets.
      3. norm2: Normalizes the output of the attention module.
      4. mlp: Applies an MLP.

    The module accepts inputs with the signature:
         (reduced_coord, reduced_feat, reduced_sep)
    and uses reduced_coord for the attention module while ignoring reduced_sep.
    NVTX ranges are annotated if enabled.
    """
    NVTX_SCOPE = "XFMR"

    def __init__(self, channels: int, hid_chann: int, num_heads: int, buck_size: int, qkv_bias: bool,
                 swin_plan: bucket_scope.SwinPlan, norm_layer: callable):
        """
        Initializes the XFMR module.

        Parameters:
            channels (int): Number of input channels.
            hid_chann (int): Number of hidden channels inside the MLP
            num_heads (int): Number of heads to divide the input channels.
            buck_size (int): Bucket size for the BucketSwinAttentionModule.
            qkv_bias (bool): Whether to use bias in the QKV projection.
            swin_plan (bucket_scope.SwinPlan): Parameters to construct bucket_swin scopes.
                The constructor only accepts swin_plan parameters while we generate scopes during runtime.
                Bucket scopes depend on input buffer sizes.
            norm_layer (callable): Normalization layer constructor (e.g., lambda ch: nn.LayerNorm(ch)).
        """
        super().__init__()
        self.set_chnn = channels
        self.swin_plan = swin_plan
        self.norm = norm_layer(channels)
        self.bwa = buck_swin.BucketSwinAttentionModule(channels, num_heads, buck_size, qkv_bias)
        self.mlp = elem_wise.MLP(channels, hid_chann)

    def forward(self, reduced_coord: torch.Tensor, reduced_feat: torch.Tensor,
                reduced_sep: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the XFMR module.
        Following the submodule interface to accept a triplet input

        Parameters:
            reduced_coord (torch.Tensor): Reduced coordinate tensor (unused).
            reduced_feat (torch.Tensor): Reduced feature tensor.
            reduced_sep (torch.Tensor): Reduced batch separator tensor (unused).

        Returns:
            torch.Tensor: Processed feature tensor.
        """
        assert reduced_feat.dim() == 2, \
            f"Expect input features have two axes [N, C], but got shape={reduced_feat.shape}"
        N, C = reduced_feat.shape
        assert C == self.set_chnn, \
            f"Expect input feat_dim={self.set_chnn}, but got {C}"
        scope_bucks = self.swin_plan.gen_scopes_from_plan(N, reduced_feat.device)

        if f3ddev.get_nvtx():
            nvtx.range_push(self.NVTX_SCOPE)
        # Step 1: Pre-norm.
        x = self.norm(reduced_feat)
        # Step 2: Apply BucketSwinAttentionModule using calculated bucket scopes.
        mid = self.bwa(x.view(1, N, C), scope_bucks).view(N, C) + reduced_feat
        # Step 3: Apply MLP.
        x = self.mlp(mid) + mid
        if f3ddev.get_nvtx():
            nvtx.range_pop()
        return x


class Stage(nn.Module):
    """
    Stage module that conforms to the submodule interface.

    The Stage module is composed of:
      - A sequence of encoder blocks.
      - A middle submodule (provided by the user; ownership is acquired).
      - A sequence of decoder blocks.

    The forward pass accepts inputs:
        (reduced_coord, reduced_feat, reduced_sep)
    and processes them by first applying the encoder blocks, then the middle submodule,
    and finally the decoder blocks. Each block uses its own scope_buckets.
    NVTX ranges are annotated if enabled.
    """
    NVTX_SCOPE = "Stage"

    def __init__(self,
                 # Encoder block parameters
                 encoder_specs: List[XFMRSpecs],
                 buck_size: int,
                 # Middle submodule (must follow the submodule signature)
                 submodule: nn.Module,
                 # Decoder block parameters
                 decoder_specs: List[XFMRSpecs]):
        """
        Initializes the Stage module.

        Parameters:
            encoder_specs List[XFMRSpecs]: List of XFMRSpecs defining the encoder blocks.
            submodule (nn.Module): A submodule that processes features. It must accept inputs as
                                   (reduced_coord, reduced_feat, reduced_sep).
            decoder_specs List[XFMRSpecs]: List of XFMRSpecs defining the decoder blocks
        """
        super().__init__()
        # Build encoder blocks as a ModuleList.
        self.encoder_blocks = nn.ModuleList([
            XFMR(
                spec.channels,
                spec.hid_channels,
                spec.num_heads,
                buck_size,
                spec.qkv_bias,
                spec.swin_plan,
                spec.norm_layer
            )
            for spec in encoder_specs
        ])
        # Acquire ownership of the provided middle submodule.
        if submodule is None:
            raise ValueError("A submodule must be provided for a Stage.")
        self.middle = submodule
        # Build decoder blocks as a ModuleList.
        self.decoder_blocks = nn.ModuleList([
            XFMR(
                spec.channels,
                spec.hid_channels,
                spec.num_heads,
                buck_size,
                spec.qkv_bias,
                spec.swin_plan,
                spec.norm_layer
            )
            for spec in decoder_specs
        ])

    def forward(self, reduced_coord: torch.Tensor, reduced_feat: torch.Tensor,
                reduced_sep: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Stage module.

        Parameters:
            reduced_coord (torch.Tensor): Reduced coordinate tensor.
            reduced_feat (torch.Tensor): Reduced feature tensor.
            reduced_sep (torch.Tensor): Reduced batch separator tensor.

        Returns:
            torch.Tensor: Processed feature tensor after encoder blocks, the middle submodule, and decoder blocks.
        """
        if f3ddev.get_nvtx():
            nvtx.range_push(self.NVTX_SCOPE)
        x = reduced_feat
        # Pass through the encoder blocks sequentially.
        for block in self.encoder_blocks:
            x = block(reduced_coord, x, reduced_sep)
        # Process through the middle submodule.
        x = self.middle(reduced_coord, x, reduced_sep)
        # Pass through the decoder blocks sequentially.
        for block in self.decoder_blocks:
            x = block(reduced_coord, x, reduced_sep)
        if f3ddev.get_nvtx():
            nvtx.range_pop()
        return x
