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
import transformer_engine.pytorch as te
from typing import Union, Tuple, List
from dataclasses import dataclass

from ..psh import dev_context as f3ddev
from ..layers import psh_ops
from ..layers import hourglass as f3dhourglass
from ..layers import stage as f3dstage


class LeafModule(nn.Module):
    """
    LeafModule does nothing
    """

    def __init__(self, in_dim, out_dim):
        """
        Initializes the LeafModule.
        """
        super().__init__()
        self.proj = te.Linear(in_dim, out_dim)

    def forward(self, reduced_coord: torch.Tensor,
                reduced_feat: torch.Tensor,
                reduced_sep: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the LeafModule.

        Parameters:
            reduced_coord (torch.Tensor): Reduced coordinate data (ignored).
            reduced_feat (torch.Tensor): Reduced feature tensor.
            reduced_sep (torch.Tensor): Reduced separations (ignored).

        Returns:
            torch.Tensor: Aligned feature tensor.
        """
        return self.proj(reduced_feat)


@dataclass(slots=True)
class F3DLevelSpecs:
    encoder_specs: List[f3dstage.XFMRSpecs]
    decoder_specs: List[f3dstage.XFMRSpecs]
    reduction_op: Union[int, str]
    pool_align: int
    pool_norm: callable
    pool_act: callable
    unpool_norm: callable
    unpool_act: callable


###############################################
# Top-Level Flash3D Module (Recursive)
###############################################

class Flash3D(nn.Module):
    """
    Flash3D recursively builds a hierarchical module based on a list of level specifications.

    Specs are defined as a `F3DLevelSpecs` instance
    At non-leaf nodes, a Stage is constructed whose submodule is a HourglassModule wrapping the next-level Stage.
    At leaf nodes, the Stage's submodule is a LeafModule, an identity.

    NVTX ranges are used if enabled.
    """
    NVTX_SCOPE = "Flash3D"

    def __init__(self, level_specs: List[F3DLevelSpecs], buck_size: int, scope_size: int, top_level_hash):
        """
        Initializes the Flash3D module.

        Parameters:
            level_specs (list of F3DLevelSpecs): Defines every parameter in a stage
            buck_size (int): bucket size for this transformer
            scope_size (int): attention scope size in number of buckets
            top_level_hash: hashtype for the leading stage
        """
        super().__init__()
        f3ddev.set_sync_stream(False)
        emb_chann = level_specs[0].encoder_specs[0].channels
        self.first_hashtype = top_level_hash
        # We need to further pad for more buck_swin striding options.
        self.psh_scatter = psh_ops.PSH3DCoordEmbedding(
            emb_chann, buck_size, buck_size * scope_size * 6)

        self.level_specs = level_specs
        self.buck_size = buck_size
        self.scope_size = scope_size
        # Recursively build the module tree starting from level 0.
        self.module_tree = self.build_module(0)

    def build_module(self, level: int) -> nn.Module:
        """
        Recursively builds the module tree from level_specs.

        At the leaf node, returns a Stage with a LeafModule as the submodule.
        At non-leaf nodes, returns a Stage whose submodule is a HourglassModule wrapping the next level's Stage.
        """
        if level >= len(self.level_specs):
            raise ValueError("Level exceeds level_specs length.")
        level_spec = self.level_specs[level]
        next_spec = self.level_specs[level + 1] if level + 1 < len(self.level_specs) else None
        last_enc_spec = level_spec.encoder_specs[-1]
        next_enc_spec = None if next_spec is None else next_spec.encoder_specs[0]
        first_dec_spec = level_spec.decoder_specs[0]
        next_dec_spec = None if next_spec is None else next_spec.decoder_specs[-1]

        if level + 1 >= len(self.level_specs):
            # Leaf node: submodule is a LeafModule.
            submodule = LeafModule(last_enc_spec.channels, first_dec_spec.channels)
        else:
            # Non-leaf node: recursively build the next level Stage and wrap it in a HourglassModule.
            next_stage = self.build_module(level + 1)
            submodule = f3dhourglass.HourglassModule(
                pooling_in_channels=last_enc_spec.channels,
                # When next_enc_spec is None, this branch won't be called
                pooling_out_channels=next_enc_spec.channels,
                pooling_reduction_op=level_spec.reduction_op,
                pooling_length_alignment=level_spec.pool_align,
                unpooling_in_channels=next_dec_spec.channels,
                unpooling_out_channels=first_dec_spec.channels,
                pooling_norm_layer=level_spec.pool_norm,
                pooling_act_layer=level_spec.pool_act,
                unpooling_norm_layer=level_spec.unpool_norm,
                unpooling_act_layer=level_spec.unpool_act,
                submodule=next_stage
            )
        # Build the Stage at the current level.
        stage = f3dstage.Stage(
            encoder_specs=level_spec.encoder_specs,
            buck_size=self.buck_size,
            submodule=submodule,
            decoder_specs=level_spec.decoder_specs
        )
        return stage

    def forward(self, coords: torch.Tensor, input_feat: torch.Tensor, seps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Flash3D.

        Parameters:
            coords (torch.Tensor): Coordinate data tensor.
            input_feat (torch.Tensor): Input feature tensor (top-level channels).
            seps (torch.Tensor): Separation data tensor.

        Returns:
            torch.Tensor: Final output tensor.
        """
        if f3ddev.get_nvtx():
            nvtx.range_push(self.NVTX_SCOPE)
        psh_feat = self.psh_scatter(coords, seps, self.first_hashtype)
        out = self.module_tree(coords, psh_feat, seps)
        if f3ddev.get_nvtx():
            nvtx.range_pop()
        return out

