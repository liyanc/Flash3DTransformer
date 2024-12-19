#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
#  Authored by Liyan Chen (liyanc@cs.utexas.edu) on 11/19/24
#

import torch
from torch.autograd import Function
from torch import nn
from typing import Tuple

from ..psh.subbuck import SubbuckBuffers, UnpoolBuffers, UnpoolGradients
from ..lib import pshattn


class AdditiveUnpoolFunction(Function):
    @staticmethod
    def forward(ctx, residual_feat: torch.Tensor, down_feat: torch.Tensor, buffers: SubbuckBuffers,
                subbuck_size: int) -> torch.Tensor:
        """
        Forward pass for additive unpooling.

        Args:
            residual_feat (torch.Tensor): Residual features at the upsampled level.
            down_feat (torch.Tensor): Downsampled features (output from pooling).
            buffers (SubbuckBuffers): Buffer containing auxiliary information,
                in particular the unpool_ind tensor.
            subbuck_size (int): Number of slots per subbucket.

        Returns:
            torch.Tensor: The upsampled output tensor (up_add_feat), which is differentiable.
        """
        # Create the output buffer using the helper.
        unpool_buf: UnpoolBuffers = UnpoolBuffers.create_unpool_buff(residual_feat)
        # Call the forward unpooling kernel.
        pshattn.additive_unpool_fwd(
            residual_feat,
            down_feat,
            unpool_buf.up_add_feat,
            buffers.unpool_ind,
            subbuck_size
        )
        ctx.subbuck_buffers = buffers  # Save the entire UnpoolBuffers instance.
        ctx.subbuck_size = subbuck_size
        return unpool_buf.up_add_feat

    @staticmethod
    def backward(ctx, grad_up_add_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, None, None]:
        """
        Backward pass for additive unpooling.

        Args:
            grad_up_add_feat (torch.Tensor): Gradient with respect to the unpooled output.

        Returns:
            Tuple:
              - Gradient for residual_feat.
              - Gradient for down_feat.
              - None for buffers (non-differentiable).
              - None for subbuck_size.
        """
        buffers: SubbuckBuffers = ctx.subbuck_buffers
        subbuck_size: int = ctx.subbuck_size

        # Create gradient buffers using the helper; use buffers for shape info.
        unpool_grad: UnpoolGradients = UnpoolGradients.create_unpool_grad(grad_up_add_feat, buffers)
        # Call the backward unpooling kernel.
        pshattn.additive_unpool_bwd(
            unpool_grad.grad_res,
            unpool_grad.grad_down,
            grad_up_add_feat,
            buffers.unpool_ind,
            subbuck_size
        )
        return unpool_grad.grad_res, unpool_grad.grad_down, None, None


class AdditiveUnpoolLayer(nn.Module):
    """
    nn.Module wrapper for additive unpooling.

    This layer wraps the additive unpooling kernel into a PyTorch layer.
    It accepts a SubbuckBuffers instance (from pooling) and a residual feature tensor,
    and outputs the unpooled feature tensor along with auxiliary unpool buffers.

    Forward Args:
        residual_feat (torch.Tensor): Residual features at the upsampled level.
        buffers (SubbuckBuffers): Contains the downsampled features and unpool indices.
        subbuck_size (int): Number of slots per subbucket.

    Returns:
        Tuple[torch.Tensor, UnpoolBuffers]:
          - up_add_feat: Output tensor from unpooling (differentiable).
          - unpool_buf: An UnpoolBuffers instance containing auxiliary info.
    """

    def __init__(self, subbuck_size: int = 2) -> None:
        super(AdditiveUnpoolLayer, self).__init__()
        assert subbuck_size == 2, f"Currently only subbuck_size=2 is supported, input is {subbuck_size}"
        self.subbuck_size = subbuck_size

    def forward(self, residual_feat: torch.Tensor, down_feat: torch.Tensor, buffers: SubbuckBuffers) -> torch.Tensor:
        return AdditiveUnpoolFunction.apply(residual_feat, down_feat, buffers, self.subbuck_size)
