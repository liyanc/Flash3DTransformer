#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
#  Authored by Liyan Chen (liyanc@cs.utexas.edu) on 7/18/24


import torch
from torch import nn
import torch.cuda.nvtx as nvtx
from torch.autograd import Function
from typing import Tuple

from ..psh import swin_attn as sa
from ..psh import dev_context as f3ddev
from ..lib import pshattn


class BuckSwinFwdFunction(Function):
    """
    A custom autograd function for the buck_swin forward pass.

    This function computes the output tensor O based on input tensors Q, K, V,
    along with additional parameters scope_buckets and buck_size. It stores auxiliary
    buffers in the context for use in the backward pass.
    """
    @staticmethod
    def forward(ctx,
                Q: torch.Tensor,
                K: torch.Tensor,
                V: torch.Tensor,
                scope_buckets: torch.Tensor,
                buck_size: int) -> torch.Tensor:
        """
        Performs the forward pass of the buck_swin operation.

        This dummy implementation creates auxiliary buffers using the provided query tensor Q.
        It computes a dummy LSE by applying log-sum-exp over the last dimension of Q and computes
        the output tensor O as the element-wise sum of Q and V. The auxiliary buffers are stored
        in the ctx for backward computation.

        Parameters:
            Q (torch.Tensor): Query tensor of shape (B, L, H, D).
            K (torch.Tensor): Key tensor of shape (B, L, H, D).
            V (torch.Tensor): Value tensor of shape (B, L, H, D).
            scope_buckets (torch.Tensor): Tensor defining bucket scopes.
            buck_size (int): Size of each bucket.

        Returns:
            torch.Tensor: The output tensor O.
        """
        # Create auxiliary buffers using the provided query tensor Q.
        buff = sa.BuckSwinFwdBuffers.create_buckswin_fwd_buff(Q)

        # Invoke the actual implementation; it populates buff.O and buff.LSE.
        pshattn.buck_swin_fwd(Q, K, V, buff.O, buff.LSE, scope_buckets, buck_size)

        # Save necessary tensors and auxiliary buffers for the backward pass.
        ctx.save_for_backward(Q, K, V, buff.O, buff.LSE)
        ctx.scope_buckets = scope_buckets
        ctx.buck_size = buck_size
        ctx.buff = buff

        return buff.O


    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
        """
        Performs the backward pass of the buck_swin operation by connecting
        the gradient dO to the actual backward implementation.

        Parameters:
            grad_output (torch.Tensor): Gradient tensor dO of shape (B, L, H, D)
                                        corresponding to the output O.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None]:
                Gradients with respect to Q, K, and V, and None for scope_buckets
                and buck_size.
        """
        # Retrieve saved tensors from the forward pass.
        Q, K, V, O, lse = ctx.saved_tensors
        scope_buckets = ctx.scope_buckets
        buck_size = ctx.buck_size

        # Create backward auxiliary buffers.
        bwd_buff = sa.BuckSwinBwdBuffers.create_buckswin_buffers(O, scope_buckets, buck_size)
        # Call the actual backward implementation.
        pshattn.buck_swin_bwd(Q, K, V, O, grad_output, lse, bwd_buff.Delta,
                              bwd_buff.dQ, bwd_buff.dK, bwd_buff.dV, scope_buckets, buck_size)

        # Return gradients for Q, K, V; non-tensor inputs receive None.
        return bwd_buff.dQ, bwd_buff.dK, bwd_buff.dV, None, None


class BucketSwinAttention(nn.Module):
    """
    Torch module wrapper for BuckSwinFwdFunction.

    This module encapsulates the buck_swin operation provided by the underlying
    implementation in pshattn. The buck_size parameter is set during initialization,
    while scope_buckets is provided as an argument during the forward pass.
    """
    def __init__(self, buck_size: int) -> None:
        """
        Initializes the BuckSwin module.

        Parameters:
            buck_size (int): The size of each bucket for the buck_swin operation.
        """
        super().__init__()
        self.buck_size = buck_size

    def forward(self,
                Q: torch.Tensor,
                K: torch.Tensor,
                V: torch.Tensor,
                scope_buckets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the BucketSwinAttention module.

        This method calls the BuckSwinFwdFunction with the provided Q, K, V tensors
        and scope_buckets, using the buck_size specified during initialization.

        Parameters:
            Q (torch.Tensor): Query tensor of shape (B, L, H, D).
            K (torch.Tensor): Key tensor of shape (B, L, H, D).
            V (torch.Tensor): Value tensor of shape (B, L, H, D).
            scope_buckets (torch.Tensor): Tensor defining bucket scopes.

        Returns:
            torch.Tensor: The output tensor O computed by BuckSwinFwdFunction.
        """
        assert Q.dim() == 4, \
            f"Expect Q with four axes [B, L, H, D], but got shape={Q.shape}"
        return BuckSwinFwdFunction.apply(Q, K, V, scope_buckets, self.buck_size)


class BucketSwinAttentionModule(nn.Module):
    NVTX_SCOPE = "BucketSwinAttention"

    def __init__(self, channels: int, num_heads: int, buck_size: int, qkv_bias: bool = True):
        """
        Initializes the BucketSwinAttentionModule.

        Parameters:
            channels (int): The number of input/output channels, must be divisible by num_heads.
            num_heads: The number of heads in MHSA
            buck_size (int): The size of each bucket for the BucketSwinAttention.
            qkv_bias (bool): Whether to include a bias term in the QKV projection.
        """
        super().__init__()
        assert channels % num_heads == 0, f"Channels={channels} NOT divisible by num_heads={num_heads}"
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        # Linear layer to project input to Q, K, and V (concatenated)
        self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        # BucketSwinAttention module
        self.attn = BucketSwinAttention(buck_size)
        # Output projection layer
        self.proj = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, scope_buckets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the BucketSwinAttentionModule.

        Parameters:
            x (torch.Tensor): Input tensor of shape [B, L, channels].
            scope_buckets (torch.Tensor): Bucket indices for the attention operation.

        Returns:
            torch.Tensor: Output tensor of shape [B, L, channels].
        """
        assert x.dim() == 3, \
            f"Expect input features with three axes [B, L, C], but got shape={x.shape}"
        B, L, C = x.shape
        assert C == self.num_heads * self.head_dim, \
            f"Expect input channels are {self.num_heads * self.head_dim}, but got {C}"

        if f3ddev.get_nvtx():
            nvtx.range_push(self.NVTX_SCOPE)
        # Apply QKV projection; output shape: [B, L, 3 * channels]
        qkv = self.qkv(x)
        # Split concatenated QKV into separate Q, K, V tensors; each of shape: [B, L, channels]
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q_mh = q.view(B, L, self.num_heads, self.head_dim).contiguous()
        k_mh = k.view(B, L, self.num_heads, self.head_dim).contiguous()
        v_mh = v.view(B, L, self.num_heads, self.head_dim).contiguous()
        # Apply bucket swin attention using provided scope_buckets
        attn_out = self.attn(q_mh, k_mh, v_mh, scope_buckets)
        # Apply output projection
        out = self.proj(attn_out.view(B, L, C))
        if f3ddev.get_nvtx():
            nvtx.range_pop()
        return out