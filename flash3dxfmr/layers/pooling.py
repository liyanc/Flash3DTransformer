#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
#  Authored by Liyan Chen (liyanc@cs.utexas.edu) on 11/10/24
#

import torch
from torch.autograd import Function
from torch import nn

from ..psh import subbuck as f3dsubbuck
from ..psh import batching as f3dbatch
from ..lib import pshattn


class SubbuckReduceFunction(Function):
    @staticmethod
    def forward(ctx, coords: torch.Tensor, input_feat: torch.Tensor, seps:torch.Tensor,
                subbuck_size, reduction_op, out_alignment=1):
        """
        Forward pass for subbucket reduction.

        Inputs:
          coords       : Tensor with coordinate data.
          input_feat   : Tensor of shape (total_N, feat_dim).
          seps         : Tensor with separation data.
          subbuck_size : Integer, number of slots per subbucket.
          reduction_op : Integer indicating the reduction operation:
                         0 = O_MEAN, 1 = O_SUM, 2 = O_MIN, 3 = O_MAX.
          out_alignment: Integer indicating the alignment for output tensor length
                         (reduced_N).

        Outputs:
          A tuple (reduced_feat, buffers) where:
            - reduced_feat: Tensor of shape (reduced_N, feat_dim) that is differentiable.
            - buffers     : A SubbuckBuffers instance containing auxiliary info
                            (bbox_min, bbox_max, expect_sep, subbuck_id, subbuck_off,
                             reduced_sep, reduced_coord, unpool_ind) used for backward
                             and later layers.

        Note:
          PyTorch’s autograd will only track the Tensor (reduced_feat); the dataclass
          is merely used to carry additional information.
        """
        # Compute MaxNInstance and set num_vox.
        MaxNInstance = f3dbatch.max_inslen_from_seps(seps)
        num_vox = 0xFFFF

        # Create intermediate buffers via SubbuckBuffers.
        sb = f3dsubbuck.SubbuckBuffers.create_subbuck_buff(
            coords, input_feat, seps, subbuck_size, out_alignment=out_alignment)

        # Call the forward kernel; it fills in the buffers inside sb.
        pshattn.batch_subbuck_reduce(
            coords,
            input_feat,
            seps,
            MaxNInstance,
            subbuck_size,
            num_vox,
            sb.bbox_min,
            sb.bbox_max,
            sb.subbuck_id,
            sb.subbuck_off,
            sb.reduced_sep,
            sb.reduced_coord,
            sb.reduced_feat,
            sb.unpool_ind,
            1,  # Extra parameter required by the kernel.
            reduction_op
        )

        # Save necessary tensors for backward.
        ctx.save_for_backward(input_feat, sb.unpool_ind)
        ctx.subbuck_size = subbuck_size
        ctx.reduction_op = reduction_op

        # Remove unused buffers
        sb.remove_unused_buffers()

        # Return a tuple: (differentiable output, auxiliary buffers)
        return sb.reduced_feat, sb

    @staticmethod
    def backward(ctx, grad_reduced_feat: torch.Tensor, grad_buffers: f3dsubbuck.SubbuckBuffers):
        """
        Backward pass for subbucket reduction.

        Only reduced_feat is differentiable; grad_reduced_feat is the gradient with respect
        to the first output. The second output (buffers) is auxiliary and not differentiable.

        Returns:
          Gradients for the inputs: (None, grad_input_feat, None, None, None, None)
          Only input_feat receives a non-None gradient.
        """
        input_feat, unpool_ind = ctx.saved_tensors
        subbuck_size = ctx.subbuck_size
        reduction_op = ctx.reduction_op

        grad_input_feat = torch.zeros_like(input_feat)

        # Call the backward kernel to compute grad_input_feat.
        pshattn.batch_subbuck_reduce_backward(
            grad_reduced_feat,  # gradient with respect to reduced_feat
            input_feat,
            unpool_ind,
            grad_input_feat,
            subbuck_size,
            reduction_op
        )

        # The forward received (coords, input_feat, seps, subbuck_size, reduction_op, out_alignment);
        # only input_feat is differentiable.
        return None, grad_input_feat, None, None, None, None


class InbucketPoolingLayer(nn.Module):
    """
    PyTorch module for in-bucket pooling reduction.

    This layer encapsulates the forward and backward operations.

    Forward:
      Inputs:
        - coords       : Tensor with coordinate data.
        - input_feat   : Tensor of shape (total_N, feat_dim).
        - seps         : Tensor with separation data.
        - subbuck_size : Number of slots per subbucket.

      Outputs:
        - reduced_coord: Reduced coordinates.
        - reduced_sep  : Reduced separations.
        - reduced_feat : Reduced features (differentiable).
        - unpool_ind   : Unpool indices (uint32), for later use.

    Backward:
      Only reduced_feat is differentiable; the backward pass computes grad_input_feat from grad_reduced_feat.
    """
    REDOP_MAP = {
        "mean": 0,
        "sum": 1,
        "min": 2,
        "max": 3
    }

    def __init__(self, reduction_op, out_alignment=1024, subbuck_size=2):
        """
        Initialize the layer.

        Parameters:
          reduction_op : Integer specifying the reduction operation:
                         0 = O_MEAN, 1 = O_SUM, 2 = O_MIN, 3 = O_MAX.
          out_alignment: Integer specifying the subsequent BucketSwin alignment requirement
        """
        super(InbucketPoolingLayer, self).__init__()
        self.out_align = out_alignment
        assert subbuck_size == 2, f"Currently only subbuck_size=2 is supported, input is {subbuck_size}"
        if isinstance(reduction_op, int):
            self.reduction_op = reduction_op
        elif isinstance(reduction_op, str):
            assert reduction_op in self.REDOP_MAP, f"Unsupported reduction op name:{reduction_op}"
            self.reduction_op = self.REDOP_MAP[reduction_op]
        self.subbuck_size = subbuck_size

    def forward(self, coords, input_feat, seps):
        """
        Forward pass for subbucket reduction.

        Parameters:
          coords       : Tensor with coordinate data.
          input_feat   : Tensor of shape (total_N, feat_dim).
          seps         : Tensor with separation data.
          subbuck_size : Integer, number of slots per subbucket.

        Returns:
          A tuple (reduced_feat, buffers) where:
            - reduced_feat: Tensor of shape (reduced_N, feat_dim) that is differentiable.
            - buffers     : A SubbuckBuffers instance containing auxiliary info
                            (bbox_min, bbox_max, expect_sep, subbuck_id, subbuck_off,
                             reduced_sep, reduced_coord, unpool_ind) used for backward
                             and later layers.
        """
        return SubbuckReduceFunction.apply(
            coords, input_feat, seps, self.subbuck_size, self.reduction_op, self.out_align)
