#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
#  Authored by Liyan Chen (liyanc@cs.utexas.edu) on 2/01/25
#

import os
import glob

import tqdm
import torch
import unittest
import numpy as np
from torch import nn

import flash3dxfmr.psh.subbuck as f3dsubbuck
import flash3dxfmr.psh.batching as f3dbatch
import flash3dxfmr.psh.dev_context as f3ddev
import flash3dxfmr.layers.pooling as f3dpool
import flash3dxfmr.layers.unpool as f3dunpool
from flash3dxfmr.lib import pshattn


KITTI_RT = os.getenv("KITTI_RT", None)
torch.manual_seed(2)

class TestInbucketUnpool(unittest.TestCase):
    BATCH_SIZE = 64
    def get_kitti_seq_iter(self):
        return glob.iglob(f'{KITTI_RT}/dataset/sequences/*/velodyne/*.bin')

    def build_batch(self, num_spls, down_sample_ratio=96, coord_dtype=torch.float16):
        spls = []
        assert KITTI_RT is not None, \
            "KITTI_RT environment variable is not set! Provide KITTI_RT in environment variables."
        for f, _ in zip(self.get_kitti_seq_iter(), range(num_spls)):
            npar = np.fromfile(f, dtype=np.float32).reshape(-1, 4)
            slen = npar.shape[0]
            spls.append(torch.from_numpy(npar[:slen // down_sample_ratio, :]))

        cb, seps = f3dbatch.collate_coords_flat_sep(spls, coord_dtype=coord_dtype, sep_dtype=torch.uint32)
        return cb, seps

    def test_additive_unpool_forward(self) -> None:
        """
        Verify that the additive unpooling forward kernel computes the output correctly.
        This test first obtains pooling (downsampled) data and unpool_ind via a validated pooling forward call.
        Then it calls the unpooling forward kernel and compares its output with the expected result computed on the CPU.

        Expected behavior:
          For each downsampled row i and each subbucket slot, if
            up_row = unpool_ind[i, slot] < up_N,
          then:
            up_add_feat[up_row, :] = down_feat[i, :] + residual_feat[up_row, :]
        """
        # -------------------------------------------------------------
        # 1. Prepare input data
        # -------------------------------------------------------------
        coords, seps = f3dbatch.bulk_to(self.build_batch(self.BATCH_SIZE), "cuda:0")
        subbuck_size: int = 2
        feat_dim: int = 256
        total_N, _ = coords.shape

        # Create a random input feature tensor in BF16.
        input_feat: torch.Tensor = torch.randn(total_N, feat_dim, device="cuda:0", dtype=torch.bfloat16)

        # -------------------------------------------------------------
        # 2. Setup device and compute kernel parameters
        # -------------------------------------------------------------
        dev_mgr = f3ddev.get_dev_manager()
        f3ddev.init_dev(0)
        dev_mgr.set_sync_stream(True)
        num_vox: int = 0xFFFF
        MaxNInstance: int = f3dbatch.max_inslen_from_seps(seps)

        # -------------------------------------------------------------
        # 3. Obtain pooling (downsampled) data and unpool_ind via pooling forward
        # -------------------------------------------------------------
        sb = f3dsubbuck.SubbuckBuffers.create_subbuck_buff(coords, input_feat, seps, subbuck_size)
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
            1,  # extra parameter if needed
            1  # reduction_op: O_SUM
        )
        # Use the pooling output as the downsampled feature.
        down_feat: torch.Tensor = sb.reduced_feat
        # Assume the upsampled space is the original space.
        up_N: int = total_N

        # -------------------------------------------------------------
        # 4. Prepare unpooling inputs and call the unpooling forward kernel
        # -------------------------------------------------------------
        # residual_feat: additional feature tensor at the upsampled level.
        residual_feat: torch.Tensor = torch.randn(up_N, feat_dim, device="cuda:0", dtype=torch.bfloat16)
        # Allocate output tensor for the unpooled result.
        up_add_feat: torch.Tensor = torch.empty(up_N, feat_dim, device="cuda:0", dtype=torch.bfloat16)

        pshattn.additive_unpool_fwd(
            residual_feat,
            down_feat,
            up_add_feat,
            sb.unpool_ind,
            subbuck_size
        )

        # -------------------------------------------------------------
        # 5. Compute expected output using tensor operations
        # -------------------------------------------------------------
        # For each down row i and each subbucket slot, if up_row = unpool_ind[i, slot] < up_N,
        # then expected[up_row, :] = down_feat[i, :] + residual_feat[up_row, :].
        # We convert unpool_ind to int64 for safe indexing.
        unpool_ind_test: torch.Tensor = sb.unpool_ind.to(torch.int64)
        expected: torch.Tensor = torch.zeros_like(up_add_feat)
        down_rows: int = down_feat.size(0)
        for i in range(down_rows):
            for sb_iter in range(subbuck_size):
                up_row = int(unpool_ind_test[i, sb_iter])
                if up_row < up_N:
                    expected[up_row, :] = down_feat[i, :] + residual_feat[up_row, :]

        # -------------------------------------------------------------
        # 6. Verify the unpooling forward result.
        # -------------------------------------------------------------
        self.assertTrue(
            torch.allclose(up_add_feat, expected, atol=1e-3),
            f"Unpool forward output mismatch:\nGot: {up_add_feat}\nExpected: {expected}"
        )

    def test_additive_unpool_backward(self) -> None:
        """
        Verify that the additive unpooling backward kernel computes gradients correctly.
        This test first obtains pooling (downsampled) data and unpool_ind via a validated pooling forward call.
        Then it simulates a gradient for the unpooling output (grad_up_added) and calls the unpooling backward kernel.

        Expected behavior:
          - For each up_row, grad_res[up_row, :] should equal grad_up_added[up_row, :].
          - For each down row i, grad_down[i, :] should equal the sum over subbucket slots of
            grad_up_added[ up_row, : ], where up_row = unpool_ind[i, slot] and up_row < up_N.
        """
        # -------------------------------------------------------------
        # 1. Prepare input data
        # -------------------------------------------------------------
        coords, seps = f3dbatch.bulk_to(self.build_batch(self.BATCH_SIZE), "cuda:0")
        subbuck_size: int = 2
        feat_dim: int = 256
        total_N, _ = coords.shape

        input_feat: torch.Tensor = torch.randn(total_N, feat_dim, device="cuda:0", dtype=torch.bfloat16)

        # -------------------------------------------------------------
        # 2. Setup device and compute kernel parameters
        # -------------------------------------------------------------
        dev_mgr = f3ddev.get_dev_manager()
        f3ddev.init_dev(0)
        dev_mgr.set_sync_stream(True)
        num_vox: int = 0xFFFF
        MaxNInstance: int = f3dbatch.max_inslen_from_seps(seps)

        # -------------------------------------------------------------
        # 3. Obtain pooling (downsampled) data and unpool_ind via pooling forward
        # -------------------------------------------------------------
        sb = f3dsubbuck.SubbuckBuffers.create_subbuck_buff(coords, input_feat, seps, subbuck_size)
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
            1,  # extra parameter if needed
            1  # reduction_op: O_SUM
        )
        down_feat: torch.Tensor = sb.reduced_feat
        up_N: int = total_N

        # -------------------------------------------------------------
        # 4. Prepare gradients for the unpooling output and allocate output gradient tensors
        # -------------------------------------------------------------
        grad_up_added: torch.Tensor = torch.randn(up_N, feat_dim, device="cuda:0", dtype=torch.bfloat16)
        grad_res: torch.Tensor = torch.empty(up_N, feat_dim, device="cuda:0", dtype=torch.bfloat16)
        grad_down: torch.Tensor = torch.empty(down_feat.size(0), feat_dim, device="cuda:0", dtype=torch.bfloat16)

        # -------------------------------------------------------------
        # 5. Call the unpooling backward kernel via its Python binding.
        # -------------------------------------------------------------
        pshattn.additive_unpool_bwd(
            grad_res,
            grad_down,
            grad_up_added,
            sb.unpool_ind,
            subbuck_size
        )

        # -------------------------------------------------------------
        # 6. Compute expected gradients using tensor operations
        # -------------------------------------------------------------
        # Expected grad_res: For each up_row, grad_res[up_row, :] should equal grad_up_added[up_row, :].
        expected_grad_res: torch.Tensor = grad_up_added.clone()

        # Expected grad_down: For each down row i, grad_down[i, :] should equal the sum over subbuck slots (if valid)
        # of grad_up_added[ up_row, : ], where up_row = unpool_ind[i, slot].
        expected_grad_down: torch.Tensor = torch.zeros_like(grad_down)
        unpool_ind_test: torch.Tensor = sb.unpool_ind.to(torch.int64)
        down_rows: int = down_feat.size(0)
        for i in range(down_rows):
            acc = torch.zeros(feat_dim, device="cuda:0", dtype=torch.float32)
            for sb_iter in range(subbuck_size):
                up_row: int = int(unpool_ind_test[i, sb_iter].item())
                if up_row < up_N:
                    acc += grad_up_added[up_row, :].to(torch.float32)
            expected_grad_down[i, :] = acc

        # -------------------------------------------------------------
        # 7. Verify the unpooling backward gradients.
        # -------------------------------------------------------------
        self.assertTrue(
            torch.allclose(grad_res, expected_grad_res, atol=1e-3),
            f"Unpool backward grad_res mismatch:\nGot: {grad_res}\nExpected: {expected_grad_res}"
        )
        self.assertTrue(
            torch.allclose(grad_down, expected_grad_down, atol=1e-3),
            f"Unpool backward grad_down mismatch:\nGot: {grad_down}\nExpected: {expected_grad_down}"
        )

    def test_unpooling(self) -> None:
        """
        Verify that the additive unpooling layer correctly propagates gradients.
        This test uses the pooling layer to obtain a SubbuckBuffers instance (sb) from the input.
        It then uses the AdditiveUnpoolLayer with the residual features (feats) as input.
        After computing a loss on the unpooled output, gradients should flow back to both
        the input coordinates and the features (feats). If both gradients are non-None, the test passes.
        """
        # -------------------------------------------------------------
        # 1. Prepare input data
        # -------------------------------------------------------------
        coords, seps = f3dbatch.bulk_to(self.build_batch(self.BATCH_SIZE), "cuda:0")
        coords.requires_grad_(True)
        coords.retain_grad()

        # Create a pooling layer and a simple MLP to compute features from coords.
        pl = f3dpool.InbucketPoolingLayer("mean", 2)
        mlp = nn.Linear(3, 256, dtype=torch.float16, device="cuda:0")

        # Compute features; these will serve as the residual features for unpooling.
        feats = mlp(coords).to(torch.bfloat16)
        feats.retain_grad()

        # -------------------------------------------------------------
        # 2. Perform pooling to obtain SubbuckBuffers (sb)
        # -------------------------------------------------------------
        red, sb = pl(coords, feats, seps)
        red.retain_grad()

        # -------------------------------------------------------------
        # 3. Perform unpooling using AdditiveUnpoolLayer, using feats as residual.
        # -------------------------------------------------------------
        unpool_layer = f3dunpool.AdditiveUnpoolLayer(2)
        up_add_feat = unpool_layer(feats, red, sb)

        # -------------------------------------------------------------
        # 4. Compute a loss on the unpooled output and backpropagate.
        # -------------------------------------------------------------
        loss = up_add_feat.mean()
        loss.backward()

        # -------------------------------------------------------------
        # 5. Verify gradients propagate to both coords and feats.
        # -------------------------------------------------------------
        self.assertIsNotNone(coords.grad, "Coordinates gradients are None, backward malfunction")
        self.assertIsNotNone(feats.grad, "Feature gradients are None, backward malfunction")
        self.assertIsNotNone(red.grad, "Downsampled feature gradients are None, backward malfunction")