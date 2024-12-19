#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
#  Authored by Liyan Chen (liyanc@cs.utexas.edu) on 1/15/25
#

import os
import glob

import tqdm
import torch
import unittest
import numpy as np

import flash3dxfmr.psh.subbuck as f3dsubbuck
import flash3dxfmr.psh.debug as f3ddebug
import flash3dxfmr.psh.batching as f3dbatch
import flash3dxfmr.psh.dev_context as f3ddev
from flash3dxfmr.lib import pshattn


KITTI_RT = os.getenv("KITTI_RT", None)
torch.manual_seed(2)

class TestSubbucketBackward(unittest.TestCase):
    BATCH_SIZE = 64

    def get_kitti_seq_iter(self):
        return glob.iglob(f'{KITTI_RT}/dataset/sequences/*/velodyne/*.bin')

    def build_batch(self, num_spls, down_sample_ratio=96, coord_dtype=torch.float16):
        spls = []
        for f, _ in zip(self.get_kitti_seq_iter(), range(num_spls)):
            npar = np.fromfile(f, dtype=np.float32).reshape(-1, 4)
            slen = npar.shape[0]
            spls.append(torch.from_numpy(npar[:slen // down_sample_ratio, :]))

        cb, seps = f3dbatch.collate_coords_flat_sep(spls, coord_dtype=coord_dtype, sep_dtype=torch.uint32)
        return cb, seps

    def test_000_kitti_root_provided(self):
        self.assertIsNotNone(
            KITTI_RT,
            "KITTI_RT environment variable is not set! Provide KITTI_RT in environment variables.")


    def test_subbucket_reduced_feat_sum_backward(self):
        """
        Verify that the backward gradient is correctly computed for O_SUM reduction.
        For each subbucket (each row in reduced_feat), the backward kernel should propagate
        the d_red_feat gradient to the corresponding input feature rows (as recorded in unpool_ind).
        This unit test leverages a validated forward computation to obtain unpool_ind.
        """
        # -------------------------------------------------------------
        # 1. Prepare input data
        # -------------------------------------------------------------
        coords, seps = f3dbatch.bulk_to(self.build_batch(self.BATCH_SIZE), "cuda:0")
        subbuck_size = 2
        feat_dim = 256
        total_N, _ = coords.shape

        # Create a random input feature tensor in BF16
        input_feat = torch.randn(total_N, feat_dim, device="cuda:0", dtype=torch.bfloat16)

        # -------------------------------------------------------------
        # 2. Setup device and compute kernel parameters
        # -------------------------------------------------------------
        dev_mgr = f3ddev.get_dev_manager()
        f3ddev.init_dev(0)
        dev_mgr.set_sync_stream(True)
        bsize = seps.shape[0]
        num_vox = 0xFFFF
        MaxNInstance = f3dbatch.max_inslen_from_seps(seps)

        # Create subbucket buffers using our helper
        sb = f3dsubbuck.SubbuckBuffers.create_subbuck_buff(coords, input_feat, seps, subbuck_size)

        # -------------------------------------------------------------
        # 3. Call the forward kernel with O_SUM to obtain unpool_ind and reduced features
        # -------------------------------------------------------------
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
            1,  # dummy variable to keep consistent with the call
            1  # mode parameter: O_SUM
        )

        # -------------------------------------------------------------
        # 4. Prepare d_red_feat (gradient from forward) and allocate d_input_feat
        # -------------------------------------------------------------
        # Create a random gradient tensor for the reduced features (or use ones for simplicity)
        d_red_feat = torch.randn(sb.reduced_feat.shape, device="cuda:0", dtype=torch.bfloat16) * 10
        # Initialize d_input_feat to zero (ensure that gradients for unassigned indices are zero)
        d_input_feat = torch.zeros_like(input_feat)

        # -------------------------------------------------------------
        # 5. Launch the backward kernel for O_SUM reduction
        # -------------------------------------------------------------
        # Assume pshattn.batch_subbuck_reduce_backward wraps our reduce_feat_backward_ker kernel.
        # The kernel parameters:
        #   - d_red_feat: [reduced_N, feat_dim]
        #   - input_feat: [total_N, feat_dim]
        #   - unpool_ind: [reduced_N, subbuck_size]
        #   - d_input_feat: [total_N, feat_dim]
        #   - total_N, reduced_N, feat_dim, subbuck_size, red_op, stride_featN, stride_unpoolN
        pshattn.batch_subbuck_reduce_backward(
            d_red_feat,
            input_feat,
            sb.unpool_ind,
            d_input_feat,
            subbuck_size,
            1,  # mode parameter: O_SUM
        )

        # -------------------------------------------------------------
        # 6. Compute expected gradient separately
        # -------------------------------------------------------------
        # For O_SUM, every valid input row (as recorded in unpool_ind) should receive the entire gradient from d_red_feat.
        expected_grad = torch.zeros_like(input_feat)
        reduced_N = sb.reduced_feat.shape[0]
        for r in range(reduced_N):
            for sb_iter in range(subbuck_size):
                # Get the input row from unpool_ind and check if it's valid (< total_N)
                input_row = int(sb.unpool_ind[r, sb_iter].item())
                if input_row < total_N:
                    expected_grad[input_row, :] = d_red_feat[r, :]

        # -------------------------------------------------------------
        # 7. Verify the correctness of the backward gradient
        # -------------------------------------------------------------
        self.assertTrue(torch.allclose(d_input_feat, expected_grad, atol=1e-3),
                        f"d_input_feat and expected gradient do not match.\n d_input_feat: {d_input_feat}\n expected: {expected_grad}")

    def test_subbucket_reduced_feat_mean_backward(self):
        """
        Verify that the backward gradient is correctly computed for O_MEAN reduction.
        For each subbucket (each row in reduced_feat), the backward kernel should propagate
        the d_red_feat gradient divided by the number of valid indices (from unpool_ind)
        to each valid input feature row.
        This unit test leverages a validated forward computation to obtain unpool_ind.
        """
        # -------------------------------------------------------------
        # 1. Prepare input data
        # -------------------------------------------------------------
        coords, seps = f3dbatch.bulk_to(self.build_batch(self.BATCH_SIZE), "cuda:0")
        subbuck_size = 2
        feat_dim = 256
        total_N, _ = coords.shape

        # Create a random input feature tensor in BF16
        input_feat = torch.randn(total_N, feat_dim, device="cuda:0", dtype=torch.bfloat16)

        # -------------------------------------------------------------
        # 2. Setup device and compute kernel parameters
        # -------------------------------------------------------------
        dev_mgr = f3ddev.get_dev_manager()
        f3ddev.init_dev(0)
        dev_mgr.set_sync_stream(True)
        bsize = seps.shape[0]
        num_vox = 0xFFFF
        MaxNInstance = f3dbatch.max_inslen_from_seps(seps)

        # Create subbucket buffers using our helper
        sb = f3dsubbuck.SubbuckBuffers.create_subbuck_buff(coords, input_feat, seps, subbuck_size)

        # -------------------------------------------------------------
        # 3. Call the forward kernel with O_MEAN to obtain unpool_ind and reduced features
        # -------------------------------------------------------------
        # Assume mode parameter 2 corresponds to O_MEAN
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
            1,
            0  # mode parameter: O_MEAN
        )

        # -------------------------------------------------------------
        # 4. Prepare d_red_feat (gradient from forward) and allocate d_input_feat
        # -------------------------------------------------------------
        d_red_feat = torch.randn(sb.reduced_feat.shape, device="cuda:0", dtype=torch.bfloat16)
        d_input_feat = torch.zeros_like(input_feat)

        # -------------------------------------------------------------
        # 5. Launch the backward kernel for O_MEAN reduction
        # -------------------------------------------------------------
        pshattn.batch_subbuck_reduce_backward(
            d_red_feat, input_feat, sb.unpool_ind, d_input_feat, subbuck_size, 0)

        # -------------------------------------------------------------
        # 6. Compute expected gradient separately
        # -------------------------------------------------------------
        # For each reduced row, every valid input should get d_red_feat[r, :] divided by the count of valid entries.
        expected_grad = torch.zeros_like(input_feat)
        reduced_N = sb.reduced_feat.shape[0]
        for r in range(reduced_N):
            valid_indices = []
            for sb_iter in range(subbuck_size):
                input_row = int(sb.unpool_ind[r, sb_iter].item())
                if input_row < total_N:
                    valid_indices.append(input_row)
            cnt = len(valid_indices)
            if cnt > 0:
                grad_val = d_red_feat[r, :] / float(cnt)
                for input_row in valid_indices:
                    expected_grad[input_row, :] = grad_val

        # -------------------------------------------------------------
        # 7. Verify the correctness of the backward gradient for O_MEAN
        # -------------------------------------------------------------
        self.assertTrue(torch.allclose(d_input_feat, expected_grad, atol=1e-3),
                        f"d_input_feat and expected gradient do not match for O_MEAN.\n"
                        f"d_input_feat: {d_input_feat}\n expected: {expected_grad}")

    def test_subbucket_reduced_feat_max_backward(self):
        """
        Verify that the backward gradient is correctly computed for O_MAX reduction.

        The gradient of a max operation is a "winner-takes-all" scenario. The upstream
        gradient (d_red_feat) is passed back *only* to the specific input feature element
        that was the maximum during the forward pass. All other inputs in that subbucket
        receive a gradient of zero for that feature dimension.
        """
        # -------------------------------------------------------------
        # 1. Prepare input data
        # -------------------------------------------------------------
        coords, seps = f3dbatch.bulk_to(self.build_batch(self.BATCH_SIZE), "cuda:0")
        subbuck_size = 4
        feat_dim = 128
        total_N, _ = coords.shape
        input_feat = (torch.randn(total_N, feat_dim, device="cuda:0") * 10).to(torch.bfloat16)

        # -------------------------------------------------------------
        # 2. Setup and forward pass (no changes here)
        # -------------------------------------------------------------
        dev_mgr = f3ddev.get_dev_manager()
        f3ddev.init_dev(0)
        dev_mgr.set_sync_stream(True)
        bsize = seps.shape[0]
        num_vox = 0xFFFF
        MaxNInstance = f3dbatch.max_inslen_from_seps(seps)
        sb = f3dsubbuck.SubbuckBuffers.create_subbuck_buff(coords, input_feat, seps, subbuck_size)

        pshattn.batch_subbuck_reduce(
            coords, input_feat, seps, MaxNInstance, subbuck_size, num_vox,
            sb.bbox_min, sb.bbox_max, sb.subbuck_id, sb.subbuck_off,
            sb.reduced_sep, sb.reduced_coord, sb.reduced_feat, sb.unpool_ind,
            1, 3  # mode parameter: O_MAX
        )

        # -------------------------------------------------------------
        # 4. Prepare gradients (no changes here)
        # -------------------------------------------------------------
        d_red_feat = (torch.randn(sb.reduced_feat.shape, device="cuda:0") * 5).to(torch.bfloat16)
        d_input_feat = torch.zeros_like(input_feat)

        # -------------------------------------------------------------
        # 5. Launch the backward kernel (no changes here)
        # -------------------------------------------------------------
        pshattn.batch_subbuck_reduce_backward(
            d_red_feat, input_feat, sb.unpool_ind, d_input_feat, subbuck_size, 3
        )

        # -------------------------------------------------------------
        # 6. Compute expected gradient using FAST, VECTORIZED PyTorch operations (uint32-safe)
        # -------------------------------------------------------------
        expected_grad = torch.zeros_like(input_feat)
        reduced_N, feat_dim = sb.reduced_feat.shape

        # Cast the uint32 index tensor to int64 ONCE for compatibility. This is the key fix.
        unpool_ind_i64 = sb.unpool_ind.to(torch.int64)

        # Step 6a: Create a mask for valid indices. Comparison is fine with uint32.
        valid_mask = unpool_ind_i64 < total_N

        # Step 6b: Gather all candidate features that contributed to the reduction.
        # Use the new int64 tensor for `where` and for indexing `input_feat`.
        safe_indices = torch.where(valid_mask, unpool_ind_i64, 0) # Now works with int64
        gathered_feats = input_feat[safe_indices] # Shape: [reduced_N, subbuck_size, feat_dim]

        # Set features from invalid indices to -inf so they are never chosen as the max.
        gathered_feats = torch.where(
            valid_mask.unsqueeze(-1), # Condition, shape [N, S, 1]
            gathered_feats,           # Value if True, shape [N, S, D]
            -torch.inf                # Value if False (broadcasts to [N, S, D])
        )

        # Step 6c: Find the index of the winning input within each subbucket.
        # torch.argmax returns int64 by default, which is what we need.
        # Shape: [reduced_N, feat_dim]
        winner_sb_indices = torch.argmax(gathered_feats, dim=1)

        # Step 6d: Use the winning subbucket indices to gather the global row indices.
        # We must use the int64 tensor here for torch.gather.
        # Shape: [reduced_N, feat_dim]
        winner_global_indices = torch.gather(unpool_ind_i64, 1, winner_sb_indices)

        # Step 6e: Scatter the gradients to their final destinations.
        # We only scatter gradients for reduced rows that had at least one valid input.
        any_valid_in_row = torch.any(valid_mask, dim=1, keepdim=True) # Shape: [reduced_N, 1]

        # Mask the source gradient tensor to ensure empty subbuckets don't cause issues
        src_grad = d_red_feat * any_valid_in_row

        # `scatter_add_` requires its index tensor (`winner_global_indices`) to be int64.
        # Our logic now ensures this is the case.
        expected_grad.scatter_add_(0, winner_global_indices, src_grad)

        # -------------------------------------------------------------
        # 7. Verify the correctness of the backward gradient
        # -------------------------------------------------------------
        self.assertTrue(torch.allclose(d_input_feat, expected_grad, atol=1e-2),
                        f"d_input_feat and expected gradient do not match for O_MAX.\n"
                        f"Max difference: {torch.max(torch.abs(d_input_feat - expected_grad))}")

    def test_subbucket_reduced_feat_min_backward(self):
        """
        Verify that the backward gradient is correctly computed for O_MIN reduction.

        This test uses a fast, vectorized PyTorch implementation as the ground truth.
        The gradient logic for MIN is "winner-takes-all," where the upstream gradient
        is passed back *only* to the input feature that was the minimum during the
        forward pass. It is robust to uint32 index types.
        """
        # -------------------------------------------------------------
        # 1. Prepare input data
        # -------------------------------------------------------------
        coords, seps = f3dbatch.bulk_to(self.build_batch(self.BATCH_SIZE), "cuda:0")
        subbuck_size = 4
        feat_dim = 128
        total_N, _ = coords.shape
        input_feat = (torch.randn(total_N, feat_dim, device="cuda:0") * 10).to(torch.bfloat16)

        # -------------------------------------------------------------
        # 2. Setup and forward pass
        # -------------------------------------------------------------
        dev_mgr = f3ddev.get_dev_manager()
        f3ddev.init_dev(0)
        dev_mgr.set_sync_stream(True)
        bsize = seps.shape[0]
        num_vox = 0xFFFF
        MaxNInstance = f3dbatch.max_inslen_from_seps(seps)
        sb = f3dsubbuck.SubbuckBuffers.create_subbuck_buff(coords, input_feat, seps, subbuck_size)

        # Call the forward kernel with O_MIN (mode 2)
        pshattn.batch_subbuck_reduce(
            coords, input_feat, seps, MaxNInstance, subbuck_size, num_vox,
            sb.bbox_min, sb.bbox_max, sb.subbuck_id, sb.subbuck_off,
            sb.reduced_sep, sb.reduced_coord, sb.reduced_feat, sb.unpool_ind,
            1, 2  # mode parameter: O_MIN
        )

        # -------------------------------------------------------------
        # 4. Prepare gradients
        # -------------------------------------------------------------
        d_red_feat = (torch.randn(sb.reduced_feat.shape, device="cuda:0") * 5).to(torch.bfloat16)
        d_input_feat = torch.zeros_like(input_feat)

        # -------------------------------------------------------------
        # 5. Launch the backward kernel for O_MIN
        # -------------------------------------------------------------
        pshattn.batch_subbuck_reduce_backward(
            d_red_feat, input_feat, sb.unpool_ind, d_input_feat, subbuck_size, 2 # mode: O_MIN
        )

        # -------------------------------------------------------------
        # 6. Compute expected gradient using FAST, VECTORIZED PyTorch operations
        # -------------------------------------------------------------
        expected_grad = torch.zeros_like(input_feat)
        reduced_N, feat_dim = sb.reduced_feat.shape

        unpool_ind_i64 = sb.unpool_ind.to(torch.int64)

        # --- THE FIX IS HERE: Use the int64 tensor for comparison ---
        valid_mask = unpool_ind_i64 < total_N
        # --- END FIX ---

        safe_indices = torch.where(valid_mask, unpool_ind_i64, 0)
        gathered_feats = input_feat[safe_indices]

        # Mask invalid entries with positive infinity so they are never chosen by argmin.
        gathered_feats = torch.where(
            valid_mask.unsqueeze(-1),
            gathered_feats,
            torch.inf
        )

        # Find the index of the minimum value.
        winner_sb_indices = torch.argmin(gathered_feats, dim=1)

        # The rest of the logic remains the same
        winner_global_indices = torch.gather(unpool_ind_i64, 1, winner_sb_indices)

        any_valid_in_row = torch.any(valid_mask, dim=1, keepdim=True)
        src_grad = d_red_feat * any_valid_in_row

        expected_grad.scatter_add_(0, winner_global_indices, src_grad)

        # -------------------------------------------------------------
        # 7. Verify the correctness of the backward gradient
        # -------------------------------------------------------------
        self.assertTrue(torch.allclose(d_input_feat, expected_grad, atol=1e-2),
                        f"d_input_feat and expected gradient do not match for O_MIN.\n"
                        f"Max difference: {torch.max(torch.abs(d_input_feat - expected_grad))}")