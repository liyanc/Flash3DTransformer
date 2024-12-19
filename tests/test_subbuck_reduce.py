#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
#  Authored by Liyan Chen (liyanc@cs.utexas.edu) on 11/15/24
#

import os
import glob

import tqdm
import torch
import unittest
import numpy as np

import flash3dxfmr.psh.subbuck as f3dsubbuck
import flash3dxfmr.psh.batching as f3dbatch
import flash3dxfmr.psh.dev_context as f3ddev
from flash3dxfmr.lib import pshattn


KITTI_RT = os.getenv("KITTI_RT", None)
torch.manual_seed(2)

class TestSubbucketReduce(unittest.TestCase):
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

    def test_subbucket_reduced_feat_mean(self):
        """
        Verify that reduced_feat is correctly computed.
        For each subbucket (each row in reduced_feat), the kernel should compute
        the average of input_feat values for all valid indices recorded in unpool_ind.
        If a subbucket is partially filled, only the valid entries (i.e. indices < total_N)
        are used for averaging.
        """
        # -------------------------------------------------------------
        # 1. Prepare input data
        # -------------------------------------------------------------
        coords, seps = f3dbatch.bulk_to(self.build_batch(self.BATCH_SIZE), "cuda:0")
        subbuck_size = 3
        # Assume a fixed feature dimension, e.g. 256
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

        # Create subbucket buffers (sb) using our helper
        sb = f3dsubbuck.SubbuckBuffers.create_subbuck_buff(coords, input_feat, seps, subbuck_size)

        # -------------------------------------------------------------
        # 3. Call the subbucket reduce operation kernel directly
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
            1,
            0
        )
        # -------------------------------------------------------------
        # 4. Verify the correctness of reduced_feat
        # -------------------------------------------------------------
        # The reduced_feat tensor is expected to have shape [reduced_N, feat_dim]
        # and unpool_ind is global (of shape [reduced_N, subbuck_size]).
        #
        # For each subbucket (global row in reduced_feat), we gather the valid
        # indices from unpool_ind (i.e. those with value < total_N) and compute the average
        # of the corresponding rows in input_feat.
        #
        # We then compare this expected average with the value in reduced_feat.
        #
        reduced_feat_test = sb.reduced_feat.to(torch.float32)
        unpool_ind_test = sb.unpool_ind.to(torch.int64)
        input_feat_test = input_feat.to(torch.float32)

        # sb.reduced_sep is used to determine the number of subbuckets per batch.
        reduced_sep = sb.reduced_sep.to(torch.int64)

        prev_end = 0
        for i in range(bsize):
            end = int(seps[i])
            # The expected number of subbuckets for batch i is given by reduced_sep[i] - reduced_sep[i-1] (or reduced_sep[0] if i==0)
            if i == 0:
                expected_num_subbuckets = int(reduced_sep[0].item())
                start_sb = 0
            else:
                expected_num_subbuckets = int((reduced_sep[i] - reduced_sep[i - 1]).item())
                start_sb = int(reduced_sep[i - 1].item())

            # For each subbucket in this batch, verify the computed reduced feature
            for sb_local in range(expected_num_subbuckets):
                sb_global = start_sb + sb_local
                # Gather valid indices from unpool_ind for this subbucket row
                inds = unpool_ind_test[sb_global]  # shape: [subbuck_size]
                valid_mask = inds < total_N
                valid_inds = inds[valid_mask].tolist()
                if valid_inds:
                    feats = [input_feat_test[idx] for idx in valid_inds]
                    feats = torch.stack(feats, dim=0)  # shape: [n, feat_dim]
                    expected_avg = feats.mean(dim=0)
                else:
                    # Should not happen normally; use zero vector if no valid entries found.
                    expected_avg = torch.zeros(feat_dim, dtype=torch.float32)
                diff = torch.abs(reduced_feat_test[sb_global].to(0) - expected_avg.to(0))
                max_diff = diff.max().item()
                self.assertTrue(torch.all(diff < 0.2),
                                f"Batch {i}, subbucket {sb_local}: reduced_feat mismatch, max diff {max_diff}")

    def test_subbucket_reduced_feat_sum(self):
        """
        Verify that reduced_feat is correctly computed for O_SUM reduction.
        For each subbucket (each row in reduced_feat), the kernel should compute
        the sum of input_feat values for all valid indices recorded in unpool_ind.
        If a subbucket is partially filled, only the valid entries (indices < total_N)
        are used for summation.
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
        # 3. Call the subbucket reduce operation kernel with O_SUM (mode = 1)
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
            1,
            1  # mode parameter: O_SUM
        )

        # -------------------------------------------------------------
        # 4. Verify the correctness of reduced_feat for O_SUM
        # -------------------------------------------------------------
        reduced_feat_test = sb.reduced_feat.to(torch.float32)
        unpool_ind_test = sb.unpool_ind.to(torch.int64)
        input_feat_test = input_feat.to(torch.float32)
        reduced_sep = sb.reduced_sep.to(torch.int64)

        prev_end = 0
        for i in range(bsize):
            end = int(seps[i])
            if i == 0:
                expected_num_subbuckets = int(reduced_sep[0].item())
                start_sb = 0
            else:
                expected_num_subbuckets = int((reduced_sep[i] - reduced_sep[i - 1]).item())
                start_sb = int(reduced_sep[i - 1].item())

            for sb_local in range(expected_num_subbuckets):
                sb_global = start_sb + sb_local
                inds = unpool_ind_test[sb_global]  # shape: [subbuck_size]
                valid_mask = inds < total_N
                valid_inds = inds[valid_mask].tolist()
                if valid_inds:
                    feats = [input_feat_test[idx] for idx in valid_inds]
                    feats = torch.stack(feats, dim=0)
                    expected_sum = feats.sum(dim=0)
                else:
                    expected_sum = torch.zeros(feat_dim, dtype=torch.float32)
                diff = torch.abs(reduced_feat_test[sb_global].to(0) - expected_sum.to(0))
                max_diff = diff.max().item()
                self.assertTrue(torch.all(diff < 0.2),
                                f"Batch {i}, subbucket {sb_local}: reduced_feat (sum) mismatch, max diff {max_diff}")
            prev_end = end

    def test_subbucket_reduced_feat_min(self):
        """
        Verify that reduced_feat is correctly computed for O_MIN reduction.
        For each subbucket (each row in reduced_feat), the kernel should compute
        the minimum of input_feat values for all valid indices recorded in unpool_ind.
        If a subbucket is partially filled, only the valid entries (indices < total_N)
        are used for the minimum calculation.
        """
        # -------------------------------------------------------------
        # 1. Prepare input data
        # -------------------------------------------------------------
        coords, seps = f3dbatch.bulk_to(self.build_batch(self.BATCH_SIZE), "cuda:0")
        subbuck_size = 2
        feat_dim = 256
        total_N, _ = coords.shape

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

        sb = f3dsubbuck.SubbuckBuffers.create_subbuck_buff(coords, input_feat, seps, subbuck_size)

        # -------------------------------------------------------------
        # 3. Call the subbucket reduce operation kernel with O_MIN (mode = 2)
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
            1,
            2  # mode parameter: O_MIN
        )

        # -------------------------------------------------------------
        # 4. Verify the correctness of reduced_feat for O_MIN
        # -------------------------------------------------------------
        reduced_feat_test = sb.reduced_feat.to(torch.float32)
        unpool_ind_test = sb.unpool_ind.to(torch.int64)
        input_feat_test = input_feat.to(torch.float32)
        reduced_sep = sb.reduced_sep.to(torch.int64)

        prev_end = 0
        for i in range(bsize):
            end = int(seps[i])
            if i == 0:
                expected_num_subbuckets = int(reduced_sep[0].item())
                start_sb = 0
            else:
                expected_num_subbuckets = int((reduced_sep[i] - reduced_sep[i - 1]).item())
                start_sb = int(reduced_sep[i - 1].item())

            for sb_local in range(expected_num_subbuckets):
                sb_global = start_sb + sb_local
                inds = unpool_ind_test[sb_global]
                valid_mask = inds < total_N
                valid_inds = inds[valid_mask].tolist()
                if valid_inds:
                    feats = [input_feat_test[idx] for idx in valid_inds]
                    feats = torch.stack(feats, dim=0)
                    expected_min, _ = feats.min(dim=0)
                else:
                    expected_min = torch.full((feat_dim,), float('inf'), dtype=torch.float32)
                diff = torch.abs(reduced_feat_test[sb_global].to(0) - expected_min.to(0))
                max_diff = diff.max().item()
                self.assertTrue(torch.all(diff < 0.2),
                                f"Batch {i}, subbucket {sb_local}: reduced_feat (min) mismatch, max diff {max_diff}")
            prev_end = end

    def test_subbucket_reduced_feat_max(self):
        """
        Verify that reduced_feat is correctly computed for O_MAX reduction.
        For each subbucket (each row in reduced_feat), the kernel should compute
        the maximum of input_feat values for all valid indices recorded in unpool_ind.
        If a subbucket is partially filled, only the valid entries (indices < total_N)
        are used for the maximum calculation.
        """
        # -------------------------------------------------------------
        # 1. Prepare input data
        # -------------------------------------------------------------
        coords, seps = f3dbatch.bulk_to(self.build_batch(self.BATCH_SIZE), "cuda:0")
        subbuck_size = 2
        feat_dim = 256
        total_N, _ = coords.shape

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

        sb = f3dsubbuck.SubbuckBuffers.create_subbuck_buff(coords, input_feat, seps, subbuck_size)

        # -------------------------------------------------------------
        # 3. Call the subbucket reduce operation kernel with O_MAX (mode = 3)
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
            1,
            3  # mode parameter: O_MAX
        )

        # -------------------------------------------------------------
        # 4. Verify the correctness of reduced_feat for O_MAX
        # -------------------------------------------------------------
        reduced_feat_test = sb.reduced_feat.to(torch.float32)
        unpool_ind_test = sb.unpool_ind.to(torch.int64)
        input_feat_test = input_feat.to(torch.float32)
        reduced_sep = sb.reduced_sep.to(torch.int64)

        prev_end = 0
        for i in range(bsize):
            end = int(seps[i])
            if i == 0:
                expected_num_subbuckets = int(reduced_sep[0].item())
                start_sb = 0
            else:
                expected_num_subbuckets = int((reduced_sep[i] - reduced_sep[i - 1]).item())
                start_sb = int(reduced_sep[i - 1].item())

            for sb_local in range(expected_num_subbuckets):
                sb_global = start_sb + sb_local
                inds = unpool_ind_test[sb_global]
                valid_mask = inds < total_N
                valid_inds = inds[valid_mask].tolist()
                if valid_inds:
                    feats = [input_feat_test[idx] for idx in valid_inds]
                    feats = torch.stack(feats, dim=0)
                    expected_max, _ = feats.max(dim=0)
                else:
                    expected_max = torch.full((feat_dim,), -float('inf'), dtype=torch.float32)
                diff = torch.abs(reduced_feat_test[sb_global].to(0) - expected_max.to(0))
                max_diff = diff.max().item()
                self.assertTrue(torch.all(diff < 0.2),
                                f"Batch {i}, subbucket {sb_local}: reduced_feat (max) mismatch, max diff {max_diff}")
            prev_end = end


    def test_subbucket_id_in_each_interval(self):
        """
        Ensure subbucket_id in each interval (defined by the reduced_batch_sep)
        has no duplicates and ranges from 0 to max.
        """

        # -------------------------------------------------------------
        # 1. Prepare input data
        # -------------------------------------------------------------
        coords, seps = f3dbatch.bulk_to(self.build_batch(self.BATCH_SIZE), "cuda:0")

        # Subbucket size, currently fixed at 2
        subbuck_size = 2

        # -------------------------------------------------------------
        # 2. (Reorganized) batch_subbuck logic
        # -------------------------------------------------------------
        dev_mgr = f3ddev.get_dev_manager()
        f3ddev.init_dev(0)  # Initialize device index 0
        dev_mgr.set_sync_stream(True)  # Sync stream setting for demonstration

        coord_dtype = torch.float16
        # Basic checks
        assert seps.device == coords.device, (
            f"Coords.device={coords.device} and seps.device={seps.device} mismatch"
        )

        # Some parameters used by batch_subbuck
        num_vox = 0xFFFF
        total_N, _ = coords.shape
        B = seps.shape[0]
        MaxNInstance = f3dbatch.max_inslen_from_seps(seps)

        # Create a dummy feature tensor for demonstration
        input_feat = torch.randn(
            total_N, 256, dtype=torch.bfloat16, device=coords.device
        )

        # Create subbucket buffers
        sb = f3dsubbuck.SubbuckBuffers.create_subbuck_buff(coords, input_feat, seps, subbuck_size)

        # Call the subbucket reduce operation
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
            0
        )

        # -------------------------------------------------------------
        # 3. Cast subbuck_id and subbuck_off to int64 before testing
        # -------------------------------------------------------------
        subbuck_id_test = sb.subbuck_id.to(torch.int64)
        subbuck_off_test = sb.subbuck_off.to(torch.int64)
        reduced_sep_test = sb.reduced_sep.to(torch.int64)
        unpool_ind_test = sb.unpool_ind.to(torch.int64)
        reduced_coord_test = sb.reduced_coord.to(torch.float32)

        # -------------------------------------------------------------
        # 4. Validate subbucket_id and subbuck_off within each interval
        # -------------------------------------------------------------
        prev_end = 0
        prev_red = 0  # accumulated sub-bucket count for previous batches
        for i in range(B):
            # The original batch segment in coords/features
            end = int(seps[i])
            subrange_size = end - prev_end

            # The number of unique subbucket IDs
            # in this batch is determined by sb.reduced_sep (the "reduced" layout).
            # If i=0, use reduced_sep[0], else use reduced_sep[i] - reduced_sep[i-1].
            # How many sub-buckets are expected for this batch?
            if i == 0:
                expected_num_subbuckets = reduced_sep_test[0].item()
                start_sb_id = 0
            else:
                expected_num_subbuckets = (reduced_sep_test[i] - reduced_sep_test[i - 1]).item()
                start_sb_id = reduced_sep_test[i - 1].item()

            # Slice subbucket_id / subbuck_off in the original layout
            subrange_ids = subbuck_id_test[prev_end:end]
            subrange_off = subbuck_off_test[prev_end:end]

            # 4.1 Check subbucket_id
            unique_ids = torch.unique(subrange_ids)
            # The count of unique subbucket_id in this batch should match what's in reduced_sep
            self.assertEqual(
                len(unique_ids),
                expected_num_subbuckets,
                f"Batch {i} mismatch: expected {expected_num_subbuckets} unique IDs, got {len(unique_ids)}"
            )

            # subrange_ids itself can still be from 0..(subrange_size-1),
            # but typically the guaranteed continuity is up to 'expected_num_subbuckets - 1'.
            # If we also want to assert subrange_ids.min() == 0 / subrange_ids.max() == subrange_size-1,
            # we can do so if that is indeed the contract:
            self.assertEqual(
                subrange_ids.min().item(),
                0,
                f"Batch {i} subbucket_id does not start from 0"
            )
            # Here we check it ends at (subrange_size - 1), or you may want
            # to check it ends at (expected_num_subbuckets - 1), depending on the exact contract:
            self.assertEqual(
                subrange_ids.max().item(),
                expected_num_subbuckets - 1,
                f"Batch {i} subbucket_id does not go up to (expected_num_subbuckets - 1)"
            )

            # 4.2 Encode (subbucket_id, subbuck_off) into 1D code
            #     code = subbucket_id * subbuck_size + subbuck_off
            #     The maximum code range = expected_num_subbuckets * subbuck_size
            max_code_count = expected_num_subbuckets * subbuck_size

            # offset must not exceed subbuck_size, otherwise it's invalid
            # (except the last subbucket might not be fully filled, but still <= subbuck_size)
            self.assertTrue(
                torch.all(subrange_off < subbuck_size),
                f"Batch {i}: some subbuck_off >= subbuck_size"
            )

            code = subrange_ids * subbuck_size + subrange_off
            code_counts = torch.bincount(code, minlength=max_code_count)

            # 4.3 Check code_counts distribution
            #     Let counts[i] = number of points in subbucket i
            #     Then code_counts[i*subbuck_size : i*subbuck_size + counts[i]] should be == 1,
            #     while the remainder in that subrange = 0.
            counts = torch.bincount(subrange_ids, minlength=expected_num_subbuckets)

            for sb_id in range(expected_num_subbuckets):
                c_i = counts[sb_id].item()  # number of points in subbucket sb_id

                # For subbucket sb_id, check [start_offset, start_offset + c_i) all == 1
                # and [start_offset + c_i, start_offset + subbuck_size) all == 0
                start_offset = sb_id * subbuck_size

                # slice1: indices that should all be 1
                slice1 = code_counts[start_offset: start_offset + c_i]
                if not torch.all(slice1 == 1):
                    raise AssertionError(
                        f"Batch {i}: subbucket_id={sb_id} does not have consecutive offsets from 0..{c_i - 1}"
                    )

                # slice2: indices that should be 0 (remaining part)
                slice2 = code_counts[start_offset + c_i: start_offset + subbuck_size]
                if not torch.all(slice2 == 0):
                    raise AssertionError(
                        f"Batch {i}: subbucket_id={sb_id} has offset out of range or duplicated (some > c_i-1)"
                    )

            # 4.4 unpool_ind check: unpool_ind[ row, col ] == global_index
            #     where row = start_sb_id + subbucket_id, col = subbuck_off,
            #     and global_index = (prev_end + local_idx).
            subrange_size = end - prev_end
            local_indices = torch.arange(subrange_size, device=coords.device, dtype=torch.int64)
            global_indices = local_indices + prev_end
            row_indices = start_sb_id + subrange_ids
            col_indices = subrange_off

            # advanced indexing: unpool_vals[i] = unpool_ind[row_indices[i], col_indices[i]]
            unpool_vals = unpool_ind_test[row_indices, col_indices]

            # we expect unpool_vals == global_indices
            if not torch.all(unpool_vals == global_indices):
                # find first mismatch for debugging
                mismatch_idx = (unpool_vals != global_indices).nonzero()[0].item()
                raise AssertionError(
                    f"Batch {i}: unpool_ind mismatch at subrange local index={mismatch_idx}, "
                    f"expected={global_indices[mismatch_idx].item()}, got={unpool_vals[mismatch_idx].item()}"
                )

            # -------------------------------------------------------------
            # 5. Now check reduced_coord correctness: average of original coords
            #    For each sub-bucket row j in [start_sb_id, start_sb_id+expected_num_subbuckets)
            # -------------------------------------------------------------
            for sb_id_local in range(expected_num_subbuckets):
                sb_id_global = start_sb_id + sb_id_local
                # Gather local indices for which subbucket_id == sb_id_local
                mask = (subrange_ids == sb_id_local)
                local_idxs = mask.nonzero().squeeze(-1)
                group_size = local_idxs.shape[0]

                # If group_size == 0, it should not happen if subbucket_id range is correct
                if group_size == 0:
                    raise AssertionError(f"Batch {i}, subbucket_id={sb_id_local} has no points, unexpected")

                # Gather original coords from the global indices
                # global = prev_end + local_idx
                these_global_idxs = local_idxs + prev_end
                # coords shape [N, D], gather them
                these_coords = coords[these_global_idxs, :].to(torch.float32)

                # Compute average across dimension 0
                avg_coords = these_coords.mean(dim=0)  # shape = [D]

                # Compare with reduced_coord[sb_id_global, :]
                rc = reduced_coord_test[sb_id_global, :]
                # We'll allow a small numerical tolerance since half precision can introduce rounding
                diff = torch.abs(rc - avg_coords)
                if not torch.all(diff < 0.2):
                    raise AssertionError(
                        f"Batch {i}, subbucket_id={sb_id_local} mismatch in reduced_coord. "
                        f"Expected {avg_coords}, got {rc}, diff={diff}"
                    )

            prev_end = end
            prev_red += expected_num_subbuckets


if __name__ == "__main__":
    unittest.main()
