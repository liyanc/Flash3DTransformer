#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
#  Authored by Liyan Chen (liyanc@cs.utexas.edu) on 7/18/24

import os
import glob

import math
import torch
import unittest
import numpy as np

import flash3dxfmr.psh.bucket as f3dbuck
import flash3dxfmr.psh.batching as f3dbatch
import flash3dxfmr.psh.arithmatic as f3darith
import flash3dxfmr.psh.dev_context as f3ddev
from flash3dxfmr.lib import pshattn
from flash3dxfmr.psh.psh_main import batch_bucket_scatter
from flash3dxfmr.layers.psh_ops import PSH3DCoordEmbedding


KITTI_RT = os.getenv("KITTI_RT", None)
torch.manual_seed(2)

class TestPSHMain(unittest.TestCase):
    BATCH_SIZE = 64

    def get_kitti_seq_iter(self):
        return glob.iglob(f'{KITTI_RT}/dataset/sequences/*/velodyne/*.bin')

    def build_batch(self, num_spls, down_sample_ratio=2, coord_dtype=torch.float16):
        spls = []
        assert KITTI_RT is not None, \
            "KITTI_RT environment variable is not set! Provide KITTI_RT in environment variables."
        for f, _ in zip(self.get_kitti_seq_iter(), range(num_spls)):
            npar = np.fromfile(f, dtype=np.float32).reshape(-1, 4)
            slen = npar.shape[0]
            spls.append(torch.from_numpy(npar[:slen // down_sample_ratio, :]))

        cb, seps = f3dbatch.collate_coords_flat_sep(spls, coord_dtype=coord_dtype, sep_dtype=torch.uint32)
        return cb, seps

    def test_batch_bucket_scatter_pad(self) -> None:
        """
        Unit test for batch_bucket_scatter_pad functionality.

        This test verifies that:
          - The scattered_coord tensor contains no NaN values.
          - The cumulative sum of bucket_cntr matches bucket_cumsum.
        """
        # -------------------------------------------------------------
        # 1. Prepare input data
        # -------------------------------------------------------------
        coords, seps = f3dbatch.bulk_to(
            self.build_batch(self.BATCH_SIZE, down_sample_ratio=1), "cuda:0")
        bucket_size = 512

        # -------------------------------------------------------------
        # 2. Setup device and compute kernel parameters
        # -------------------------------------------------------------
        dev_mgr = f3ddev.get_dev_manager()
        f3ddev.init_dev(0)
        dev_mgr.set_sync_stream(True)
        pad_to = f3darith.cdiv(coords.shape[0], bucket_size) * bucket_size
        coord_dtype = torch.float16
        self.assertTrue(coords.is_contiguous(), "Coords must be contiguous")
        self.assertEqual(coords.dtype, coord_dtype, f"Coords dtype must be {coord_dtype}")
        self.assertEqual(seps.device, coords.device, "Coords.device and seps.device mismatch")
        bbox_min, _ = torch.min(coords, dim=0)
        bbox_max, _ = torch.max(coords, dim=0)
        self.assertEqual(bbox_min.dtype, coord_dtype, "bbox_min dtype must match coord_dtype")
        Total_N, D = coords.shape
        dev = coords.device
        num_vox = 0xFFFF
        BLOCK_N = 1024
        N = f3dbatch.sep2sizes(seps).to(torch.float32).mean()
        B = seps.shape[0]
        MaxNInstance = int(f3dbatch.sep2sizes(seps.to(torch.int64)).max())
        num_buck = f3darith.round_next_two_power(int(N / bucket_size))
        bucket_divisor_heuristic = int(math.ceil(num_vox / num_buck))

        for hash_op in range(1, 5):
            # -------------------------------------------------------------
            # 3. Allocate buffers and prepare output tensor
            # -------------------------------------------------------------
            cnt_dtype = torch.uint32
            hash_dtype = torch.uint16
            bucket_id = torch.ones(Total_N, dtype=hash_dtype, device=dev)
            bucket_cntr = torch.zeros(B, num_buck, dtype=cnt_dtype, device=dev)
            bucket_offs = torch.empty(Total_N, dtype=cnt_dtype, device=dev)
            bucket_cumsum = torch.zeros_like(bucket_cntr)
            probe_offs = f3dbuck.generate_probe_offsets(dev, row_major=True)
            scattered_coord = torch.full((pad_to, D), float('nan'), dtype=coord_dtype, device=dev)

            # -------------------------------------------------------------
            # 4. Call the scatter_pad kernel dispatcher
            # -------------------------------------------------------------
            pshattn.batch_psh_scatter_pad_hash(
                coords, bucket_id, bucket_cntr, bucket_offs,
                seps, MaxNInstance, num_buck, bucket_divisor_heuristic,
                bucket_size, num_vox, bbox_min, bbox_max, probe_offs, bucket_cumsum, scattered_coord,
                0.0, hash_op
            )

            offset_maxcnt, _ = torch.max(bucket_cntr.to(torch.int32), dim=-1)
            offset_maxcnt = offset_maxcnt.to(torch.uint32)
            max_cntr = int(torch.max(bucket_cntr.to(torch.int32)))
            offset_check_table = torch.zeros((self.BATCH_SIZE, num_buck, max_cntr), dtype=torch.int8, device=dev)
            bucket_res = bucket_cntr.detach().clone()

            # -------------------------------------------------------------
            # 5. Verify output conditions
            # -------------------------------------------------------------
            self.assertFalse(
                torch.any(torch.isnan(scattered_coord)),
                "Scattered_coord contains NaN values"
            )
            cumsum_cntr = torch.cumsum(bucket_cntr, dim=1)
            self.assertTrue(
                torch.all(cumsum_cntr == bucket_cumsum.to(torch.int32)),
                f"Cumsum mismatch: expected {cumsum_cntr}, got {bucket_cumsum}"
            )

            pshattn.batch_bucket_sanitizer_cpu(
                bucket_id, bucket_cntr, bucket_offs, offset_check_table, bucket_res, seps,
                offset_maxcnt, self.BATCH_SIZE, num_buck)


    def test_batch_bucket_scatter_pad_60k_scene(self) -> None:
        """
        Unit test for batch_bucket_scatter_pad functionality.

        This test verifies that:
          - The scattered_coord tensor contains no NaN values.
          - The cumulative sum of bucket_cntr matches bucket_cumsum.
        """
        # -------------------------------------------------------------
        # 1. Prepare input data
        # -------------------------------------------------------------
        coords, seps = f3dbatch.bulk_to(
            self.build_batch(self.BATCH_SIZE, down_sample_ratio=2), "cuda:0")
        bucket_size = 512

        # -------------------------------------------------------------
        # 2. Setup device and compute kernel parameters
        # -------------------------------------------------------------
        dev_mgr = f3ddev.get_dev_manager()
        f3ddev.init_dev(0)
        dev_mgr.set_sync_stream(True)
        pad_to = f3darith.cdiv(coords.shape[0], bucket_size) * bucket_size
        coord_dtype = torch.float16
        self.assertTrue(coords.is_contiguous(), "Coords must be contiguous")
        self.assertEqual(coords.dtype, coord_dtype, f"Coords dtype must be {coord_dtype}")
        self.assertEqual(seps.device, coords.device, "Coords.device and seps.device mismatch")
        bbox_min, _ = torch.min(coords, dim=0)
        bbox_max, _ = torch.max(coords, dim=0)
        self.assertEqual(bbox_min.dtype, coord_dtype, "bbox_min dtype must match coord_dtype")
        Total_N, D = coords.shape
        dev = coords.device
        num_vox = 0xFFFF
        BLOCK_N = 1024
        N = f3dbatch.sep2sizes(seps).to(torch.float32).mean()
        B = seps.shape[0]
        MaxNInstance = int(f3dbatch.sep2sizes(seps.to(torch.int64)).max())
        num_buck = f3darith.round_next_two_power(int(N / bucket_size))
        bucket_divisor_heuristic = int(math.ceil(num_vox / num_buck))

        for hash_op in range(1, 5):
            # -------------------------------------------------------------
            # 3. Allocate buffers and prepare output tensor
            # -------------------------------------------------------------
            cnt_dtype = torch.uint32
            hash_dtype = torch.uint16
            bucket_id = torch.ones(Total_N, dtype=hash_dtype, device=dev)
            bucket_cntr = torch.zeros(B, num_buck, dtype=cnt_dtype, device=dev)
            bucket_offs = torch.empty(Total_N, dtype=cnt_dtype, device=dev)
            bucket_cumsum = torch.zeros_like(bucket_cntr)
            probe_offs = f3dbuck.generate_probe_offsets(dev, row_major=True)
            scattered_coord = torch.full((pad_to, D), float('nan'), dtype=coord_dtype, device=dev)

            # -------------------------------------------------------------
            # 4. Call the scatter_pad kernel dispatcher
            # -------------------------------------------------------------
            pshattn.batch_psh_scatter_pad_hash(
                coords, bucket_id, bucket_cntr, bucket_offs,
                seps, MaxNInstance, num_buck, bucket_divisor_heuristic,
                bucket_size, num_vox, bbox_min, bbox_max, probe_offs, bucket_cumsum, scattered_coord,
                0.0, hash_op
            )

            offset_maxcnt, _ = torch.max(bucket_cntr.to(torch.int32), dim=-1)
            offset_maxcnt = offset_maxcnt.to(torch.uint32)
            max_cntr = int(torch.max(bucket_cntr.to(torch.int32)))
            offset_check_table = torch.zeros((self.BATCH_SIZE, num_buck, max_cntr), dtype=torch.int8, device=dev)
            bucket_res = bucket_cntr.detach().clone()

            # -------------------------------------------------------------
            # 5. Verify output conditions
            # -------------------------------------------------------------
            self.assertFalse(
                torch.any(torch.isnan(scattered_coord)),
                "Scattered_coord contains NaN values"
            )
            cumsum_cntr = torch.cumsum(bucket_cntr, dim=1)
            self.assertTrue(
                torch.all(cumsum_cntr == bucket_cumsum.to(torch.int32)),
                f"Cumsum mismatch: expected {cumsum_cntr}, got {bucket_cumsum}"
            )

            pshattn.batch_bucket_sanitizer_cpu(
                bucket_id, bucket_cntr, bucket_offs, offset_check_table, bucket_res, seps,
                offset_maxcnt, self.BATCH_SIZE, num_buck)


    def test_batch_bucket_scatter_pad_30k_scene(self) -> None:
        """
        Unit test for batch_bucket_scatter_pad functionality.

        This test verifies that:
          - The scattered_coord tensor contains no NaN values.
          - The cumulative sum of bucket_cntr matches bucket_cumsum.
        """
        # -------------------------------------------------------------
        # 1. Prepare input data
        # -------------------------------------------------------------
        coords, seps = f3dbatch.bulk_to(
            self.build_batch(self.BATCH_SIZE, down_sample_ratio=4), "cuda:0")
        bucket_size = 512

        # -------------------------------------------------------------
        # 2. Setup device and compute kernel parameters
        # -------------------------------------------------------------
        dev_mgr = f3ddev.get_dev_manager()
        f3ddev.init_dev(0)
        dev_mgr.set_sync_stream(True)
        pad_to = f3darith.cdiv(coords.shape[0], bucket_size) * bucket_size
        coord_dtype = torch.float16
        self.assertTrue(coords.is_contiguous(), "Coords must be contiguous")
        self.assertEqual(coords.dtype, coord_dtype, f"Coords dtype must be {coord_dtype}")
        self.assertEqual(seps.device, coords.device, "Coords.device and seps.device mismatch")
        bbox_min, _ = torch.min(coords, dim=0)
        bbox_max, _ = torch.max(coords, dim=0)
        self.assertEqual(bbox_min.dtype, coord_dtype, "bbox_min dtype must match coord_dtype")
        Total_N, D = coords.shape
        dev = coords.device
        num_vox = 0xFFFF
        BLOCK_N = 1024
        N = f3dbatch.sep2sizes(seps).to(torch.float32).mean()
        B = seps.shape[0]
        MaxNInstance = int(f3dbatch.sep2sizes(seps.to(torch.int64)).max())
        num_buck = f3darith.round_next_two_power(int(N / bucket_size))
        bucket_divisor_heuristic = int(math.ceil(num_vox / num_buck))

        for hash_op in range(1, 5):
            # -------------------------------------------------------------
            # 3. Allocate buffers and prepare output tensor
            # -------------------------------------------------------------
            cnt_dtype = torch.uint32
            hash_dtype = torch.uint16
            bucket_id = torch.ones(Total_N, dtype=hash_dtype, device=dev)
            bucket_cntr = torch.zeros(B, num_buck, dtype=cnt_dtype, device=dev)
            bucket_offs = torch.empty(Total_N, dtype=cnt_dtype, device=dev)
            bucket_cumsum = torch.zeros_like(bucket_cntr)
            probe_offs = f3dbuck.generate_probe_offsets(dev, row_major=True)
            scattered_coord = torch.full((pad_to, D), float('nan'), dtype=coord_dtype, device=dev)

            # -------------------------------------------------------------
            # 4. Call the scatter_pad kernel dispatcher
            # -------------------------------------------------------------
            pshattn.batch_psh_scatter_pad_hash(
                coords, bucket_id, bucket_cntr, bucket_offs,
                seps, MaxNInstance, num_buck, bucket_divisor_heuristic,
                bucket_size, num_vox, bbox_min, bbox_max, probe_offs, bucket_cumsum, scattered_coord,
                0.0, hash_op
            )

            offset_maxcnt, _ = torch.max(bucket_cntr.to(torch.int32), dim=-1)
            offset_maxcnt = offset_maxcnt.to(torch.uint32)
            max_cntr = int(torch.max(bucket_cntr.to(torch.int32)))
            offset_check_table = torch.zeros((self.BATCH_SIZE, num_buck, max_cntr), dtype=torch.int8, device=dev)
            bucket_res = bucket_cntr.detach().clone()

            # -------------------------------------------------------------
            # 5. Verify output conditions
            # -------------------------------------------------------------
            self.assertFalse(
                torch.any(torch.isnan(scattered_coord)),
                "Scattered_coord contains NaN values"
            )
            cumsum_cntr = torch.cumsum(bucket_cntr, dim=1)
            self.assertTrue(
                torch.all(cumsum_cntr == bucket_cumsum.to(torch.int32)),
                f"Cumsum mismatch: expected {cumsum_cntr}, got {bucket_cumsum}"
            )

            pshattn.batch_bucket_sanitizer_cpu(
                bucket_id, bucket_cntr, bucket_offs, offset_check_table, bucket_res, seps,
                offset_maxcnt, self.BATCH_SIZE, num_buck)


    def test_batch_bucket_scatter_func_30k_scene(self) -> None:
        """
        Unit test for batch_bucket_scatter functionality.

        This test verifies that:
          - The scattered_coord tensor contains no NaN values.
          - The cumulative sum of bucket_cntr matches bucket_cumsum.

        It calls the batch_bucket_scatter function (which wraps the scatter_pad kernel)
        using various hash_op values.
        """
        # -------------------------------------------------------------
        # 1. Prepare input data
        # -------------------------------------------------------------
        coords, seps = f3dbatch.bulk_to(self.build_batch(self.BATCH_SIZE, down_sample_ratio=4), "cuda:0")
        bucket_size = 512

        # -------------------------------------------------------------
        # 2. Compute pad_to from coords and bucket_size
        # -------------------------------------------------------------
        pad_to = f3darith.cdiv(coords.shape[0], bucket_size) * bucket_size

        # -------------------------------------------------------------
        # 3. Test for different hash operations
        # -------------------------------------------------------------
        for hash_op in range(1, 5):
            scattered_coord, buffers = batch_bucket_scatter(coords, seps, bucket_size, pad_to, hash_op)

            # -------------------------------------------------------------
            # 4. Verify that scattered_coord contains no NaN values
            # -------------------------------------------------------------
            self.assertFalse(
                torch.any(torch.isnan(scattered_coord)),
                "Scattered_coord contains NaN values"
            )

            # -------------------------------------------------------------
            # 5. Verify that the cumulative sum of bucket_cntr matches bucket_cumsum
            # -------------------------------------------------------------
            cumsum_cntr = torch.cumsum(buffers.bucket_cntr, dim=1)
            self.assertTrue(
                torch.all(cumsum_cntr == buffers.bucket_cumsum.to(torch.int32)),
                f"Cumsum mismatch: expected {cumsum_cntr}, got {buffers.bucket_cumsum}"
            )

            # -------------------------------------------------------------
            # 6. Run the bucket sanitizer to validate the bucket scatter result
            # -------------------------------------------------------------
            N = f3dbatch.sep2sizes(seps).to(torch.float32).mean().item()
            num_buck = f3darith.round_next_two_power(int(N / bucket_size))
            dev = coords.device
            offset_maxcnt, _ = torch.max(buffers.bucket_cntr.to(torch.int32), dim=-1)
            offset_maxcnt = offset_maxcnt.to(torch.uint32)
            max_cntr = int(torch.max(buffers.bucket_cntr.to(torch.int32)))
            offset_check_table = torch.zeros((self.BATCH_SIZE, num_buck, max_cntr), dtype=torch.int8, device=dev)
            bucket_res = buffers.bucket_cntr.detach().clone()

            pshattn.batch_bucket_sanitizer_cpu(
                buffers.bucket_id, buffers.bucket_cntr, buffers.bucket_offs,
                offset_check_table, bucket_res, seps,
                offset_maxcnt, self.BATCH_SIZE, num_buck
            )

    def test_psh3d_coord_embedding_grad(self) -> None:
        """
        Unit test for PSH3DCoordEmbedding.

        This test verifies that after a backward pass, the linear layer's weight gradients
        are non-zero.
        """
        # -------------------------------------------------------------
        # 1. Prepare input data
        # -------------------------------------------------------------
        coords, seps = f3dbatch.bulk_to(self.build_batch(self.BATCH_SIZE), "cuda:0")
        emb_dim = 256
        bucket_size = 512
        hash_op = 2  # e.g., H_XORSUM_DIV

        # -------------------------------------------------------------
        # 2. Create the PSH3DCoordEmbedding layer and compute output
        # -------------------------------------------------------------
        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            model = PSH3DCoordEmbedding(emb_dim, bucket_size, bucket_size).to("cuda:0")
            output = model(coords, seps, hash_op)
            loss = output.mean()

        # -------------------------------------------------------------
        # 3. Backpropagate and verify gradients
        # -------------------------------------------------------------
        loss.backward()
        self.assertIsNotNone(model.lin.weight.grad, "Linear layer weight gradients are None")
        grad_norm = model.lin.weight.grad.norm().item()
        self.assertGreater(grad_norm, 0, f"Linear layer weight gradients are zero (norm={grad_norm})")
