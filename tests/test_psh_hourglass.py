#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
#  Authored by Liyan Chen (liyanc@cs.utexas.edu) on 1/18/25

import os
import glob

import math
import torch
import unittest
import numpy as np
from torch import nn
from functools import partial

import transformer_engine.pytorch as te

import flash3dxfmr.psh.batching as f3dbatch
from flash3dxfmr import layers
from flash3dxfmr.lib import pshattn
from flash3dxfmr.psh import bucket_scope
from flash3dxfmr.layers import psh_ops
from flash3dxfmr.layers import hourglass
from flash3dxfmr.layers import stage
from flash3dxfmr.layers import flash3d

KITTI_RT = os.getenv("KITTI_RT", None)
torch.manual_seed(2)


class TestPSHMain(unittest.TestCase):
    BATCH_SIZE = 4

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

    def test_layers(self):
        # -------------------------------------------------------------
        # 1. Prepare input data
        # -------------------------------------------------------------
        coords, seps = f3dbatch.bulk_to(self.build_batch(self.BATCH_SIZE), "cuda:0")
        bucket_size = 512
        scope_in_buckets = 8
        hash_op = 2  # e.g., H_XORSUM_DIV

        nl = te.LayerNorm
        act = nn.GELU

        ms = [
            layers.F3DLevelSpecs(
                encoder_specs=[
                    layers.XFMRSpecs(
                        channels=32,
                        hid_channels=128,
                        num_heads=2,
                        qkv_bias=True,
                        swin_plan=bucket_scope.SwinPlan("stride", bucket_size, scope_in_buckets, 1),
                        norm_layer=nl
                    ),
                    layers.XFMRSpecs(
                        channels=32,
                        hid_channels=128,
                        num_heads=2,
                        qkv_bias=True,
                        swin_plan=bucket_scope.SwinPlan("swin", bucket_size, scope_in_buckets, 2),
                        norm_layer=nl
                    )
                ],
                decoder_specs=[
                    layers.XFMRSpecs(
                        channels=64,
                        hid_channels=256,
                        num_heads=4,
                        qkv_bias=True,
                        swin_plan=bucket_scope.SwinPlan("swin", bucket_size, scope_in_buckets, 3),
                        norm_layer=nl
                    ),
                    layers.XFMRSpecs(
                        channels=64,
                        hid_channels=256,
                        num_heads=4,
                        qkv_bias=True,
                        swin_plan=bucket_scope.SwinPlan("swin", bucket_size, scope_in_buckets, 4),
                        norm_layer=nl
                    )
                ],
                reduction_op="mean",
                pool_align=bucket_scope.swinable_alignment(bucket_size, scope_in_buckets),
                pool_norm=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01),
                pool_act=nn.GELU,
                unpool_norm=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01),
                unpool_act=nn.GELU
            ),
            layers.F3DLevelSpecs(
                encoder_specs=[
                    layers.XFMRSpecs(
                        channels=64,
                        hid_channels=128,
                        num_heads=4,
                        qkv_bias=True,
                        swin_plan=bucket_scope.SwinPlan("swin", bucket_size, scope_in_buckets, 1),
                        norm_layer=nl
                    ),
                    layers.XFMRSpecs(
                        channels=64,
                        hid_channels=128,
                        num_heads=4,
                        qkv_bias=True,
                        swin_plan=bucket_scope.SwinPlan("swin", bucket_size, scope_in_buckets, 2),
                        norm_layer=nl
                    )
                ],
                decoder_specs=[
                    layers.XFMRSpecs(
                        channels=64,
                        hid_channels=256,
                        num_heads=4,
                        qkv_bias=True,
                        swin_plan=bucket_scope.SwinPlan("swin", bucket_size, scope_in_buckets, 3),
                        norm_layer=nl
                    ),
                    layers.XFMRSpecs(
                        channels=64,
                        hid_channels=256,
                        num_heads=4,
                        qkv_bias=True,
                        swin_plan=bucket_scope.SwinPlan("swin", bucket_size, scope_in_buckets, 4),
                        norm_layer=nl
                    )
                ],
                reduction_op="mean",
                pool_align=bucket_scope.swinable_alignment(bucket_size, scope_in_buckets),
                pool_norm=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01),
                pool_act=nn.GELU,
                unpool_norm=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01),
                unpool_act=nn.GELU
            )
        ]

        f3d = layers.Flash3D(ms, bucket_size, scope_in_buckets, hash_op).to("cuda")

        # -------------------------------------------------------------
        # 2. Create the PSH3DCoordEmbedding layer and compute output
        # -------------------------------------------------------------
        with torch.autocast(dtype=torch.bfloat16, device_type="cuda"):
            feat = f3d(coords, coords, seps)
            loss = feat.mean()

        loss.backward()
