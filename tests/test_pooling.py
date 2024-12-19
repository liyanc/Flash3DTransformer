#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
#  Authored by Liyan Chen (liyanc@cs.utexas.edu) on 11/19/24
#

import os
import glob

import tqdm
import torch
import unittest
import numpy as np

import flash3dxfmr.psh.subbuck as f3dsubbuck
import flash3dxfmr.psh.batching as f3dbatch
import flash3dxfmr.layers.pooling as f3dpool


KITTI_RT = os.getenv("KITTI_RT", None)
torch.manual_seed(2)

class TestSubbucketReduce(unittest.TestCase):
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

    def test_000_kitti_root_provided(self):
        self.assertIsNotNone(
            KITTI_RT,
            "KITTI_RT environment variable is not set! Provide KITTI_RT in environment variables.")

    def test_pooling(self):
        coords, seps = f3dbatch.bulk_to(self.build_batch(self.BATCH_SIZE), "cuda:0")
        coords.requires_grad_(True).retain_grad()
        pl = f3dpool.InbucketPoolingLayer("mean", 2)
        mlp = torch.nn.Linear(3, 256, dtype=torch.float16, device="cuda:0")
        feats = mlp(coords).to(torch.bfloat16)
        red, sb = pl(coords, feats, seps)
        loss = red.mean()

        loss.backward()
        self.assertIsNotNone(
            coords.grad, "Gradients to inputs are None, backward passes malfunction")


