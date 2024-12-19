#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
#  Authored by Liyan Chen (liyanc@cs.utexas.edu) on 1/15/25
#

import torch
import unittest
from flash3dxfmr.lib import pshattn
from flash3dxfmr.psh import dev_context

dev_context.set_sync_stream(True)


def measure_error_stats(pred, label):
    abs_diff = torch.abs(pred - label)
    max_err = abs_diff.max().item()
    avg_err = abs_diff.mean().item()

    return max_err, avg_err


class TestGEMM(unittest.TestCase):
    def test_perfect_tile_bf16(self):
        # M=128 (2*64), N=256 (2*128), K=128 (8*16)
        # These dimensions are perfect multiples of the kernel's internal tiles.
        M, N, K = 128, 256, 128
        a = torch.randn(M, K, dtype=torch.bfloat16, device=0)
        b = torch.randn(K, N, dtype=torch.bfloat16, device=0)
        o = torch.zeros(M, N, dtype=torch.bfloat16, device=0)

        pshattn.gemm_sm_bf16(a, b, o)
        o_ref = a @ b

        max_err, avg_err = measure_error_stats(o_ref, o)

        # Use a reasonable tolerance for bf16 computations
        self.assertTrue(
            torch.allclose(o, o_ref, atol=1e-2, rtol=1e-2),
            f"Custom GEMM(bf16) with ideal dimensions failed, max_err={max_err}, avg_err={avg_err}"
        )

if __name__ == "__main__":
    unittest.main()
