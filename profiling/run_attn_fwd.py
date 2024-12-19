# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Authored by Liyan Chen (liyanc@cs.utexas.edu) on 4/16/25
#

import math
import torch
import unittest

from torch.nn import functional as F
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
from flash3dxfmr.lib import pshattn
from flash3dxfmr.psh import debug

torch.manual_seed(2)


@torch.no_grad()
def buck_swin(Q, K, V, scope_buckets, buck_size, return_lse=False):
    B, T, H, D = Q.shape
    RT = torch.zeros_like(Q)
    LSE = torch.zeros(B, T, H, dtype=torch.float32, device=Q.device)
    pshattn.buck_swin_fwd(Q, K, V, RT, LSE, scope_buckets, buck_size)
    if return_lse:
        return RT, LSE
    else:
        return RT


def test_flat_buckswin_match_fa2_random_qk_dim16():
    """
    Control group test and validation test:
    Control test between Pytorch SDP and FlashAttention-2.
    Validation test between Bucket-Swin and SDP.
    Validation test between Bucket-Swin and Torch Emulations.

    :return: None
    """
    torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)
    B, L, H, D = 64, 4096, 4, 16
    buck_size = 512
    num_buck = L // buck_size

    # Stretch to extreme numerical ranges
    Q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0) * 16
    K = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0) * 16
    V = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0) * 12
    scopes = torch.arange(num_buck, dtype=torch.int32, device=0).to(torch.uint32)[None, ...].contiguous()

    O_bw = buck_swin(Q, K, V, scopes, buck_size)


if __name__ == '__main__':
    test_flat_buckswin_match_fa2_random_qk_dim16()
