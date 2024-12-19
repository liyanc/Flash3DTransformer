#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
#  Authored by Liyan Chen (liyanc@cs.utexas.edu) on 2/3/25
#

import math
import torch
import unittest

from torch.nn import functional as F
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
from flash3dxfmr.lib import pshattn
from flash3dxfmr.psh import debug
from tests import test_lib

torch.manual_seed(2)


def print_tile(tile, desc, flag):
    if flag:
        print(f"{desc}\n{str(tile)}")

@torch.no_grad()
def flash_attn_emu(Q, K, V, block_size=16, debug_print=False):
    """
    Emulate FlashAttention 2 (global attention, no causal mask) using block-wise key processing.

    For each query block, the keys are processed in blocks. For each key block,
    the block-wise partial sums (i.e. sum of exp(logits) and the weighted sum of V)
    are computed vectorized, then combined with the previously accumulated results using
    a log-sum-exp trick for numerical stability.

    Args:
        Q: bf16 tensor of shape [B, T, H, D]
        K: bf16 tensor of shape [B, T, H, D]
        V: bf16 tensor of shape [B, T, H, D]
        block_size: int, the block size to process queries and keys.

    Returns:
        RT: bf16 tensor of shape [B, T, H, D]
    """
    B, T, H, D = Q.shape
    scale = 1.0 / math.sqrt(D)

    # Reshape inputs to [B*H, T, D] and convert to float32 for accurate computation
    Q_ = Q.transpose(1, 2).contiguous().reshape(B * H, T, D).to(torch.float32)
    K_ = K.transpose(1, 2).contiguous().reshape(B * H, T, D).to(torch.float32)
    V_ = V.transpose(1, 2).contiguous().reshape(B * H, T, D).to(torch.float32)

    #print_tile(Q_[0, :16, :], "Q tile after reshape", debug_print)

    # Prepare output tensor of shape [B*H, T, D]
    output = torch.empty_like(Q_)  # float32

    # Process each batch*head independently
    for idx in range(Q_.shape[0]):
        q_seq = Q_[idx]  # [T, D]
        k_seq = K_[idx]  # [T, D]
        v_seq = V_[idx]  # [T, D]
        out_seq = torch.empty((T, D), device=Q_.device, dtype=torch.float32)

        # Process queries in blocks along the sequence dimension
        for i in range(0, T, block_size):
            # q_block: [b_q, D] where b_q <= block_size
            q_block = q_seq[i: i + block_size, :]
            b_q = q_block.shape[0]
            print_tile(q_block, f"Batch={idx},Qi={i}", debug_print and i < 2)

            # Initialize accumulators for the current query block:
            # acc: accumulated weighted sum of V, shape [b_q, D]
            # s: accumulated sum of exponentials, shape [b_q, 1]
            # l_val: current maximum logit for numerical stability, shape [b_q, 1]
            acc = torch.zeros(b_q, D, device=Q_.device, dtype=torch.float32)
            s = torch.zeros(b_q, 1, device=Q_.device, dtype=torch.float32)
            l_val = torch.full((b_q, 1), -float('inf'), device=Q_.device, dtype=torch.float32)

            # Process keys in blocks
            for j in range(0, T, block_size):
                # k_block and v_block: [b_k, D] where b_k <= block_size
                k_block = k_seq[j: j + block_size, :]
                v_block = v_seq[j: j + block_size, :]
                b_k = k_block.shape[0]

                # Compute logits for the current key block:
                # logits: [b_q, b_k] = q_block @ k_block^T * scale
                logits = (q_block @ k_block.T) * scale  # [b_q, b_k]
                # Compute maximum logit per query for the current block, shape [b_q, 1]
                L_block = logits.max(dim=1, keepdim=True)[0]
                # Compute exponentials normalized by L_block, shape [b_q, b_k]
                exp_logits = torch.exp(logits - L_block)
                # Sum of exponentials for the block, shape [b_q, 1]
                sum_exp = exp_logits.sum(dim=1, keepdim=True)
                # Compute the weighted sum of V over the key block, shape [b_q, D]
                weighted_v = exp_logits @ v_block

                # Combine current block's results with previous accumulators using the log-sum-exp trick:
                # new_max = max(l_val, L_block)
                new_max = torch.maximum(l_val, L_block)  # [b_q, 1]
                # Scale previous accumulators and new block results:
                factor_prev = torch.exp(l_val - new_max)  # [b_q, 1]
                factor_block = torch.exp(L_block - new_max)  # [b_q, 1]
                # Update accumulators:
                acc = acc * factor_prev + weighted_v * factor_block  # [b_q, D]
                s = s * factor_prev + sum_exp * factor_block  # [b_q, 1]
                l_val = new_max

            # Compute the output for the current query block: [b_q, D]
            out_seq[i: i + b_q, :] = acc / s

        output[idx] = out_seq

    # Reshape output back to [B, H, T, D] and then transpose to [B, T, H, D]
    RT = output.view(B, H, T, D).transpose(1, 2)
    # Cast output back to bf16 to match input dtype
    return RT.to(Q.dtype)


@torch.no_grad()
def flash_attn_emu_sequential(Q, K, V):
    """
    Emulate FlashAttention 2（

    :param Q, K, V: bf16[B, T, H, D]

    :return RT: bf16[B, T, H, D]
    """
    B, T, H, D = Q.shape
    scale = 1.0 / math.sqrt(D)

    Q_ = Q.transpose(1, 2).reshape(B * H, T, D).to(torch.float32)
    K_ = K.transpose(1, 2).reshape(B * H, T, D).to(torch.float32)
    V_ = V.transpose(1, 2).reshape(B * H, T, D).to(torch.float32)

    output = torch.empty_like(Q_, dtype=torch.float32)

    for idx in range(Q_.shape[0]):
        q_seq = Q_[idx]  # [T, D]
        k_seq = K_[idx]  # [T, D]
        v_seq = V_[idx]  # [T, D]
        out_seq = torch.empty((T, D), device=Q.device, dtype=torch.float32)

        for t in range(T):
            r = torch.zeros(D, device=Q.device, dtype=torch.float32)
            s = 0.0
            l_val = -float('inf')
            q_t = q_seq[t]  # [D]
            for j in range(T):
                z = (q_t @ k_seq[j]) * scale  # scalar
                new_l = max(l_val, z.item())
                factor = math.exp(l_val - new_l) if l_val != -float('inf') else 0.0
                r = r * factor + math.exp(z.item() - new_l) * v_seq[j]
                s = s * factor + math.exp(z.item() - new_l)
                l_val = new_l
            out_seq[t] = r / s
        output[idx] = out_seq

    # Recover [B, H, T, D], transpose to [B, T, H, D]
    RT = output.view(B, H, T, D).transpose(1, 2)
    # Downcast back to bfloat16
    return RT.to(torch.bfloat16)


@torch.no_grad()
def naive_attn_fwd(Q, K, V):
    B, T, H, D = Q.shape
    scale = 1.0 / (D ** 0.5)

    Q_ = Q.transpose(1, 2).reshape(B * H, T, D)
    K_ = K.transpose(1, 2).reshape(B * H, T, D)
    V_ = V.transpose(1, 2).reshape(B * H, T, D)

    logits = Q_.to(torch.float32) * scale @ K_.transpose(1, 2).to(torch.float32)  # [B*H, T, T]
    diff = logits - logits.max(dim=-1, keepdim=True)[0]
    attn = torch.softmax(diff, dim=-1)  # [B*H, T, T]
    output = attn @ V_.to(torch.float32)  # [B*H, T, D]

    RT = output.view(B, H, T, D).transpose(1, 2)
    return RT


@torch.no_grad()
def sdp_attn(Q, K, V):
    Q_ = Q.transpose(1, 2).to(torch.float32)
    K_ = K.transpose(1, 2).to(torch.float32)
    V_ = V.transpose(1, 2).to(torch.float32)
    out = F.scaled_dot_product_attention(Q_, K_, V_, dropout_p=0.0, is_causal=False)
    return out.transpose(1, 2)


@torch.no_grad()
def fa2_attn(Q, K, V, return_lse=False):
    B, T, H, D = Q.shape
    qkv = torch.stack([Q, K, V], dim=2)
    qkv = qkv.reshape(B * T, 3, H, D)

    cu_seqlens = torch.arange(0, (B+1) * T, step=T, dtype=torch.int32, device=Q.device)
    max_seqlen = T
    if return_lse:
        o, lse, _ = flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, False, return_attn_probs=True)
        return o.view(B, T, H, D), lse.view(B, H, T).permute(0, 2, 1)
    else:
        o = flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, False, return_attn_probs=False)
        return o.view(B, T, H, D)


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


def measure_error_stats(pred, label):
    abs_diff = torch.abs(pred - label)
    max_err = abs_diff.max().item()
    avg_err = abs_diff.mean().item()

    return max_err, avg_err


def print_err_stats(stats):
    max_err, avg_err = stats
    print(f"Max error: {max_err}\nAvg error: {avg_err}")


class TestFlashAttnFwd(unittest.TestCase):
    def test_softmax_invariant_under_subtraction(self):
        m = torch.randn(32, 128, dtype=torch.float32, device=0)
        mm = m.max(dim=-1, keepdim=True)[0]

        rsm = torch.softmax(m, dim=-1)
        dsm = torch.softmax(m - mm, dim=-1)
        self.assertTrue(torch.allclose(rsm, dsm), "Softmax differs after deducting row-max")

    def test_naive_sdp_fa2_emu_match(self):
        """
        Control group test to ensure reference attentions match with each other.
        Reference attentions include Naive, PyTorch SDP, FlashAttention-2, and Torch Emulations.

        :return: None
        """
        B, L, H, D = 2, 256, 4, 128
        Q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0) * 10
        K = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0)
        V = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0) * 10

        O_naive = test_lib.naive_attn_fwd(Q, K, V)
        O_sdp = sdp_attn(Q, K, V)
        O_emu = flash_attn_emu(Q, K, V)
        O_fa = fa2_attn(Q, K, V)
        #O_emu_seq = flash_attn_emu_sequential(Q, K, V)

        max_err, avg_err = measure_error_stats(O_sdp, O_fa)
        self.assertTrue(max_err < 0.3 and avg_err < 0.01,
                        f"FA2 mismatches SDP, max_err={max_err}, avg_err={avg_err}")
        max_err, avg_err = measure_error_stats(O_naive, O_sdp)
        self.assertTrue(max_err < 0.2 and avg_err < 0.01,
                        f"Naive mismatches SDP, max_err={max_err}, avg_err={avg_err}")
        max_err, avg_err = measure_error_stats(O_naive, O_fa)
        self.assertTrue(max_err < 0.3 and avg_err < 0.01,
                        f"Naive mismatches FA2, max_err={max_err}, avg_err={avg_err}")
        max_err, avg_err = measure_error_stats(O_emu, O_fa)
        self.assertTrue(max_err < 0.3 and avg_err < 0.01,
                        f"Emulated mismatches FA2, max_err={max_err}, avg_err={avg_err}")


    def test_emulation_match_naive(self):
        """
        Control group test to ensure emulated attentions match with FP32 Ground-Truth.

        :return: None
        """
        B, T, H, D = 2, 32, 4, 64
        Q = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=0) * 10
        K = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=0) * 10
        V = torch.randn(B, T, H, D, dtype=torch.bfloat16, device=0)

        O_emu = flash_attn_emu(Q, K, V).to(torch.bfloat16)
        O_gt = test_lib.naive_attn_fwd(Q, K, V).to(torch.bfloat16)
        max_err, avg_err = measure_error_stats(O_emu, O_gt)

        self.assertTrue(
            torch.allclose(O_gt, O_emu, rtol=3e-2, atol=3e-2),
            f"Emulated RT mismatches to GT RT\nMaxErr={max_err}, AvgErr={avg_err}")


    def test_flat_buckswin_match_fa2_artificial_qk(self):
        """
        Control group test and validation test:
        Control test between Pytorch SDP and FlashAttention-2.
        Validation test between Bucket-Swin and SDP.
        Validation test between Bucket-Swin and Torch Emulations.

        :return: None
        """
        torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)
        B, L, H, D = 3, 1280, 2, 128
        buck_size = 256
        num_buck = L // buck_size

        Q = debug.create_BLHD_dbg_tensor_repeatB(B, L, H, D, torch.bfloat16, dev=0)
        K = debug.create_BLHD_dbg_tensor_repeatB(B, L, H, D, torch.bfloat16, dev=0)
        V = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0) * 10
        scopes = torch.arange(num_buck, dtype=torch.int32, device=0).to(torch.uint32)[None, ...].contiguous()


        O_fa2 = fa2_attn(Q, K, V)
        O_sdp = sdp_attn(Q, K, V)
        O_bw = test_lib.buck_swin(Q, K, V, scopes, buck_size)

        max_err, avg_err = measure_error_stats(O_sdp, O_fa2)
        self.assertTrue(max_err < 0.3 and avg_err < 0.01,
                        f"SDP mismatches FA2, max_err={max_err}, avg_err={avg_err}")

        max_err, avg_err = measure_error_stats(O_bw, O_sdp)
        self.assertTrue(max_err < 0.3 and avg_err < 0.01,
                        f"Buckswin mismatches SDP, max_err={max_err}, avg_err={avg_err}")

        max_err, avg_err = measure_error_stats(O_bw, O_fa2)
        self.assertTrue(max_err < 0.3 and avg_err < 0.01,
                        f"Buckswin mismatches FA2, max_err={max_err}, avg_err={avg_err}")


    def test_flat_buckswin_match_fa2_random_qk(self):
        """
        Control group test and validation test:
        Control test between Pytorch SDP and FlashAttention-2.
        Validation test between Bucket-Swin and SDP.
        Validation test between Bucket-Swin and Torch Emulations.

        :return: None
        """
        torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)
        B, L, H, D = 16, 4096, 4, 16
        buck_size = 512
        num_buck = L // buck_size

        Q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0) * 2
        K = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0) * 8
        V = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0) * 16
        scopes = torch.arange(num_buck, dtype=torch.int32, device=0).to(torch.uint32)[None, ...].contiguous()

        O_fa2 = fa2_attn(Q, K, V)
        O_sdp = sdp_attn(Q, K, V)
        O_bw = test_lib.buck_swin(Q, K, V, scopes, buck_size)

        with self.subTest("Control test"):
            max_err, avg_err = measure_error_stats(O_sdp, O_fa2)
            self.assertTrue(max_err < 0.5 and avg_err < 0.02,
                            f"SDP mismatches FA2, max_err={max_err}, avg_err={avg_err}")

        with self.subTest("Buckswin vs FA2"):
            max_err, avg_err = measure_error_stats(O_bw, O_fa2)
            self.assertTrue(max_err < 0.6 and avg_err < 0.02,
                            f"Buckswin mismatches FA2, max_err={max_err}, avg_err={avg_err}")

        with self.subTest("Buckswin vs SDP"):
            max_err, avg_err = measure_error_stats(O_bw, O_sdp)
            self.assertTrue(max_err < 0.5 and avg_err < 0.02,
                            f"Buckswin mismatches SDP, max_err={max_err}, avg_err={avg_err}")



    def test_flat_buckswin_LSE_fa2_random_qk(self):
        """
        Control group test and validation test:
        Control test between Pytorch SDP and FlashAttention-2.
        Validation test between Bucket-Swin and SDP.
        Validation test between Bucket-Swin and Torch Emulations.

        :return: None
        """
        torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)
        B, L, H, D = 2, 4096, 4, 16
        buck_size = 512
        num_buck = L // buck_size

        Q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0)
        K = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0) * 10
        V = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0) * 10
        scopes = torch.arange(num_buck, dtype=torch.int32, device=0).to(torch.uint32)[None, ...].contiguous()

        O_fa2, L_fa2 = fa2_attn(Q, K, V, return_lse=True)
        O_bw, L_bw = test_lib.buck_swin(Q, K, V, scopes, buck_size, return_lse=True)
        non_positive = (L_bw <= 0.01).sum().item()
        total_elem = L_bw.numel()

        print(f"close to zero {non_positive}")
        test_lib.print_stange_lse(L_bw)
        max_err, avg_err = measure_error_stats(O_bw, O_fa2)

        with self.subTest("Buckswin Out vs FA2 Out"):
            self.assertTrue(max_err < 1.8 and avg_err < 0.1,
                            f"Buckswin mismatches FA2, max_err={max_err}, avg_err={avg_err}")

        with self.subTest("Num close to zeros"):
            self.assertTrue(non_positive == 0,
                            f"Log-Sum-Exp(LSE) values are Not strictly positive. #{non_positive}-numbers are non-positive. Non-positive ratio is {non_positive / total_elem * 100}%.")

        with self.subTest("Buckswin LSE vs FA2 LSE"):
            max_err, avg_err = measure_error_stats(L_bw, L_fa2)
            self.assertTrue(max_err < 0.3 and avg_err < 0.03,
                            f"Buckswin LSE mismatches FA2, max_err={max_err}, avg_err={avg_err}")

        with self.subTest("Buckswin LSE nnp"):
            self.assertTrue(non_positive == 0,
                            f"Log-Sum-Exp(LSE) values are Not strictly positive. #{non_positive}-numbers are non-positive. Non-positive ratio is {non_positive / total_elem * 100}%.")


    def test_flat_buckswin_invariant_under_random_bucketid(self):
        """
        Control group test and validation test:
        Control test between Pytorch SDP and FlashAttention-2.
        Validation test between Bucket-Swin and SDP.
        Validation test between Bucket-Swin and Torch Emulations.

        :return: None
        """
        torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)
        B, L, H, D = 8, 1280, 2, 128
        buck_size = 256
        num_buck = L // buck_size

        Q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0)
        K = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0)
        V = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0) * 10
        scopes = torch.arange(num_buck, dtype=torch.int32, device=0)
        scopes = scopes[torch.randperm(scopes.shape[0])].to(torch.uint32)[None, ...].contiguous()

        O_fa2 = fa2_attn(Q, K, V)
        O_sdp = sdp_attn(Q, K, V)
        O_bw = test_lib.buck_swin(Q, K, V, scopes, buck_size)

        max_err, avg_err = measure_error_stats(O_sdp, O_fa2)
        self.assertTrue(max_err < 0.3 and avg_err < 0.01,
                        f"SDP mismatches FA2, max_err={max_err}, avg_err={avg_err}")

        max_err, avg_err = measure_error_stats(O_bw, O_sdp)
        self.assertTrue(max_err < 0.3 and avg_err < 0.01,
                        f"Buckswin mismatches SDP, max_err={max_err}, avg_err={avg_err}")

        max_err, avg_err = measure_error_stats(O_bw, O_fa2)
        self.assertTrue(max_err < 0.3 and avg_err < 0.01,
                        f"Buckswin mismatches FA2, max_err={max_err}, avg_err={avg_err}")



if __name__ == "__main__":
    unittest.main(failfast=False)
