#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
#  Authored by Liyan Chen (liyanc@cs.utexas.edu) on 2/9/25
#

from dataclasses import dataclass
import torch
import unittest

from torch.nn import functional as F
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
from flash3dxfmr.lib import pshattn
from flash3dxfmr.psh.dev_context import DEV_MGR
from flash3dxfmr.layers.buck_swin import BucketSwinAttention
from tests import test_lib

torch.manual_seed(2)
DEV_MGR.set_sync_stream(True)

@torch.no_grad()
def preprocess_delta_emu(dO, O, keepdims=False):
    # [B, L, H]
    Delta = (dO.to(torch.float32) * O.to(torch.float32)).sum(dim=-1, keepdims=keepdims)
    return Delta

@dataclass
class TestInputs:
    Q: torch.Tensor
    K: torch.Tensor
    V: torch.Tensor
    O: torch.Tensor
    dO: torch.Tensor

def zero_test_inputs(B, L, H, D) -> TestInputs:
    Q = torch.zeros(B, L, H, D, dtype=torch.bfloat16, device=0) 
    K = torch.zeros(B, L, H, D, dtype=torch.bfloat16, device=0)
    V = torch.zeros(B, L, H, D, dtype=torch.bfloat16, device=0) 
    O = torch.zeros(B, L, H, D, dtype=torch.bfloat16, device=0) 
    dO = torch.zeros(B, L, H, D, dtype=torch.bfloat16, device=0)
    return TestInputs(Q=Q, K=K, V=V, O=O, dO=dO)

def random_test_inputs(B, L, H, D) -> TestInputs:
    Q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0) * 10
    K = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0)
    V = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0) * 10
    O = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0) * 10
    dO = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0) * 10
    return TestInputs(Q=Q, K=K, V=V, O=O, dO=dO)


def buck_swin_bwd(Q, K, V, O, dO, lse, scope_buckets, bucket_size):
    B, L, H, D = Q.shape
    Delta = torch.zeros(B, L, H, dtype=torch.float32, device=Q.device)

    dQ = torch.zeros_like(Q)
    dK = torch.zeros_like(K)
    dV = torch.zeros_like(V)
    pshattn.buck_swin_bwd(Q, K, V, O, dO, lse, Delta, dQ, dK, dV, scope_buckets, bucket_size)
    return dQ, dK, dV, Delta


def measure_error_stats(pred, label):
    abs_diff = torch.abs(pred - label)
    max_err = abs_diff.max().item()
    avg_err = abs_diff.mean().item()

    return max_err, avg_err


def print_err_stats(stats):
    max_err, avg_err = stats
    print(f"Max error: {max_err}\nAvg error: {avg_err}")


class TestFlashAttnBwd(unittest.TestCase):
    def test_flat_delta_bwd_small(self):
        max_err_thres = 3.0
        B, L, H, D = 4, 8192, 4, 64
        buck_size = 256
        num_buck = L // buck_size

        Q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0) * 10
        K = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0)
        V = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0) * 10
        O = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0)
        dO = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0) * 10
        scopes = torch.arange(num_buck, dtype=torch.int32, device=0).to(torch.uint32)[None, ...].contiguous()

        _, gt_lse = test_lib.fa2_attn(Q, K, V, return_lse=True)

        Delta_emu = preprocess_delta_emu(dO, O).to(torch.bfloat16)
        dQ, dK, dV, Delta = buck_swin_bwd(
            Q, K, V, O, dO, gt_lse, scopes, buck_size)

        max_err, avg_err = measure_error_stats(Delta, Delta_emu)
        self.assertTrue(max_err < max_err_thres and avg_err < 0.2,
                        f"buckswin_delta mismatches naive_delta, max_err={max_err}, avg_err={avg_err}")

    def test_flat_delta_bwd_one(self):
        max_err_thres = 3.0
        B, L, H, D = 4, 8192, 4, 128
        buck_size = 256
        num_buck = L // buck_size

        Q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0) * 10
        K = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0)
        V = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0) * 10
        O = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0)
        dO = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0) * 10
        scopes = torch.arange(num_buck, dtype=torch.int32, device=0).to(torch.uint32)[None, ...].contiguous()

        _, gt_lse = test_lib.fa2_attn(Q, K, V, return_lse=True)

        Delta_emu = preprocess_delta_emu(dO, O).to(torch.bfloat16)
        dQ, dK, dV, Delta = buck_swin_bwd(
            Q, K, V, O, dO, gt_lse, scopes, buck_size)

        max_err, avg_err = measure_error_stats(Delta, Delta_emu)
        self.assertTrue(max_err < max_err_thres and avg_err < 0.2,
                        f"buckswin_delta mismatches naive_delta, max_err={max_err}, avg_err={avg_err}")


    def test_flat_delta_bwd_two(self):
        max_err_thres = 3.0
        B, L, H, D = 4, 8192, 4, 128
        buck_size = 256
        num_buck = L // buck_size

        Q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0) * 3
        K = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0)
        V = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0) * 10
        O = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0)
        dO = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0) * 10
        scopes = torch.arange(num_buck, dtype=torch.int32, device=0).to(torch.uint32)[None, ...].contiguous()

        _, gt_lse = test_lib.fa2_attn(Q, K, V, return_lse=True)

        Delta_emu = preprocess_delta_emu(dO, O).to(torch.bfloat16)
        dQ, dK, dV, Delta = buck_swin_bwd(
            Q, K, V, O, dO, gt_lse, scopes, buck_size)

        max_err, avg_err = measure_error_stats(Delta, Delta_emu)
        self.assertTrue(max_err < max_err_thres and avg_err < 0.2,
                        f"buckswin_delta mismatches naive_delta, max_err={max_err}, avg_err={avg_err}")


    def test_dq_dk_dv_bwd(self):
        max_err_thres = 4.0
        B, L, H, D = 2, 16384, 5, 128
        buck_size = 256
        num_buck = L // buck_size

        Q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0)
        K = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0)
        V = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0)
        dO = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0)
        scopes = torch.arange(num_buck, dtype=torch.int32, device=0).to(torch.uint32)[None, ...].contiguous()

        Q_diff, K_diff, V_diff = test_lib.set_require_grad(*test_lib.clone_tensors(Q, K, V))
        O_fwd = test_lib.fa2_pure_func(Q_diff, K_diff, V_diff)
        dQ_fa2, dK_fa2, dV_fa2 = torch.autograd.grad(
            outputs=O_fwd, inputs=[Q_diff, K_diff, V_diff], grad_outputs=dO, retain_graph=True)


        with torch.no_grad():
            Q_fa, K_fa, V_fa = test_lib.clone_tensors(Q, K, V)
            O_fa, LSE_fa = test_lib.fa2_attn(Q_fa, K_fa, V_fa, return_lse=True)
            delta_emu = preprocess_delta_emu(dO, O_fa)

            Q_bw, K_bw, V_bw = test_lib.clone_tensors(Q, K, V)
            O_bw, LSE_bw = test_lib.buck_swin(Q_bw, K_bw, V_bw, scopes, buck_size, return_lse=True)

        max_err, avg_err = measure_error_stats(O_bw, O_fa)
        self.assertTrue(max_err < 0.7 and avg_err < 0.04,
                        f"Buckswin mismatches FA2, max_err={max_err}, avg_err={avg_err}")

        max_err, avg_err = measure_error_stats(LSE_fa, LSE_bw)
        print(f"LSE margins: max_err {max_err}, avg_err {avg_err}")
        self.assertTrue(max_err < max_err_thres and avg_err < 0.2,
                        f"buckswin_lse mismatches fa_lse, max_err={max_err}, avg_err={avg_err}")

        dQ, dK, dV, Delta = buck_swin_bwd(
            Q, K, V, O_bw, dO, LSE_bw, scopes, buck_size)

        max_err, avg_err = measure_error_stats(Delta, delta_emu)
        print(f"Delta margins: max_err {max_err}, avg_err {avg_err}")
        self.assertTrue(max_err < max_err_thres and avg_err < 0.2,
                        f"buckswin_delta mismatches naive_delta, max_err={max_err}, avg_err={avg_err}")

        max_err, avg_err = measure_error_stats(dK, dK_fa2)
        self.assertTrue(max_err < max_err_thres and avg_err < 0.08,
                        f"buckswin_dK mismatches fa2_dK, max_err={max_err}, avg_err={avg_err}")

        max_err, avg_err = measure_error_stats(dV, dV_fa2)
        self.assertTrue(max_err < max_err_thres and avg_err < 0.08,
                        f"buckswin_dV mismatches fa2_dV, max_err={max_err}, avg_err={avg_err}")

        max_err, avg_err = measure_error_stats(dQ, dQ_fa2)
        self.assertTrue(max_err < max_err_thres and avg_err < 0.08,
                        f"buckswin_dQ mismatches fa2_dQ, max_err={max_err}, avg_err={avg_err}")
        

    def test_qkv_bwd_dim32(self):
        max_err_thres = 8.0
        B, L, H, D = 2, 4096, 16, 32
        buck_size = 256
        num_buck = L // buck_size

        Q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0)
        K = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0)
        V = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0)
        dO = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0)
        scopes = torch.arange(num_buck, dtype=torch.int32, device=0).to(torch.uint32)[None, ...].contiguous()

        Q_diff, K_diff, V_diff = test_lib.set_require_grad(*test_lib.clone_tensors(Q, K, V))
        O_fwd = test_lib.fa2_pure_func(Q_diff, K_diff, V_diff)
        dQ_fa2, dK_fa2, dV_fa2 = torch.autograd.grad(
            outputs=O_fwd, inputs=[Q_diff, K_diff, V_diff], grad_outputs=dO, retain_graph=True)


        with torch.no_grad():
            Q_fa, K_fa, V_fa = test_lib.clone_tensors(Q, K, V)
            O_fa, LSE_fa = test_lib.fa2_attn(Q_fa, K_fa, V_fa, return_lse=True)
            delta_emu = preprocess_delta_emu(dO, O_fa)

            Q_bw, K_bw, V_bw = test_lib.clone_tensors(Q, K, V)
            O_bw, LSE_bw = test_lib.buck_swin(Q_bw, K_bw, V_bw, scopes, buck_size, return_lse=True)


        max_err, avg_err = measure_error_stats(O_bw, O_fa)
        self.assertTrue(max_err < 0.7 and avg_err < 0.04,
                        f"Buckswin mismatches FA2, max_err={max_err}, avg_err={avg_err}")

        max_err, avg_err = measure_error_stats(LSE_fa, LSE_bw)
        print(f"LSE margins: max_err {max_err}, avg_err {avg_err}")
        self.assertTrue(max_err < max_err_thres and avg_err < 0.2,
                        f"buckswin_lse mismatches fa_lse, max_err={max_err}, avg_err={avg_err}")

        dQ, dK, dV, Delta = buck_swin_bwd(
            Q, K, V, O_fa, dO, LSE_fa, scopes, buck_size)

        max_err, avg_err = measure_error_stats(Delta, delta_emu)
        print(f"Delta margins: max_err {max_err}, avg_err {avg_err}")
        self.assertTrue(max_err < max_err_thres and avg_err < 0.2,
                        f"buckswin_delta mismatches naive_delta, max_err={max_err}, avg_err={avg_err}")

        max_err, avg_err = measure_error_stats(dK, dK_fa2)
        self.assertTrue(max_err < max_err_thres and avg_err < 0.2,
                        f"buckswin_dK mismatches fa2_dK, max_err={max_err}, avg_err={avg_err}")

        max_err, avg_err = measure_error_stats(dV, dV_fa2)
        self.assertTrue(max_err < max_err_thres and avg_err < 0.08,
                        f"buckswin_dV mismatches fa2_dV, max_err={max_err}, avg_err={avg_err}")

        max_err, avg_err = measure_error_stats(dQ, dQ_fa2)
        self.assertTrue(max_err < max_err_thres and avg_err < 0.2,
                        f"buckswin_dQ mismatches fa2_dQ, max_err={max_err}, avg_err={avg_err}")


    def test_qkv_bwd_dim64(self):
        max_err_thres = 7.0
        B, L, H, D = 2, 4096, 16, 64
        buck_size = 256
        num_buck = L // buck_size

        Q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0)
        K = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0)
        V = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0)
        dO = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0)
        scopes = torch.arange(num_buck, dtype=torch.int32, device=0).to(torch.uint32)[None, ...].contiguous()

        Q_diff, K_diff, V_diff = test_lib.set_require_grad(*test_lib.clone_tensors(Q, K, V))
        O_fwd = test_lib.fa2_pure_func(Q_diff, K_diff, V_diff)
        dQ_fa2, dK_fa2, dV_fa2 = torch.autograd.grad(
            outputs=O_fwd, inputs=[Q_diff, K_diff, V_diff], grad_outputs=dO, retain_graph=True)


        with torch.no_grad():
            Q_fa, K_fa, V_fa = test_lib.clone_tensors(Q, K, V)
            O_fa, LSE_fa = test_lib.fa2_attn(Q_fa, K_fa, V_fa, return_lse=True)
            delta_emu = preprocess_delta_emu(dO, O_fa)

            Q_bw, K_bw, V_bw = test_lib.clone_tensors(Q, K, V)
            O_bw, LSE_bw = test_lib.buck_swin(Q_bw, K_bw, V_bw, scopes, buck_size, return_lse=True)


        max_err, avg_err = measure_error_stats(O_bw, O_fa)
        self.assertTrue(max_err < 0.7 and avg_err < 0.04,
                        f"Buckswin mismatches FA2, max_err={max_err}, avg_err={avg_err}")

        max_err, avg_err = measure_error_stats(LSE_fa, LSE_bw)
        print(f"LSE margins: max_err {max_err}, avg_err {avg_err}")
        self.assertTrue(max_err < max_err_thres and avg_err < 0.2,
                        f"buckswin_lse mismatches fa_lse, max_err={max_err}, avg_err={avg_err}")

        dQ, dK, dV, Delta = buck_swin_bwd(
            Q, K, V, O_fa, dO, LSE_fa, scopes, buck_size)

        max_err, avg_err = measure_error_stats(Delta, delta_emu)
        print(f"Delta margins: max_err {max_err}, avg_err {avg_err}")
        self.assertTrue(max_err < max_err_thres and avg_err < 0.2,
                        f"buckswin_delta mismatches naive_delta, max_err={max_err}, avg_err={avg_err}")

        max_err, avg_err = measure_error_stats(dK, dK_fa2)
        print(f"dK_dim{D} margins: max_err {max_err}, avg_err {avg_err}")
        self.assertTrue(max_err < max_err_thres and avg_err < 0.4,
                        f"buckswin_dK mismatches fa2_dK, max_err={max_err}, avg_err={avg_err}")

        max_err, avg_err = measure_error_stats(dV, dV_fa2)
        print(f"dV_dim{D} margins: max_err {max_err}, avg_err {avg_err}")
        self.assertTrue(max_err < max_err_thres and avg_err < 0.08,
                        f"buckswin_dV mismatches fa2_dV, max_err={max_err}, avg_err={avg_err}")

        max_err, avg_err = measure_error_stats(dQ, dQ_fa2)
        self.assertTrue(max_err < max_err_thres and avg_err < 0.4,
                        f"buckswin_dQ mismatches fa2_dQ, max_err={max_err}, avg_err={avg_err}")


    def test_qkv_bwd_dim128(self):
        max_err_thres = 8.0
        B, L, H, D = 2, 4096, 16, 128
        buck_size = 256
        num_buck = L // buck_size

        Q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0)
        K = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0)
        V = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0)
        dO = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0)
        scopes = torch.arange(num_buck, dtype=torch.int32, device=0).to(torch.uint32)[None, ...].contiguous()

        Q_diff, K_diff, V_diff = test_lib.set_require_grad(*test_lib.clone_tensors(Q, K, V))
        O_fwd = test_lib.fa2_pure_func(Q_diff, K_diff, V_diff)
        dQ_fa2, dK_fa2, dV_fa2 = torch.autograd.grad(
            outputs=O_fwd, inputs=[Q_diff, K_diff, V_diff], grad_outputs=dO, retain_graph=True)


        with torch.no_grad():
            Q_fa, K_fa, V_fa = test_lib.clone_tensors(Q, K, V)
            O_fa, LSE_fa = test_lib.fa2_attn(Q_fa, K_fa, V_fa, return_lse=True)
            delta_emu = preprocess_delta_emu(dO, O_fa)

            Q_bw, K_bw, V_bw = test_lib.clone_tensors(Q, K, V)
            O_bw, LSE_bw = test_lib.buck_swin(Q_bw, K_bw, V_bw, scopes, buck_size, return_lse=True)


        max_err, avg_err = measure_error_stats(O_bw, O_fa)
        self.assertTrue(max_err < 0.7 and avg_err < 0.04,
                        f"Buckswin mismatches FA2, max_err={max_err}, avg_err={avg_err}")

        max_err, avg_err = measure_error_stats(LSE_fa, LSE_bw)
        print(f"LSE stats: max_err {max_err}, avg_err {avg_err}")
        self.assertTrue(max_err < max_err_thres and avg_err < 0.2,
                        f"buckswin_lse mismatches fa_lse, max_err={max_err}, avg_err={avg_err}")

        dQ, dK, dV, Delta = buck_swin_bwd(
            Q, K, V, O_fa, dO, LSE_fa, scopes, buck_size)

        max_err, avg_err = measure_error_stats(Delta, delta_emu)
        print(f"Delta stats: max_err {max_err}, avg_err {avg_err}")
        self.assertTrue(max_err < max_err_thres and avg_err < 0.2,
                        f"buckswin_delta mismatches naive_delta, max_err={max_err}, avg_err={avg_err}")

        max_err, avg_err = measure_error_stats(dK, dK_fa2)
        self.assertTrue(max_err < max_err_thres and avg_err < 0.2,
                        f"buckswin_dK mismatches fa2_dK, max_err={max_err}, avg_err={avg_err}")

        max_err, avg_err = measure_error_stats(dV, dV_fa2)
        self.assertTrue(max_err < max_err_thres and avg_err < 0.08,
                        f"buckswin_dV mismatches fa2_dV, max_err={max_err}, avg_err={avg_err}")

        max_err, avg_err = measure_error_stats(dQ, dQ_fa2)
        self.assertTrue(max_err < max_err_thres and avg_err < 0.2,
                        f"buckswin_dQ mismatches fa2_dQ, max_err={max_err}, avg_err={avg_err}")


    def test_layer(self):
        B, L, H, D = 2, 4096, 16, 128
        buck_size = 256
        num_buck = L // buck_size

        Q = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0)
        K = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0)
        V = torch.randn(B, L, H, D, dtype=torch.bfloat16, device=0)
        scopes = torch.arange(num_buck, dtype=torch.int32, device=0).to(torch.uint32)[None, ...].contiguous()
        bwa = BucketSwinAttention(buck_size)

        Q.requires_grad_(True).retain_grad()
        K.requires_grad_(True).retain_grad()
        V.requires_grad_(True).retain_grad()

        O = bwa(Q, K, V, scopes)
        l = O.mean()
        l.backward()

        # -------------------------------------------------------------
        # Verify gradients propagate to Q, K, V.
        # -------------------------------------------------------------
        self.assertIsNotNone(Q.grad, "Q gradients are None, backward malfunction")
        self.assertIsNotNone(K.grad, "K gradients are None, backward malfunction")
        self.assertIsNotNone(V.grad, "V gradients are None, backward malfunction")


    def test_v_bwd_uniform_attn(self):
        # With zero queries and keys, we expect the attentions to be uniform.

        max_err_thres = 4.0
        B, L, H, D = 1, 256, 1, 16
        buck_size = 256 
        num_buck = L // buck_size

        tin = zero_test_inputs(B, L, H, D)
        tin.dO += 1
        
        scopes = torch.arange(num_buck, dtype=torch.int32, device=0).to(torch.uint32)[None, ...].contiguous()

        # vjp_fn(dO)[I] = the derivative with respect to the Ith input, 
        # applied to the output variation d0.
        (out, vjp_fn) = torch.func.vjp(test_lib.naive_attn_fwd, tin.Q, tin.K, tin.V)

        Q_fa, K_fa, V_fa = test_lib.clone_tensors(tin.Q, tin.K, tin.V)
        O_fa, LSE_fa = test_lib.fa2_attn(Q_fa, K_fa, V_fa, return_lse=True)

        dQ, dK, dV, Delta = buck_swin_bwd(
            tin.Q, tin.K, tin.V, Q_fa, tin.dO, LSE_fa, scopes, buck_size)

        dV_emu = vjp_fn(tin.dO)[2]

        #print('computed dV')
        #print(dV)

        #print('expected dV')
        #print(dV_emu)

        max_err, avg_err = measure_error_stats(dV, dV_emu)
        self.assertTrue(max_err < max_err_thres and avg_err < 0.08,
                        f"buckswin_dQ mismatches naive_dQ, max_err={max_err}, avg_err={avg_err}")

       



if __name__ == "__main__":
    unittest.main()
