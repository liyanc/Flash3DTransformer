#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
#  Authored by Liyan Chen (liyanc@cs.utexas.edu) and Paul Vernaza (paul.vernaza@getcruise.com) on 2/9/25

import torch
from flash3dxfmr.lib import pshattn

from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func


def buck_swin(Q, K, V, scope_buckets, buck_size, return_lse=False):
    B, T, H, D = Q.shape
    RT = torch.zeros_like(Q)
    LSE = torch.zeros(B, T, H, dtype=torch.float32, device=Q.device)
    pshattn.buck_swin_fwd(Q, K, V, RT, LSE, scope_buckets, buck_size)
    if return_lse:
        return RT, LSE
    else:
        return RT

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

def buck_swin(Q, K, V, scope_buckets, buck_size, return_lse=False):
    B, L, H, D = Q.shape
    RT = torch.zeros_like(Q)
    LSE = torch.zeros(B, L, H, dtype=torch.float32, device=Q.device)
    pshattn.buck_swin_fwd(Q, K, V, RT, LSE, scope_buckets, buck_size)
    if return_lse:
        return RT, LSE
    else:
        return RT


def print_stange_lse(LSE):
    near_zero = (LSE <= 0.01)
    indices = near_zero.nonzero(as_tuple=False)
    print(indices.shape)
    print(indices.tolist())


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


def fa2_pure_func(Q, K, V):
    B, T, H, D = Q.shape
    qkv = torch.stack([Q, K, V], dim=2)
    qkv = qkv.reshape(B * T, 3, H, D)

    cu_seqlens = torch.arange(0, (B+1) * T, step=T, dtype=torch.int32, device=Q.device)
    max_seqlen = T

    o = flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, False, return_attn_probs=False)
    return o.view(B, T, H, D)


def clone_tensors(*args):
    return [t.clone() for t in args]

def set_require_grad(*args):
    return [t.detach().requires_grad_() for t in args]