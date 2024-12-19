#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
#  Authored by Liyan Chen (liyanc@cs.utexas.edu) on 7/18/24


import torch
from dataclasses import dataclass


@dataclass(slots=True)
class BuckSwinFwdBuffers:
    O: torch.Tensor
    LSE: torch.Tensor

    @staticmethod
    def create_buckswin_fwd_buff(Q: torch.Tensor):
        B, L, H, D = Q.shape
        LSE = torch.zeros(B, L, H, dtype=torch.float32, device=Q.device)
        O = torch.empty_like(Q)
        return BuckSwinFwdBuffers(O, LSE)


@dataclass(slots=True)
class BuckSwinBwdBuffers:
    O: torch.Tensor
    dQ: torch.Tensor
    dK: torch.Tensor
    dV: torch.Tensor
    Scope_buckets: torch.Tensor
    Delta: torch.Tensor
    buck_size: int

    @staticmethod
    def create_buckswin_buffers(O: torch.Tensor, Scope_buckets: torch.Tensor, buck_size: int):
        B, L, H, D = O.shape
        Delta = torch.zeros(B, L, H, dtype=torch.float32, device=O.device)
        dQ = torch.zeros_like(O)
        dK = torch.zeros_like(O)
        dV = torch.zeros_like(O)
        return BuckSwinBwdBuffers(O, dQ, dK, dV, Scope_buckets, Delta, buck_size)
