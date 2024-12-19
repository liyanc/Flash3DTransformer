#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
#  Authored by Liyan Chen (liyanc@cs.utexas.edu)


import torch
from torch import nn
import torch.cuda.nvtx as nvtx

from transformer_engine.pytorch import LayerNormMLP

from ..psh import dev_context as f3ddev


class MLP(nn.Module):
    NVTX_SCOPE = "MLP"
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.mlp = LayerNormMLP(
            hidden_size=in_dim,
            ffn_hidden_size=in_dim if hidden_dim is None else hidden_dim,
            activation="swiglu"
        )

    def forward(self, x):
        if f3ddev.get_nvtx():
            nvtx.range_push(self.NVTX_SCOPE)
        res = self.mlp(x)
        if f3ddev.get_nvtx():
            nvtx.range_pop()
        return res