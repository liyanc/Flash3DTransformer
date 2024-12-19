#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
#  Authored by Liyan Chen (liyanc@cs.utexas.edu) on 10/25/24


import torch


def create_rowcol_dbg_tensor(rows, cols, dtype, dev):
    row_numbers = torch.arange(rows, dtype=dtype, device=dev).view(-1, 1)
    col_numbers = torch.arange(cols, dtype=dtype, device=dev).view(1, -1)

    res = torch.empty((rows, cols), dtype=dtype, device=dev)

    res[:, 0::2] = row_numbers
    res[:, 1::2] = col_numbers[:, 1::2]

    return res


def create_LHD_dbg_tensor(L, H, D, dtype, dev):
    l_numbers = torch.arange(L, dtype=dtype, device=dev).view(L, 1, 1).expand(-1, H, D)
    h_numbers = torch.arange(H, dtype=dtype, device=dev).view(1, H, 1).expand(L, -1, D)
    d_numbers = torch.arange(D, dtype=dtype, device=dev).view(1, 1, D).expand(L, H, -1)

    # Create the empty tensor to store the debug tensor
    res = torch.empty((L, H, D), dtype=dtype, device=dev)

    # Fill the debug tensor with coordinates
    res[:, :, 0::3] = l_numbers[:, :, 0::3]  # Every third column starting from 0 stores L coordinates
    res[:, :, 1::3] = h_numbers[:, :, 1::3]  # Every third column starting from 1 stores H coordinates
    res[:, :, 2::3] = d_numbers[:, :, 2::3]  # Every third column starting from 2 stores D coordinates

    return res


def create_BLHD_dbg_tensor_repeatB(B, L, H, D, dtype, dev):
    l_numbers = torch.arange(L, dtype=dtype, device=dev).view(L, 1, 1).expand(-1, H, D)
    h_numbers = torch.arange(H, dtype=dtype, device=dev).view(1, H, 1).expand(L, -1, D)
    d_numbers = torch.arange(D, dtype=dtype, device=dev).view(1, 1, D).expand(L, H, -1)

    # Create the empty tensor to store the debug tensor
    res = torch.empty((L, H, D), dtype=dtype, device=dev)

    # Fill the debug tensor with coordinates
    res[:, :, 0::3] = l_numbers[:, :, 0::3]  # Every third column starting from 0 stores L coordinates
    res[:, :, 1::3] = h_numbers[:, :, 1::3]  # Every third column starting from 1 stores H coordinates
    res[:, :, 2::3] = d_numbers[:, :, 2::3]  # Every third column starting from 2 stores D coordinates

    rep = res[None, ...].repeat(B, 1, 1, 1).contiguous()
    return rep


def create_BLD_dbg_tensor(B, L, D, dtype, dev):
    LD_indices = torch.arange(L * D, dtype=dtype, device=dev).view(L, D)[None, ...]
    return LD_indices.repeat(B, 1, 1).contiguous()
