#  Copyright (c) 2018-present, Cruise LLC
#
# This source code is licensed under the Apache License, Version 2.0,
# found in the LICENSE file in the root directory of this source tree.
# You may not use this file except in compliance with the License.
# Authored by Liyan Chen (liyanc@cs.utexas.edu)

import os
import sys
import torch

from pathlib import Path
from skbuild import setup


ARCH_NAME_MAP = {
    '75': '7.5', '80': '8.0', '86': '8.6', '87': '8.7', '89': '8.9', '90': '9.0', '90a': '9.0a',
    '95': '9.5', '100': '10.0', '100a': '10.0a', '101': '10.1', '101a': '10.1a', '120': '12.0',
    '120a': '12.0a'
}


def cmakearch2torcharch(arch_list):
    return ';'.join(ARCH_NAME_MAP[k] for k in arch_list.split(';'))


torch_rt = os.path.dirname(torch.__file__)
print(torch_rt)

if 'CXX' in os.environ:
    cxx = os.environ['CXX']
    print(f'env `CXX` set as {cxx}')
else:
    print(f'env `CXX` unset, using system-default cc')

if 'F3D_CUDA_ARCH' in os.environ:
    cuda_arch = os.environ['F3D_CUDA_ARCH']
    # Ad-hoc macro for Orin. One can further specify Orin Max Shared Memory to be 192KB according to spec.
    # This flag will set Max Shared Memory to be 100KB for Orin, but still allow nvcc for Orin-specific optimizations. 
    if '89' in cuda_arch or '87' in cuda_arch:
        tk_arch = 'KITTENS_4090'
    elif '80' in cuda_arch:
        tk_arch = 'KITTENS_A100'
    elif '90' in cuda_arch:
        tk_arch = 'KITTENS_HOPPER'
    # Ad-hoc mac for Thor. One can further specify Thor Max Shared Memory to be ???. (Spec sheets not available now)
    # This flag will set Max Shared Memory to be 227KB for Thor, but still allow nvcc for Thor-specific optimizations.
    elif '100' in cuda_arch:
        tk_arch = 'KITTENS_HOPPER'
    else:
        print(f'ThunderKitten has NO support for `CUDA_ARCHITECTURES`={cuda_arch}')

    os.environ['TORCH_CUDA_ARCH_LIST'] = cmakearch2torcharch(cuda_arch)
    tk_flag = f'-D{tk_arch}'
    print(f'env `CUDA_ARCHITECTURES` set as {cuda_arch}, TK arch macro set as {tk_arch}')
else:
    raise RuntimeError(f'No `F3D_CUDA_ARCH` set in env')


nvcc_path_flag = []
if 'CUDA_HOME' in os.environ:
    cudadir = os.environ['CUDA_HOME']
    nvcc_path_flag.append(f'-DCMAKE_CUDA_COMPILER={cudadir}/bin/nvcc')
    print(f'env `CUDA_HOME` set as {cudadir}')
else:
    print(f'env `CUDA_HOME` unset, using system-default nvcc')


setup(
    name="flash3dxfmr",
    author='Liyan Chen',
    author_email='liyanc@cs.utexas.edu',
    version="0.3.dev1",
    description="Flash3D Point Transformers",
    license="Apache License 2.0",
    packages=['flash3dxfmr', 'flash3dxfmr.lib'],
    cmake_args=[f'-DCMAKE_PREFIX_PATH={torch_rt}', f'-DTK_FLAGS={tk_flag}'] + nvcc_path_flag,
    cmake_minimum_required_version='3.25'
)