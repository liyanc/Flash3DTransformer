#!/bin/bash

set -euo pipefail

# apt dependencies
apt update
apt install -y nfs-common

# pip dependencies
python3 -m pip install --root-user-action=ignore spconv-cu126==2.3.8 scikit-build pybind11

# clear apt caches
rm -rf /var/lib/apt/lists/*
# clear pip caches
rm -rf ~/.cache/pip

echo "All dependencies installed for the docker image"