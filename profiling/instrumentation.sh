#!/bin/bash
#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.
#  Authored by Liyan Chen (liyanc@cs.utexas.edu) on 11/10/24


profile_nsys() {
  if [ $# -lt 1 ]; then
    echo "Usage: profile_nsys <output_file> [additional_args...]"
    return 1
  fi

  local output_file="$1"
  shift  # remove the first parameter and use the rest for the profiling commands

  sudo nsys profile --gpu-metrics-devices all \
    --trace=nvtx,cuda -o "${output_file}" "$@"
}

nsys_to_sqlite() {
  if [ $# -lt 2 ]; then
    echo "Usage: nsys_to_sqlite <input_file> <output_file>"
    return 1
  fi

  local input_file="$1"
  local output_file="$2"

  nsys export -t sqlite "$input_file" -o "$output_file"
}

# profile_ncu:
#   Locks clocks on a specified GPU, profiles a script with Nsight Compute, then resets clocks.
#
# Parameters:
#   $1   GPU ID (as shown by nvidia-smi -L)
#   $2   SM (core) clock in MHz
#   $3   DRAM (memory) clock in MHz
#   $4   Output file path for the .ncu-rep report
#   $5…  One or more kernel name filters (for -k), terminated by "--"
#   Last argument: Path to the script/command to profile
profile_ncu() {
  #–– Quick sanity check on argument count
  if [ $# -lt 6 ]; then
    echo "Usage: profile_ncu <gpu_id> <sm_clk> <mem_clk> <output_file> <kernel1> [<kernel2> …] -- <script>"
    return 1
  fi

  #–– Parse fixed parameters
  local gpu_id="$1"
  local sm_clk="$2"
  local mem_clk="$3"
  local outfile="$4"
  shift 4

  #–– Collect kernel filters up to "--"
  local kernels=()
  while [ "$1" != "--" ]; do
    kernels+=( "$1" )
    shift
  done
  shift  # drop the "--"
  local script="$1"

  #–– Locate tools
  local NCU=$(which ncu)
  local PY3=$(which python3)

  #–– Lock clocks for stable profiling on the chosen GPU
  sudo nvidia-smi -i "$gpu_id" -pm 1                            # enable persistence
  sudo nvidia-smi -i "$gpu_id" --lock-gpu-clocks="$sm_clk"      # lock core clocks
  sudo nvidia-smi -i "$gpu_id" --lock-memory-clocks="$mem_clk"  # lock memory clocks

  #–– Build kernel filter flags
  local kflags=()
  for k in "${kernels[@]}"; do
    kflags+=( -k "$k" )
  done

  #–– Run Nsight Compute
  sudo "$NCU" \
    --target-processes=all \
    --devices="$gpu_id" \
    --clock-control none \
    -f -o "$outfile" \
    "${kflags[@]}" \
    --set full \
    "$PY3" "$script"

  #–– Reset clocks and persistence
  sudo nvidia-smi -i "$gpu_id" --reset-gpu-clocks     # reset core clocks
  sudo nvidia-smi -i "$gpu_id" --reset-memory-clocks  # reset memory clocks
  sudo nvidia-smi -i "$gpu_id" -pm 0                  # disable persistence
}
