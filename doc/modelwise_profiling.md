# User Guide: Nsight Systems Metric Analysis & Plotting 📊

## Overview ✨

Welcome! These scripts help you analyze performance metrics captured by NVIDIA Nsight Systems and stored in SQLite databases (`.sqlite` files, often generated alongside `.nsys-rep` files).

Specifically, they allow you to:

1.  Extract a particular GPU performance metric (e.g., "SMs Active [Throughput %]") within specific code sections (scopes) across multiple runs (multiple `.sqlite` files). *We assume you have already marked these sections using NVTX in your code.*
2.  Optionally skip the first `k` occurrences of the scope in each run (useful for ignoring warmup iterations).
3.  Calculate the average and standard deviation of the metric *per scope*.
4.  Calculate the overall average and standard deviation across all relevant scopes within *each run*.
5.  Plot the final average metric value (with standard deviation error bars) against an experimental variable (like batch size or input point count) across all your runs.

## Prerequisites & Assumptions ✅

Before using these scripts, please ensure your data collection process meets the following assumptions:

*   **Consistent Setup per Session:** Each Nsight Systems profiling session (resulting in one `.sqlite` file) samples the *same configuration* (e.g., same batch size, same model variant) using potentially different data samples (e.g., different point clouds from KITTI).
*   **Multiple Iterations:** Each session runs *multiple iterations* of the core workload (e.g., model forward/backward pass).
*   **NVTX Scopes:** Each iteration you want to analyze is clearly marked using NVTX ranges via `nvtx.range_push("ScopeName")` and `nvtx.range_pop()` **in your source code**. The "ScopeName" (e.g., "Backbone") will be used by these scripts to identify the regions of interest.
*   **CUDA Synchronization:** `torch.cuda.synchronize()` (or equivalent) is called **before the start** (`nvtx.range_push`) and **before the end** (`nvtx.range_pop`) of the NVTX scope you want to measure. This ensures accurate timing and association of GPU work with the NVTX range.
*   **Output Files:** Each profiling session produces one `.nsys-rep` file and one corresponding `.sqlite` file.
*   **Data Organization:** All the `.sqlite` files you want to compare are gathered under a *single directory*.

## Files 📁

You'll primarily interact with these two files:

1.  `analyze_metrics.py`: The main Python script you run. It handles argument parsing, orchestrates the analysis across multiple database files, executes the SQL logic, and generates the final plot.
2.  `metric_stats.sql`: Contains the SQL query logic executed by the Python script on each database file. It finds the NVTX scopes, skips initial ones if requested, calculates per-scope averages, and computes the overall average and standard deviation for the specified metric within the relevant scopes. You generally don't need to run this file directly.

## Usage ⚙️

You run the analysis using the `analyze_metrics.py` script from your command line. It accepts several arguments to control its behavior:

*   `-m`, `--metric` ( **Required** ): The exact name of the GPU metric you want to analyze as it appears in the Nsight Systems database.
    *   _Example:_ `"SMs Active [Throughput %]"` (Use quotes if the name contains spaces or special characters).
*   `-d`, `--db_dir` ( **Required** ): The path to the directory containing all the `.sqlite` database files you want to process.
    *   _Example:_ `"/path/to/your/results/b200_runs"`
*   `-k`, `--skip_k` (Optional): The number of initial NVTX scopes (ordered by start time) to skip within each database file. Defaults to `0` (no skipping).
    *   _Example:_ `-k 1` (Skips the very first scope found).
*   `-t`, `--title` (Optional): The title for the generated plot. Defaults to `"Metric vs Input Size"`.
    *   _Example:_ `--title "Macbeth Model Performance"`
*   `-y`, `--ylabel` (Optional): The label for the Y-axis of the plot. Defaults to `"Metric Value"`.
    *   _Example:_ `--ylabel "Achieved Occupancy (%)"`
*   `--xlabel` (Optional): The label for the X-axis of the plot. Defaults to `"Input Size"`.
    *   _Example:_ `--xlabel "Number of Input Points"`
*   `-o`, `--output` (Optional): The filename for the saved plot image. Defaults to `"metric_plot.png"`.
    *   _Example:_ `-o "macbeth_sm_utilization.png"`
*   `--legend` (Optional): The label for the data series shown in the plot legend. Defaults to `"Data Series"`.
    *   _Example:_ `--legend "Macbeth-o8 Pro Preview 03-27 - B200"`

## Example Command 🚀

Here's an example command demonstrating how to use the script:

```bash
python analyze_metrics.py \
    -m "SMs Active [Throughput %]" \
    -k 1 \
    -d "/path/to/your/results/b200_runs" \
    --title "SM Active vs Input Size" \
    --ylabel "SMs Active Rates(%)" \
    --legend "Macbeth-o8 Pro Preview 03-27 - B200"
```

This command will:

1.  Analyze the metric `"SMs Active [Throughput %]"`.
2.  Skip the first (`k=1`) NVTX scope named "Backbone" (the default scope name set inside `analyze_metrics.py`) in each database.
3.  Look for `.sqlite` files inside the `/path/to/your/results/b200_runs` directory (replace this with your actual path!).
4.  Generate a plot titled "SM Active vs Input Size".
5.  Label the Y-axis "SMs Active Rates(%)".
6.  Label the data line in the legend as "Macbeth-o8 Pro Preview 03-27 - B200".
7.  Use the default X-axis label ("Input Size") and output filename ("metric\_plot.png").

After executing the command, you can generate a plot like such:
![sample plot](https://www.cs.utexas.edu/~liyanc/files/metric_plot.png)

## SQL Schema Reference 📖

The `metric_stats.sql` script queries tables like `NVTX_EVENTS`, `GPU_METRICS`, and `TARGET_INFO_GPU_METRICS`. If you need to understand the structure of the Nsight Systems SQLite database in detail (e.g., to find other metric names or understand different tables), please refer to the official NVIDIA documentation:

🔗 **Nsight Systems SQLite Schema Reference:** [https://docs.nvidia.com/nsight-systems/UserGuide/index.html#sqlite-schema-reference](https://docs.nvidia.com/nsight-systems/UserGuide/index.html#sqlite-schema-reference)

## Available Metric Names (Example) <a name="available-metrics"></a>

To help you choose which metric to analyze with the `-m` or `--metric` argument, here is a list of metric names extracted from an example Nsight Systems `.sqlite` file. The exact list might vary slightly depending on your GPU architecture and the profiling settings used.

You can always query your specific `.sqlite` file to get the exact list using the following SQL command:

```sql
SELECT DISTINCT metricName FROM TARGET_INFO_GPU_METRICS ORDER BY metricName;
```

**Example Metric List:**

*   `Async Compute in Flight [Throughput %]`
*   `Async Copy Engine Active 0 [Throughput %]`
*   `Async Copy Engine Active 0 [Workloads]`
*   `Async Copy Engine Active 1 [Throughput %]`
*   `Async Copy Engine Active 1 [Workloads]`
*   `Async Copy Engine Active 2 [Throughput %]`
*   `Async Copy Engine Active 2 [Workloads]`
*   `Compute Warps in Flight [Avg Warps per Cycle]`
*   `Compute Warps in Flight [Avg]`
*   `Compute Warps in Flight [Throughput %]`
*   `DRAM Read Bandwidth [Throughput %]`
*   `DRAM Write Bandwidth [Throughput %]`
*   `GPC Clock Frequency [MHz]`
*   `GR Active [Throughput %]`
*   `GR Active [Workloads]`
*   `PCIe RX Throughput [Throughput %]`
*   `PCIe Read Requests to BAR1 [Requests]`
*   `PCIe TX Throughput [Throughput %]`
*   `PCIe Write Requests to BAR1 [Requests]`
*   `Pixel Warps in Flight [Avg Warps per Cycle]`
*   `Pixel Warps in Flight [Avg]`
*   `Pixel Warps in Flight [Throughput %]`
*   `SM Issue [Throughput %]`
*   `SMs Active [Throughput %]`
*   `SYS Clock Frequency [MHz]`
*   `Sync Compute in Flight [Throughput %]`
*   `Sync Copy Engine Active [Throughput %]`
*   `Sync Copy Engine Active [Workloads]`
*   `Tensor Active [Throughput %]`
*   `Unallocated Warps in Active SMs [Avg Warps per Cycle]`
*   `Unallocated Warps in Active SMs [Avg]`
*   `Unallocated Warps in Active SMs [Throughput %]`
*   `Vertex/Tess/Geometry Warps in Flight [Avg Warps per Cycle]`
*   `Vertex/Tess/Geometry Warps in Flight [Avg]`
*   `Vertex/Tess/Geometry Warps in Flight [Throughput %]`


## Customization 🔧

While command-line arguments control most aspects, you might need to modify `analyze_metrics.py` directly for:

*   **Database Files (`DBS`):** Change the list of specific `.sqlite` filenames to process within the `--db_dir`.
*   **X-axis Values (`X_VALUES`):** Adjust the list of numerical values corresponding to each database file in `DBS`. **Ensure `DBS` and `X_VALUES` have the same number of elements and are in the correct order!** These values form the X-axis of your plot.
