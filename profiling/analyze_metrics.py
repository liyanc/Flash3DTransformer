# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Authored by Liyan Chen (liyanc@cs.utexas.edu) on 4/28/25
#

import sqlite3
import matplotlib
import numpy as np
import argparse
import os
import sys

from matplotlib import pyplot as plt


# --- Configuration ---
# List of database files to process
# Could also be made dynamic or an argument if needed
DBS = ["somemodel-b1.sqlite", "somemodel-b2.sqlite", "somemodel-b3.sqlite", "somemodel-b4.sqlite", "somemodel-b5.sqlite"]
# Corresponding X-axis values for the plot (e.g., batch sizes, point counts)
# Ensure this matches the order and number of databases in DBS
X_VALUES = [124087, 248174, 372261, 496348, 620435]
# Name of the SQL script file
SQL_SCRIPT_FILE = "metric_stats.sql"
# Default scope name (can be overridden if needed, maybe via args later)
DEFAULT_SCOPE_NAME = "Backbone"
# --- End Configuration ---

def load_sql_script(filename):
    """Loads SQL content from a file."""
    try:
        with open(filename, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: SQL script file '{filename}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading SQL script file '{filename}': {e}", file=sys.stderr)
        sys.exit(1)

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze Nsight metrics across multiple databases and plot results.")
    parser.add_argument("-m", "--metric", required=True,
                        help="Name of the GPU metric to analyze (e.g., 'SMs Active [Throughput %]')")
    parser.add_argument("-k", "--skip_k", type=int, default=0,
                        help="Number of initial NVTX scopes to skip (default: 0)")
    parser.add_argument("-t", "--title", default="Metric vs Input Size",
                        help="Title for the plot")
    parser.add_argument("-y", "--ylabel", default="Metric Value",
                        help="Label for the Y-axis")
    parser.add_argument("-o", "--output", default="metric_plot.png",
                        help="Output filename for the plot (e.g., 'sm_active.png')")
    parser.add_argument("--xlabel", default="Input Size",
                        help="Label for the X-axis")
    parser.add_argument("--legend", default="Some Model",
                        help="Label for the plotted line in the legend")
    parser.add_argument("-d", "--db_dir", required=True,
                        help="Directory containing the SQLite database files. (Required)")

    return parser.parse_args()

def run_analysis(sql_script_content, db_path, metric_name, scope_name, k):
    """Connects to a database, executes the SQL script, and returns results."""
    conn = None
    try:
        conn = sqlite3.connect(db_path)

        cur = conn.cursor()
        params = {
            "scope_name": scope_name,
            "metric_name": metric_name,
            "k": k
        }
        cur.execute(sql_script_content, params)
        result = cur.fetchone() # Should return one row (overall_avg, overall_stddev)

        # Handle cases where no scopes or metric data were found
        if result is None or result[0] is None:
            print(f"Warning: No data found for metric '{metric_name}' in scope '{scope_name}' (skipping {k}) in database '{os.path.basename(db_path)}'. Returning NaN.")
            return np.nan, np.nan # Return NaN if no results

        # If stddev couldn't be calculated (e.g., only one scope), it might be None
        mean_val = result[0]
        std_val = result[1] if result[1] is not None else 0.0 # Treat stddev of single point as 0

        return mean_val, std_val

    except sqlite3.Error as e:
        print(f"SQLite error connecting to or querying '{db_path}': {e}", file=sys.stderr)
        return np.nan, np.nan # Return NaN on error
    except Exception as e:
        print(f"An unexpected error occurred with database '{db_path}': {e}", file=sys.stderr)
        return np.nan, np.nan # Return NaN on error
    finally:
        if conn:
            conn.close()
            # print(f"Closed connection to {db_path}") # Optional: for debugging

if __name__ == "__main__":
    args = parse_arguments()

    # Load the SQL script
    sql_content = load_sql_script(SQL_SCRIPT_FILE)

    metric_means = []
    metric_stds = []

    print(f"Analyzing metric '{args.metric}' in scope '{DEFAULT_SCOPE_NAME}', skipping first {args.skip_k} scopes.")

    if len(DBS) != len(X_VALUES):
        print(f"Error: Number of databases ({len(DBS)}) does not match number of X-values ({len(X_VALUES)}).", file=sys.stderr)
        sys.exit(1)

    for db_file, x_val in zip(DBS, X_VALUES):
        db_full_path = os.path.join(args.db_dir, db_file)
        print(f"Processing {db_full_path} (X={x_val})...")

        mean, std = run_analysis(sql_content, db_full_path, args.metric, DEFAULT_SCOPE_NAME, args.skip_k)
        metric_means.append(mean)
        metric_stds.append(std)

    # Convert results to numpy arrays, handling potential NaNs
    m = np.array(metric_means, dtype=float)
    ms = np.array(metric_stds, dtype=float)
    x = np.array(X_VALUES, dtype=float)

    # Filter out NaN values for plotting
    valid_indices = ~np.isnan(m) & ~np.isnan(ms)
    x_plot = x[valid_indices]
    m_plot = m[valid_indices]
    ms_plot = ms[valid_indices]

    print("\n--- Results ---")
    print(f"X Values: {x.tolist()}")
    print(f"Means: {m.tolist()}")
    print(f"Std Devs: {ms.tolist()}")
    print("---------------")

    if len(m_plot) == 0:
        print("No valid data points to plot.", file=sys.stderr)
        sys.exit(1)

    # Calculate bounds for fill_between
    m_lo = m_plot - ms_plot
    m_hi = m_plot + ms_plot

    # Plotting
    plt.figure(figsize=(6, 5)) # Adjust figure size if needed
    plt.plot(x_plot, m_plot, marker='o', linestyle='-', label=args.legend)
    plt.fill_between(x_plot, m_lo, m_hi, alpha=0.2)

    plt.xlabel(args.xlabel)
    plt.ylabel(args.ylabel)
    plt.title(args.title)
    plt.ticklabel_format(style='sci', axis='x', useMathText=True, scilimits=(0,0))
    plt.legend(loc=4) # Use default location or specify loc= if needed
    plt.grid(linestyle='--')
    plt.tight_layout()

    # Save the plot
    try:
        plt.savefig(args.output)
        print(f"Plot saved to {args.output}")
    except Exception as e:
        print(f"Error saving plot to '{args.output}': {e}", file=sys.stderr)
