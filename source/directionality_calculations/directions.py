import glob
import os
import numpy as np
import pandas as pd
import argparse
import re
import time

from utils import *

from dask.distributed import Client
import dask


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        required=True,
        help="Input file or folder with trajectory files.",
    )
    parser.add_argument(
        "-ml",
        "--min_length",
        type=int,
        default=10,
        required=False,
        help="Minimum number of timepoints per trajectory.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="angle_d_alpha.csv",
        required=False,
        help="Output name",
    )
    parser.add_argument(
        "-t",
        "--tmp",
        type=str,
        default="./scratch/",
        required=False,
        help="Scratch folder for temporary files.",
    )
    parser.add_argument(
        "-ur",
        "--uncorrected_residual",
        dest="uncorrected_residual",
        action="store_true",
        help="If defined, it will look for uncorrected and residual files as well.",
    )
    args = parser.parse_args()
    return args


def main():
    """Calculate angles, D, alpha for a given trajectory file."""
    t0 = time.time()

    # Parse input
    args = _parse_args()

    if not os.path.exists(args.tmp):
        os.mkdir(args.tmp)

    # Start parallelization
    dask.config.set(shuffle="disk")
    dask.config.set(
        {"temporary_directory": args.tmp}
    ) 
    client = Client()

    trajectory_files = []
    if os.path.isdir(args.input):
        trajectory_files.extend(glob.glob(f"{args.input}/*_corrected.csv"))
        if args.uncorrected_residual:
            trajectory_files.extend(glob.glob(f"{args.input}/*_residual.csv"))
            trajectory_files.extend(glob.glob(f"{args.input}/*_uncorrected.csv"))
    elif os.path.isfile(args.input):
        trajectory_files = [args.input]
    else:
        raise ValueError(f"{args.input} must be a directory or a file!")

    # Filter tracks based on quality (length, number of tracks)
    res = client.map(
        filter_track_single_movie, trajectory_files, min_length=args.min_length
    )
    results = client.gather(res)

    path = os.path.abspath(args.input)
    trajectory_files = glob.glob(f"{path}/*pure.csv")

    # Calculate all tamsd
    res = client.map(
        calculate_directions_all,
        trajectory_files,
        min_length=args.min_length,
    )
    results = client.gather(res)

    df = pd.concat(results)

    # Add more info to results
    df[["date", "cell_line", "induction_time", "rep", "motion_correction_type"]] = df[
        "traj_file"
    ].str.extract(
        r"(20[0-9]*)_[\w\W_]*?([^_]*)_([^_]*)_[\d]*?[perc_]*?([0-9])_[\w\W]*?_([\w]*)\.csvpure",
        expand=True,
    )
    df.to_csv(args.output, index=False)
    # Stop parallelization
    client.close()


if __name__ == "__main__":
    main()
