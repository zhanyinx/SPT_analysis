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
        "-mp",
        "--min_points",
        type=int,
        default=5,
        required=False,
        help="Minimum number of points to calculate tamsd.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output.csv",
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
    args = parser.parse_args()
    return args


def main():
    """Calculate MSD given trajectory file."""
    t0 = time.time()

    # Parse input
    args = _parse_args()

    if not os.path.exists(args.tmp):
        os.mkdir(args.tmp)

    # Start parallelization
    dask.config.set(shuffle="disk")
    dask.config.set(
        {"temporary_directory": args.tmp}
    )  #'/work3/ggiorget/kospave/temp/'})
    client = Client()

    if os.path.isdir(args.input):
        trajectory_files = glob.glob(f"{args.input}/*_corrected.csv")
    elif os.path.isfile(args.input):
        trajectory_files = [args.input]
    else:
        raise ValueError(f"{args.input} must be a directory or a file!")

    # Filter tracks based on quality (length, number of tracks)
    filter_tracks(trajectory_files, args.min_length)

    path = os.path.abspath(args.input)
    trajectory_files = glob.glob(f"{path}/*pure.csv")

    # Calculate all tamsd
    res = client.map(
        calculate_all_tamsd,
        trajectory_files,
        min_points=args.min_points,
        min_length=args.min_length,
    )
    results = client.gather(res)

    df = pd.concat(results)

    # Add more info to results
    df[["date", "cell_line", "induction_time", "rep"]] = df["traj_file"].str.extract(
        r"(20[0-9]*)_[\w\W_]*?([^_]*)_([^_]*)_[\d]*?[perc_]*?([0-9])_",
        expand=True,
    )
    df.to_csv(args.output, index=False)
    # Stop parallelization
    client.close()
    print("Fast version took", time.time() - t0)


if __name__ == "__main__":
    main()
