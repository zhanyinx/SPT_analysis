import glob
import os
import numpy as np
import pandas as pd
import argparse
import re

from utils import *


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

    args = parser.parse_args()
    return args


def main():
    """Calculate MSD given trajectory file."""
    args = _parse_args()
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
    results = [
        calculate_all_tamsd(f, min_points=args.min_points, min_length=args.min_length)
        for f in trajectory_files
    ]
    df = pd.concat(results)

    # Add more info to results
    allinfo = df["traj_file"]
    df["cell_line"] = [re.search("laminin_2i_([^_]*)", x)[1] for x in allinfo]
    df["induction_time"] = [
        re.search("laminin_2i_[^_]*_([^_]*)_", x)[1] for x in allinfo
    ]
    df["rep"] = [re.search("_([0-9])_", x)[1] for x in allinfo]
    df["date"] = [re.search("([0-9]*)_laminin_2i_", x)[1] for x in allinfo]
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
