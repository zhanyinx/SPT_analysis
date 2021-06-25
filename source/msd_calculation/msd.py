from genericpath import isfile
import glob
import os
import numpy as np
import pandas as pd
import argparse


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
        trajectory_files = glob.glob(f"{args.input}/*pure*csv")
    elif os.path.isfile(args.input):
        trajectory_files = [args.input]
    else:
        raise ValueError(f"{args.input} must be a directory or a file!")
    results = [calculate_all_tamsd(f) for f in trajectory_files]
    df = pd.concat(results)

    df.to_csv(args.output)


if __name__ == "__main__":
    main()
