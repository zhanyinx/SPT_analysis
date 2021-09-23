import glob
import os
import numpy as np
import pandas as pd
import argparse
import re
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.use("Agg")

from utils import *

from dask.distributed import Client


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        required=True,
        help="Input folder to trajectories file from 2 channels.",
    )
    parser.add_argument(
        "-d",
        "--distance_cutoff",
        type=float,
        default=2.0,
        required=False,
        help="Maximum distance between trajectories to be considered as matching.",
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
        "-c",
        "--cost",
        dest="cost",
        action="store_true",
        help="If defined, it will apply cost to partially overlapping tracks. See util.calculate_single_dist for more info",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        required=False,
        help="Output folder. Default pair_wise_distance within the input folder.",
    )
    args = parser.parse_args()
    return args


def main():
    """Match tracks from channels."""
    # Parse input
    args = _parse_args()

    client = Client()

    channel1_files = sorted(glob.glob(f"{args.input}/*w1*csv"))
    names = [re.search(r"(^.*)w1", os.path.basename(x))[1] for x in channel1_files]
    channel2_files = [glob.glob(f"{args.input}/{name}*w2*csv")[0] for name in names]

    outdir = args.output
    if args.output is None:
        outdir = f"{args.input}/pair_wise_distance/"

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for channel1, channel2 in zip(channel1_files, channel2_files):
        outname = os.path.basename(channel1).replace("w1", "w1.w2")
        channel1 = pd.read_csv(channel1)
        channel1 = channel1[[X, Y, Z, FRAME, TRACKID, CELLID]]
        channel2 = pd.read_csv(channel2)
        channel2 = channel2[[X, Y, Z, FRAME, TRACKID, CELLID]]

        channel2 = filter_tracks(channel2, min_length=args.min_length)
        channel1 = filter_tracks(channel1, min_length=args.min_length)
        res = merge_channels(
            channel1, channel2, cost=args.cost, distance_cutoff=args.distance_cutoff
        )
        with PdfPages(f"{outdir}/{outname}.pdf") as pdf:
            for _, sub in res.groupby("uniqueid"):
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                sub = sub.dropna()
                ax.plot(sub[f"{X}_x"], sub[f"{Y}_x"])
                ax.plot(sub[f"{X}_y"], sub[f"{Y}_y"])
                pdf.savefig(fig)
                plt.close()

        res = calculate_pairwise_distance(res)

        res.to_csv(f"{outdir}/{outname}", index=False)

    client.close()


if __name__ == "__main__":
    main()
