import argparse
import dask
from dask.distributed import Client
import glob
import os
import pandas as pd

from utils import rototranslation_correction_movie


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        required=True,
        help="Input folder where uncorrected tracks are.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output",
        required=False,
        help="Output folder where corrected tracks are saved.",
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
    args = _parse_args()

    if not os.path.exists(args.tmp):
        os.mkdir(args.tmp)

    dask.config.set(shuffle="disk")
    dask.config.set({"temporary_directory": args.tmp})
    client = Client()

    if not os.path.isdir(args.input):
        raise ValueError(f"Input folder does not exist or is not a folder!")

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    files = glob.glob(f"{args.input}/*uncorrected.csv")

    def save_file(file):

        u_data = pd.read_csv(file)
        u_data = u_data[["x", "y", "z", "track", "frame", "cell"]].copy()
        u_data = u_data[u_data.cell != 0]

        # rigid body corrected data
        rtc_data = rototranslation_correction_movie(u_data)
        if len(rtc_data):
            basename = os.path.basename(file)
            outuncorrected = f"{args.output}/{basename}"
            outcorrected = (
                f"{args.output}/{basename.replace('uncorrected', 'corrected')}"
            )
            outresidual = f"{args.output}/{basename.replace('uncorrected', 'residual')}"

            rtc_data.drop(["xres", "yres", "zres"], axis=1).to_csv(
                outcorrected,
                index=False,
            )

            rtc_data_uncorrected = rtc_data.copy()
            rtc_data_uncorrected[["x", "y", "z"]] = (
                rtc_data_uncorrected[["x", "y", "z"]].values
                - rtc_data_uncorrected[["xres", "yres", "zres"]].values
            )
            rtc_data_uncorrected.drop(["xres", "yres", "zres"], axis=1).to_csv(
                outuncorrected,
                index=False,
            )

            rtc_data_residual = rtc_data.copy()
            rtc_data_residual[["x", "y", "z"]] = rtc_data_residual[
                ["xres", "yres", "zres"]
            ].values
            rtc_data_residual.drop(["xres", "yres", "zres"], axis=1).to_csv(
                outresidual,
                index=False,
            )

    res = client.map(save_file, files)
    results = client.gather(res)

    client.close()


if __name__ == "__main__":
    main()
