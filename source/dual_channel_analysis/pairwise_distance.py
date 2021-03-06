import argparse
import re
from utils import *

matplotlib.use("Agg")

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
        default=1.5,
        required=False,
        help="Maximum distance between trajectories to be considered as matching.",
    )
    parser.add_argument(
        "-ml",
        "--min_length",
        type=int,
        default=25,
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
        "-r",
        "--recursive",
        dest="recursive",
        action="store_true",
        help="If defined, recursively matches the leftover of partially matched tracks.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        required=False,
        help="Output folder. Default pair_wise_distance within the input folder.",
    )
    parser.add_argument(
        "-b",
        "--beads",
        type=str,
        default=None,
        required=False,
        help="Folder containing beads spots, if provided, it automatically performs chromatic aberration correction.",
    )
    parser.add_argument(
        "-t",
        "--twod",
        dest="twod",
        action="store_true",
        help="If defined, it will perform chromatic aberration correction in 2d.",
    )
    parser.add_argument(
        "-a",
        "--affine",
        dest="affine",
        action="store_true",
        help="If defined, it will calculate the affine transformation between the two channels.",
    )
    parser.add_argument(
        "-dp",
        "--distance_cutoff_points",
        type=float,
        default=0.1,
        required=False,
        help="Maximum distance between points to be considered as corresponding in matching beads.",
    )
    parser.add_argument(
        "-s",
        "--stitch",
        dest="stitch",
        action="store_true",
        help="If defined, it will try to stich tracks from the same cells",
    )
    parser.add_argument(
        "-ds",
        "--distance_cutoff_stitching",
        type=float,
        default=1.6,
        required=False,
        help="Maximum distance between allowed to joining tracks from same cells. Distance defined between closest frames.",
    )
    parser.add_argument(
        "-mo",
        "--max_overlaps",
        type=float,
        default=0.5,
        required=False,
        help="Maximum fraction overlapping allowed for overlapping tracks.",
    )
    parser.add_argument(
        "-dd",
        "--debug",
        dest="debug",
        action="store_true",
        help="If defined, it will print out debug information.",
    )

    args = parser.parse_args()
    return args


def main():
    """Match tracks from channels."""
    # Parse input
    args = _parse_args()

    # start dask client
    client = Client()

    if not os.path.isdir(args.input):
        raise ValueError(f"Input directory {args.input} does not exist.")

    # list files
    channel1_files = sorted(glob.glob(f"{args.input}/*w1*csv"))
    names = [re.search(r"(^.*)w1", os.path.basename(x))[1] for x in channel1_files]
    channel2_files = sorted(
        glob.glob(f"{args.input}/*w2*csv")
    )  # [glob.glob(f"{args.input}/{name}*w2*csv")[0] for name in names]

    if len(channel1_files) == 0:
        raise ValueError(f"No files found in the input directory {args.input}.")

    outdir = args.output
    if args.output is None:
        outdir = f"{args.input}/pair_wise_distance/"

    if args.debug:
        outdir_debug = f"{outdir}/debug/"
        os.makedirs(outdir_debug, exist_ok=True)

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    # loop over movies
    for channel1_name, channel2_name in zip(channel1_files, channel2_files):
        try:
            outname = os.path.basename(channel1_name).replace("w1", "w1.w2")

            # read movies
            channel1 = pd.read_csv(channel1_name)
            channel2 = pd.read_csv(channel2_name)

            # correct for chromatic aberration
            if args.beads:
                coords = channel2[[X, Y, Z]].values
                coords_corrected, sx, sy, sz = chromatic_aberration_correction(
                    directory=args.beads,
                    coords=coords,
                    channel_to_correct=2,
                    distance_cutoff=args.distance_cutoff_points,
                    quality=f"{outdir}/chromatic_aberration_correction_quality.pdf",
                    twod=args.twod,
                    affine=args.affine,
                )
                channel2[[X, Y, Z]] = coords_corrected

            channel1 = channel1[channel1[CELLID] != 0]
            channel2 = channel2[channel2[CELLID] != 0]

            if args.stitch:
                channel1 = stitch(
                    df=channel1,
                    max_dist=args.distance_cutoff_stitching,
                    max_overlaps=args.max_overlaps,
                )
                channel2 = stitch(
                    df=channel2,
                    max_dist=args.distance_cutoff_stitching,
                    max_overlaps=args.max_overlaps,
                )

            # filter too short tracks
            channel1 = filter_tracks(channel1, min_length=args.min_length)
            channel2 = filter_tracks(channel2, min_length=args.min_length)

            if args.debug:
                channel1.to_csv(
                    f"{outdir_debug}/{os.path.basename(channel1_name)}", index=False
                )
                channel2.to_csv(
                    f"{outdir_debug}/{os.path.basename(channel2_name)}", index=False
                )

            # assign tracks between channels
            res = merge_channels(
                channel1,
                channel2,
                cost=args.cost,
                distance_cutoff=args.distance_cutoff,
                recursive=args.recursive,
            )

            # filter too short matched tracks
            distribution_length = res["uniqueid"].value_counts()
            selection = distribution_length.index.values[
                distribution_length.values > args.min_length
            ]
            res = res[res["uniqueid"].isin(selection)]

            # plot matched tracks
            with PdfPages(f"{outdir}/{outname}.pdf") as pdf:
                for idx, sub in res.groupby("uniqueid"):
                    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                    sub = sub.dropna()
                    legends = ["c1", "c2"]
                    ax.plot(sub[f"{X}_x"], sub[f"{Y}_x"], "-o")
                    ax.plot(sub[f"{X}_y"], sub[f"{Y}_y"], "-o")
                    plt.legend(legends)
                    plt.title(f"Length tracks {len(sub)}; id {idx}")
                    pdf.savefig(fig)
                    plt.close()

            if args.debug:
                out1 = res[["x_x", "y_x", "z_x", FRAME, "track_x"]]
                out1.columns = ["x", "y", "z", "frame", "track"]
                out1.to_csv(
                    f"{outdir_debug}/{os.path.basename(channel1_name)}.merged.csv",
                    index=False,
                )

                out1 = res[["x_y", "y_y", "z_y", FRAME, "track_y"]]
                out1.columns = ["x", "y", "z", "frame", "track"]
                out1.to_csv(
                    f"{outdir_debug}/{os.path.basename(channel2_name)}.merged.csv",
                    index=False,
                )

            # calculate pair-wise distance
            res = calculate_pairwise_distance(res)
            res["distance"] = np.sqrt(np.sum(np.square(res[[X, Y, Z]].values), axis=1))

            # calculate errors if bead images provided
            if args.beads:
                res["sigma_x"] = sx
                res["sigma_y"] = sy
                res["sigma_z"] = sz
                res["sigma_d"] = (
                    np.sqrt(
                        (res[X].values * sx) ** 2
                        + (res[Y].values * sy) ** 2
                        + (res[Z].values * sz) ** 2
                    )
                    / res["distance"]
                )

            # log whether chromatic aberration has been performed
            res["chromatic_correction"] = bool(args.beads)
            res["filename"] = outname

            # save to file
            res.to_csv(f"{outdir}/{outname}", index=False)
        except:
            pass

    client.close()


if __name__ == "__main__":
    main()
