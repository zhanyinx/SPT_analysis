import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import skimage.io

from utils import *
from trackmate_xml_3d import create_trackmate_xml

matplotlib.use("Agg")


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        required=True,
        help="Input movie to be used for spot detection",
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default=None,
        required=True,
        help="deepBlink model.",
    )
    parser.add_argument(
        "-xy",
        "--xy_crop",
        type=int,
        default=6,
        help="Crop size on xy for gaussian fitting. Default 6.",
    )
    parser.add_argument(
        "-z",
        "--z_crop",
        type=int,
        default=4,
        help="Crop size along z for gaussian fitting. Default 4.",
    )
    parser.add_argument(
        "-sr",
        "--search_range",
        type=int,
        default=5,
        help="The maximum distance spots can move between frames. Default 5.",
    )
    parser.add_argument(
        "-gf",
        "--gap_frames",
        type=int,
        default=0,
        help="The maximum gap frame allowed to link spots. Default 0.",
    )
    parser.add_argument(
        "-mf",
        "--min_frames",
        type=int,
        default=3,
        help="The minimum number of frames necessary to call a true 3D spots. Default 3.",
    )
    parser.add_argument(
        "-r",
        "--radius",
        type=int,
        default=2,
        help="Refinement radius used for trackpy.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output name of the pdf report. Default None.",
    )

    parser.add_argument(
        "-su",
        "--spatialunits",
        type=str,
        default="pixel",
        help="Spatial unit of the image. Default pixel, alternative um",
    )
    parser.add_argument(
        "-tu",
        "--timeunits",
        type=str,
        default="sec",
        help="Spatial unit of the image. Default sec, alternative min, hour",
    )
    parser.add_argument(
        "-pw",
        "--pixelwidth",
        type=float,
        default=1,
        help="Conversion pixel to spatial units along x, default 1.",
    )
    parser.add_argument(
        "-hw",
        "--pixelheight",
        type=float,
        default=1,
        help="Conversion pixel to spatial units along y, default 1.",
    )
    parser.add_argument(
        "-vd",
        "--voxeldepth",
        type=float,
        default=1,
        help="Conversion pixel to spatial units along z, default 1.",
    )
    parser.add_argument(
        "-ti",
        "--timeinterval",
        type=float,
        default=1,
        help="Time resolution, default 1 time unit.",
    )

    args = parser.parse_args()
    return args


def main():
    """Detect spots and pdf with spot projection in 2D."""

    # Avoid verbosity from trackpy
    tp.quiet()

    args = _parse_args()

    search_range = args.search_range
    gap_frames = args.gap_frames
    min_frames = args.min_frames
    outname = args.output

    # Import model
    model = pink.io.load_model(args.model)
    outdir = "."

    # Change relative path to abs path
    args.input = os.path.abspath(args.input)

    if outname is None:
        outname = "".join([os.path.splitext(os.path.basename(args.input))[0], ".pdf"])

    # create output directory
    if not os.path.exists(f"{outdir}/pixel_based"):
        os.mkdir(f"{outdir}/pixel_based")

    if not os.path.exists(f"{outdir}/um_based"):
        os.mkdir(f"{outdir}/um_based")

    # Read movie
    movie = skimage.io.imread(args.input)
    if len(movie.shape) == 3:
        movie = np.expand_dims(movie, axis=0)

    if len(movie.shape) < 3:
        raise ValueError(
            f"Expected at least 3 dimension, found only {len(movie.shape)}"
        )

    # Detect spots
    dfs = []
    for image in movie:
        dfs.append(detect_spots(image, model, args.radius))

    # Link spots across z stacks
    tracks = []
    for df in dfs:
        tracks.append(link_3d(df, search_range, gap_frames, min_frames))

    # Non maximum suppression using brightest frame
    df_cleans = []
    for track in tracks:
        # Index of brightest particles
        idx = track.groupby(["particle"])["mass"].transform(max) == track["mass"]
        df_cleans.append(track[idx])

    # Gaussian fitting around the spots
    coord_list = []
    coord_list_gauss = []
    for df_clean, image in zip(df_cleans, movie):
        coord_list.append(np.array(df_clean[["x", "y", "slice"]]))
        coord_list_gauss.append(
            gauss_single_image(
                image,
                np.array(df_clean[["x", "y", "slice"]]),
                args.xy_crop,
                args.z_crop,
            )
        )

    # save to xml pixel based
    df = pd.DataFrame()
    frame = 0

    for coord in coord_list_gauss:
        if len(coord > 0):
            df_new = pd.DataFrame(coord, columns=["x", "y", "z"])
            df_new["frame"] = frame
            df = pd.concat([df, df_new])
            frame = frame + 1

    outxml = outname.replace("pdf", "xml")

    # create and save trackmate xml file. It also saves a csv file with coordinate
    create_trackmate_xml(
        file_image=args.input,
        spots_df=df[["x", "y", "z", "frame"]].copy(),
        file_output=f"{outdir}/pixel_based/{outxml}",
    )

    # micron based
    if args.spatialunits == "um":
        create_trackmate_xml(
            file_image=args.input,
            spots_df=df[["x", "y", "z", "frame"]].copy(),
            file_output=f"{outdir}/um_based/{outxml}",
            spatialunits=args.spatialunits,
            timeunits=args.timeunits,
            pixelwidth=args.pixelwidth,
            pixelheight=args.pixelheight,
            voxeldepth=args.voxeldepth,
            timeinterval=args.timeinterval,
        )

    # Save pdf
    with PdfPages(outname) as pdf:
        for idx in np.arange(start=0, stop=len(movie), step=30):
            try:
                fig, ax = plt.subplots(1, 3, figsize=(30, 10))
                ax[0].imshow(
                    np.max(movie[idx], axis=0), vmax=np.quantile(movie[idx], 0.999)
                )
                ax[0].scatter(
                    coord_list[idx][..., 0],
                    coord_list[idx][..., 1],
                    color="red",
                    marker="+",
                )
                ax[1].imshow(
                    np.max(movie[idx], axis=0), vmax=np.quantile(movie[idx], 0.999)
                )
                ax[1].scatter(
                    coord_list_gauss[idx][..., 0],
                    coord_list_gauss[idx][..., 1],
                    color="red",
                    marker="+",
                )
                ax[2].imshow(
                    np.max(movie[idx], axis=0), vmax=np.quantile(movie[idx], 0.999)
                )
                pdf.savefig(fig)
                plt.close()
            except:
                continue


if __name__ == "__main__":
    main()
