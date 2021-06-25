import argparse
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import skimage.io

from utils import *
from trackmate_xml_2d import create_trackmate_xml

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

    outname = args.output

    # Import model
    model = pink.io.load_model(args.model)
    outdir = "./"

    if outname is None:
        outname = os.path.basename(args.input).replace("tiff", "pdf")
        outname = outname.replace("tif", "pdf")

    # create output directory
    if not os.path.exists(f"{outdir}/pixel_based"):
        os.mkdir(f"{outdir}/pixel_based")

    if not os.path.exists(f"{outdir}/um_based"):
        os.mkdir(f"{outdir}/um_based")

    # Read movie
    movie = skimage.io.imread(args.input)

    # Detect spots
    df = detect_spots(movie, model)

    outxml = outname.replace("pdf", "xml")

    # create and save trackmate xml file. It also saves a csv file with coordinate
    create_trackmate_xml(
        file_image=args.input,
        spots_df=df[["x", "y", "slice"]].copy(),
        file_output=f"{outdir}/pixel_based/{outxml}",
    )

    # micron based
    if args.spatialunits == "um":
        create_trackmate_xml(
            file_image=args.input,
            spots_df=df[["x", "y", "slice"]].copy(),
            file_output=f"{outdir}/um_based/{outxml}",
            spatialunits=args.spatialunits,
            timeunits=args.timeunits,
            pixelwidth=args.pixelwidth,
            pixelheight=args.pixelheight,
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
                    df[df["slice" == idx]]["x"],
                    df[df["slice" == idx]]["y"],
                    color="red",
                    marker="+",
                )
                ax[1].imshow(
                    np.max(movie[idx], axis=0), vmax=np.quantile(movie[idx], 0.999)
                )
                ax[1].scatter(
                    df[df["slice" == idx]]["x"],
                    df[df["slice" == idx]]["y"],
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
