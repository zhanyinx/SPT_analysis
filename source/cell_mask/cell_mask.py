import argparse
import glob
import numpy as np
import os
import skimage.io
import torch
import tifffile

from cellpose import models


def _parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        required=True,
        help="Input image or folder with images to mask.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        required=False,
        help="Output folder, default mask within input folder",
    )
    args = parser.parse_args()
    return args


def main():
    """Create cell masks and save them into mask folder within input folder."""

    args = _parse_args()
    if os.path.isdir(args.input):
        inputs = glob.glob(f"{args.input}/*tif")
    elif os.path.isfile(args.input):
        inputs = [args.input]
    else:
        raise ValueError(f"Expected input folder or file. Provided {args.input}.")

    output = args.output
    if output is None:
        output = f"{os.path.abspath(args.input)}/mask"

    if not os.path.exists(output):
        os.mkdir(output)

    cellpose_model = models.Cellpose(model_type="cyto", gpu=False)
    for input_file in inputs:
        img = skimage.io.imread(input_file)
        middle_slice = len(img) // 2

        mask_nucl, *_ = cellpose_model.eval(
            [np.max(img, axis=1)[middle_slice]],
            diameter=150,
            channels=[0, 0],
            min_size=15,
        )

        name = os.path.basename(input_file)
        out = f"{output}/{name}"
        tifffile.imsave(out, mask_nucl[0])


if __name__ == "__main__":
    main()
