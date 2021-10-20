import os
import glob

import deepblink as pink
import numpy as np
import pandas as pd
import scipy.optimize as opt
import tensorflow as tf
import trackpy as tp
import scipy


def detect_spots(
    image: np.ndarray, model: tf.keras.models.Model, radius: int = 1
) -> pd.DataFrame:
    """Detect spots and gather supplementary information on each some using trackpy.refine_com.

    Args:
        image: 3D image to be used for spot detection.
        radius: radius used to gather supplementary information of detected spots.

    Return:
        df: DataFrame containing all detected spots along with supplementary information
    """
    df = pd.DataFrame()
    pad_width = radius + 1

    for slice, image_curr in enumerate(image):
        # deepBlink prediction
        yx = pink.inference.predict(image=image_curr, model=model)
        y, x = yx.T.copy()

        # pad to refine spot close to the edges
        yx = yx + pad_width
        image_curr = np.pad(
            image_curr, pad_width=pad_width, mode="constant", constant_values=0
        )

        # Refinement with trackpy
        df_curr = tp.refine_com(
            raw_image=image_curr, image=image_curr, radius=radius, coords=yx
        )
        df_curr["x"] = x
        df_curr["y"] = y
        df_curr["slice"] = slice
        df = df.append(df_curr, ignore_index=True)

    return df


def link_3d(
    df: pd.DataFrame, search_range: float, gap_frames: int, min_frames: int
) -> pd.DataFrame:
    """Link detected spots across slices using trackpy.

    Args:
        search_range: the maximum distance features can move between frames, optionally per dimension.
        df: DataFrame containing frame information along with x and y coordinates.
        gap_frames: maximum gap frames allowed.
        min_frames: minimum number of frames to be considered a real spot.

    Returns:
        track: DataFrame containing the filtered tracks (3D spots).
    """
    track = tp.link(
        df.rename({"slice": "frame"}, axis=1),
        search_range=search_range,
        memory=gap_frames,
    )
    track = tp.filter_stubs(track, threshold=min_frames).rename(
        {"frame": "slice"}, axis=1
    )
    return track


EPS = 1e-4


def gauss_3d(xyz, amplitude, x0, y0, z0, sigma_xy, sigma_z, offset):
    """3D gaussian."""
    x, y, z = xyz
    x0 = float(x0)
    y0 = float(y0)
    z0 = float(z0)

    gauss = offset + amplitude * np.exp(
        -(
            ((x - x0) ** (2) / (2 * sigma_xy ** (2)))
            + ((y - y0) ** (2) / (2 * sigma_xy ** (2)))
            + ((z - z0) ** (2) / (2 * sigma_z ** (2)))
        )
    )
    return gauss


def find_start_end(coord, img_size, crop_size):
    start_dim = np.max([int(np.round(coord - crop_size // 2)), 0])
    if start_dim < img_size - crop_size:
        end_dim = start_dim + crop_size
    else:
        start_dim = img_size - crop_size
        end_dim = img_size

    return start_dim, end_dim


def gauss_single_spot(
    image: np.ndarray,
    c_coord: float,
    r_coord: float,
    z_coord: float,
    crop_size: int,
    crop_size_z: int,
):
    """Gaussian prediction on a single crop centred on spot."""

    start_dim1, end_dim1 = find_start_end(c_coord, image.shape[0], crop_size)
    start_dim2, end_dim2 = find_start_end(r_coord, image.shape[1], crop_size)
    start_dim3, end_dim3 = find_start_end(z_coord, image.shape[2], crop_size_z)

    crop = image[start_dim1:end_dim1, start_dim2:end_dim2, start_dim3:end_dim3]

    x = np.arange(0, crop.shape[0], 1)
    y = np.arange(0, crop.shape[1], 1)
    z = np.arange(0, crop.shape[2], 1)
    xx, yy, zz = np.meshgrid(x, y, z)

    # Guess intial parameters
    x0 = int(crop.shape[0] // 2)  # Center of gaussian, middle of the crop
    y0 = int(crop.shape[1] // 2)  # Center of gaussian, middle of the crop
    z0 = int(crop.shape[2] // 2)  # Center of gaussian, middle of the crop
    sigma = max(*crop.shape[:-1]) * 0.1  # SD of gaussian, 10% of the crop
    sigmaz = crop.shape[-1] * 0.1  # SD of gaussian, 10% of the crop
    amplitude_max = max(
        np.max(crop) / 2, np.min(crop)
    )  # Height of gaussian, maximum value
    initial_guess = [amplitude_max, x0, y0, z0, sigma, sigmaz, 0]

    # Parameter search space bounds
    lower = [np.min(crop), 0, 0, 0, 0, 0, -np.inf]
    upper = [
        np.max(crop) + EPS,
        crop_size,
        crop_size,
        crop_size_z,
        np.inf,
        np.inf,
        np.inf,
    ]
    bounds = [lower, upper]
    try:
        popt, _ = opt.curve_fit(
            gauss_3d,
            (xx.ravel(), yy.ravel(), zz.ravel()),
            crop.ravel(),
            p0=initial_guess,
            bounds=bounds,
        )
    except RuntimeError:
        return r_coord, c_coord, z_coord

    x0 = popt[1] + start_dim2
    y0 = popt[2] + start_dim1
    z0 = popt[3] + start_dim3

    # If predicted spot is out of the border of the image
    if x0 >= image.shape[0] or y0 >= image.shape[1] or z0 >= image.shape[2]:
        return r_coord, c_coord, z_coord

    return x0, y0, z0


def gauss_single_image(
    image: np.ndarray, coord_list: np.ndarray, crop_size: int = 4, crop_size_z: int = 4
):
    """Gaussian localization on a single image initialized on deepblink output (mask)."""
    image = np.moveaxis(image, 0, -1).copy()  # move z axis as last

    prediction_coord = []
    for coord in coord_list:
        r_coord = min(len(image) - EPS, coord[0])
        c_coord = min(len(image) - EPS, coord[1])
        z_coord = min(image.shape[-1] - EPS, coord[2])

        prediction_coord.append(
            gauss_single_spot(image, c_coord, r_coord, z_coord, crop_size, crop_size_z)
        )

    return np.array(prediction_coord)


def print_results(results: pd.DataFrame):
    """Print the average f1-score and f1-integral across the 4 images with Ground truth."""
    for name in results.name.unique():
        df = results[results["name"] == name]

        p = 4
        f1_m = df[df["cutoff"] == 3]["f1_score"].mean().round(p)
        f1_s = df[df["cutoff"] == 3]["f1_score"].std().round(p)
        f1i_m = df["f1_integral"].mean().round(p)
        f1i_s = df["f1_integral"].std().round(p)
        rmse_m = df["mean_euclidean"].mean().round(p)
        rmse_s = df["mean_euclidean"].std().round(p)

        print(
            f"Evaluation of {name}:\n"
            f"    F1 @3px: {f1_m} ± {f1_s}\n"
            f"    F1 integral @3px: {f1i_m} ± {f1i_s}\n"
            f"    RMSE @3px: {rmse_m} ± {rmse_s}\n"
        )


def calculate_rototranslation_3D(A, B, distance_cutoff=1):
    """Return translation and rotation matrices.
    Args:
        A: coordinates of fixed set of points.
        B: coodinates of moving set of points (to which roto translation needs to be applied).

    Return:
        R, t: rotation and translation matrix
    """

    cdist = scipy.spatial.distance.cdist(A, B, metric="euclidean")
    rows, cols = scipy.optimize.linear_sum_assignment(cdist)
    for r, c in zip(rows, cols):
        if cdist[r, c] > distance_cutoff:
            rows = rows[rows != r]
            cols = cols[cols != c]

    if len(rows) < 4:
        return None, None

    A = np.array([A[i] for i in rows])
    B = np.array([B[i] for i in cols])

    A = np.transpose(A)
    B = np.transpose(B)

    num_rows, num_cols = A.shape

    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


def rototranslation_correction_cell(df):
    """Return corrected dataframe given a dataframe with coordinate as input."""
    frames = sorted(df["frame"].unique())
    prev_df = df[df["frame"] == frames[0]].copy()
    out = prev_df.copy()
    out[["xres", "yres", "zres"]] = 0

    rotation = []
    degree_rotation = []
    translation = []
    degree_translation = []
    for idx in np.arange(1, len(frames)):
        curr_df = df[df["frame"] == frames[idx]].copy()
        prev = prev_df[["x", "y", "z"]].values
        curr = curr_df[["x", "y", "z"]].values
        R, t = calculate_rototranslation_3D(curr, prev)
        if R is None:
            continue
        # from http://motion.cs.illinois.edu/RoboticSystems/3DRotations.html
        degree_rotation.append(np.arccos((np.trace(R) - 1) / 2))
        degree_translation.append(np.sqrt(np.sum(np.square(t))))
        rotation.append(R)
        translation.append(t)

        prev_df = curr_df.copy()
        rang = np.arange(len(rotation))
        original_pos = curr.copy()
        for j in rang[::-1]:
            if rotation[j] is not None:
                curr = np.transpose(rotation[j] @ curr.T + translation[j])
        curr_df[["x", "y", "z"]] = curr
        curr_df[["xres", "yres", "zres"]] = curr - original_pos
        out = pd.concat([out, curr_df])
    out["degree_av_rotation"] = np.mean(degree_rotation)
    out["degree_av_translation"] = np.mean(degree_translation)
    return out


def rototranslation_correction_movie(df):
    out = pd.DataFrame()
    for _, cell_df in df.groupby("cell"):
        corrected = rototranslation_correction_cell(cell_df)
        out = pd.concat([out, corrected])

    return out
