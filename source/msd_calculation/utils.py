import os
import copy
import sys

import numpy as np
import pandas as pd


def calculate_single_tamsd(single_traj: pd.DataFrame, min_points: int = 10):
    """Calculate trajectory average MSD at all lags.

    Inputs:
        coord: pd.DataFrame containing the coordinates of a given trajectory
        min_points: minimum number of points to calculate the time average MSD
    Return:
        df: pd.DataFrame containing lags and time average MSD"""
    # Calculate pair-wise differences between all timepoints in the trajectory and store it
    # in a matrix
    tvalues = single_traj["frame"].values
    tvalues = tvalues[:, None] - tvalues

    # list of lags
    lags = np.arange(len(single_traj) - min_points) + 1

    final_lags = []
    tamsd = []
    # Loop over lags
    for lag in lags:
        # find indexes of pairs of timepoints with lag equal to the selected lag
        x, y = np.where(tvalues == lag)

        if len(x) < min_points:
            continue

        tmp_tamsd = np.mean(
            np.sum(
                np.square(
                    single_traj.iloc[x][["x", "y", "z"]].values
                    - single_traj.iloc[y][["x", "y", "z"]].values
                ),
                axis=1,
            )
        )

        final_lags.append(lag)
        tamsd.append(tmp_tamsd)

    df = pd.DataFrame({"lags": final_lags, "tamsd": tamsd})
    df["cellid"] = single_traj["cell"].values[0]

    return df


def calculate_all_tamsd(traj_file: str, min_points: int = 10, min_length: int = 10):
    """Calculate all time average MSD given a movie and return a DataFrame containing them.

    Inputs:
        traj_file: path to the trajectories file.
        min_points: minimum number of points to consider time average MSD.
        min_length: minimum length of trajectory accepted.

    Return:
        results: pd.DataFrame containing all time average MSD given the trajectories of a movie.
    """

    # Read data from trajectory file
    df = pd.read_csv(traj_file)

    # output data frame result holder
    results = pd.DataFrame()

    # Loop of tracks
    for track_id in df["track"].unique():
        # Extract single trajectory and sort based on time (frame)
        single_traj = df[df["track"] == track_id].copy().sort_values(by="frame")

        # filter on too short tracks
        if len(single_traj) < min_length:
            continue

        df_tmp = calculate_single_tamsd(single_traj, min_points=min_points)
        results = pd.concat([results, df_tmp])

    results["traj_file"] = os.path.basename(traj_file)

    return results


def filter_track_single_movie(
    filename: str,
    min_length: int = 10,
    max_num_trajs_per_cell: int = 1000000,
    min_tracks: int = 2,
):
    """
    Given a file containing the trajectories of a movie, it filters out all tracks with
    less than min_length timepoints; in addition it filters out cells with too little (<min_tracks)
    or too many trajectories (> max_num_traj_per_cell).
    Input:
        filename: file containing the trajectories
        min_length - minimal track length (in timepoints)
        max_num_trajs_per_cell - maximum number of trajectories (particles) per cell
    """
    ini = True
    trajs = []
    pure_df = pd.DataFrame()
    df = pd.read_csv(filename)
    df = df.mask(df["track"].astype(object).eq("None")).dropna()

    cells = set(df["cell"].values)
    cells.discard(0)
    print("Total length", len(df))
    print(
        "number of duplicates",
        np.sum(df.duplicated(subset=["track", "cell", "frame"]).values),
    )
    for cell in cells:
        sdf = df[(df["cell"] == cell)]
        trajs.append([])
        trs = set(sdf["track"].values)
        for track in trs:
            check_it = sdf[(sdf["track"] == track)]
            if len(check_it) < min_length:
                sdf = sdf.drop(check_it.index, axis=0)
                df = df.drop(check_it.index, axis=0)
                continue
            if ini:
                pure_df = check_it
                ini = False
            else:
                pure_df = pd.concat([pure_df, check_it])
            trajs[-1].append(1)
        if len(trajs[-1]) < min_tracks or len(trajs[-1]) > max_num_trajs_per_cell:
            for track in trs:
                ssdf = sdf[sdf["track"] == track]
                df = df.drop(ssdf.index, axis=0, errors="ignore")
                pure_df = pure_df.drop(ssdf.index, axis=0, errors="ignore")

    if len(pure_df) > 0:
        pure_df[["track", "x", "y", "z", "frame", "cell"]].to_csv(
            filename + "pure.csv", index=False
        )


def filter_tracks(list_files: list, min_length: int = 10):
    """Given folder of tracks, performs quality filters on all tracks,
    see filter_track_single_movie function for more info."""

    for f in list_files:
        filter_track_single_movie(f, min_length=min_length)
        print(f, " is done")
