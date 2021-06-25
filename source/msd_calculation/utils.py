import numpy as np
import pandas as pd
import os


def calculate_single_tamsd(single_traj: pd.DataFrame, min_points: int = 10):
    """Calculate trajectory average MSD at all lags.

    Inputs:
        coord: pd.DataFrame containing the coordinates of a given trajectory
        min_points: minimum number of points to calculate the time average MSD
    Return:
        df: pd.DataFrame containing lags and time average MSD"""

    # Calculate pair-wise differences between all timepoints in the trajectory
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

        final_lags.append(lag)
        tmp_tamsd = np.mean(
            np.square(
                single_traj.iloc[x][["x", "y", "z"]].values
                - single_traj.iloc[y][["x", "y", "z"]].values
            )
        )
        tamsd.append(tmp_tamsd)

    df = pd.DataFrame({"lags": final_lags, "tamsd": tamsd})

    return df


def calculate_all_tamsd(
    traj_file: str, min_points: int = 10, min_length: int = 10, log=False
):
    """Calculate all time average MSD given a movie and return a DataFrame containing them.

    Inputs:
        traj_file: path to the trajectories file.
        min_points: minimum number of points to consider time average MSD.
        min_length: minimum length of trajectory accepted.
        log: transform in log prior ensamble average, default False.

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
