import os
import copy
import sys

import numpy as np
import pandas as pd


def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'."""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def calculate_directions_single_track(
    single_traj: pd.DataFrame, dt: int = 1, time_step: float = 10
):
    """Calculate distribution of angles for a particular dt.
    dt in arbitrary units.
    If an appropriate frame is missing than simply do not
    calculate any angle.

    inputs:
        single track in a DataFrame format, containing [x, y, z, frame]
    outputs:
        distribution of angles for a particular track
    """
    angles = []
    Ds = []
    slopes = []
    time_start = []
    time_end = []
    displacements = []
    for t in single_traj.frame.values:
        # assign
        # i -> t
        # j -> t + dt
        # k -> t + 2 * dt
        if np.count_nonzero(single_traj.frame.values == t) == 1:
            i = single_traj.index[single_traj.frame == t][0]
        else:
            continue
        if np.count_nonzero(single_traj.frame.values == t + dt) == 1:
            j = single_traj.index[single_traj.frame == t + dt][0]
        else:
            continue
        if np.count_nonzero(single_traj.frame.values == t + 2 * dt) == 1:
            k = single_traj.index[single_traj.frame == t + 2 * dt][0]
        else:
            continue
        # create vectors v1 = (j-i) and v2 = (k-j)
        v1 = np.array(
            [
                single_traj.loc[j].x - single_traj.loc[i].x,
                single_traj.loc[j].y - single_traj.loc[i].y,
                single_traj.loc[j].z - single_traj.loc[i].z,
            ]
        )
        v2 = np.array(
            [
                single_traj.loc[k].x - single_traj.loc[j].x,
                single_traj.loc[k].y - single_traj.loc[j].y,
                single_traj.loc[k].z - single_traj.loc[j].z,
            ]
        )
        # calculate angle between v1 and v2
        angles.append(angle_between(v1, v2) / np.pi * 180)

        # calculate vector v3 = (k-i)
        v3 = np.array(
            [
                single_traj.loc[k].x - single_traj.loc[i].x,
                single_traj.loc[k].y - single_traj.loc[i].y,
                single_traj.loc[k].z - single_traj.loc[i].z,
            ]
        )

        # calculate slope and intercept (D) for MSD using shift vectors (v1, v2) and v3
        x = np.arange(1, 3) * dt * time_step
        y = np.zeros(2)
        y[0] = np.mean(
            np.sum(
                [v1 ** 2, v2 ** 2],
                axis=1,
            ),
        )
        y[1] = np.sum(v3 ** 2)
        loc_displacement = y[1]
        x = np.log10(x)
        y = np.log10(y)
        slope, lgD = np.polyfit(x, y, 1)

        # add slope, D, timestep to the corresponding list
        slopes.append(slope)
        Ds.append(10 ** lgD)
        time_start.append(t)
        time_end.append(t + 2 * dt)
        displacements.append(np.sqrt(loc_displacement))

    # create dataframe using lists
    df = pd.DataFrame(
        {
            "dt": dt,
            "start": time_start,
            "angle": angles,
            "D": Ds,
            "slope": slopes,
            "displacement": displacements,
        }
    )
    df["cell_id"] = single_traj["cell"].unique()[0]
    df["track_id"] = single_traj["track"].unique()[0]
    return df


def calculate_directions_all(
    traj_file: str, min_length: int = 10, time_step: float = 10
):
    """Calculate all time average MSD given a movie and return a DataFrame containing them.

    Inputs:
        traj_file: path to the trajectories file.
        min_length: minimum length of trajectory accepted.

    Return:
        results: pd.DataFrame containing angles, (D, alpha) for a 3-point piece of a track.
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
        for delta_t in [1, 5, 10, 15, 20]:
            df_tmp = calculate_directions_single_track(single_traj, dt=delta_t)
            results = pd.concat([results, df_tmp])

    results["traj_file"] = os.path.basename(traj_file)

    return results


def filter_track_single_movie(filename, min_length=10):
    """
    Given a file containing the trajectories of a movie, it filters out all tracks with
    less than min_length timepoints; in addition it filters out cells with too little (<2)
    or too many trajectories (> max_num_traj_per_cell).
    Input:
        filename: file containing the trajectories
        min_length - minimal track length (in timepoints)
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
        if len(trajs[-1]) < 2:
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
