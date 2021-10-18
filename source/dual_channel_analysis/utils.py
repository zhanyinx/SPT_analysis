import dask
import glob
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
import pandas as pd
from pymicro.view.vol_utils import compute_affine_transform
import scipy
import scipy.optimize
import secrets

matplotlib.use("Agg")
TRACKID = "track"
X = "x"
Y = "y"
Z = "z"
FRAME = "frame"
CELLID = "cell"


def drop_matched(matched: pd.DataFrame, df1: pd.DataFrame, df2: pd.DataFrame):
    """Remove the matched rows from df1 and df2. Matched results from merging df1 with df2.
    Important order of df1 and df2 matters."""

    # extract df1 and df2 from matched
    matched_x = matched[[x for x in matched.columns if "_x" in x]].copy()
    matched_y = matched[[y for y in matched.columns if "_y" in y]].copy()
    matched_x.columns = [x.replace("_x", "") for x in matched_x.columns]
    matched_y.columns = [y.replace("_y", "") for y in matched_y.columns]

    # Add frame column and reorder
    matched_x[FRAME] = matched[FRAME]
    matched_y[FRAME] = matched[FRAME]
    matched_x[CELLID] = matched[CELLID]
    matched_y[CELLID] = matched[CELLID]

    matched_x = matched_x[df1.columns]
    matched_y = matched_y[df2.columns]

    df1_new = pd.concat([df1, matched_x])
    df2_new = pd.concat([df2, matched_y])

    df1_new = df1_new.drop_duplicates(keep=False)
    df2_new = df2_new.drop_duplicates(keep=False)

    return df1_new, df2_new


def filter_tracks(df: pd.DataFrame, min_length: int = 10) -> pd.DataFrame:
    """Filter tracks based on length.

    Arg:
       df: dataframe containing the tracked data
       min_length: integer specifying the min track length

    Return:
       filtered data frame."""

    df = df[[X, Y, Z, FRAME, TRACKID, CELLID]]
    df = df[df[CELLID] != 0].copy()
    distribution_length = df[TRACKID].value_counts()
    selection = distribution_length.index.values[
        distribution_length.values > min_length
    ]

    df = df[df[TRACKID].isin(selection)]
    return df


def filter_overlapping(df: pd.DataFrame, max_overlaps: float = 0.5):
    """Return data.frame where tracks with overlaps higher than max_overlaps are filtered out.

    Args:
        df: dataframe with tracks to stitch
        max_overlaps: maximum fraction of track that can overlap.
                      Tracks with higher overlaps will be filtered out.

    Return:
        filtered dataframe.
    """

    while True:
        # count number of duplicated timepoints per track
        duplicated = df[df[FRAME].isin(df[df[FRAME].duplicated()][FRAME])][
            TRACKID
        ].value_counts()

        if len(duplicated) < 1:
            return df

        # duplicated track id
        duplicated_tracks = duplicated.index.values
        # number of duplication
        duplicated_values = duplicated.values

        # count number of timepoints per track
        count_tracks_length = df[TRACKID].value_counts()

        # if number of track is 1, by definition there is no overlapping
        if len(count_tracks_length) == 1:
            return df

        # count track length of overlapping tracks
        count_tracks_overlapping = count_tracks_length[
            count_tracks_length.index.isin(duplicated_tracks)
        ]

        # extract track id of shortest overlapping tracks
        shortest_track_overlapping_idx = count_tracks_overlapping.idxmin()

        # too long overlaps?
        toolong = False
        for track, value in zip(duplicated_tracks, duplicated_values):
            fraction = value / len(df[df[TRACKID] == track])
            if fraction > max_overlaps:
                toolong = True

        # if we found too many overlaps, remove shortest track and restart
        if toolong:
            df = df[df[TRACKID] != shortest_track_overlapping_idx].copy()

        # if no too long overlaps, remove duplicates and return dataframe
        if not toolong:
            df = df.drop_duplicates(FRAME)
            return df


def stitch(df: pd.DataFrame, max_dist: float = 1.6, max_overlaps: float = 0.5):
    """Stitch tracks with the same cell id. If tracks overlap, filters out
    tracks with overlap higher than max_overlaps. Overlapping frames are filtered out randomly.

    Arg:
       df: dataframe containing the tracked data.
       max_dist: maximum distance to match tracks from the same cell.
       max_overlaps: maximum overlap allowed for each track.

    Return:
       dataframe with stitched tracks."""

    res = pd.DataFrame()
    # loop over cell (stitch only tracks from same cell)
    for cell, sub in df.groupby(CELLID):

        # if we find any overlapping tracks, filter them out (either whole track or partial tracks)
        if np.sum(sub[FRAME].duplicated()) > 0:
            sub = filter_overlapping(df=sub, max_overlaps=max_overlaps)
        sub = sub.sort_values(FRAME).reset_index(drop=True)

        # if we have only 1 track, skip stitching
        if len(sub[TRACKID].unique()) == 1:
            res = pd.concat([res, sub])
            continue

        # look for jumps between tracks in time
        idx = sub[sub[TRACKID].diff() != 0].index.values[
            1:
        ]  # remove first value id df which is always different from none

        # all jumping track ids
        trackids = sub.loc[np.unique([idx - 1, idx]), TRACKID].values
        indexes = np.unique(trackids, return_index=True)[1]
        trackids = np.array([trackids[index] for index in sorted(indexes)])

        # find prev and after jump data frames (sub2 before, sub1 after jump)
        sub1 = sub.loc[idx]
        sub2 = sub.loc[idx - 1]

        # vector containing all the stitched ids
        lastidx = 0
        stitched_trackids = [trackids[lastidx]]

        for index in np.arange(len(trackids) - 1):
            # if no jumping found between current and next track id, go back to previous tracks
            back_iteration = 0

            # loop until you find jumping between next track and some previous tracks
            while True:
                if index - back_iteration < 0:
                    raise ValueError(
                        "No jumping found between next track and any of previous one. Something is off.."
                    )

                # select all transition between two tracks
                selection = (
                    (sub1[TRACKID] == trackids[index - back_iteration]).values
                    & (sub2[TRACKID] == trackids[index + 1]).values
                ) | (
                    (sub1[TRACKID] == trackids[index + 1]).values
                    & (sub2[TRACKID] == trackids[index - back_iteration]).values
                )

                # if jumping between tracks occur multiple times, take the shortest distance
                if np.sum(selection) > 0:
                    dists = np.sqrt(
                        np.sum(
                            np.square(
                                sub1.loc[selection, [X, Y, Z]].values
                                - sub2.loc[selection, [X, Y, Z]].values
                            ),
                            axis=1,
                        )
                    )
                    dist = np.min(dists)
                    break
                # if no jumping has been found, take previous track
                else:
                    back_iteration += 1

            # if jumping has found, check distance
            if dist < max_dist:
                sub.loc[
                    sub[TRACKID] == trackids[index + 1], TRACKID
                ] = stitched_trackids[-1 - back_iteration]
            # if jumping is too big, record the unstitched track id
            else:
                lastidx = index + 1

            stitched_trackids.append(trackids[lastidx])
        res = pd.concat([res, sub])

    return res


def calculate_single_dist(sub_df1, sub_df2, cost=True):
    min_len = np.min([len(sub_df1), len(sub_df2)])
    merged = pd.merge(sub_df1, sub_df2, how="inner", on=[FRAME])
    if not len(merged):
        return 9999999999
    s1 = merged[[x + "_x" for x in [X, Y, Z]]].values
    s2 = merged[[x + "_y" for x in [X, Y, Z]]].values
    dist = np.mean(np.sum((s1 - s2) ** 2, axis=1)) / np.sqrt(len(merged))
    if cost:
        scaling = np.mean([len(sub_df1), len(sub_df2)]) / len(merged)
        dist *= scaling
    return dist


@dask.delayed
def calculate_dist(sub_df1, df2, cost=True):
    dist = []
    for _, sub_df2 in df2.groupby(TRACKID):
        dist.append(calculate_single_dist(sub_df1, sub_df2, cost=cost))
    return dist


def calculate_distance(df1: pd.DataFrame, df2: pd.DataFrame, cost: bool = True):
    """Return the matrix of distances between tracks in true and pred.
    Since we want to privilege long overlapping tracks, we will divide the average distance by the
    square-root of the number of overlapping points.

    Args:
        df1: dataframe containing the first dataset (ground truth, channel 1 etc..)
        df2: dataframe containing the second dataset (ground truth, channel 2 etc..)
        cost: if defined, the distance will be scaled by the fraction of no overlapping points
    Return:
        matrix of all pairwise distances.
    """

    dist = []
    column_length = df1[TRACKID].nunique()
    row_length = df2[TRACKID].nunique()
    for _, sub_df1 in df1.groupby(TRACKID):
        dist.append(calculate_dist(sub_df1, df2, cost=cost))

    dist = dask.compute(dist)
    dist = np.array(dist)

    return dist


def merge_channels(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    cost: bool = True,
    distance_cutoff: int = 2,
    recursive: bool = True,
):
    """Return data frame containing the merged channel tracks

    Args:
        df1: dataframe containing the first dataset (ground truth, channel 1 etc..)
        df2: dataframe containing the second dataset (ground truth, channel 2 etc..)
        cost: if defined, the distance will be scaled by the fraction of no overlapping points
        distance_cutoff: cutoff on the average distance between tracks to be considered as corresponding track.
        recursive: apply recursive matching of leftover of partially matched tracks.
    Return:
        data frame of merged datasers.
    """
    results = pd.DataFrame()
    while True:
        # calculate distance between tracks
        dist = calculate_distance(df1, df2, cost)
        dist = dist.squeeze()

        # match tracks
        rows, cols = scipy.optimize.linear_sum_assignment(dist)
        remove = 0
        for r, c in zip(rows, cols):
            if dist[r, c] > distance_cutoff:
                rows = rows[rows != r]
                cols = cols[cols != c]
                remove += 1

        if len(rows) == 0:
            break

        # extract matched track ids
        track_ids_df1 = []
        for trackid, _ in df1.groupby(TRACKID):
            track_ids_df1.append(trackid)
        track_ids_df2 = []
        for trackid, _ in df2.groupby(TRACKID):
            track_ids_df2.append(trackid)

        track_list_df1 = np.array([track_ids_df1[i] for i in rows])
        track_list_df2 = np.array([track_ids_df2[i] for i in cols])

        # record and drop matched part of tracks
        for idx1, idx2 in zip(track_list_df1, track_list_df2):
            sub1 = df1[df1[TRACKID] == idx1].copy()
            sub2 = df2[df2[TRACKID] == idx2].copy()
            tmp = pd.merge(sub1, sub2, on=[FRAME, CELLID], how="inner").sort_values(
                FRAME
            )
            df1, df2 = drop_matched(tmp, df1, df2)
            tmp["uniqueid"] = secrets.token_hex(16)
            results = pd.concat([results, tmp])

        if not recursive:
            return results

    return results


def register_points_using_euclidean_distance(
    reference_file: str, moving_file: str, distance_cutoff: float = 0.1
):
    """Given file containing reference and moving coordinates, get the two sets of matched points"""
    reference = pd.read_csv(reference_file)
    moving = pd.read_csv(moving_file)

    reference = reference[[X, Y, Z]].values
    moving = moving[[X, Y, Z]].values

    cdist = scipy.spatial.distance.cdist(reference, moving, metric="euclidean")
    rows, cols = scipy.optimize.linear_sum_assignment(cdist)
    for r, c in zip(rows, cols):
        if cdist[r, c] > distance_cutoff:
            rows = rows[rows != r]
            cols = cols[cols != c]

    reference = np.array([reference[i] for i in rows])
    moving = np.array([moving[i] for i in cols])

    return reference, moving


def chrom_aberration_quality(original1, original2, original2_corrected, outfile):
    """Plot graphs to check quality of chromatic aberration correction and
    returns sigma x, y and z."""

    axis = ["x", "y", "z"]

    range1 = [0, 0, 1]
    range2 = [1, 2, 2]
    with PdfPages(outfile) as pdf:
        for i, j in zip(range1, range2):
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].scatter(original1[..., i], original1[..., j])
            ax[0].scatter(
                original2_corrected[..., i],
                original2_corrected[..., j],
                color="r",
                marker="+",
            )
            ax[0].set_title(f"After correction {axis[i]}{axis[j]}")
            ax[1].scatter(original1[..., i], original1[..., j])
            ax[1].scatter(original2[..., i], original2[..., j], color="r", marker="+")
            ax[1].set_title(f"Before correction {axis[i]}{axis[j]}")
            pdf.savefig(fig)
            plt.close()

        sigma = []
        for i in range(len(axis)):
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            diff1 = original1[..., i] - original2_corrected[..., i]
            diff2 = original1[..., i] - original2[..., i]

            std1 = round(np.std(diff1), 5)
            sigma.append(std1)
            std2 = round(np.std(diff2), 5)
            minimum = np.min(np.concatenate([diff1, diff2]))
            maximum = np.max(np.concatenate([diff1, diff2]))
            ax[0].hist(diff1)
            ax[0].set_title(f"After correction {axis[i]}, sigma {std1}")
            ax[0].set_xlim(minimum, maximum)
            ax[1].hist(diff2)
            ax[1].set_title(f"Before correction {axis[i]}, sigma {std2}")
            ax[1].set_xlim(minimum, maximum)
            pdf.savefig(fig)
            plt.close()

    return sigma


def compute_affine_transformation3d(
    reference_files: list,
    moving_files: list,
    quality: str,
    channel_to_correct: int = 2,
    distance_cutoff: float = 0.1,
):
    """Find affine transformation to go from spots within moving files to reference files.

    Return A, t so that reference ~= np.transpose(np.dot(A, moving.T)) + t.
    In addition returns the error on x,y,z from bead images.

    Args:
        reference_files: list of files containing the tracks from reference (fixed).
        moving_files: list of files containing the tracks from channel to be moved (moving).
        quality: Filename where to save quality of chromatic aberration correction on bead images.
        channel_to_correct: channel of moving channel, default channel 2.
        distance_cutoff: max distance for matching spots between reference and moving.

    Return:
        A (3x3): Affine transformation matrix.
        t (1x3): Affine translation vector.
        sx: error in x.
        sy: error in y.
        sz: error in z."""

    references = []
    movings = []

    # pool registred points from all bead images
    for reference, moving in zip(reference_files, moving_files):
        reference, moving = register_points_using_euclidean_distance(
            reference_file=reference,
            moving_file=moving,
            distance_cutoff=distance_cutoff,
        )

        references.append(reference)
        movings.append(moving)

    references = np.concatenate(references)
    movings = np.concatenate(movings)

    if len(references) < 4:
        raise ValueError(
            f"Computing affine trasformation requires at least 4 points, provided {len(references)}"
        )

    t, A = compute_affine_transform(references, movings)

    newcoords = np.transpose(np.dot(A, movings.T)) + t
    sx, sy, sz = chrom_aberration_quality(
        original1=references,
        original2=movings,
        original2_corrected=newcoords,
        outfile=quality,
    )

    return A, t, sx, sy, sz


def chromatic_aberration_correction(
    directory: str,
    coords: np.ndarray,
    quality: str,
    channel_to_correct: int = 2,
    distance_cutoff: float = 0.1,
) -> np.ndarray:
    """Perform chromatic aberration correction and return corrected DataFrame.

    Args:
        directory: directory containing spots from bead images.
        coords: np.ndarray with coordinates (shape n,3).
        quality: Filename where to save quality of chromatic aberration correction on bead images.
        channel_to_correct: channel to correct, default channel 2.
        distance_cutoff: max distance for matching spots between reference and moving.

    Return:
        Corrected coordinates and errors on x,y,z calculated from bead images.
    """

    if not os.path.isdir(directory):
        raise ValueError(f"Beads directory does not exist!")

    reference_files = sorted(glob.glob(f"{directory}/*w1*csv"))
    moving_files = sorted(glob.glob(f"{directory}/*w2*csv"))

    assert len(reference_files) == len(moving_files)
    assert len(reference_files) != 0

    if channel_to_correct == 1:
        moving_files, reference_files = reference_files, moving_files

    if channel_to_correct != 1 and channel_to_correct != 2:
        raise ValueError(
            f"Choose either channel 1 or channel 2 to be corrected! Provided channel {channel_to_correct}"
        )

    A, t, sx, sy, sz = compute_affine_transformation3d(
        reference_files=reference_files,
        moving_files=moving_files,
        channel_to_correct=channel_to_correct,
        distance_cutoff=distance_cutoff,
        quality=quality,
    )

    newcoords = np.transpose(np.dot(A, coords.T)) + t

    return newcoords, sx, sy, sz


def calculate_pairwise_distance(df: pd.DataFrame):
    """Return pairwise distance trajectories.

    Args:
        df: DataFrame containing the matched tracks across two channels.

    Return:
        DataFrame of pairwise distance trajectories."""
    # df = df.dropna()
    channel1 = df[[x + "_x" for x in [X, Y, Z]]].values
    channel2 = df[[x + "_y" for x in [X, Y, Z]]].values

    res = pd.DataFrame(channel1 - channel2)
    res.columns = ["x", "y", "z"]
    res["frame"] = df[FRAME].values
    # res["cell"] = [
    #     f"w1.{w1}_w2.{w2}"
    #     for w1, w2 in zip(df[CELLID + "_x"].values, df[CELLID + "_y"].values)
    # ]
    res["cell"] = df[CELLID].values
    res["track"] = [
        f"w1.{w1}_w2.{w2}"
        for w1, w2 in zip(df[TRACKID + "_x"].values, df[TRACKID + "_y"].values)
    ]

    res["uniqueid"] = df["uniqueid"].values

    return res
