import dask
import numpy as np
import pandas as pd
import scipy.optimize
import secrets

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

    distribution_length = df[TRACKID].value_counts()
    selection = distribution_length.index.values[
        distribution_length.values > min_length
    ]

    df = df[df[TRACKID].isin(selection)]
    return df


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
        dist = calculate_distance(df1, df2, cost)
        dist = dist.squeeze()

        rows, cols = scipy.optimize.linear_sum_assignment(dist)
        remove = 0
        for r, c in zip(rows, cols):
            if dist[r, c] > distance_cutoff:
                rows = rows[rows != r]
                cols = cols[cols != c]
                remove += 1

        if len(rows) == 0:
            break

        track_ids_df1 = []
        for trackid, _ in df1.groupby(TRACKID):
            track_ids_df1.append(trackid)

        track_ids_df2 = []
        for trackid, _ in df2.groupby(TRACKID):
            track_ids_df2.append(trackid)

        track_list_df1 = np.array([track_ids_df1[i] for i in rows])
        track_list_df2 = np.array([track_ids_df2[i] for i in cols])

        for idx1, idx2 in zip(track_list_df1, track_list_df2):
            sub1 = df1[df1[TRACKID] == idx1].copy()
            sub2 = df2[df2[TRACKID] == idx2].copy()
            tmp = pd.merge(sub1, sub2, on=FRAME, how="outer").sort_values(FRAME)
            df1, df2 = drop_matched(tmp, df1, df2)
            tmp["uniqueid"] = secrets.token_hex(16)
            results = pd.concat([results, tmp])

        if not recursive:
            return results

    return results


def calculate_pairwise_distance(df: pd.DataFrame):
    """Return pairwise distance trajectories.

    Args:
        df: DataFrame containing the matched tracks across two channels.

    Return:
        DataFrame of pairwise distance trajectories."""
    df = df.dropna()
    channel1 = df[[x + "_x" for x in [X, Y, Z]]].values
    channel2 = df[[x + "_y" for x in [X, Y, Z]]].values

    res = pd.DataFrame(channel1 - channel2)
    res.columns = ["x", "y", "z"]
    res["frame"] = df[FRAME].values
    res["cell"] = [
        f"w1.{w1}_w2.{w2}"
        for w1, w2 in zip(df[CELLID + "_x"].values, df[CELLID + "_y"].values)
    ]
    res["track"] = [
        f"w1.{w1}_w2.{w2}"
        for w1, w2 in zip(df[TRACKID + "_x"].values, df[TRACKID + "_y"].values)
    ]

    return res
