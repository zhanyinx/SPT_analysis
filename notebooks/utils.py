import numpy as np
import pandas as pd

def rle(inarray, full=False):
    """Run length encoding. Partial credit to R rle function.
    Multi datatype arrays catered for including non Numpy
    Returns: tuple (runlengths, startpositions, values)"""
    ia = np.asarray(inarray)  # convert to numpy array
    n = len(ia)
    if n == 0:
        return (None, None, None)
    else:
        y = ia[1:] != ia[:-1]  # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)  # must include last element posi
        z = np.diff(np.append(-1, i))  # run lengths
        p = np.cumsum(np.append(0, z))[:-1]  # positions
        if not full:
            return (
                z[1:][:-1],
                p[1:][:-1],
                ia[i][1:][:-1],
            )  # remove first and last elements for which time is ill defined
        return (
            z,
            p,
            ia[i],
        )


def contact_duration_second_passage_time_inclusive(
    df: pd.DataFrame,
    resolution: float,
    contact_cutoff: float = 0.1,
    trackid: str = "uniqueid",
    distance: str = "distance",
    split: str = "condition",
):
    """Return DataFrame of contact duration and second passage time across all matched tracks within the provided DataFrame.
    All values, including those overlapping with gaps in tracking/stitching will be included.

    Args:
        df: dataframe containing the distance between two channels across all matched tracks.
        contact_cutoff: distance to define a contact.
        trackid: column name of the trackid in df.
        distance: column name of distance in df.
        split: column name to split the df into different conditions.
    """
    duration_df = pd.DataFrame()
    second_passage_time_df = pd.DataFrame()
    for condition, subgroup in df.groupby(split):
        duration = []
        second_passage_time = []
        for _, sub in subgroup.groupby(trackid):
            # calculate length, start position and type of a vector elements
            length, position, types = rle(sub[distance] < contact_cutoff)

            # calculate the start index of True (contact)
            start = position[np.where(types == True)]
            # calculate the end index of True (contact)
            end = np.array(
                [x + y for x, y in zip(start, length[np.where(types == True)])]
            )
            # calculate the duration of contact
            duration_tmp = (
                sub.iloc[end]["frame"].values - sub.iloc[start]["frame"].values
            )
            duration.append(duration_tmp)

            # calculate the start index of loss of contact (False)
            start = position[np.where(types == False)][1:]
            # calculate the end index of missing contact (false)
            end = np.array(
                [x + y for x, y in zip(start, length[np.where(types == False)][1:])]
            )

            # calculate the duration of contact
            second_passage_time_tmp = (
                sub.iloc[end]["frame"].values - sub.iloc[start]["frame"].values
            )
            second_passage_time.append(second_passage_time_tmp)

        tmp = pd.DataFrame(np.concatenate(duration), columns=["contact_duration"])
        tmp[split] = condition
        duration_df = pd.concat([duration_df, tmp])

        tmp = pd.DataFrame(
            np.concatenate(second_passage_time), columns=["second_passage_time"]
        )
        tmp[split] = condition
        second_passage_time_df = pd.concat([second_passage_time_df, tmp])

    duration_df["contact_duration"] *= resolution
    second_passage_time_df["second_passage_time"] *= resolution
    duration_df = duration_df.reset_index()
    second_passage_time_df = second_passage_time_df.reset_index()
    return duration_df, second_passage_time_df


def contact_duration_second_passage_time_exclusive(
    df: pd.DataFrame,
    resolution: float,
    contact_cutoff: float = 0.1,
    trackid: str = "uniqueid",
    distance: str = "distance",
    split: str = "condition",
):
    """Return DataFrame of contact duration and second passage time across all matched tracks within the provided DataFrame.
    Values overlapping with gaps in tracking/stitching are excluded.

    Args:
        df: dataframe containing the distance between two channels across all matched tracks.
        contact_cutoff: distance to define a contact.
        trackid: column name of the trackid in df.
        distance: column name of distance in df.
        split: column name to split the df into different conditions.
    """
    duration_df = pd.DataFrame()
    second_passage_time_df = pd.DataFrame()
    for condition, subgroup in df.groupby(split):
        duration = []
        second_passage_time = []
        for _, sub1 in subgroup.groupby(trackid):
            print(sub1.frame - np.arange(len(sub1)))
            for _, sub in sub1.groupby(sub1.frame - np.arange(len(sub1))):
                length, position, types = rle(sub[distance] < contact_cutoff)
                duration.append(length[np.where(types == True)][1:])
                second_passage_time.append(length[np.where(types == False)][1:][:-1])

        tmp = pd.DataFrame(np.concatenate(duration), columns=["contact_duration"])
        tmp[split] = condition
        duration_df = pd.concat([duration_df, tmp])
        tmp = pd.DataFrame(
            np.concatenate(second_passage_time), columns=["second_passage_time"]
        )
        tmp[split] = condition
        second_passage_time_df = pd.concat([second_passage_time_df, tmp])

    duration_df["contact_duration"] *= resolution
    second_passage_time_df["second_passage_time"] *= resolution
    duration_df = duration_df.reset_index()
    second_passage_time_df = second_passage_time_df.reset_index()
    print(duration_df)
    return duration_df, second_passage_time_df


def contact_duration_second_passage_time_different_gaps(
    df: pd.DataFrame,
    resolution: float,
    contact_cutoff: float = 0.1,
    trackid: str = "uniqueid",
    distance: str = "distance",
    split: str = "condition",
    max_ngap: int = 2,
):

    """Return DataFrame of contact duration and second passage time across all matched tracks within the provided DataFrame
    for all the number of gaps until max_ngap

    Args:
        df: dataframe containing the distance between two channels across all matched tracks.
        contact_cutoff: distance to define a contact.
        trackid: column name of the trackid in df.
        distance: column name of distance in df.
        split: column name to split the df into different conditions.
    """
    durations = pd.DataFrame()
    second_passage_times = pd.DataFrame()
    for ngap in np.arange(max_ngap + 1):
        for _, sub in df.groupby("uniqueid"):
            sub = sub.sort_values("frame")
            # calculate the differences between consecutive frames
            sub["diff"] = sub.frame - sub.frame.shift(1)
            sub = sub.fillna(
                1.0
            )  # first frame is always nan as there is no frame before the first

            # check where we have gaps bigger than ngap
            length, start, value = rle(sub["diff"] <= ngap + 1, full=True)

            # find indexes of gaps
            gap_indexes = [i for i, x in enumerate(value) if not x]

            if len(gap_indexes):
                # loop over fraction subset of contiguous dataframe with gaps lower than ngap
                for gap_idx in gap_indexes:
                    subset = sub.iloc[(max(0, start[gap_idx - 1] - 1)) : start[gap_idx]]
                    (
                        duration,
                        second_passage_time,
                    ) = contact_duration_second_passage_time_inclusive(
                        df=subset,
                        resolution=resolution,
                        contact_cutoff=contact_cutoff,
                        trackid=trackid,
                        distance=distance,
                        split=split,
                    )
                    duration["ngap"] = ngap
                    second_passage_time["ngap"] = ngap
                    durations = pd.concat([durations, duration])
                    second_passage_times = pd.concat(
                        [second_passage_times, second_passage_time]
                    )

                (
                    duration,
                    second_passage_time,
                ) = contact_duration_second_passage_time_inclusive(
                    df=sub.iloc[start[gap_indexes[-1]] :],
                    resolution=resolution,
                    contact_cutoff=contact_cutoff,
                    trackid=trackid,
                    distance=distance,
                    split=split,
                )
                duration["ngap"] = ngap
                second_passage_time["ngap"] = ngap
                durations = pd.concat([durations, duration])
                second_passage_times = pd.concat(
                    [second_passage_times, second_passage_time]
                )
            else:
                (
                    duration,
                    second_passage_time,
                ) = contact_duration_second_passage_time_inclusive(
                    df=sub,
                    resolution=resolution,
                    contact_cutoff=contact_cutoff,
                    trackid=trackid,
                    distance=distance,
                    split=split,
                )
                duration["ngap"] = ngap
                second_passage_time["ngap"] = ngap
                durations = pd.concat([durations, duration])
                second_passage_times = pd.concat(
                    [second_passage_times, second_passage_time]
                )

    return durations, second_passage_times