import numpy as np
import pandas as pd


def fill_gaps(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Given a dataframe with gaps along the column, fills the gaps with the previous value.
    Args:
        df: input data dataframe.
        column: column name of the variable to assess whether there is gap.

    Return:
        pd.DataFrame of input data with no gaps."""

    df = df.set_index(column).reindex(np.arange(df[column].min(), df[column].max() + 1))
    c = df["distance"].isna().sum()
    df[column] = df.index
    df = df.ffill()

    return df, c


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
    full: bool = False,
):
    """Return DataFrame of contact duration and second passage time across all matched tracks within the provided DataFrame.
    All values, including those overlapping with gaps in tracking/stitching will be included.

    Args:
        df: dataframe containing the distance between two channels across all matched tracks.
        contact_cutoff: distance to define a contact.
        trackid: column name of the trackid in df.
        distance: column name of distance in df.
        split: column name to split the df into different conditions.
        full: if true, it returns also the boundary affected duration and second passage time
    """
    duration_df = pd.DataFrame()
    second_passage_time_df = pd.DataFrame()
    for condition, subgroup in df.groupby(split):
        duration = []
        second_passage_time = []
        for _, sub in subgroup.groupby(trackid):
            # calculate length, start position and type of a vector elements
            length, position, types = rle(sub[distance] < contact_cutoff, full=full)
            # calculate the start index of True (contact)
            if len(length) == 1 and types[0] == True and full:
                duration_tmp = np.max(sub.frame) - np.min(sub.frame)
            else:
                start = position[np.where(types == True)]
                start[start > len(sub) - 1] = len(sub) - 1

                # calculate the end index of True (contact)
                end = np.array(
                    [x + y for x, y in zip(start, length[np.where(types == True)])]
                )
                end[end > len(sub) - 1] = len(sub) - 1

                # calculate the duration of contact

                duration_tmp = (
                    sub.iloc[end]["frame"].values - sub.iloc[start]["frame"].values
                )
            duration.append(duration_tmp)

            # calculate the start index of loss of contact (False)
            if len(length) == 1 and types[0] == False and full:
                second_passage_time_tmp = np.max(sub.frame) - np.min(sub.frame)
            else:
                start = position[np.where(types == False)]
                start[start > len(sub) - 1] = len(sub) - 1

                # calculate the end index of missing contact (false)
                end = np.array(
                    [x + y for x, y in zip(start, length[np.where(types == False)])]
                )
                end[end > len(sub) - 1] = len(sub) - 1

                # calculate the duration of contact
                second_passage_time_tmp = (
                    sub.iloc[end]["frame"].values - sub.iloc[start]["frame"].values
                )
            second_passage_time.append(second_passage_time_tmp)

        try:
            tmp = pd.DataFrame(np.concatenate(duration), columns=["contact_duration"])
        except:
            tmp = pd.DataFrame({"contact_duration": duration[0]}, index=[0])

        tmp[split] = condition
        duration_df = pd.concat([duration_df, tmp])
        try:
            tmp = pd.DataFrame(
                np.concatenate(second_passage_time), columns=["second_passage_time"]
            )
        except:
            tmp = pd.DataFrame(
                {"second_passage_time": second_passage_time[0]}, index=[0]
            )
        tmp[split] = condition
        second_passage_time_df = pd.concat([second_passage_time_df, tmp])

    duration_df["contact_duration"] *= resolution
    second_passage_time_df["second_passage_time"] *= resolution
    duration_df = duration_df.reset_index()
    second_passage_time_df = second_passage_time_df.reset_index()
    return duration_df, second_passage_time_df
