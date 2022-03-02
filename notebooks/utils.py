from hmmlearn import hmm
import hmmlearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats

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
    """Return DataFrame of contact duration and second passage time and contact frequency across all matched tracks within the provided DataFrame.
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
    frequency_df = pd.DataFrame()

    for condition, subgroup in df.groupby(split):
        duration = []
        second_passage_time = []
        frequency = []
        for _, sub in subgroup.groupby(trackid):
            # calculate length, start position and type of a vector elements
            length, position, types = rle(sub[distance] < contact_cutoff, full=full)

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

            if len(length) == 1:
                frequency_tmp = np.max(sub.frame) - np.min(sub.frame)
            else:
                start = position[np.where(types == True)]
                start[start > len(sub) - 1] = len(sub) - 1
                frequency_tmp = (
                    sub.iloc[start[1:]]["frame"].values
                    - sub.iloc[start[:-1]]["frame"].values
                )

            duration.append(duration_tmp)
            frequency.append(frequency_tmp)

        try:
            tmp = pd.DataFrame(np.concatenate(duration), columns=["contact_duration"])
        except:
            tmp = pd.DataFrame({"contact_duration": duration[0]}, index=[0])

        tmp[split] = condition
        duration_df = pd.concat([duration_df, tmp])

        try:
            tmp = pd.DataFrame(np.concatenate(frequency), columns=["frequency"])
        except:
            tmp = pd.DataFrame({"frequency": frequency[0]}, index=[0])

        tmp[split] = condition
        frequency_df = pd.concat([frequency_df, tmp])

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
    frequency_df["frequency"] *= resolution
    frequency_df["frequency"] = frequency_df["frequency"]

    duration_df = duration_df.reset_index()
    second_passage_time_df = second_passage_time_df.reset_index()
    frequency_df = frequency_df.reset_index()
    return duration_df, second_passage_time_df, frequency_df


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


def calculate_duration_second_passage_time(
    data: pd.DataFrame,
    resolution: float,
    model: hmmlearn.hmm.GaussianHMM,
    fraction_nan_max: float = 0.1,
    gt: bool = False,
    full: bool = True,
):
    """Calculate duration and second passage time of contact and loss of contact.

    Args:
        data: dataframe containing the distance between two channels across all matched tracks.
        resolution: time resolution of the data.
        model: hmm model used to calculate the duration and second passage time.
        fraction_nan_max: maximum fraction of nan allowed in the data.
        gt: if true, it returns the ground truth duration and second passage time.
        full: if true, it returns also the boundary affected duration and second passage time
    """

    durations = pd.DataFrame()
    second_passage_times = pd.DataFrame()
    frequencies = pd.DataFrame()
    fraction_time = []
    conditions = []

    original = pd.DataFrame()
    length = []
    try:
        for _, sub in data.groupby("uniqueid"):
            sub, c = fill_gaps(sub, "frame")
            if c / len(sub) < fraction_nan_max:
                original = pd.concat([original, sub])
                length.append(len(sub))
    except:
        original = data.copy()

    data = original.copy()
    means = model.means_
    covars = model.covars_
    data["prediction"] = -1
    # inference and calculation of contact duration and second passage time
    for condition, df in data.groupby("condition"):
        av = []
        d = df.distance.values.reshape(-1, 1)
        nstates = 2
        model = hmm.GaussianHMM(
            n_components=nstates,
            covariance_type="full",
            min_covar=0.1,
            n_iter=10000,
            params="mtc",
            init_params="mtc",
            hack=True,
            hack_mean = means[0],
            hack_covar = covars.squeeze()[0]
        )

        # instead of fitting
        model.startprob_ = [1/nstates] * nstates
        model.fit(d)

        for uniqueid, sub in df.groupby("uniqueid"):
            distance = sub.distance.values.reshape(-1, 1)
            if not gt:
                states = model.predict(distance)
                states[states > 1] = 1
                data.loc[
                    (data["condition"] == condition) & (data["uniqueid"] == uniqueid),
                    "prediction",
                ] = states
                states = states[2:]  # remove starting condition
            else:
                states = sub.bond[2:]
            time = sub.frame.values[2:]  # remove starting condition
            df_tmp = pd.DataFrame({"state": states, "frame": time})
            df_tmp["uniqueid"] = uniqueid
            df_tmp["condition"] = condition

            av.append(df_tmp.state)

            (
                duration,
                second_passage_time,
                frequency,
            ) = contact_duration_second_passage_time_inclusive(
                df=df_tmp,
                resolution=resolution,
                contact_cutoff=0.5,
                distance="state",
                full=full,
            )

            durations = pd.concat([durations, duration])
            second_passage_times = pd.concat(
                [second_passage_times, second_passage_time]
            )
            frequencies = pd.concat([frequencies, frequency])
        fraction_time.append(np.mean(np.concatenate(av)))
        conditions.append(condition)
    durations[["cell_line", "induction_time"]] = durations["condition"].str.split(
        "_", expand=True
    )

    second_passage_times[["cell_line", "induction_time"]] = second_passage_times[
        "condition"
    ].str.split("_", expand=True)

    frequencies[["cell_line", "induction_time"]] = frequencies["condition"].str.split(
        "_", expand=True
    )

    return durations, second_passage_times, frequencies, fraction_time, conditions, data


def plot(df, x, y, title):
    fig = plt.figure()
    box_plot = sns.boxplot(data=df, x=x, y=y)

    ax = box_plot.axes
    lines = ax.get_lines()
    categories = ax.get_xticks()

    for cat in categories:
        y = round(lines[4 + cat * 6].get_ydata()[0], 1)

        ax.text(
            cat,
            y,
            f"{y}",
            ha="center",
            va="center",
            fontweight="bold",
            size=10,
            color="white",
            bbox=dict(facecolor="#445A64"),
        )

    box_plot.figure.tight_layout()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.title(title)
    plt.show()


def reorder_hmm_model_parameters(model):
    newmodel = model
    ordering = np.argsort(model.means_.squeeze())
    newmodel.means_ = model.means_[ordering]
    newmodel.covars_ = model.covars_[ordering]
    newmodel.transmat_ = model.transmat_[ordering, :][:, ordering]
    return newmodel

def confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return h

def calc_loops_gaps_total_len_gt(df, rev=True):
    # 1 - unlooped
    # 0 - looped
    arr=df.bond.values
    if reversed:
        arr=1-arr
    loops = []
    gaps = []
    streak = 1
    
    for i in range(len(arr)-1):
        if arr[i+1] - arr[i] == 1:
            loops.append(streak)
            streak = 1
        elif arr[i+1] - arr[i] == -1:
            gaps.append(streak)
            streak = 1
        elif arr[i+1] - arr[i] == 0:
            streak += 1
        if i == len(arr)-2:
            if arr[i+1] > 0:
                gaps.append(streak)
            else:
                gaps.append(streak)
    return loops, gaps
