import base64
import io

import gdown
import hmmlearn
from hmmlearn import hmm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import streamlit as st

# load data and cache it
@st.cache
def load_data(data: str, interval: float = None):
    """Load data from csv file using pandas."""
    df = pd.read_csv(data)
    if interval is not None:
        df["lags"] = df["lags"] * interval
    return df


@st.cache
def load_model(name: str):
    """Load model from pickle."""
    with open(name, "rb") as f:
        model = pickle.load(f)
    return model


def myround(x):
    return np.around(x, 6)


def fit_alpha_d(subset: pd.DataFrame, end1: float, end2: float):
    """Fit the alpha and D under the 3 different regimes separated by end1 and end2 values."""
    subset["loglags"] = np.log10(subset["lags"].values)
    subset["logtamsd"] = np.log10(subset["tamsd"].values)

    subset = subset.dropna()
    r1 = subset[subset["lags"] < end1].copy()
    r2 = subset[(subset["lags"] >= end1) & (subset["lags"] <= end2)].copy()
    r3 = subset[subset["lags"] > end2].copy()

    (a1, d1), cov1 = np.polyfit((r1["loglags"]), (r1["logtamsd"]), 1, cov=True)
    sda1, sdd1 = np.sqrt(np.diag(cov1))

    (a2, d2), cov2 = np.polyfit((r2["loglags"]), (r2["logtamsd"]), 1, cov=True)
    sda2, sdd2 = np.sqrt(np.diag(cov2))

    (a3, d3), cov3 = np.polyfit((r3["loglags"]), (r3["logtamsd"]), 1, cov=True)
    sda3, sdd3 = np.sqrt(np.diag(cov3))

    minimum = np.min(r1["lags"])
    maximum = np.max(r3["lags"])
    regimes = [f"{minimum}-{end1}", f"{end1}-{end2}", f"{end2}-{maximum}"]
    alphas = [
        f"{myround(a1)} +/- {myround(sda1)}",
        f"{myround(a2)} +/- {myround(sda2)}",
        f"{myround(a3)} +/- {myround(sda3)}",
    ]
    ds = [
        f"{myround(10**d1)} +/- {myround(sdd1 * np.abs(10**d1) * np.log(10))}",
        f"{myround(10**d2)} +/- {myround(sdd2 * np.abs(10**d2) * np.log(10))}",
        f"{myround(10**d3)} +/- {myround(sdd3 * np.abs(10**d3) * np.log(10))}",
    ]
    df = pd.DataFrame(
        {
            "alphas": alphas,
            "Ds": ds,
            "Regimes": regimes,
        }
    )
    return df


def sample_specific_systematic_error(data: pd.DataFrame):
    """Subtract sample specific systematic error."""
    for celline in data["cell_line"].unique():
        subset = data[data["cell_line"] == celline].copy()
        try:
            systematic_error = round(
                subset[
                    (subset["induction_time"] == "fixed")
                    & (subset["motion_correction_type"] == "cellIDs_corrected")
                ]
                .groupby(["lags"])
                .mean()["tamsd"]
                .values[0],
                4,
            )
        except:
            systematic_error = round(
                subset[(subset["induction_time"] == "fixed")]
                .groupby(["lags"])
                .mean()["tamsd"]
                .values[0],
                4,
            )
        data.loc[data["cell_line"] == celline, "tamsd"] = data["tamsd"].apply(
            lambda x: x - systematic_error
        )
    data = data[~(data["induction_time"] == "fixed")]
    return data


def download_plot(download_filename: str, download_link_text: str) -> str:
    """Generates a link to download a plot.

    Args:
        download_filename: filename and extension of file. e.g. myplot.pdf
        download_link_text: Text to display for download link.
    """
    file = io.BytesIO()
    plt.savefig(file, format="pdf")
    file = base64.b64encode(file.getvalue()).decode("utf-8").replace("\n", "")

    return f'<a href="data:application/pdf;base64,{file}" download="{download_filename}">{download_link_text}</a>'


def download_csv(
    df: pd.DataFrame, download_filename: str, download_link_text: str
) -> str:
    """Generates link to download csv of DataFrame.

    Args:
        df: DataFrame to download.
        download_filename: filename and extension of file. e.g. myplot.pdf
        download_link_text: Text to display for download link.
    """
    csv = df.to_csv().encode()
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{download_filename}" target="_blank">{download_link_text}</a>'
    return href


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


@st.cache
def calculate_duration_second_passage_time(
    data: pd.DataFrame, resolution: float, model: hmmlearn.hmm.GaussianHMM
):
    """Calculate duration and second passage time of contact and loss of contact.

    Args:
        data: dataframe containing the distance between two channels across all matched tracks.
        resolution: time resolution of the data.
        model: hmm model used to calculate the duration and second passage time.
    """

    durations = pd.DataFrame()
    second_passage_times = pd.DataFrame()
    fraction_time = []
    conditions = []

    # inference and calculation of contact duration and second passage time
    for condition, df in data.groupby("condition"):
        av = []
        for uniqueid, sub in df.groupby("uniqueid"):
            distance = sub.distance.values.reshape(-1, 1)
            states = (model.predict(distance))[2:]  # remove starting condition
            time = sub.frame.values[2:]  # remove starting condition
            df_tmp = pd.DataFrame({"state": states, "frame": time})
            df_tmp["uniqueid"] = uniqueid
            df_tmp["condition"] = condition

            av.append(df_tmp.state)

            (
                duration,
                second_passage_time,
            ) = contact_duration_second_passage_time_inclusive(
                df=df_tmp,
                resolution=resolution,
                contact_cutoff=0.5,
                distance="state",
                full=True,
            )

            durations = pd.concat([durations, duration])
            second_passage_times = pd.concat(
                [second_passage_times, second_passage_time]
            )

        fraction_time.append(np.mean(np.concatenate(av)))
        conditions.append(condition)

    durations[["cell_line", "induction_time"]] = durations["condition"].str.split(
        "_", expand=True
    )

    second_passage_times[["cell_line", "induction_time"]] = second_passage_times[
        "condition"
    ].str.split("_", expand=True)

    return durations, second_passage_times, fraction_time, conditions
