import base64
import io

import streamlit as st
import gdown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# load data and cache it
@st.cache
def load_data(data: str, interval: float = None):
    """Load data from csv file using pandas."""
    df = pd.read_csv(data)
    if interval is not None:
        df["lags"] = df["lags"] * interval
    return df


# Fit alpha and diffusion coefficient given 3 regimes
def fit_alpha_d(subset: pd.DataFrame, end1: float, end2: float):
    """Fit the alpha and D under the 3 different regimes separated by end1 and end2 values."""
    r1 = subset[subset["lags"] < end1].copy()
    r2 = subset[(subset["lags"] >= end1) & (subset["lags"] <= end2)].copy()
    r3 = subset[subset["lags"] > end2].copy()

    a1, d1 = np.polyfit(np.log10(r1["lags"]), np.log10(r1["tamsd"]), 1)
    a2, d2 = np.polyfit(np.log10(r2["lags"]), np.log10(r2["tamsd"]), 1)
    a3, d3 = np.polyfit(np.log10(r3["lags"]), np.log10(r3["tamsd"]), 1)

    minimum = np.min(r1["lags"])
    maximum = np.max(r3["lags"])
    regimes = [f"{minimum}-{end1}", f"{end1}-{end2}", f"{end2}-{maximum}"]

    df = pd.DataFrame(
        {
            "alphas": [a1, a2, a3],
            "Ds": [10 ** d1, 10 ** d2, 10 ** d3],
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
        data[data["cell_line"] == celline, "tamsd"] = (
            data[data["cell_line"] == celline]["tamsd"] - systematic_error
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


def rle(inarray):
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
        return (z, p, ia[i])


@st.cache
def contact_duration_second_passage_time(
    df: pd.DataFrame,
    resolution: float,
    contact_cutoff: float = 0.1,
    trackid: str = "uniqueid",
    distance: str = "distance",
    split: str = "condition",
):
    """Return DataFrame of contact duration and second passage time across all matched tracks within the provided DataFrame.

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
            length, position, types = rle(sub[distance] < contact_cutoff)
            duration.append(length[np.where(types == True)])
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
    return duration_df, second_passage_time_df
