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


@st.cache
def filter_data(df: pd.DataFrame, min_points: int):
    """Filter tracks with lower number of points"""
    df_filtered = pd.DataFrame()
    for _, sub in df.groupby("uniqueid"):
        if len(sub) > min_points:
            df_filtered = pd.concat([df_filtered, sub])
    return df_filtered


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
