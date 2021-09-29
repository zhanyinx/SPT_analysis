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
