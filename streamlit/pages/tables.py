import streamlit as st
import os
from .dictionary import LIST_SAMPLES_MSD, SYSTEMATIC_ERRORS
from .utils import *


def systematic_error_table(list_samples):
    """Calculate the systematic error given a list of samples (experiments)."""

    df = pd.DataFrame()
    # Loop over experiments and download the data if missing
    for sample_name, sample_link in list_samples.items():
        if not os.path.isfile(f"{sample_name}.csv.zip"):
            sample_link = list_samples[sample_name]
            gdown.download(sample_link, f"{sample_name}.csv.zip")

        # load and make a copy of original data
        try:
            original_data = load_data(f"{sample_name}.csv.zip")
        except:
            continue
        data = original_data.copy()

        tmp_df = pd.DataFrame()

        # take only fixed data
        data = data[data["induction_time"] == "fixed"]
        for celline in data["cell_line"].unique():
            subset = data[data["cell_line"] == celline].copy()

            # calculate systematic error
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

            # Keep systematic error in a temporary df
            tmp = pd.DataFrame(
                {
                    "cell_line": celline,
                    "systematic_error (2rho**2) (um2)": systematic_error,
                },
                index=[None],
            )
            tmp_df = pd.concat([tmp_df, tmp], ignore_index=True)

        tmp_df["sample_id"] = sample_name
        df = pd.concat([df, tmp_df], ignore_index=True)

    # Show systematic error in a table and provide link for download
    st.subheader("Systematic error table")
    st.dataframe(df)
    st.markdown(
        download_csv(df, "systematic_error.csv", "Download systematic errors table"),
        unsafe_allow_html=True,
    )


def rousetime_table(data, sample, time_resolution=0.1):
    """Calculate rouse time given the rouse time experimental data."""
    df = pd.DataFrame()

    # Subtract cell line specific specific experimental error
    if st.checkbox("Subtract cell line specific experimental error instead of global systematic error", False):
        data = sample_specific_systematic_error(data)
    else:
        if st.checkbox("Add fixed samples in the table", False):
            data = data.copy()
        else:
            data = data[~(data["induction_time"] == "fixed")].copy()

        systematic_error = SYSTEMATIC_ERRORS[sample]
        data["tamsd"] = data["tamsd"] - systematic_error

    # create experimental condition variable
    if st.checkbox("Check to merge cell lines at given induction time", True):
        data["condition"] = data["induction_time"]
    else:
        data["condition"] = [
            f"{cl}_itime{time}"
            for cl, time in zip(
                data["cell_line"],
                data["induction_time"],
            )
        ]

    # Loop over experimental condition
    for condition in data["condition"].unique():
        # calculate EATAmsd for each condition
        subset = data[data["condition"] == condition].copy()
        subset = pd.DataFrame(subset.groupby(["lags", "condition"]).mean()["tamsd"])
        subset.reset_index(inplace=True)
        subset.columns = ["delay", "condition", "EATAmsd"]

        # extract the delay with the displacement that is the closest to teto array size (0.015*sqrt(40))**2
        teto = (
            subset.iloc[(subset["EATAmsd"] - 0.009).abs().argsort()[:1]][
                "delay"
            ].values[0]
            * time_resolution
        )

        # extract the delay with the displacement that is the closest to laco array size (0.015*sqrt(30))**2
        laco = (
            subset.iloc[(subset["EATAmsd"] - 0.00675).abs().argsort()[:1]][
                "delay"
            ].values[0]
            * time_resolution
        )

        # save in temporary df
        tmp = pd.DataFrame(
            {
                "condition": [condition, condition],
                "rousetime (s)": [teto, laco],
                "teto or laco": ["teto", "laco"],
            }
        )
        df = pd.concat([df, tmp], ignore_index=True)

    # display and provide link for Rouse time table
    st.subheader("Rouse time table")
    st.dataframe(df)
    st.markdown(
        download_csv(df, "rousetime.csv", "Download rouse time table"),
        unsafe_allow_html=True,
    )


def tables():
    st.title("Systematic error table and rouse time estimation")

    # Samples
    list_samples = LIST_SAMPLES_MSD
    # if st.checkbox("Check to calculate Systematic error", False):
    #     systematic_error_table(list_samples)

    rousetime = {key: value for key, value in LIST_SAMPLES_MSD.items() if "rouse" in key}

    sample = st.selectbox("Select sample for rouse time", list(rousetime.keys()))
    # download data if not already present
    if not os.path.isfile(f"{sample}.csv.zip"):
        sample_link = list_samples[sample]
        gdown.download(sample_link, f"{sample}.csv.zip")
    original_data = load_data(f"{sample}.csv.zip")

    # Take input from user for time resultion of acquisition
    time_resolution = float(st.sidebar.text_input("Acquisition time resolution", "0.1"))

    rousetime_table(original_data.copy(), sample, time_resolution=time_resolution)
