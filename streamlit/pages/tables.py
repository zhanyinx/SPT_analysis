import streamlit as st
import os
from .dictionary import LIST_SAMPLES
from .utils import *


def tables():
    st.title("Systematic error table and rouse time estimation")

    # Samples
    list_samples = LIST_SAMPLES

    df = pd.DataFrame()
    for sample_name, sample_link in list_samples.items():
        if not os.path.isfile(f"{sample_name}.csv.zip"):
            sample_link = list_samples[sample_name]
            gdown.download(sample_link, f"{sample_name}.csv.zip")

        # load and make a copy of original data
        original_data = load_data(f"{sample_name}.csv.zip")
        data = original_data.copy()

        # take only fixed data
        tmp_df = pd.DataFrame()
        data = data[data["induction_time"] == "fixed"]
        for celline in data["cell_line"].unique():
            subset = data[data["cell_line"] == celline].copy()
            systematic_error = round(
                subset[subset["induction_time"] == "fixed"]
                .groupby(["lags"])
                .mean()["tamsd"]
                .values[0],
                4,
            )
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

    st.subheader("Systematic error table")
    st.dataframe(df)
    st.markdown(
        download_csv(df, "systematic_error.csv", "Download systematic errors table"),
        unsafe_allow_html=True,
    )
