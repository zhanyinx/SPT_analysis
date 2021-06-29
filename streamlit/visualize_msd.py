import streamlit as st

# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# Set title
st.title("MSD visualisation app")

# load data and cache it
@st.cache
def load_data(data: str):
    """Load data from csv file using pandas."""
    if not os.path.isfile(data):
        raise ValueError(f"{data} file does not exist.")

    df = pd.read_csv(data)
    return df


# Take input from user and load file and make a copy
filename = st.sidebar.text_input(
    "Enter the input file containing the tamsd data.", "rad21_with_gaps.csv"
)
original_data = load_data(filename)
data = original_data.copy()

# Take input from user for time resultion of acquisition
interval = int(st.sidebar.text_input("Acquisition time resolution", "10"))
data["lags"] = data["lags"] * interval

# Select the upper limit for trustable data
limit = int(
    st.sidebar.slider(
        "Until where you trust the data (in second)?",
        min_value=0,
        max_value=max(data["lags"]),
        value=500,
        step=interval,
    )
)

# Select cell lines and induction time to show
cell_lines = st.sidebar.multiselect(
    "Choose your cell lines (multiple)", data["cell_line"].unique()
)
induction_time = st.sidebar.multiselect(
    "Choose the induction times to keep", data["induction_time"].unique()
)

# Filter data to keep only the selected lines and induction time
data = data[(data["lags"] < limit)]
data = data[data["cell_line"].isin(cell_lines)]
data = data[data["induction_time"].isin(induction_time)]


# Options for plot
pool_replicates = st.sidebar.checkbox("Pool replicates")
if pool_replicates:
    data["condition"] = [
        f"{cl}_itime{time}"
        for cl, time in zip(data["cell_line"], data["induction_time"])
    ]
else:
    data["condition"] = [
        f"{d}_{cl}_time{time}"
        for cl, d, time in zip(data["cell_line"], data["date"], data["induction_time"])
    ]

# Plot
fig = plt.figure()
if st.checkbox("Plot standard deviation instead of 68 confidence interval"):
    sns.lineplot(
        data=data, x="lags", y="tamsd", hue="condition", err_style="bars", ci="sd"
    )
else:
    sns.lineplot(
        data=data, x="lags", y="tamsd", hue="condition", err_style="bars", ci=68
    )
plt.xscale("log")
plt.yscale("log")
plt.xlabel("dt (sec)")
plt.ylabel("EA-tamsd (um^2)")
plt.ylim(0.01, 2)

st.pyplot(fig)

# Create table of alphas and Ds
