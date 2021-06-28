import streamlit as st

# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt


st.title("MSD visualisation app")


@st.cache
def load_data(data: str):
    """Load data from csv file using pandas."""
    if not os.path.isfile(data):
        raise ValueError(f"{data} file does not exist.")

    df = pd.read_csv(data)
    return df


filename = st.text_input("Enter the input file containing the data", "./output.csv")
interval = int(st.text_input("Time step acquisition"))
limit = int(st.text_input("Until where you trust the data?"))


original_data = load_data(filename)
data = original_data.copy()

cell_lines = st.multiselect(
    "Choose your cell lines (multiple)", data["cell_line"].unique()
)
induction_time = st.multiselect(
    "Choose the induction times to keep", data["induction_time"].unique()
)

data["lags"] = data["lags"] * interval
data = data[(data["lags"] < limit)]
data = data[data["cell_line"].isin(cell_lines)]
data = data[data["induction_time"].isin(induction_time)]


data["condition"] = [
    f"cl_{cl}_rep{rep}_time{time}"
    for cl, rep, time in zip(data["cell_line"], data["rep"], data["induction_time"])
]

if st.checkbox("Show raw data"):
    st.subheader("Raw data")
    datahead = data.head()
    st.write(datahead)

fig = plt.figure()

sns.lineplot(data=data, x="lags", y="tamsd", hue="condition", err_style="bars", ci=68)
st.pyplot(fig)
