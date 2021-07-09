import os
from pandas.core.groupby.generic import AggScalar
import seaborn as sns
import matplotlib.pyplot as plt

from utils import *


# Set title
st.title("MSD visualisation app")

# Samples
list_samples = {
    "rad21": "https://drive.google.com/uc?export=download&id=1UHfqftidv1MKOtnzhyXHNKVFQZPKIj2Z",
    "rousetime": "https://drive.google.com/uc?export=download&id=1aMj2c6TkIN3NfKmRzD6-bCjcUz424Mf8",
}

systematic_errors = {"rad21": 0.0087, "rousetime": 0.0025}


# Take input from user and load file and make a copy
sample_name = st.sidebar.selectbox(
    "Enter the input file containing the tamsd data.",
    list(list_samples.keys()),
)

systematic_error = systematic_errors[sample_name]

if not os.path.isfile(f"{sample_name}.csv.zip"):
    sample_link = list_samples[sample_name]
    gdown.download(sample_link, f"{sample_name}.csv.zip")


original_data = load_data(f"{sample_name}.csv.zip")
data = original_data.copy()

# Take input from user for time resultion of acquisition
interval = float(st.sidebar.text_input("Acquisition time resolution", "10"))
data["lags"] = data["lags"] * interval

# Select the upper limit for trustable data
limit = float(
    st.sidebar.slider(
        "Until where you trust the data (in second)?",
        min_value=0.0,
        max_value=max(data["lags"]),
        value=500.0,
        step=interval,
    )
)

# Select cell lines and induction time to show
clines = list(data["cell_line"].unique())
clines.append("All")
cell_lines = st.sidebar.multiselect("Choose your cell lines (multiple)", clines)
induction_time = st.sidebar.multiselect(
    "Choose the induction times to keep", list(data["induction_time"].unique())
)
correction_type = st.sidebar.multiselect(
    "Choose the motion correction type", list(data["motion_correction_type"].unique())
)

# Filter data to keep only the selected lines and induction time
data = data[(data["lags"] <= limit)]
if not "All" in cell_lines:
    data = data[data["cell_line"].isin(cell_lines)]
data = data[data["induction_time"].isin(induction_time)]
data = data[data["motion_correction_type"].isin(correction_type)]

avoid_se = st.sidebar.checkbox("Avoid systematic error correction")
sample_specific_se_correction = st.sidebar.checkbox(
    "Sample specific systematic error correction"
)

if sample_specific_se_correction:
    if "fixed" not in data["induction_time"].unique():
        raise ValueError(
            "Fixed data must be present to calculate sample specific Error!"
        )
    data = sample_specific_systematic_error(data)

elif not avoid_se:
    if "fixed" not in data["induction_time"].unique():
        data["tamsd"] = data["tamsd"] - systematic_error

# Options for plot
pool_clones_replicates = st.checkbox("Pool clones and replicates")
pool_replicates = st.checkbox("Pool replicates")


if pool_clones_replicates:
    data["condition"] = [
        f"itime{time}_{correction}"
        for time, correction in zip(
            data["induction_time"], data["motion_correction_type"]
        )
    ]
elif pool_replicates:
    data["condition"] = [
        f"{cl}_itime{time}_{correction}"
        for cl, time, correction in zip(
            data["cell_line"], data["induction_time"], data["motion_correction_type"]
        )
    ]
else:
    data["condition"] = [
        f"{d}_{cl}_time{time}_{correction}"
        for d, cl, time, correction in zip(
            data["date"],
            data["cell_line"],
            data["induction_time"],
            data["motion_correction_type"],
        )
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

# Show systematic error
if ("fixed" in induction_time) and ("fixed" in data["induction_time"].unique()):
    systematic_error_new = round(
        data[data["induction_time"] == "fixed"]
        .groupby(["lags"])
        .mean()["tamsd"]
        .values[0],
        4,
    )
    plt.title(f"Systematic error (from selected fixed samples): {systematic_error_new}")

if st.checkbox("Fixed y axis values to [0.01:2]"):
    plt.ylim(0.01, 2)

st.pyplot(fig)

plt.savefig("plot.pdf")
st.markdown(
    download_plot(
        "plot.pdf",
        "Download plot",
    ),
    unsafe_allow_html=True,
)

st.markdown(
    download_csv(data, "table.csv", "Download data used in the plot"),
    unsafe_allow_html=True,
)

if st.checkbox("Show raw data"):
    res = pd.DataFrame(data.groupby(["lags", "condition"]).mean()["tamsd"])
    res.reset_index(inplace=True)
    res.columns = ["delay", "condition", "EATAmsd"]
    st.dataframe(res)


# Create table of alphas and Ds
if st.checkbox("Show alpha and D?"):
    df_alphas = pd.DataFrame(data.groupby(["lags", "condition"]).mean()["tamsd"])
    df_alphas.reset_index(inplace=True)

    # Select the upper limit first range of fit
    end1 = float(
        st.number_input(
            "End of first regime for fitting a and D", value=60.0, step=interval
        )
    )

    # Select the upper limit second range of fit
    end2 = float(
        st.number_input(
            "End of second regime for fitting a and D", value=200.0, step=interval
        )
    )

    df = pd.DataFrame()
    for condition in df_alphas["condition"].unique():
        subset = df_alphas[df_alphas["condition"] == condition]
        res = fit_alpha_d(subset, end1, end2)
        res["condition"] = condition
        df = pd.concat([df, res])
    df["regime"] = df.index

    st.dataframe(df)
    st.markdown(
        download_csv(df, "alpha_d.csv", "Download alphas and d table"),
        unsafe_allow_html=True,
    )
