import os
import seaborn as sns
import matplotlib.pyplot as plt

from utils import *


# Set title
st.title("MSD visualisation app")

# Samples
list_samples = {
    "rad21_gaps": "https://drive.google.com/uc?export=download&id=1EzPK_nxJ3CmBA_Sw_21zX0J2q7cMUOdK"
}

# Take input from user and load file and make a copy
sample_name = st.sidebar.selectbox(
    "Enter the input file containing the tamsd data.",
    list(list_samples.keys()),
)

if not os.path.isfile(f"{sample_name}.csv.zip"):
    sample_link = list_samples[sample_name]
    gdown.download(sample_link, f"{sample_name}.csv.zip")


original_data = load_data(f"{sample_name}.csv.zip")
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
data = data[(data["lags"] <= limit)]
data = data[data["cell_line"].isin(cell_lines)]
data = data[data["induction_time"].isin(induction_time)]

systematic_error = st.sidebar.number_input(
    "Systematic error (rho) from fixed cells. 2*rho**2 will be subtracted",
    value=0.086,
    min_value=0.0,
    format="%.5f",
)

data["tamsd"] = data["tamsd"] - 2 * systematic_error ** 2

# Options for plot
pool_replicates = st.sidebar.checkbox("Pool replicates")
pool_clones_replicates = st.sidebar.checkbox("Pool clones and replicates")

if pool_clones_replicates:
    data["condition"] = data["induction_time"]
elif pool_replicates:
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
    st.dataframe(res)


# Create table of alphas and Ds
if st.checkbox("Show alpha and D?"):
    df_alphas = pd.DataFrame(data.groupby(["lags", "condition"]).mean()["tamsd"])
    df_alphas.reset_index(inplace=True)

    # Select the upper limit first range of fit
    end1 = int(
        st.number_input("End of first regime for fitting a and D", 60, step=interval)
    )

    # Select the upper limit second range of fit
    end2 = int(
        st.number_input("End of second regime for fitting a and D", 200, step=interval)
    )

    df = pd.DataFrame()
    for condition in df_alphas["condition"].unique():
        subset = df_alphas[df_alphas["condition"] == condition]
        res = fit_alpha_d(subset, end1, end2)
        res["condition"] = condition
        df = pd.concat([df, res])
    df["regime"] = df.index

    st.dataframe(df)
