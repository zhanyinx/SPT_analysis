import os
import seaborn as sns
import matplotlib.pyplot as plt


from .utils import *
from .dictionary import LIST_SAMPLES_PAIRWISE


def pairwise_analysis():

    # Set title
    st.title("Mean square displacement analysis")

    # Samples
    list_samples = LIST_SAMPLES_PAIRWISE

    # Take input from user and load file and make a copy
    sample_name = st.sidebar.selectbox(
        "Enter the input file containing the tamsd data.",
        list(list_samples.keys()),
    )

    # download data if not already present
    if not os.path.isfile(f"{sample_name}.csv.zip"):
        sample_link = list_samples[sample_name]
        gdown.download(sample_link, f"{sample_name}.csv.zip")

    # Take input from user for time resultion of acquisition
    interval = float(st.sidebar.text_input("Acquisition time resolution", "10"))

    # load and make a copy of original data
    original_data = load_data(f"{sample_name}.csv.zip")
    data = original_data.copy()
    data["frame"] *= interval

    # extract cell lines
    clines = list(data["cell_line"].unique())
    clines.append("All")
    cell_lines = st.sidebar.multiselect(
        "Choose your cell lines (multiple)", clines, default=["All"]
    )

    # Filter data to keep only the selected lines, induction time and correction type
    if not "All" in cell_lines:
        data = data[data["cell_line"].isin(cell_lines)]

    induction_time = st.sidebar.multiselect(
        "Choose the induction times to keep",
        list(data["induction_time"].unique()),
        default=list(data["induction_time"].unique()),
    )
    data = data[data["induction_time"].isin(induction_time)]

    # Take input from user for time resultion of acquisition
    contact_cutoff = float(st.sidebar.text_input("contact cutoff (um)", "0.1"))
    ## calculate contact duration and second passage time
    duration, second_passage_time = contact_duration_second_passage_time(
        df=data,
        resolution=interval,
        contact_cutoff=contact_cutoff,
        trackid="uniqueid",
        distance="distance",
        split="condition",
    )

    # Plotting
    st.subheader("Radial distance and ecdf plots")
    options = data.uniqueid.unique()
    select = st.slider(
        "Choose an example trajectory index",
        min_value=0,
        max_value=len(options) - 1,
        value=500,
        step=1,
    )
    col1, col2 = st.columns(2)
    fig = plt.figure()
    sub = data[data["uniqueid"] == options[select]].copy()
    plt.errorbar(x=sub["frame"], y=sub["distance"], yerr=sub["sigma_d"])
    plt.xlabel("Time (seconds)")
    plt.ylabel("Radial distance (um)")
    col1.pyplot(fig)
    plt.show()
    plt.savefig("plot.pdf")
    st.markdown(
        download_plot(
            "plot.pdf",
            "Download left plot",
        ),
        unsafe_allow_html=True,
    )

    fig = plt.figure()
    ax = sns.ecdfplot(data, x="distance", hue="condition")
    plt.xlabel("Radial distance (um)")
    plt.ylabel("ECDF")
    col2.pyplot(fig)
    plt.show()
    plt.savefig("plot.pdf")
    st.markdown(
        download_plot(
            "plot.pdf",
            "Download right plot",
        ),
        unsafe_allow_html=True,
    )

    st.subheader("Contact duration and first passage time")
    col1, col2 = st.columns(2)
    fig = plt.figure()
    legend = []
    maximum = np.max(duration["contact_duration"])
    bins = np.arange(0, maximum, interval)
    for name, sub in duration.groupby("condition"):
        plt.hist(sub["contact_duration"], alpha=0.5, density=True, bins=bins)
        legend.append(name)
    plt.xlim(0, maximum + 1)
    plt.legend(legend)
    plt.xlabel("Contact duration")
    plt.ylabel("Density")
    col1.pyplot(fig)
    plt.show()
    plt.savefig("plot.pdf")
    st.markdown(
        download_plot(
            "plot.pdf",
            "Download right plot",
        ),
        unsafe_allow_html=True,
    )

    fig = plt.figure()
    legend = []
    maximum = np.max(second_passage_time["second_passage_time"])
    for name, sub in second_passage_time.groupby("condition"):
        plt.hist(sub["second_passage_time"], alpha=0.5, density=True)
        legend.append(name)
    plt.xlim(0, maximum + 1)
    plt.legend(legend)
    plt.xlabel("Second passage time")
    plt.ylabel("Density")
    col2.pyplot(fig)
    plt.show()
    plt.savefig("plot.pdf")
    st.markdown(
        download_plot(
            "plot.pdf",
            "Download right plot",
        ),
        unsafe_allow_html=True,
    )
