import os
import seaborn as sns
import matplotlib.pyplot as plt


from .utils import *
from .dictionary import (
    LIST_SAMPLES_PAIRWISE_DISTANCE,
    LIST_SAMPLES_DURATION,
    LIST_SAMPLES_SECOND_PASSAGE_TIME,
)


def pairwise_analysis():

    # Set title
    st.title("Pairwise distance analysis")
    st.write(
        "Analysis is done on tracks with at least 25 timepoints. \
        Noisy timepoints are filtered out. Noisy is defined as top 5% of \
        ratio between distances in 3D/2D, where 2D distance is on xy only."
    )

    # Samples
    list_samples = LIST_SAMPLES_PAIRWISE_DISTANCE
    list_durations = LIST_SAMPLES_DURATION
    list_second_passage_time = LIST_SAMPLES_SECOND_PASSAGE_TIME

    # Take input from user and load file and make a copy
    sample_name = st.sidebar.selectbox(
        "Enter the input file containing the tamsd data.",
        list(list_samples.keys()),
    )

    # download data if not already present
    if not os.path.isfile(f"{sample_name}.csv.zip"):
        sample_link = list_samples[sample_name]
        gdown.download(sample_link, f"{sample_name}.csv.zip")

    if not os.path.isfile(f"{sample_name}_duration.csv.zip"):
        sample_link = list_durations[sample_name]
        gdown.download(sample_link, f"{sample_name}_duration.csv.zip")

    if not os.path.isfile(f"{sample_name}_second_passage_time.csv.zip"):
        sample_link = list_second_passage_time[sample_name]
        gdown.download(sample_link, f"{sample_name}_second_passage_time.csv.zip")

    # load and make a copy of original data
    original_data = load_data(f"{sample_name}.csv.zip")
    data = original_data.copy()

    # load and make a copy of original data
    original_duration = load_data(f"{sample_name}_duration.csv.zip")
    duration = original_duration.copy()
    duration[["cell_line", "induction_time"]] = duration["condition"].str.split(
        "_", expand=True
    )

    # load and make a copy of original data
    original_second_passage_time = load_data(
        f"{sample_name}_second_passage_time.csv.zip"
    )
    second_passage_time = original_second_passage_time.copy()
    second_passage_time[["cell_line", "induction_time"]] = second_passage_time[
        "condition"
    ].str.split("_", expand=True)

    # Take input from user for time resultion of acquisition
    interval = float(st.sidebar.text_input("Acquisition time resolution", "10"))

    # User input: select cell lines
    clines = list(data["cell_line"].unique())
    clines.append("All")
    cell_lines = st.sidebar.multiselect(
        "Choose your cell lines (multiple)", clines, default=["All"]
    )

    # Filter data to keep only the selected lines, induction time and correction type
    if not "All" in cell_lines:
        data = data[data["cell_line"].isin(cell_lines)]
        duration = duration[duration["cell_line"].isin(cell_lines)]
        second_passage_time = second_passage_time[
            second_passage_time["cell_line"].isin(cell_lines)
        ]

    induction_time = st.sidebar.multiselect(
        "Choose the induction times to keep",
        list(data["induction_time"].unique()),
        default=list(data["induction_time"].unique()),
    )
    data = data[data["induction_time"].isin(induction_time)]
    duration = duration[duration["induction_time"].isin(induction_time)]
    second_passage_time = second_passage_time[
        second_passage_time["induction_time"].isin(induction_time)
    ]

    # Plotting
    st.subheader("Distribution of radial distances across all selected movies")
    fig = plt.figure()
    legend = []
    custombins = st.checkbox("Choose your own bins (format: bin1,bin2,bin3...)")
    if custombins:
        bins = st.text_input("Your own list of bins:", "0.5,1,1.5,2,2.5,3,3.5")
        bins = np.array(bins.split(",")).astype(float)
    for name, sub in data.groupby("condition"):
        if custombins:
            plt.hist((sub["distance"]), density=True, alpha=0.5, bins=bins)
        else:
            plt.hist((sub["distance"]), density=True, alpha=0.5)
        legend.append(name)
    if st.checkbox("Manually set y axis"):
        ymax = float(st.text_input("y-axis max", "0.25"))
        plt.ylim(0, ymax)
    if st.checkbox("Check to have y axis in log"):
        plt.yscale("log")
    plt.legend(legend)
    plt.xlabel("Distances (um)")
    plt.ylabel("Density")
    st.pyplot(fig)
    plt.show()
    plt.savefig("plot.pdf")

    # Dowload options
    st.markdown(
        download_plot(
            "plot.pdf",
            "Download plot",
        ),
        unsafe_allow_html=True,
    )

    st.subheader("Radial distance and ecdf plots")
    col1, col2 = st.columns(2)
    options = data.uniqueid.unique()
    select = st.number_input(
        f"Choose an example trajectory index. Max value {len(options) -1}",
        value=0,
        step=1,
    )
    isfixed = st.checkbox("Check to fix y-axis.", value=True)
    if isfixed:
        ymax = st.number_input("Max y-axis", value=1.5)
    fig = plt.figure()
    sub = data[data["uniqueid"] == options[select]].copy()
    st.text(
        f"Info of selected track:\n Movie: {sub.filename.unique()} \n cellid: {sub.cell.unique()} \n track: {sub.track.unique()}"
    )

    plt.errorbar(x=sub["frame"], y=sub["distance"], yerr=sub["sigma_d"])
    plt.xlabel("Time (seconds)")
    plt.ylabel("Radial distance (um)")
    plt.title(sub.condition.values[0])
    if isfixed:
        plt.ylim(0, ymax)
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

    st.subheader("Contact duration and second passage time across ngap, contact radius")
    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    for i in range(4):
        for j in range(2):
            cutoff = duration.contact_cutoff.unique()[i + j]
            duration_cutoff = pd.DataFrame(
                duration[duration.contact_cutoff == cutoff]
                .groupby(["condition", "ngap"])
                .mean()["contact_duration"]
            )
            duration_cutoff["condition"], duration_cutoff["ngap"] = zip(
                *duration_cutoff.index
            )

            sns.lineplot(
                x="ngap",
                y="contact_duration",
                hue="condition",
                data=duration_cutoff,
                ax=ax[j, i],
            )
            ax[j, i].set_xlabel("ngap allowed")
            ax[j, i].set_ylabel("Average contact duration")
            ax[j, i].set_title(f"distance cutoff {cutoff}")

    st.pyplot(fig)
    plt.show()
    plt.savefig("plot.pdf")

    # Dowload options
    st.markdown(
        download_plot(
            "plot.pdf",
            "Download plot",
        ),
        unsafe_allow_html=True,
    )

    fig, ax = plt.subplots(2, 4, figsize=(20, 10))
    for i in range(4):
        for j in range(2):
            cutoff = second_passage_time.contact_cutoff.unique()[i + j]
            second_passage_time_cutoff = pd.DataFrame(
                second_passage_time[second_passage_time.contact_cutoff == cutoff]
                .groupby(["condition", "ngap"])
                .mean()["second_passage_time"]
            )
            (
                second_passage_time_cutoff["condition"],
                second_passage_time_cutoff["ngap"],
            ) = zip(*second_passage_time_cutoff.index)

            sns.lineplot(
                x="ngap",
                y="second_passage_time",
                hue="condition",
                data=second_passage_time_cutoff,
                ax=ax[j, i],
            )
            ax[j, i].set_xlabel("ngap allowed")
            ax[j, i].set_ylabel("Average second passage time")
            ax[j, i].set_title(f"distance cutoff {cutoff}")

    st.pyplot(fig)
    plt.show()
    plt.savefig("plot.pdf")

    # Dowload options
    st.markdown(
        download_plot(
            "plot.pdf",
            "Download plot",
        ),
        unsafe_allow_html=True,
    )

    st.subheader(
        "Select specific contact cutoff and number of allowed gaps for the all plots that follows"
    )
    # User input: cutoff for contact
    contact_cutoffs = list(duration["contact_cutoff"].unique())
    list_ngaps = list(duration["ngap"].unique())
    contact_cutoff = float(
        st.selectbox("Specific contact cutoff (um)", contact_cutoffs, index=0)
    )

    ngaps = int(st.selectbox("Specific number of gaps allowed", list_ngaps, index=0))

    duration = duration[
        (duration["ngap"] == ngaps) & (duration["contact_cutoff"] == contact_cutoff)
    ]
    second_passage_time = second_passage_time[
        (second_passage_time["ngap"] == ngaps)
        & (second_passage_time["contact_cutoff"] == contact_cutoff)
    ]

    st.subheader("Contact duration histograms and ecdf")
    col1, col2 = st.columns(2)

    isfixed = st.checkbox("Check to fix x-axis.", value=True)
    if isfixed:
        xmax = st.number_input("Max x-axis", value=500.0)

    fig = plt.figure()
    legend = []
    maximum = np.max(duration["contact_duration"])
    bins = np.arange(0, maximum, interval)
    for name, sub in duration.groupby("condition"):
        plt.hist(sub["contact_duration"], alpha=0.5, density=True, bins=bins)
        legend.append(name)

    if isfixed:
        plt.xlim(0, xmax)
    else:
        plt.xlim(0, maximum)
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
    ax = sns.ecdfplot(duration, x="contact_duration", hue="condition")
    plt.xlabel("contact duration (seconds)")
    plt.ylabel("ECDF")
    if isfixed:
        plt.xlim(0, xmax)

    col2.pyplot(fig)
    plt.show()
    plt.savefig("plot.pdf")
    st.markdown(
        download_plot(
            "plot.pdf",
            "Download left plot",
        ),
        unsafe_allow_html=True,
    )

    st.subheader("First passage time histogram and ecdf")
    col1, col2 = st.columns(2)
    isfixed = st.checkbox("Check to fix x-axis of first passage time.", value=True)
    if isfixed:
        xmax = st.number_input("Max x-axis", value=1500.0)

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
    if isfixed:
        plt.xlim(0, xmax)
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
    ax = sns.ecdfplot(second_passage_time, x="second_passage_time", hue="condition")
    plt.xlabel("second passage time (seconds)")
    plt.ylabel("ECDF")
    if isfixed:
        plt.xlim(0, xmax)
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
