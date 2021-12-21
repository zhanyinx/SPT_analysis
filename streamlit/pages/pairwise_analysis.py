import os
import seaborn as sns
import matplotlib.pyplot as plt


from .utils import *
from .dictionary import (
    LIST_SAMPLES_PAIRWISE_DISTANCE,
    LIST_SAMPLES_MODEL,
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
    list_models = LIST_SAMPLES_MODEL

    # Take input from user and load file and make a copy
    sample_name = st.sidebar.selectbox(
        "Enter the input file containing the tamsd data.",
        list(list_samples.keys()),
    )

    # download data if not already present
    if not os.path.isfile(f"{sample_name}.csv.zip"):
        sample_link = list_samples[sample_name]
        gdown.download(sample_link, f"{sample_name}.csv.zip")

    if not os.path.isfile(f"{sample_name}.obj"):
        sample_link = list_models[sample_name]
        gdown.download(sample_link, f"{sample_name}.obj")

    # load and make a copy of original data
    original_data = load_data(f"{sample_name}.csv.zip")
    data = original_data.copy()

    model = load_model(f"{sample_name}.obj")

    # Take input from user for time resultion of acquisition
    interval = float(st.sidebar.text_input("Acquisition time resolution", "10"))
    max_nan_allowed = float(
        st.sidebar.text_input(
            "Maximum fraction of gaps for contact duration and second passage time calculation",
            "0.2",
        )
    )

    (
        original_durations,
        original_second_passage_times,
        fraction_time,
        conditions,
        data_filtered_original,
    ) = calculate_duration_second_passage_time(
        data=data, resolution=interval, model=model, fraction_nan_max=max_nan_allowed
    )

    duration = original_durations.copy()
    second_passage_time = original_second_passage_times.copy()
    data_filtered = data_filtered_original.copy()

    data["frame"] *= interval
    data_filtered["frame"] *= interval

    # User input: select cell lines
    clines = list(data["cell_line"].unique())
    clines.append("All")
    cell_lines = st.sidebar.multiselect(
        "Choose your cell lines (multiple)", clines, default=["All"]
    )

    # Filter data to keep only the selected lines, induction time and correction type
    if not "All" in cell_lines:
        data = data[data["cell_line"].isin(cell_lines)]
        data_filtered = data_filtered[data_filtered["cell_line"].isin(cell_lines)]
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
    data_filtered = data_filtered[data_filtered["induction_time"].isin(induction_time)]
    duration = duration[duration["induction_time"].isin(induction_time)]
    second_passage_time = second_passage_time[
        second_passage_time["induction_time"].isin(induction_time)
    ]

    if st.sidebar.checkbox("Split by date", value=False):
        data["condition"] = data["date"].astype(str) + data["condition"]

    # Plotting
    st.subheader("Distribution of radial distances across all selected movies")
    col1, col2 = st.columns(2)
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
            plt.hist(
                (sub["distance"]), density=True, alpha=0.5, bins=np.arange(0, 10, 0.01)
            )
        legend.append(name)
    if st.checkbox("Manually set y axis"):
        ymax = float(st.text_input("y-axis max", "0.25"))
        plt.ylim(0, ymax)
    if st.checkbox("Check to have y axis in log"):
        plt.yscale("log")
    plt.legend(legend)
    plt.xlabel("Distances (um)")
    plt.ylabel("Density")
    plt.xlim(0, 2)
    col1.pyplot(fig)
    plt.show()
    plt.savefig("histogram.pdf")

    # Dowload options
    st.markdown(
        download_plot(
            "histogram.pdf",
            "Download histogram plot",
        ),
        unsafe_allow_html=True,
    )

    fig = plt.figure()
    ax = sns.ecdfplot(data, x="distance", hue="condition")
    plt.xlabel("Radial distance (um)")
    plt.ylabel("ECDF")
    col2.pyplot(fig)
    plt.show()
    plt.savefig("ecdf.pdf")
    st.markdown(
        download_plot(
            "ecdf.pdf",
            "Download ecdf plot",
        ),
        unsafe_allow_html=True,
    )

    st.header("Contact duration and second passage time across ngap, contact radius")

    options = data_filtered.uniqueid.unique()

    isfixed = st.checkbox("Check to fix y-axis.", value=True)
    if isfixed:
        ymax = st.number_input("Max y-axis", value=1.5)
    fig = plt.figure()
    select = st.number_input(
        f"Choose an example trajectory index. Max value {len(options) -1}",
        value=0,
        step=1,
    )
    sub = data_filtered[data_filtered["uniqueid"] == options[select]].copy()
    distance = sub.distance.values.reshape(-1, 1)
    states = model.predict(distance)

    st.text(
        f"Info of selected track:\n Movie: {sub.filename.unique()} \n cellid: {sub.cell.unique()} \n track: {sub.track.unique()}"
    )

    plt.errorbar(x=sub["frame"], y=sub["distance"], yerr=sub["sigma_d"])
    plt.plot(sub["frame"], states)

    plt.xlabel("Time (seconds)")
    plt.ylabel("Radial distance (um)")
    plt.title(sub.condition.values[0])
    if isfixed:
        plt.ylim(-0.1, ymax)
    st.pyplot(fig)
    plt.show()
    plt.savefig("single_track.pdf")
    st.markdown(
        download_plot(
            "single_track.pdf",
            "Download single track plot",
        ),
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    a = pd.DataFrame(data_filtered.groupby("condition")["uniqueid"].nunique())
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(a.index, a.uniqueid.values)
    ax.set_xticklabels(a.index, rotation=90)
    ax.set_ylabel("Number of tracks used")
    col1.pyplot(fig)
    plt.show()
    plt.savefig("number_tracks_used.pdf")
    st.markdown(
        download_plot(
            "number_tracks_used.pdf",
            "Download number of tracks plot",
        ),
        unsafe_allow_html=True,
    )

    # Fraction of time spent in looped state
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(conditions, [1 - x for x in fraction_time])
    ax.set_ylabel("Fraction of time at state looped")
    ax.set_xticklabels(conditions, rotation=90)
    plt.ylim(0, 1)
    col2.pyplot(fig)
    plt.show()
    plt.savefig("loop_time.pdf")
    st.markdown(
        download_plot(
            "loop_time.pdf",
            "Download looped time plot",
        ),
        unsafe_allow_html=True,
    )

    # Contact duration
    fig = plt.figure()
    box_plot = sns.boxplot(data=duration, x="condition", y="contact_duration")

    ax = box_plot.axes
    lines = ax.get_lines()
    categories = ax.get_xticks()

    for cat in categories:
        y = round(lines[4 + cat * 6].get_ydata()[0], 1)

        ax.text(
            cat,
            y,
            f"{y}",
            ha="center",
            va="center",
            fontweight="bold",
            size=10,
            color="white",
            bbox=dict(facecolor="#445A64"),
        )

    box_plot.figure.tight_layout()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    col1.pyplot(fig)
    plt.show()
    plt.savefig("contact_duration.pdf")
    st.markdown(
        download_plot(
            "contact_duration.pdf",
            "Download contact duration plot",
        ),
        unsafe_allow_html=True,
    )

    # ecdf = sns.ecdfplot(data=durations, x="contact_duration", hue="condition")

    # second passage time
    fig = plt.figure()
    box_plot = sns.boxplot(
        data=second_passage_time, x="condition", y="second_passage_time"
    )

    ax = box_plot.axes
    lines = ax.get_lines()
    categories = ax.get_xticks()

    for cat in categories:
        y = round(lines[4 + cat * 6].get_ydata()[0], 1)

        ax.text(
            cat,
            y,
            f"{y}",
            ha="center",
            va="center",
            fontweight="bold",
            size=10,
            color="white",
            bbox=dict(facecolor="#445A64"),
        )

    box_plot.figure.tight_layout()
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

    col2.pyplot(fig)
    plt.show()
    plt.savefig("second_passage_time.pdf")
    st.markdown(
        download_plot(
            "second_passage_time.pdf",
            "Download second passage time plot",
        ),
        unsafe_allow_html=True,
    )

    # ecdf = sns.ecdfplot(
    #     data=second_passage_time, x="second_passage_time", hue="condition"
    # )
