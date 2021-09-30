import os
import seaborn as sns
import matplotlib.pyplot as plt

from .utils import *
from .dictionary import LIST_SAMPLES_MSD, SYSTEMATIC_ERRORS


def visualize_msd():
    # Set title
    st.title("Mean square displacement analysis")

    # Samples
    list_samples = LIST_SAMPLES_MSD
    systematic_errors = SYSTEMATIC_ERRORS

    # Take input from user and load file and make a copy
    sample_name = st.sidebar.selectbox(
        "Enter the input file containing the tamsd data.",
        list(list_samples.keys()),
    )

    #  Set systematic error from global options
    systematic_error = systematic_errors[sample_name]

    # download data if not already present
    if not os.path.isfile(f"{sample_name}.csv.zip"):
        sample_link = list_samples[sample_name]
        gdown.download(sample_link, f"{sample_name}.csv.zip")

    # Take input from user for time resultion of acquisition
    interval = float(st.sidebar.text_input("Acquisition time resolution", "10"))

    # load and make a copy of original data
    original_data = load_data(f"{sample_name}.csv.zip", interval)
    data = original_data.copy()

    # Select the upper limit for trustable data
    limit = float(
        st.sidebar.slider(
            "Until where you trust the data (in second)?",
            min_value=0.0,
            max_value=2000.0,
            value=500.0,
            step=interval,
        )
    )

    # Select cell lines, induction time and correction type to show
    clines = list(data["cell_line"].unique())
    clines.append("All")
    cell_lines = st.sidebar.multiselect(
        "Choose your cell lines (multiple)", clines, default=clines[0]
    )

    # Filter data to keep only the selected lines, induction time and correction type
    data = data[(data["lags"] <= limit)]
    if not "All" in cell_lines:
        data = data[data["cell_line"].isin(cell_lines)]

    induction_time = st.sidebar.multiselect(
        "Choose the induction times to keep",
        list(data["induction_time"].unique()),
        default=list(data["induction_time"].unique()),
    )

    data = data[data["induction_time"].isin(induction_time)]

    correction_type = st.sidebar.multiselect(
        "Choose the motion correction type",
        list(data["motion_correction_type"].unique()),
    )
    data = data[data["motion_correction_type"].isin(correction_type)]

    # Select type of systematic error correction correction to perform
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
    standard_deviation = st.checkbox(
        "Plot standard deviation instead of 68 confidence interval"
    )
    yaxis = st.checkbox("Fixed y axis values to [0.01:2]")
    laura = st.checkbox("Pool cells from same date (Laura data)")

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
                data["cell_line"],
                data["induction_time"],
                data["motion_correction_type"],
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

    if laura:
        try:
            data["condition"] = [
                f"{d}_{cl}_time{time}_{correction}"
                for d, cl, time, correction in zip(
                    data["rep"],
                    data["cell_line"],
                    data["induction_time"],
                    data["motion_correction_type"],
                )
            ]
        except:
            pass

    # Plot
    fig = plt.figure()
    if standard_deviation:
        ax = sns.lineplot(
            data=data,
            x="lags",
            y="tamsd",
            hue="condition",
            err_style="bars",
            ci="sd",
        )
    else:
        ax = sns.lineplot(
            data=data, x="lags", y="tamsd", hue="condition", err_style="bars", ci=68
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\Delta$t, sec")
    plt.ylabel(r"EA-tamsd, $\mu$m$^2$)")
    ax.legend(fontsize=5)

    if yaxis:
        plt.ylim(0.01, 2)

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
    st.subheader("Table of alphas and Ds")
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

    if st.checkbox("Use all data instead of average to fit"):
        for condition in data["condition"].unique():
            subset = data[data["condition"] == condition]
            res = fit_alpha_d(subset, end1, end2)
            res["condition"] = condition
            df = pd.concat([df, res])
    else:
        for condition in df_alphas["condition"].unique():
            subset = df_alphas[df_alphas["condition"] == condition]
            res = fit_alpha_d(subset, end1, end2)
            res["condition"] = condition
            df = pd.concat([df, res])

    # Display and download link of table
    st.dataframe(df)
    st.markdown(
        download_csv(df, "alpha_d.csv", "Download alphas and d table"),
        unsafe_allow_html=True,
    )
