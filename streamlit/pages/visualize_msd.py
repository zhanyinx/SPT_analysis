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
            max_value=3000.0,
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
        default=[x for x in data.induction_time.unique() if x in ["0min", "90min"]],
    )

    data = data[data["induction_time"].isin(induction_time)]

    correction_type = st.sidebar.multiselect(
        "Choose the motion correction type",
        list(data["motion_correction_type"].unique()),
        default="cellIDs_corrected"
        if "cellIDs_corrected" in data["motion_correction_type"].unique()
        else data["motion_correction_type"].unique()[0],
    )
    data = data[data["motion_correction_type"].isin(correction_type)]

    # Select type of systematic error correction correction to perform
    avoid_se = st.sidebar.checkbox("Avoid systematic error correction")

    # Options for plot
    pool_clones_replicates = st.checkbox("Pool clones and replicates")
    pool_replicates = st.checkbox("Pool replicates", value=True)
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
    if avoid_se:
        if standard_deviation:
            # fig, ax = my_lineplot(df=data)
            ax = sns.lineplot(
                data=data,
                x="lags",
                y="tamsd",
                hue="condition",
                err_style="bars",
                estimator=lambda x: np.power(10, np.mean(np.log10(x))),
                ci="sd",
            )
        else:
            # fig, ax = my_lineplot(df=data, ste=True)
            ax = sns.lineplot(
                data=data,
                x="lags",
                y="tamsd",
                err_style="bars",
                estimator=lambda x: np.power(10, np.mean(np.log10(x))),
                hue="condition",
                ci=68,
            )
    else:
        if standard_deviation:
            ax = sns.lineplot(
                data=data,
                x="lags",
                y="tamsd",
                hue="condition",
                err_style="bars",
                estimator=lambda x: np.power(10, np.mean(np.log10(x)))
                - systematic_error,
                ci="sd",
            )
        else:
            ax = sns.lineplot(
                data=data,
                x="lags",
                y="tamsd",
                err_style="bars",
                estimator=lambda x: np.power(10, np.mean(np.log10(x)))
                - systematic_error,
                hue="condition",
                ci=68,
            )
    # x_vals = np.array(ax.get_xlim())
    # x_vals[x_vals < 30] = 30

    # y_vals = 0.006325 * x_vals ** 0.129346
    # print(y_vals, x_vals)
    # ax.plot(x_vals, y_vals, "--")
    # y_vals = 0.013856 * x_vals ** 0.169984
    # print(y_vals, x_vals)
    # ax.plot(x_vals, y_vals, "--")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\Delta$t, sec")
    plt.ylabel(r"EA-tamsd, average of logs ($\mu$m$^2$) ")
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
    st.subheader("Table of alphas and Ds (errors are SD of estimates)")
    if avoid_se:
        df_alphas = pd.DataFrame(
            data.groupby(["lags", "condition"])["tamsd"].apply(
                lambda x: np.mean(np.log10(x))
            )
        )
    else:
        df_alphas = pd.DataFrame(
            data.groupby(["lags", "condition"])["tamsd"].apply(
                lambda x: np.log10(
                    np.power(10, np.mean(np.log10(x))) - systematic_error
                )
            )
        )
    df_alphas.reset_index(inplace=True)

    # Select the upper limit first range of fit
    end1 = float(
        st.number_input(
            "End of first regime for fitting a and D", value=100.0, step=interval
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
        res = fit_alpha_d(subset, end1, end2, log=True)
        res["condition"] = condition
        df = pd.concat([df, res])

    # Display and download link of table
    st.dataframe(df)
    st.markdown(
        download_csv(df, "alpha_d.csv", "Download alphas and d table"),
        unsafe_allow_html=True,
    )
