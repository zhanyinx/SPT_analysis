from abc import abstractmethod
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from .utils import *
from .dictionary import LIST_SAMPLES, SYSTEMATIC_ERRORS


def visualize_3point():
    # Set title
    st.title(r"3-point segment analysis: angle, $\alpha$, D.")

    # Samples
    list_samples = LIST_SAMPLES

    # Take input from user and load file and make a copy
    sample_name = "directionality"

    # download data if not already present
    if not os.path.isfile(f"{sample_name}.npy"):
        sample_link = list_samples[sample_name]
        gdown.download(sample_link, f"{sample_name}.npy")

    rad21_cells = st.sidebar.checkbox("Tick when Rad21")
    # Extract data
    # Format - dictionary (key: data):
    # key contains {acquisition time}_{cell line}_{induction time}, i.e. "10_2B10_0min"
    # data for each key contain 4 series:
    #   1. Raw angle distribution
    #   2. Normalized angle distribution
    #   3. MSD slope, alpha
    #   4. MSD intercept, D
    with open("directionality.npy", "rb") as handle:
        data = pickle.load(handle)

    dts = []
    cell_lines = []
    induction_times = []

    for datum in data.keys():
        i, j, k = datum.split("_")
        dts.append(i)
        cell_lines.append(j)
        induction_times.append(k)

    if rad21_cells:
        conditions = ["TetO", "3xCTCF", "Cre", "Tir1"]
        times = list(map(str, sorted(list(map(int, list(set(dts)))))))
        treatment = sorted(set(induction_times))
        chosen_cell_lines1 = st.sidebar.multiselect(
            "TetO cell lines", ["1B5", "1G7", "2G9"]
        )
        chosen_cell_lines2 = st.sidebar.multiselect(
            "3xCTCF cell lines", ["2B10", "3B8", "3G4"]
        )
        chosen_cell_lines3 = st.sidebar.multiselect(
            "Cre cell lines", ["2B10-A2", "3G4-C4"]
        )
        chosen_cell_lines4 = st.sidebar.multiselect(
            "Tir1 cell lines", ["2B8", "2G2", "3B10"]
        )
        chosen_cell_lines = (
            chosen_cell_lines1
            + chosen_cell_lines2
            + chosen_cell_lines3
            + chosen_cell_lines4
        )
        lst_cells = [
            chosen_cell_lines1,
            chosen_cell_lines2,
            chosen_cell_lines3,
            chosen_cell_lines4,
        ]

    else:
        # Take input from user and load file and make a copy
        chosen_cell_lines = st.sidebar.multiselect(
            "Choose the cell lines", sorted(set(cell_lines))
        )
    pool_replicates = st.sidebar.checkbox("Pool replicates")
    chosen_induction_times = st.sidebar.multiselect(
        "Choose the induction times", treatment
    )
    chosen_acq_times = st.sidebar.multiselect("Choose the acquisition times", times)
    if rad21_cells:
        acq_times = list(map(str, list(set(chosen_acq_times))))
        lst_cells_times = []
        for t in range(len(acq_times)):
            lst_cells_times.append([])
            for aux in range(len(treatment)):
                lst_cells_times[t].append([])
                for cl in range(len(lst_cells)):
                    lst_cells_times[t][aux].append([])
                    for l_cl in lst_cells[cl]:
                        lst_cells_times[t][aux][cl].append(
                            acq_times[t] + "_" + l_cl + "_" + treatment[aux]
                        )
    # Filter data to keep only the selected lines, induction time and correction type
    selected_data = []
    sub_data = []
    for i in range(len(chosen_cell_lines)):
        for j in range(len(chosen_induction_times)):
            for k in range(len(chosen_acq_times)):
                dataset_name = "_".join(
                    [
                        chosen_acq_times[k],
                        chosen_cell_lines[i],
                        chosen_induction_times[j],
                    ]
                )
                selected_data.append(dataset_name)
                sub_data.append(data[dataset_name])

    # Plot
    if st.button("Plot!"):
        if len(sub_data) == 0:
            st.error("Chose 3 things: cell line, ind. time, and acq. time.")
        else:  # if st.button("Plot!") and len(sub_data) > 0:
            fig, ax = plt.subplots(2, 2, figsize=(14, 12))
            ax[0, 0].set_title("Raw angle distribution")
            if (
                pool_replicates
                and rad21_cells
                and len(chosen_cell_lines) > 0
                and len(chosen_induction_times) > 0
                and len(chosen_acq_times) > 0
            ):
                x = sub_data[0][0].index.values  # values for x-axis
                conditions_title = [
                    times[i] + "_" + conditions[k] + "_" + treatment[j]
                    for i in range(len(lst_cells_times))
                    for j in range(len(lst_cells_times[i]))
                    for k in range(len(lst_cells_times[i][j]))
                    if len(lst_cells_times[i][j][k]) > 0
                ]  # list of conditions
                times_cell_lines_list = list(
                    filter(lambda x: x, lst_cells_times)
                )  # list of lists with cell line names
                ys = []
                for t in range(len(times_cell_lines_list)):
                    ys.append([])
                    for a in range(len(times_cell_lines_list[t])):
                        ys[t].append([])
                        for c in range(len(times_cell_lines_list[t][a])):
                            ys[t][a].append([])
                            for j in range(len(sub_data)):
                                ti, cl, au = selected_data[j].split("_")
                                if (
                                    ti + "_" + cl + "_" + au
                                    in times_cell_lines_list[t][a][c]
                                ):
                                    ys[t][a][c].append(list(sub_data[j][0].values))
                new_y = []
                for s_y in ys:
                    for ss_y in s_y:
                        for sss_y in ss_y:
                            if len(sss_y) > 0:
                                new_y.append(np.mean(sss_y, axis=0))
                for i in range(len(new_y)):
                    ax[0, 0].plot(x, new_y[i], label=conditions_title[i])
            else:
                for i in range(len(sub_data)):
                    ax[0, 0].plot(
                        sub_data[i][0].index.values,
                        sub_data[i][0].values,
                        label=selected_data[i],
                    )
            ax[0, 0].set_xlabel(r"Angle, $^{\circ}$")
            ax[0, 0].set_ylabel("Count")
            ax[0, 0].legend()

            ax[0, 1].set_title("Normalized angle distribution")
            if (
                pool_replicates
                and rad21_cells
                and len(chosen_cell_lines) > 0
                and len(chosen_induction_times) > 0
                and len(chosen_acq_times) > 0
            ):
                x = sub_data[0][0].index.values  # values for x-axis
                conditions_title = [
                    times[i] + "_" + conditions[k] + "_" + treatment[j]
                    for i in range(len(lst_cells_times))
                    for j in range(len(lst_cells_times[i]))
                    for k in range(len(lst_cells_times[i][j]))
                    if len(lst_cells_times[i][j][k]) > 0
                ]  # list of conditions
                times_cell_lines_list = list(
                    filter(lambda x: x, lst_cells_times)
                )  # list of lists with cell line names
                ys = []
                for t in range(len(times_cell_lines_list)):
                    ys.append([])
                    for a in range(len(times_cell_lines_list[t])):
                        ys[t].append([])
                        for c in range(len(times_cell_lines_list[t][a])):
                            ys[t][a].append([])
                            for j in range(len(sub_data)):
                                ti, cl, au = selected_data[j].split("_")
                                if (
                                    ti + "_" + cl + "_" + au
                                    in times_cell_lines_list[t][a][c]
                                ):
                                    ys[t][a][c].append(list(sub_data[j][1].values))
                new_y = []
                for s_y in ys:
                    for ss_y in s_y:
                        for sss_y in ss_y:
                            if len(sss_y) > 0:
                                new_y.append(np.mean(sss_y, axis=0))
                for i in range(len(new_y)):
                    ax[0, 1].plot(x, new_y[i], label=conditions_title[i])
            else:
                for i in range(len(sub_data)):
                    ax[0, 1].plot(
                        sub_data[i][1].index.values,
                        sub_data[i][1].values,
                        label=selected_data[i],
                    )
            ax[0, 1].set_xlabel(r"Angle, $^{\circ}$")
            ax[0, 1].set_ylabel("Count")
            ax[0, 1].legend()

            ax[1, 0].set_title("MSD slope distribution")
            if (
                pool_replicates
                and rad21_cells
                and len(chosen_cell_lines) > 0
                and len(chosen_induction_times) > 0
                and len(chosen_acq_times) > 0
            ):
                x = sub_data[0][2].index.values  # values for x-axis
                conditions_title = [
                    times[i] + "_" + conditions[k] + "_" + treatment[j]
                    for i in range(len(lst_cells_times))
                    for j in range(len(lst_cells_times[i]))
                    for k in range(len(lst_cells_times[i][j]))
                    if len(lst_cells_times[i][j][k]) > 0
                ]  # list of conditions
                times_cell_lines_list = list(
                    filter(lambda x: x, lst_cells_times)
                )  # list of lists with cell line names
                ys = []
                for t in range(len(times_cell_lines_list)):
                    ys.append([])
                    for a in range(len(times_cell_lines_list[t])):
                        ys[t].append([])
                        for c in range(len(times_cell_lines_list[t][a])):
                            ys[t][a].append([])
                            for j in range(len(sub_data)):
                                ti, cl, au = selected_data[j].split("_")
                                if (
                                    ti + "_" + cl + "_" + au
                                    in times_cell_lines_list[t][a][c]
                                ):
                                    ys[t][a][c].append(list(sub_data[j][2].values))
                new_y = []
                for s_y in ys:
                    for ss_y in s_y:
                        for sss_y in ss_y:
                            if len(sss_y) > 0:
                                new_y.append(np.mean(sss_y, axis=0))
                for i in range(len(new_y)):
                    ax[1, 0].plot(x, new_y[i], label=conditions_title[i])
            else:
                for i in range(len(sub_data)):
                    ax[1, 0].plot(
                        sub_data[i][2].index.values,
                        sub_data[i][2].values,
                        label=selected_data[i],
                    )
            ax[1, 0].set_xlabel(r"Slope, $\alpha$")
            ax[1, 0].set_ylabel("Count")
            ax[1, 0].legend()

            ax[1, 1].set_title("MSD intercept distribution")
            if (
                pool_replicates
                and rad21_cells
                and len(chosen_cell_lines) > 0
                and len(chosen_induction_times) > 0
                and len(chosen_acq_times) > 0
            ):
                x = sub_data[0][3].index.values  # values for x-axis
                conditions_title = [
                    times[i] + "_" + conditions[k] + "_" + treatment[j]
                    for i in range(len(lst_cells_times))
                    for j in range(len(lst_cells_times[i]))
                    for k in range(len(lst_cells_times[i][j]))
                    if len(lst_cells_times[i][j][k]) > 0
                ]  # list of conditions
                times_cell_lines_list = list(
                    filter(lambda x: x, lst_cells_times)
                )  # list of lists with cell line names
                ys = []
                for t in range(len(times_cell_lines_list)):
                    ys.append([])
                    for a in range(len(times_cell_lines_list[t])):
                        ys[t].append([])
                        for c in range(len(times_cell_lines_list[t][a])):
                            ys[t][a].append([])
                            for j in range(len(sub_data)):
                                ti, cl, au = selected_data[j].split("_")
                                if (
                                    ti + "_" + cl + "_" + au
                                    in times_cell_lines_list[t][a][c]
                                ):
                                    ys[t][a][c].append(list(sub_data[j][3].values))
                new_y = []
                for s_y in ys:
                    for ss_y in s_y:
                        for sss_y in ss_y:
                            if len(sss_y) > 0:
                                new_y.append(np.mean(sss_y, axis=0))
                for i in range(len(new_y)):
                    ax[1, 1].plot(x, new_y[i], label=conditions_title[i])
            else:
                for i in range(len(sub_data)):
                    ax[1, 1].plot(
                        sub_data[i][3].index.values,
                        sub_data[i][3].values,
                        label=selected_data[i],
                    )
            ax[1, 1].set_xscale("log")
            ax[1, 1].set_xlabel("Intercept, D")
            ax[1, 1].set_ylabel("Count")
            ax[1, 1].legend()

            plt.tight_layout()
            st.pyplot(fig)
            plt.savefig("plot.pdf")

            # Dowload options
            st.markdown(
                download_plot(
                    "plot.pdf",
                    "Download plot",
                ),
                unsafe_allow_html=True,
            )
