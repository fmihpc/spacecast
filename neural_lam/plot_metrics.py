# flake8: noqa=W605
# Standard library
import argparse
import os

# Third-party
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.ticker import LogFormatter, SymmetricalLogLocator
from tueplots import bundles

plt.rcParams.update(bundles.neurips2023(usetex=False, family="serif"))
plt.rcParams["axes.prop_cycle"] = plt.cycler(
    color=[
        "#1f77b4",  # blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#aec7e8",  # light blue
        "#ffbb78",  # light orange
        "#98df8a",  # light green
    ]
)

FEATURES = [
    "B_x",
    "B_y",
    "B_z",
    "E_x",
    "E_y",
    "E_z",
    "v_x",
    "v_y",
    "v_z",
    r"\rho",
    "P",
    "T",
]
UNITS = [
    "nT",
    "nT",
    "nT",
    "mV/m",
    "mV/m",
    "mV/m",
    "km/s",
    "km/s",
    "km/s",
    "1/cm$^3$",
    "nPa",
    "MK",
]
TIME_STEP_S = 1.0

# Run IDs
simple_efm = "simple_efm"
multiscale_efm = "multiscale_efm"
hierarchical_efm = "hierarchical_efm"

simple_efm_before_crps = "simple_efm_before_crps"
multiscale_efm_before_crps = "multiscale_efm_before_crps"
hierarchical_efm_before_crps = "hierarchical_efm_before_crps"

simple_efm_2_members = "simple_efm_2_members"
simple_efm_5_members = "simple_efm_5_members"
simple_efm_10_members = "simple_efm_10_members"

simple_fm = "simple_fm"
multiscale_fm = "multiscale_fm"
hierarchical_fm = "hierarchical_fm"

simple_fm_1s = "simple_fm_1s"
simple_fm_2s = "simple_fm_2s"
simple_fm_3s = "simple_fm_3s"

simple_fm_before_div = "simple_fm_before_div"
simple_efm_before_div = "simple_efm_before_div"

simple_fm_100_div = "simple_fm_100_div"
simple_fm_1000_div = "simple_fm_1000_div"

vlasiator_div = "vlasiator_div"


def plot_metrics_grid(input_dir, output_dir):
    """
    Plot RMSE, CRPS, and SSR for all variables and lead times.
    """
    grid_1_vars = ["B_z", "E_z", "v_z", r"\rho", "P", "T"]
    grid_2_vars = ["B_x", "B_y", "E_x", "E_y", "v_x", "v_y"]

    latex_labels = {
        "B_x": r"$B_x$",
        "B_y": r"$B_y$",
        "B_z": r"$B_z$",
        "E_x": r"$E_x$",
        "E_y": r"$E_y$",
        "E_z": r"$E_z$",
        "v_x": r"$v_x$",
        "v_y": r"$v_y$",
        "v_z": r"$v_z$",
        r"\rho": r"$\rho$",
        "P": r"$P$",
        "T": r"$T$",
    }

    feature_to_unit = {f: u for f, u in zip(FEATURES, UNITS)}

    metric_files = {
        "RMSE": "test_rmse.csv",
        "CRPS": "test_crps_ens.csv",
        "SSR": "test_spsk_ratio.csv",
    }
    metric_order = ["RMSE", "CRPS", "SSR"]

    fm_runs = {
        "simple": simple_fm,
        "multiscale": multiscale_fm,
        "hierarchical": hierarchical_fm,
    }

    efm_runs = {
        "simple": simple_efm,
        "multiscale": multiscale_efm,
        "hierarchical": hierarchical_efm,
    }

    colors = {
        "simple": ("C3", "C0"),
        "multiscale": ("C4", "C1"),
        "hierarchical": ("C5", "C2"),
    }

    ls_fm = "--"
    ls_efm = "-"

    def plot_grid(var_list, save_name):

        nrows = len(var_list)
        ncols = len(metric_order)

        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(7.7, 11.2), sharex=True
        )
        axes = np.atleast_2d(axes)

        for i, var in enumerate(var_list):
            for j, metric in enumerate(metric_order):

                ax = axes[i, j]
                metric_file = metric_files[metric]

                for arch, run_id in efm_runs.items():
                    fp = os.path.join(input_dir, run_id, metric_file)
                    df = pd.read_csv(fp, header=None, names=FEATURES)
                    lead_s = (df.index + 1) * TIME_STEP_S

                    ax.plot(
                        lead_s,
                        df[var],
                        color=colors[arch][1],
                        linestyle=ls_efm,
                        lw=1.3,
                        label=f"Graph-EFM ({arch})" if (i == 0 and j == 0) else None,
                    )

                if metric == "RMSE":
                    for arch, run_id in fm_runs.items():
                        fp = os.path.join(input_dir, run_id, metric_file)
                        df = pd.read_csv(fp, header=None, names=FEATURES)
                        lead_s = (df.index + 1) * TIME_STEP_S

                        ax.plot(
                            lead_s,
                            df[var],
                            color=colors[arch][0],
                            linestyle=ls_fm,
                            lw=1.3,
                            label=f"Graph-FM ({arch})" if (i == 0 and j == 0) else None,
                        )

                ax.grid(True, ls=":", lw=0.5)
                ax.tick_params(labelsize=9)
                ax.set_xticks(np.arange(0, 31, 10))

                ax.set_title(latex_labels[var], fontsize=11)

                unit = feature_to_unit[var]
                if metric == "SSR":
                    ylabel = "SSR"
                else:
                    ylabel = f"{metric} ({unit})"

                ax.set_ylabel(ylabel, fontsize=10)

                if metric == "SSR":
                    ax.set_ylim(0, 0.5)

                if i == nrows - 1:
                    ax.set_xlabel("Lead time (s)", fontsize=10)

        handles, labels = axes[0, 0].get_legend_handles_labels()

        desired_order = [
            "Graph-FM (simple)",
            "Graph-EFM (simple)",
            "Graph-FM (multiscale)",
            "Graph-EFM (multiscale)",
            "Graph-FM (hierarchical)",
            "Graph-EFM (hierarchical)",
        ]

        label_to_handle = dict(zip(labels, handles))
        ordered_handles = [
            label_to_handle[lbl] for lbl in desired_order if lbl in label_to_handle
        ]

        fig.legend(
            ordered_handles,
            desired_order,
            loc="lower center",
            ncol=3,
            frameon=False,
            fontsize=10,
            bbox_to_anchor=(0.5, 0.0),
        )

        fig.tight_layout(rect=[0.05, 0.05, 0.95, 1])
        for ext in ["png", "pdf"]:
            outpath = os.path.join(output_dir, f"{save_name}.{ext}")
            plt.savefig(outpath, dpi=300)
        plt.close(fig)

        print(f"Saved {outpath}")

    plot_grid(grid_1_vars, "metrics_grid_1")
    plot_grid(grid_2_vars, "metrics_grid_2")


def plot_fm_timesteps(input_dir, output_dir):
    """
    Plot RMSE for all Graph-FM timestep variants (1s, 2s, 3s).
    """

    timestep_models = {
        "1": {"fm": simple_fm_1s},
        "2": {"fm": simple_fm_2s},
        "3": {"fm": simple_fm_3s},
    }
    colors = {"1": "C0", "2": "C1", "3": "C2"}

    fig, axes = plt.subplots(nrows=4, ncols=3, sharex=True, figsize=(6.5, 7.2))
    axes = axes.flatten()

    file_info = {"fm_file": "test_rmse.csv"}

    for ts_label, run_dict in timestep_models.items():
        for model_type, run_id in run_dict.items():
            filepath = os.path.join(input_dir, run_id, file_info["fm_file"])

            df = pd.read_csv(filepath, header=None, names=FEATURES)
            lead_time_s = (df.index + 1) * int(ts_label[0])

            for i, feature in enumerate(FEATURES):
                axes[i].plot(
                    lead_time_s,
                    df[feature].values,
                    label=f"Graph-FM ($\Delta t = {ts_label}\,\\mathrm{{s}}$)",
                    color=colors[ts_label],
                    linestyle="-",
                    lw=1.3,
                )

    for i, feature in enumerate(FEATURES):
        ax = axes[i]
        ax.set_title(f"${feature}$", fontsize=10)
        ax.grid(True, linestyle=":", linewidth=0.5)
        ax.set_xticks(np.arange(0, 31, 10))

        unit = UNITS[i] if i < len(UNITS) else ""
        ylabel = f"RMSE ({unit})"
        ax.set_ylabel(ylabel, fontsize=9)
        ax.tick_params(axis="both", labelsize=8)

    for ax in axes[-3:]:
        ax.set_xlabel("Lead time (s)", fontsize=9)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    for ext in ["png", "pdf"]:
        save_path = os.path.join(output_dir, f"rmse_1s_2s_3s.{ext}")
        plt.savefig(save_path, bbox_inches="tight")
    print(f"Saved RMSE plot for Graph-FM 1s, 2s, 3s to {save_path}")
    plt.close(fig)


def plot_div_loss_comparison(input_dir, output_dir):
    """
    Compare RMSE(Bx), RMSE(Bz), and ⟨|∇·B|⟩ between models
    trained with different divergence-loss weights.
    """
    runs = {
        "Graph-FM (before $\mathcal{L}_{\\mathrm{Div}}$)": simple_fm_before_div,
        "Graph-FM ($\\lambda_{\\mathrm{Div}}=10$)": simple_fm,
        "Graph-FM ($\\lambda_{\\mathrm{Div}}=100$)": simple_fm_100_div,
        "Graph-FM ($\\lambda_{\\mathrm{Div}}=1000$)": simple_fm_1000_div,
        "Graph-EFM (before $\mathcal{L}_{\\mathrm{Div}}$)": simple_efm_before_div,
        "Graph-EFM ($\\lambda_{\\mathrm{Div}}=10^7$)": simple_efm,
        "Vlasiator": vlasiator_div,
    }

    colors = {
        "Graph-FM (before $\mathcal{L}_{\\mathrm{Div}}$)": "C3",
        "Graph-FM ($\\lambda_{\\mathrm{Div}}=10$)": "C0",
        "Graph-FM ($\\lambda_{\\mathrm{Div}}=100$)": "C4",
        "Graph-FM ($\\lambda_{\\mathrm{Div}}=1000$)": "C1",
        "Graph-EFM (before $\mathcal{L}_{\\mathrm{Div}}$)": "C5",
        "Graph-EFM ($\\lambda_{\\mathrm{Div}}=10^7$)": "C2",
        "Vlasiator": "black",
    }

    linestyles = {
        "before": "--",
        "default": "-",
        "100": "--",
        "1000": "-",
        "vlasiator": ":",
    }

    metric_files = {
        "Bx RMSE": "test_rmse.csv",
        "Bz RMSE": "test_rmse.csv",
        "Div B": "test_div_b.csv",
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.4), sharex=False)
    axes = axes.flatten()

    # Panels 1–2: Bx, Bz RMSE
    for col, var in enumerate(["B_x", "B_z"]):
        ax = axes[col]
        for label, run_id in runs.items():
            if label == "Vlasiator":
                continue

            if "before" in label:
                ls = linestyles["before"]
            elif "1000" in label:
                ls = linestyles["1000"]
            elif "100" in label:
                ls = linestyles["100"]
            else:
                ls = linestyles["default"]

            filepath = os.path.join(
                input_dir, run_id, metric_files[f"{var.replace('_','')} RMSE"]
            )
            df = pd.read_csv(filepath, header=None, names=FEATURES)
            lead_time_s = (df.index + 1) * TIME_STEP_S

            ax.plot(
                lead_time_s,
                df[var],
                label=label,
                color=colors[label],
                linestyle=ls,
            )

        ax.set_xlabel("Lead time (s)", fontsize=9)
        ax.set_ylabel(rf"${var}$ RMSE (nT)", fontsize=9)
        ax.grid(True, ls=":", lw=0.5)
        ax.tick_params(axis="both", labelsize=8)

    # Panel 3: ⟨|∇·B|⟩
    ax = axes[2]
    for label, run_id in runs.items():
        if label == "Vlasiator":
            filepath = os.path.join(input_dir, run_id, "test_div_b.csv")
            df = pd.read_csv(filepath, header=None, names=["div_B"])
            lead_time_s = (df.index + 1) * TIME_STEP_S
            ax.plot(
                lead_time_s,
                df["div_B"],
                label="Vlasiator",
                color="black",
                linestyle=linestyles["vlasiator"],
                zorder=-1,
            )
            continue

        if "before" in label:
            ls = linestyles["before"]
        elif "1000" in label:
            ls = linestyles["1000"]
        elif "100" in label:
            ls = linestyles["100"]
        else:
            ls = linestyles["default"]

        filepath = os.path.join(input_dir, run_id, metric_files["Div B"])
        df = pd.read_csv(filepath, header=None, names=["div_B"])
        lead_time_s = (df.index + 1) * TIME_STEP_S

        ax.plot(
            lead_time_s,
            df["div_B"],
            label=label,
            color=colors[label],
            linestyle=ls,
        )

    ax.set_xlabel("Lead time (s)", fontsize=9)
    ax.set_ylabel(
        r"$\langle |\nabla \cdot \mathbf{B}| \rangle$ (nT / $R_E$)", fontsize=9
    )
    ax.grid(True, ls=":", lw=0.5)
    ax.tick_params(axis="both", labelsize=8)

    handles, labels = [], []
    for ax in axes:
        handle, label = ax.get_legend_handles_labels()
        handles += handle
        labels += label
    by_label = dict(zip(labels, handles))

    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.16),
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    for ext in ["png", "pdf"]:
        save_path = os.path.join(output_dir, f"div_loss_comparison.{ext}")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved divergence-loss comparison plot to {save_path}")


def plot_efm_crps_comparison(input_dir, output_dir):
    """
    Plot RMSE, CRPS, and SSR for Graph-EFM models and their
    before-CRPS training counterparts.
    """
    runs = {
        r"Graph-EFM (simple, $\lambda_{\mathrm{CRPS}}=10^{6}$)": (
            simple_efm,
            simple_efm_before_crps,
        ),
        r"Graph-EFM (multiscale, $\lambda_{\mathrm{CRPS}}=10^{6}$)": (
            multiscale_efm,
            multiscale_efm_before_crps,
        ),
        r"Graph-EFM (hierarchical, $\lambda_{\mathrm{CRPS}}=10^{5}$)": (
            hierarchical_efm,
            hierarchical_efm_before_crps,
        ),
    }

    paired_colors = {
        "simple": ("C0", "C3"),
        "multiscale": ("C1", "C4"),
        "hierarchical": ("C2", "C5"),
    }

    metric_files = {
        "RMSE": "test_ens_rmse.csv",
        "CRPS": "test_crps_ens.csv",
        "SSR": "test_spsk_ratio.csv",
    }

    var_subset = ["B_z", "E_z", "v_z", r"\rho"]
    var_labels = [r"$B_z$", r"$E_z$", r"$v_z$", r"$\rho$"]
    var_units = ["nT", "mV/m", "km/s", "1/cm$^3$"]

    nrows = len(var_subset)
    ncols = len(metric_files)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7.4, 8.4), sharex=True)
    axes = np.atleast_2d(axes)

    for row_idx, (var, var_label, unit) in enumerate(
        zip(var_subset, var_labels, var_units)
    ):
        for col_idx, (metric_name, filename) in enumerate(metric_files.items()):
            ax = axes[row_idx, col_idx]

            for label, (run_after, run_before) in runs.items():
                if "simple" in label:
                    dark_color, light_color = paired_colors["simple"]
                elif "multiscale" in label:
                    dark_color, light_color = paired_colors["multiscale"]
                else:
                    dark_color, light_color = paired_colors["hierarchical"]

                filepath = os.path.join(input_dir, run_before, filename)
                df = pd.read_csv(filepath, header=None, names=FEATURES)
                lead_time_s = (df.index + 1) * TIME_STEP_S
                label_text = (
                    rf"{label.split(',')[0]}, before " r"$\mathcal{L}_{\mathrm{CRPS}}$)"
                )

                ax.plot(
                    lead_time_s,
                    df[var],
                    color=light_color,
                    linestyle="--",
                    label=(label_text if (row_idx == 0 and col_idx == 0) else None),
                )

                filepath = os.path.join(input_dir, run_after, filename)
                df = pd.read_csv(filepath, header=None, names=FEATURES)
                lead_time_s = (df.index + 1) * TIME_STEP_S
                ax.plot(
                    lead_time_s,
                    df[var],
                    color=dark_color,
                    linestyle="-",
                    label=label if (row_idx == 0 and col_idx == 0) else None,
                )

            ax.grid(True, ls=":", lw=0.5)
            ax.tick_params(axis="both", labelsize=8)
            ax.set_xticks(np.arange(0, 31, 10))
            ax.set_title(var_label, fontsize=11)
            ax.set_ylabel(f"{metric_name} ({unit})", fontsize=9)

            if row_idx == nrows - 1:
                ax.set_xlabel("Lead time (s)", fontsize=9)
            if metric_name == "SSR":
                ax.set_ylim(0, 0.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
        columnspacing=0.7,
        handletextpad=0.5,
        handlelength=1.1,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    for ext in ["png", "pdf"]:
        save_path = os.path.join(output_dir, f"crps_loss_comparison.{ext}")
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved EFM CRPS comparison plot to {save_path}")


def plot_efm_ensemble_norm_diff(input_dir, output_dir):
    """
    Compare normalized difference in metrics to smallest M=2 ensemble.
    """
    runs = {
        "Graph-EFM (2 members)": simple_efm_2_members,
        "Graph-EFM (5 members)": simple_efm_5_members,
        "Graph-EFM (10 members)": simple_efm_10_members,
    }

    colors = {
        "Graph-EFM (2 members)": "C0",
        "Graph-EFM (5 members)": "C1",
        "Graph-EFM (10 members)": "C2",
    }

    metric_files = {
        "RMSE": "test_ens_rmse.csv",
        "CRPS": "test_crps_ens.csv",
        "SSR": "test_spsk_ratio.csv",
    }

    var_subset = ["B_z", "E_z", "v_z", r"\rho"]
    var_labels = [r"$B_z$", r"$E_z$", r"$v_z$", r"$\rho$"]

    fig, axes = plt.subplots(4, 3, figsize=(6.8, 7.2), sharex=True)
    axes = np.atleast_2d(axes)

    for row_idx, (var, var_label) in enumerate(zip(var_subset, var_labels)):
        for col_idx, (metric_name, filename) in enumerate(metric_files.items()):
            ax = axes[row_idx, col_idx]
            base_path = os.path.join(input_dir, runs["Graph-EFM (2 members)"], filename)
            df_base = pd.read_csv(base_path, header=None, names=FEATURES)
            base_values = df_base[var].values
            lead_time_s = (df_base.index + 1) * TIME_STEP_S

            ax.plot(
                lead_time_s,
                np.zeros_like(lead_time_s),
                color=colors["Graph-EFM (2 members)"],
                ls="-",
                lw=1.3,
                label=(
                    "Graph-EFM (2 members)" if (row_idx == 0 and col_idx == 0) else None
                ),
            )

            for label in ["Graph-EFM (5 members)", "Graph-EFM (10 members)"]:
                run_id = runs[label]
                filepath = os.path.join(input_dir, run_id, filename)
                df = pd.read_csv(filepath, header=None, names=FEATURES)
                values = df[var].values
                improvement = 100 * (values / base_values - 1)
                ax.plot(
                    lead_time_s,
                    improvement,
                    color=colors[label],
                    lw=1.3,
                    label=label if (row_idx == 0 and col_idx == 0) else None,
                )

            ax.set_yscale("symlog", linthresh=2)
            ax.grid(True, ls=":", lw=0.5)
            ax.tick_params(axis="both", labelsize=8)
            ax.set_xticks(np.arange(0, 31, 10))
            ax.set_ylim(-20, 0.5)
            ax.yaxis.set_major_locator(SymmetricalLogLocator(base=10, linthresh=2))
            ax.yaxis.set_minor_locator(
                SymmetricalLogLocator(base=10, linthresh=2, subs=np.arange(2, 10))
            )

            ax.set_title(var_label, fontsize=10)
            if row_idx == 3:
                ax.set_xlabel("Lead time (s)", fontsize=9)
            ax.set_ylabel(f"Norm. {metric_name} diff. (%)", fontsize=9)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    for ext in ["png", "pdf"]:
        save_path = os.path.join(output_dir, f"ens_norm_diff.{ext}")
        plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved EFM ensemble-member norm. diff. plot to {save_path}")


def plot_runs_forecast_grid(
    runs,
    input_dir,
    output_dir,
    file_name="example_1.zarr",
):
    """
    For each variable in the dataset, plot a grid with one run per row.
    """

    latex_vars = {
        "Bx": r"$B_x$",
        "By": r"$B_y$",
        "Bz": r"$B_z$",
        "Ex": r"$E_x$",
        "Ey": r"$E_y$",
        "Ez": r"$E_z$",
        "vx": r"$v_x$",
        "vy": r"$v_y$",
        "vz": r"$v_z$",
        "rho": r"$\rho$",
        "P": r"$P$",
        "T": r"$T$",
    }
    unit_map = {
        "Bx": "nT",
        "By": "nT",
        "Bz": "nT",
        "Ex": "mV/m",
        "Ey": "mV/m",
        "Ez": "mV/m",
        "vx": "km/s",
        "vy": "km/s",
        "vz": "km/s",
        "rho": "1/cm$^3$",
        "P": "nPa",
        "T": "MK",
    }

    # Open one file to list all state features
    example_run = next(iter(runs.values()))
    zarr_path = os.path.join(input_dir, example_run, file_name)
    ds_ref = xr.open_zarr(zarr_path)
    all_features = list(ds_ref["state_feature"].values)

    diverging_vars = [f"{p}{a}" for p in ("B", "E", "v") for a in ("x", "y", "z")]

    # Iterate through each feature
    for var_name in all_features:
        print(f"Plotting {var_name} ...")

        nrows = len(runs)
        ncols = 3
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(12, 3 * nrows),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axes = np.atleast_2d(axes)
        ims_main, ims_std = [], []

        units = unit_map.get(var_name, "")

        for row_idx, (run_label, run_id) in enumerate(runs.items()):
            zarr_path = os.path.join(input_dir, run_id, file_name)

            ds = xr.open_zarr(zarr_path)
            norm_map = {
                f.replace("_", "").lower(): f for f in ds["state_feature"].values
            }
            key = var_name.replace("_", "").lower()
            var_actual = norm_map[key]
            var_i = list(ds["state_feature"].values).index(var_actual)

            # Select variable i at lead time 30s
            target = ds["target"].isel(time=29, state_feature=var_i)
            ens_mean = ds["ens_mean"].isel(time=29, state_feature=var_i)
            ens_std = ds["ens_std"].isel(time=29, state_feature=var_i)

            if set(["x", "z"]).issubset(target.dims):
                target = target.transpose("z", "x")
                ens_mean = ens_mean.transpose("z", "x")
                ens_std = ens_std.transpose("z", "x")

            if var_name.replace("_", "") in diverging_vars:
                cmap_main = "RdBu_r"
                vmax_abs = float(np.nanmax(np.abs([target, ens_mean])))
                vmin, vmax = -vmax_abs, vmax_abs
            else:
                cmap_main = "viridis"
                vmin = float(np.nanmin([target.min(), ens_mean.min()]))
                vmax = float(np.nanmax([target.max(), ens_mean.max()]))

            cmap_std = "OrRd"

            for col_idx, (data, title, cmap) in enumerate(
                zip(
                    [target, ens_mean, ens_std],
                    [
                        f"Run {row_idx + 1}\nVlasiator Ground Truth",
                        f"Run {row_idx + 1}\nGraph-EFM Ensemble Mean",
                        f"Run {row_idx + 1}\nGraph-EFM Ensemble Std. Dev.",
                    ],
                    [cmap_main, cmap_main, cmap_std],
                )
            ):
                ax = axes[row_idx, col_idx]
                im = data.plot.imshow(
                    ax=ax,
                    cmap=cmap,
                    add_colorbar=False,
                    vmin=vmin if col_idx < 2 else None,
                    vmax=vmax if col_idx < 2 else None,
                    origin="lower",
                    yincrease=True,
                    add_labels=False,
                )
                im.colorbar = None  # suppress xarray labels

                if col_idx < 2:
                    ims_main.append(im)
                else:
                    ims_std.append(im)

                ax.set_title(title, fontsize=14)

                if row_idx == nrows - 1:
                    ax.set_xlabel(r"$x\ (R_E)$", fontsize=14)
                if col_idx == 0:
                    ax.set_ylabel("$z\ (R_E)$", fontsize=14)
                else:
                    ax.set_ylabel("")

                ax.set_xticks(np.arange(-60, 31, 10))
                ax.set_yticks(np.arange(-30, 31, 10))
                ax.tick_params(axis="both", which="major", labelsize=12)

        cax_main = fig.add_axes([0.15, -0.02, 0.45, 0.01])
        cb_main = fig.colorbar(ims_main[0], cax=cax_main, orientation="horizontal")
        cb_main.ax.tick_params(labelsize=12)
        cb_main.set_label(
            f"{latex_vars.get(var_name, var_name)} ({units})", fontsize=14
        )

        cax_std = fig.add_axes([0.72, -0.02, 0.25, 0.01])
        cb_std = fig.colorbar(ims_std[0], cax=cax_std, orientation="horizontal")
        cb_std.ax.tick_params(labelsize=12)
        cb_std.set_label(f"Std. Dev. ({units})", fontsize=14)

        for ext in ["png", "pdf"]:
            save_path = os.path.join(output_dir, f"forecast_grid_{var_name}.{ext}")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {save_path}")


def compute_power_spectrum(field2d, dx, dz):
    """
    Compute isotropic power spectrum of 2D field
    """
    field = np.nan_to_num(field2d - np.nanmean(field2d))
    ny, nx = field.shape

    F = np.fft.fftshift(np.fft.fft2(field))
    psd2d = (np.abs(F) ** 2) * (dx * dz) / (nx * ny)

    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    kz = np.fft.fftshift(np.fft.fftfreq(ny, d=dz))
    KX, KZ = np.meshgrid(kx, kz)
    k = np.sqrt(KX**2 + KZ**2)

    k_bins = np.linspace(0, np.max(k), 50)
    Pk, bin_edges = np.histogram(k, bins=k_bins, weights=psd2d)
    counts, _ = np.histogram(k, bins=k_bins)
    Pk = np.where(counts > 0, Pk / counts, np.nan)
    k_mid = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    return k_mid, Pk


def plot_power_spectra_grid(
    efm_run_id,
    fm_run_id,
    input_dir,
    output_dir,
    prefix,
    file_name="example_1.zarr",
):
    """
    Plot power spectra comparing Graph-FM, Graph-EFM, and Vlasiator
    at four forecast times (1s, 10s, 30s, 50s).
    """

    latex_vars = {
        "Bx": r"$B_x$",
        "By": r"$B_y$",
        "Bz": r"$B_z$",
        "Ex": r"$E_x$",
        "Ey": r"$E_y$",
        "Ez": r"$E_z$",
        "vx": r"$v_x$",
        "vy": r"$v_y$",
        "vz": r"$v_z$",
        "rho": r"$\rho$",
        "P": r"$P$",
        "T": r"$T$",
    }

    def load_ds(run_id):
        path = os.path.join(input_dir, run_id, file_name)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return xr.open_zarr(path).fillna(0)

    ds_efm = load_ds(efm_run_id)
    ds_fm = load_ds(fm_run_id)

    features = ["Bz", "Ez", "vz", "rho", "P", "T", "Bx", "By", "Ex", "Ey", "vx", "vy"]
    time_indices = [0, 9, 29, 49]
    time_labels = [1, 10, 30, 50]

    dx = float(ds_fm.x[1] - ds_fm.x[0])
    dz = float(ds_fm.z[1] - ds_fm.z[0])

    halves = [
        (features[:6], f"{prefix}_power_spectra_grid_1"),
        (features[6:], f"{prefix}_power_spectra_grid_2"),
    ]

    for half_features, fname in halves:
        fig, axes = plt.subplots(
            nrows=len(half_features),
            ncols=len(time_indices),
            figsize=(7.5, 9.5),
            constrained_layout=True,
            sharex=True,
            sharey=False,
        )

        for row_idx, var_name in enumerate(half_features):
            for col_idx, (t_idx, t_val) in enumerate(zip(time_indices, time_labels)):
                ax = axes[row_idx, col_idx]

                # Ground truth
                target = ds_fm["target"].sel(state_feature=var_name).isel(time=t_idx)
                if set(["x", "z"]).issubset(target.dims):
                    target = target.transpose("z", "x")

                # Models
                def _get_model(ds, key="ens_mean"):
                    if key not in ds.data_vars:
                        key = "prediction"
                    arr = ds[key].sel(state_feature=var_name).isel(time=t_idx)
                    return (
                        arr.transpose("z", "x")
                        if set(["x", "z"]).issubset(arr.dims)
                        else arr
                    )

                efm = _get_model(ds_efm)
                fm = _get_model(ds_fm, key="prediction")

                k_t, P_t = compute_power_spectrum(target.values, dx=dx, dz=dz)
                k_efm, P_efm = compute_power_spectrum(efm.values, dx=dx, dz=dz)
                k_fm, P_fm = compute_power_spectrum(fm.values, dx=dx, dz=dz)

                ax.plot(k_fm, P_fm, color="C0", ls="-", lw=1.3)
                ax.plot(k_efm, P_efm, color="C1", ls="-", lw=1.3)
                ax.plot(k_t, P_t, color="black", ls="--", lw=1.3)

                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.grid(True, which="both", ls=":", lw=0.5)
                ax.tick_params(axis="both", labelsize=9)

                if col_idx == 0:
                    ax.set_ylabel(r"Power $P(k)$", fontsize=10)
                if row_idx == len(half_features) - 1:
                    ax.set_xlabel(r"Wavenumber $k$", fontsize=10)

                ax.set_title(
                    (
                        f"{latex_vars.get(var_name, var_name)}, "
                        f"$t={t_val}\\,\\mathrm{{s}}$"
                    ),
                    fontsize=10,
                )

        handles = [
            plt.Line2D([], [], color="C0", label="Graph-FM"),
            plt.Line2D([], [], color="C1", label="Graph-EFM"),
            plt.Line2D([], [], color="black", ls="--", label="Vlasiator"),
        ]
        fig.legend(
            handles,
            [h.get_label() for h in handles],
            loc="lower center",
            ncol=3,
            frameon=False,
            fontsize=10,
            bbox_to_anchor=(0.5, -0.05),
        )

        for ext in ["png", "pdf"]:
            path = os.path.join(output_dir, f"{fname}.{ext}")
            plt.savefig(path, bbox_inches="tight", dpi=300)
        plt.close(fig)
        print(f"Saved {fname} to {path}")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    plot_metrics_grid(args.metrics_dir, args.output_dir)
    plot_efm_ensemble_norm_diff(args.metrics_dir, args.output_dir)
    plot_efm_crps_comparison(args.metrics_dir, args.output_dir)
    plot_div_loss_comparison(args.metrics_dir, args.output_dir)
    plot_fm_timesteps(args.metrics_dir, args.output_dir)

    runs = {
        "Run 1": "run_1_efm",
        "Run 2": "run_2_efm",
        "Run 3": "run_3_efm",
        "Run 4": "run_4_efm",
    }
    plot_runs_forecast_grid(runs, args.forecasts_dir, args.output_dir)

    plot_power_spectra_grid(
        "run_1_efm", "run_1_fm", args.forecasts_dir, args.output_dir, "run_1"
    )
    plot_power_spectra_grid(
        "run_2_efm", "run_2_fm", args.forecasts_dir, args.output_dir, "run_2"
    )
    plot_power_spectra_grid(
        "run_3_efm", "run_3_fm", args.forecasts_dir, args.output_dir, "run_3"
    )
    plot_power_spectra_grid(
        "run_4_efm", "run_4_fm", args.forecasts_dir, args.output_dir, "run_4"
    )


if __name__ == "__main__":
    """Plot model metrics"""
    parser = argparse.ArgumentParser(description="Plot metrics")
    parser.add_argument(
        "--metrics_dir",
        type=str,
        help="Input metrics folder",
    )
    parser.add_argument(
        "--forecasts_dir",
        type=str,
        help="Example forecasts folder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Results plots folder",
    )
    args = parser.parse_args()

    main(args)
