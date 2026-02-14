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
from matplotlib.colors import LogNorm
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
    grid_1_vars = ["B_z", "E_z", "v_z", r"\rho"]
    grid_2_vars = ["B_x", "B_y", "E_x", "E_y"]
    grid_3_vars = ["v_x", "v_y", "P", "T"]

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
            nrows=nrows,
            ncols=ncols,
            figsize=(7.7, 8),
            sharex=True,
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
            bbox_to_anchor=(0.5, -0.02),
        )

        fig.tight_layout(rect=[0.05, 0.05, 0.95, 1])
        for ext in ["png", "pdf"]:
            outpath = os.path.join(output_dir, f"{save_name}.{ext}")
            plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved {outpath}")

    plot_grid(grid_1_vars, "metrics_grid_1")
    plot_grid(grid_2_vars, "metrics_grid_2")
    plot_grid(grid_3_vars, "metrics_grid_3")


def plot_fm_efm_comparison(input_dir, output_dir):
    """
    Compare Graph-FM (simple) and Graph-EFM (simple) with and without
    CRPS loss, in three grids.

    Lines:
        - Graph-FM (simple): blue solid (RMSE only)
        - Graph-EFM (simple, with λ_CRPS): green solid
        - Graph-EFM (simple, before λ_CRPS): orange dashed

    Metrics: RMSE, CRPS, SSR.
    """

    grid_1_vars = ["B_x", "E_x", "v_x", r"\rho"]
    grid_2_vars = ["B_y", "B_z", "E_y", "E_z"]
    grid_3_vars = ["v_y", "v_z", "P", "T"]

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

    # FM has only RMSE stored
    metric_files_fm = {
        "RMSE": "test_rmse.csv",
    }

    # EFM has ensemble-based metrics
    metric_files_efm = {
        "RMSE": "test_ens_rmse.csv",
        "CRPS": "test_crps_ens.csv",
        "SSR": "test_spsk_ratio.csv",
    }

    metrics = ["RMSE", "CRPS", "SSR"]

    def plot_grid(var_list, suffix):
        nrows = len(var_list)
        ncols = len(metrics)

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(6.8, 7.2),
            sharex=True,
        )
        axes = np.atleast_2d(axes)

        for row_idx, var in enumerate(var_list):
            var_label = latex_labels[var]
            unit = feature_to_unit[var]

            for col_idx, metric_name in enumerate(metrics):
                ax = axes[row_idx, col_idx]

                # ---------- Graph-FM (simple) ----------
                if metric_name in metric_files_fm:
                    fm_file = os.path.join(
                        input_dir, simple_fm, metric_files_fm[metric_name]
                    )
                    df_fm = pd.read_csv(fm_file, header=None, names=FEATURES)
                    lead_time_s = (df_fm.index + 1) * TIME_STEP_S

                    ax.plot(
                        lead_time_s,
                        df_fm[var],
                        color="C0",
                        linestyle="-",
                        lw=1.3,
                        label=(
                            r"Graph-FM" if (row_idx == 0 and col_idx == 0) else None
                        ),
                    )

                # ---------- Graph-EFM (simple, with CRPS) ----------
                efm_after_file = os.path.join(
                    input_dir, simple_efm, metric_files_efm[metric_name]
                )
                df_efm_after = pd.read_csv(efm_after_file, header=None, names=FEATURES)
                lead_time_s_after = (df_efm_after.index + 1) * TIME_STEP_S

                ax.plot(
                    lead_time_s_after,
                    df_efm_after[var],
                    color="C1",
                    linestyle="-",
                    lw=1.3,
                    label=(r"Graph-EFM" if (row_idx == 0 and col_idx == 0) else None),
                )

                # ---------- Graph-EFM (simple, before CRPS) ----------
                efm_before_file = os.path.join(
                    input_dir, simple_efm_before_crps, metric_files_efm[metric_name]
                )
                df_efm_before = pd.read_csv(
                    efm_before_file, header=None, names=FEATURES
                )
                lead_time_s_before = (df_efm_before.index + 1) * TIME_STEP_S

                ax.plot(
                    lead_time_s_before,
                    df_efm_before[var],
                    color="C2",
                    linestyle="-",
                    lw=1.3,
                    label=(
                        r"Graph-EFM (before $\mathcal{L}_{\mathrm{CRPS}}$)"
                        if (row_idx == 0 and col_idx == 0)
                        else None
                    ),
                )

                # Cosmetics
                ax.grid(True, ls=":", lw=0.5)
                ax.tick_params(axis="both", labelsize=8)
                ax.set_xticks(np.arange(0, 31, 10))
                ax.set_title(var_label, fontsize=11)

                if metric_name == "SSR":
                    ylabel = "SSR"
                    ax.set_ylim(0, 0.5)
                else:
                    ylabel = f"{metric_name} ({unit})"
                ax.set_ylabel(ylabel, fontsize=9)

                if row_idx == nrows - 1:
                    ax.set_xlabel("Lead time (s)", fontsize=9)

        # Legend from first panel
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=3,
            frameon=False,
            fontsize=9,
            bbox_to_anchor=(0.5, 0.00),
        )

        fig.tight_layout(rect=[0, 0.05, 1, 1])

        for ext in ["png", "pdf"]:
            save_path = os.path.join(output_dir, f"fm_efm_comparison_{suffix}.{ext}")
            plt.savefig(save_path, bbox_inches="tight", dpi=300)

        plt.close(fig)
        print(f"Saved FM/EFM comparison plot to {save_path}")

    # Three grids, matching plot_metrics_grid groupings
    plot_grid(grid_1_vars, "1")
    plot_grid(grid_2_vars, "2")
    plot_grid(grid_3_vars, "3")


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

    var_subset = ["B_x", "E_x", "v_x", r"\rho"]
    var_labels = [r"$B_x$", r"$E_x$", r"$v_x$", r"$\rho$"]
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

    var_subset = ["B_x", "E_x", "v_x", r"\rho"]
    var_labels = [r"$B_x$", r"$E_x$", r"$v_x$", r"$\rho$"]

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

            ax.set_yscale("symlog", linthresh=0.2)
            ax.grid(True, ls=":", lw=0.5)
            ax.tick_params(axis="both", labelsize=8)
            ax.set_xticks(np.arange(0, 31, 10))
            ax.set_ylim(-25, 0.25)
            ax.yaxis.set_major_locator(SymmetricalLogLocator(base=10, linthresh=0.2))
            ax.yaxis.set_minor_locator(
                SymmetricalLogLocator(base=10, linthresh=0.2, subs=np.arange(0.2, 10))
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

    features = ["Bx", "Ex", "vx", "rho", "P", "T", "By", "Bz", "Ey", "Ez", "vy", "vz"]
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


def plot_power_spectra_grid_avg_runs(
    efm_run_ids,
    fm_run_ids,
    input_dir,
    output_dir,
    prefix,
    file_name="example_1.zarr",
):
    """
    Plot power spectra comparing mean-over-runs Graph-FM, Graph-EFM, and Vlasiator
    at four forecast times (1s, 10s, 30s, 50s).

    Averages are taken over runs in spectral space:
      P_avg(k) = mean_r P_r(k)
    """

    assert len(efm_run_ids) == len(
        fm_run_ids
    ), "EFM and FM run lists must match length."

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

    # Load one FM dataset to define grid spacing / bins
    ds_fm0 = load_ds(fm_run_ids[0])
    dx = float(ds_fm0.x[1] - ds_fm0.x[0])
    dz = float(ds_fm0.z[1] - ds_fm0.z[0])

    features = ["Bx", "Ex", "vx", "rho", "P", "T", "By", "Bz", "Ey", "Ez", "vy", "vz"]
    time_indices = [0, 9, 29, 49]
    time_labels = [1, 10, 30, 50]

    halves = [
        (features[:6], f"{prefix}_power_spectra_grid_1"),
        (features[6:], f"{prefix}_power_spectra_grid_2"),
    ]

    def get_target(ds_fm, var_name, t_idx):
        target = ds_fm["target"].sel(state_feature=var_name).isel(time=t_idx)
        if set(["x", "z"]).issubset(target.dims):
            target = target.transpose("z", "x")
        return target.values

    def get_fm_pred(ds_fm, var_name, t_idx):
        arr = ds_fm["prediction"].sel(state_feature=var_name).isel(time=t_idx)
        if set(["x", "z"]).issubset(arr.dims):
            arr = arr.transpose("z", "x")
        return arr.values

    def get_efm_mean(ds_efm, var_name, t_idx):
        key = "ens_mean" if "ens_mean" in ds_efm.data_vars else "prediction"
        arr = ds_efm[key].sel(state_feature=var_name).isel(time=t_idx)
        if set(["x", "z"]).issubset(arr.dims):
            arr = arr.transpose("z", "x")
        return arr.values

    # Pre-load datasets (avoids re-opening per panel)
    ds_fms = [load_ds(rid) for rid in fm_run_ids]
    ds_efms = [load_ds(rid) for rid in efm_run_ids]

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

                # Accumulate spectra over runs
                P_t_list, P_fm_list, P_efm_list = [], [], []
                k_ref = None

                for ds_fm, ds_efm in zip(ds_fms, ds_efms):
                    target = get_target(ds_fm, var_name, t_idx)
                    fm = get_fm_pred(ds_fm, var_name, t_idx)
                    efm = get_efm_mean(ds_efm, var_name, t_idx)

                    k_t, P_t = compute_power_spectrum(target, dx=dx, dz=dz)
                    k_fm, P_fm = compute_power_spectrum(fm, dx=dx, dz=dz)
                    k_efm, P_efm = compute_power_spectrum(efm, dx=dx, dz=dz)

                    # Assumes consistent k bins, same shape
                    if k_ref is None:
                        k_ref = k_t
                    P_t_list.append(P_t)
                    P_fm_list.append(P_fm)
                    P_efm_list.append(P_efm)

                # Mean over runs, ignoring NaNs
                P_t_mean = np.nanmean(np.stack(P_t_list, axis=0), axis=0)
                P_fm_mean = np.nanmean(np.stack(P_fm_list, axis=0), axis=0)
                P_efm_mean = np.nanmean(np.stack(P_efm_list, axis=0), axis=0)

                ax.plot(k_ref, P_fm_mean, color="C0", ls="-", lw=1.3)
                ax.plot(k_ref, P_efm_mean, color="C1", ls="-", lw=1.3)
                ax.plot(k_ref, P_t_mean, color="black", ls="--", lw=1.3)

                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.grid(True, which="both", ls=":", lw=0.5)
                ax.tick_params(axis="both", labelsize=9)

                if col_idx == 0:
                    ax.set_ylabel(r"Power $P(k)$", fontsize=10)
                if row_idx == len(half_features) - 1:
                    ax.set_xlabel(r"Wavenumber $k$", fontsize=10)

                ax.set_title(
                    f"{latex_vars.get(var_name, var_name)}, $t={t_val}\\,\\mathrm{{s}}$",
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


def _pearson(y_true: np.ndarray, y_pred: np.ndarray):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt = y_true[mask].astype(np.float64)
    yp = y_pred[mask].astype(np.float64)
    if yt.size < 3:
        return np.nan, np.nan

    yt0 = yt - np.mean(yt)
    yp0 = yp - np.mean(yp)
    denom = np.sqrt(np.sum(yt0**2) * np.sum(yp0**2))
    r = np.sum(yt0 * yp0) / denom if denom > 0 else np.nan
    return r


def _best_fit_line(y_true: np.ndarray, y_pred: np.ndarray):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    x = y_true[mask].astype(np.float64)
    y = y_pred[mask].astype(np.float64)
    if x.size < 2:
        return np.nan, np.nan
    a, b = np.polyfit(x, y, 1)  # y ≈ a*x + b
    return a, b


def plot_pred_vs_true_heatmaps_avg_runs(
    efm_run_ids,
    fm_run_ids,
    input_dir,
    output_dir,
    prefix,
    file_name="example_1.zarr",
    features=None,
    time_indices=(0, 9, 29, 49),
    time_labels=(1, 10, 30, 50),
    bins=120,
    standardize_per_var=True,
):
    """
    Make a 2x4 grid of pred-vs-true density heatmaps using log counts,
      - Row 0: Graph-FM (avg over runs and variables)
      - Row 1: Graph-EFM (avg over runs and variables)
      - Cols: lead times

    Overlays: 1:1 line, best-fit line, Pearson score.
    """

    assert len(efm_run_ids) == len(
        fm_run_ids
    ), "EFM and FM run lists must match length."
    assert len(time_indices) == len(
        time_labels
    ), "time_indices and time_labels must match length."

    if features is None:
        features = [
            "Bx",
            "Ex",
            "vx",
            "rho",
            "P",
            "T",
            "By",
            "Bz",
            "Ey",
            "Ez",
            "vy",
            "vz",
        ]

    def load_ds(run_id):
        path = os.path.join(input_dir, run_id, file_name)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return xr.open_zarr(path).fillna(0)

    ds_fms = [load_ds(rid) for rid in fm_run_ids]
    ds_efms = [load_ds(rid) for rid in efm_run_ids]

    def _get_target(ds_fm, var_name, t_idx):
        arr = ds_fm["target"].sel(state_feature=var_name).isel(time=t_idx)
        if set(["x", "z"]).issubset(arr.dims):
            arr = arr.transpose("z", "x")
        return arr.values

    def _get_fm_pred(ds_fm, var_name, t_idx):
        arr = ds_fm["prediction"].sel(state_feature=var_name).isel(time=t_idx)
        if set(["x", "z"]).issubset(arr.dims):
            arr = arr.transpose("z", "x")
        return arr.values

    def _get_efm_mean(ds_efm, var_name, t_idx):
        key = "ens_mean" if "ens_mean" in ds_efm.data_vars else "prediction"
        arr = ds_efm[key].sel(state_feature=var_name).isel(time=t_idx)
        if set(["x", "z"]).issubset(arr.dims):
            arr = arr.transpose("z", "x")
        return arr.values

    def _standardize(x, mean, std):
        std = std if std > 0 else 1.0
        return (x - mean) / std

    fig, axes = plt.subplots(
        nrows=2,
        ncols=len(time_indices),
        figsize=(10.5, 5.2),
        constrained_layout=True,
        sharex=False,
        sharey=False,
    )

    im_for_cbar = None

    for col_idx, (t_idx, t_val) in enumerate(zip(time_indices, time_labels)):

        pooled = {
            "fm": {"yt": [], "yp": []},
            "efm": {"yt": [], "yp": []},
        }

        # Pool across runs and variables
        for ds_fm, ds_efm in zip(ds_fms, ds_efms):
            for var in features:
                yt = _get_target(ds_fm, var, t_idx).ravel()
                yp_fm = _get_fm_pred(ds_fm, var, t_idx).ravel()
                yp_efm = _get_efm_mean(ds_efm, var, t_idx).ravel()

                mask_fm = np.isfinite(yt) & np.isfinite(yp_fm)
                mask_efm = np.isfinite(yt) & np.isfinite(yp_efm)

                idx_fm = np.where(mask_fm)[0]
                idx_efm = np.where(mask_efm)[0]

                yt_fm = yt[idx_fm]
                yp_fm = yp_fm[idx_fm]
                yt_efm = yt[idx_efm]
                yp_efm = yp_efm[idx_efm]

                # Optional per-variable standardization
                if standardize_per_var:
                    m = np.nanmean(yt)
                    s = np.nanstd(yt)
                    yt_fm = _standardize(yt_fm, m, s)
                    yp_fm = _standardize(yp_fm, m, s)
                    yt_efm = _standardize(yt_efm, m, s)
                    yp_efm = _standardize(yp_efm, m, s)

                pooled["fm"]["yt"].append(yt_fm)
                pooled["fm"]["yp"].append(yp_fm)
                pooled["efm"]["yt"].append(yt_efm)
                pooled["efm"]["yp"].append(yp_efm)

        for row_idx, model in enumerate(["fm", "efm"]):
            ax = axes[row_idx, col_idx]

            yt = (
                np.concatenate(pooled[model]["yt"])
                if pooled[model]["yt"]
                else np.array([])
            )
            yp = (
                np.concatenate(pooled[model]["yp"])
                if pooled[model]["yp"]
                else np.array([])
            )

            if yt.size == 0 or yp.size == 0:
                ax.set_axis_off()
                continue

            # Bounds
            pooled_vals = np.concatenate([yt, yp])
            lo = np.nanpercentile(pooled_vals, 0.5)
            hi = np.nanpercentile(pooled_vals, 99.5)
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = np.nanmin(pooled_vals), np.nanmax(pooled_vals)
                if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                    lo, hi = -1.0, 1.0

            # 2D histogram heatmap (log counts)
            H, _, _ = np.histogram2d(yt, yp, bins=bins, range=[[lo, hi], [lo, hi]])
            im = ax.imshow(
                H.T,
                origin="lower",
                extent=[lo, hi, lo, hi],
                aspect="equal",
                cmap="viridis",
                norm=LogNorm(vmin=1, vmax=max(2, float(np.nanmax(H)))),
            )
            im_for_cbar = im

            # 1:1 line
            ax.plot([lo, hi], [lo, hi], color="black", ls="--", lw=1.0)

            # Best-fit line
            a, b = _best_fit_line(yt, yp)
            if np.isfinite(a) and np.isfinite(b):
                xs = np.array([lo, hi])
                ax.plot(xs, a * xs + b, color="white", lw=1.2, alpha=0.9)

            # Metrics
            r = _pearson(yt, yp)
            ax.text(
                0.05,
                0.95,
                rf"$r={r:.3f}$",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=14,
                color="white",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="black",
                    alpha=0.75,
                    edgecolor="none",
                ),
            )

            # Titles / labels
            if row_idx == 0:
                ax.set_title(rf"$t={t_val}\,\mathrm{{s}}$", fontsize=14)

            if col_idx == 0:
                ax.set_ylabel("Graph-FM" if row_idx == 0 else "Graph-EFM", fontsize=14)

            if row_idx == 1:
                ax.set_xlabel("Vlasiator", fontsize=14)
            ax.tick_params(axis="both", labelsize=12)

    # Shared colorbar
    if im_for_cbar is not None:
        cbar = fig.colorbar(
            im_for_cbar,
            ax=axes.ravel().tolist(),
            fraction=0.03,
            aspect=35,
            pad=0.02,
            shrink=0.92,
        )
        cbar.set_label("Count (log)", fontsize=14)
        cbar.ax.tick_params(labelsize=12)

    # Save
    for ext in ["png", "pdf"]:
        out_path = os.path.join(output_dir, f"{prefix}_pred_vs_true.{ext}")
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved pooled all-vars pred-vs-true heatmap grid to {out_path}")


def plot_efm_pred_vs_true_heatmaps_corr_per_run(
    efm_run_ids,
    input_dir,
    output_dir,
    file_name="example_1.zarr",
    features=None,
    time_indices=(0, 9, 29, 49),
    time_labels=(1, 10, 30, 50),
    bins=120,
    standardize_per_var=True,
    sample_max_per_var=None,  # e.g. 100_000 to sample a subset
    seed=0,
):
    """
    4x4 figure: rows = Run 1..4 (EFM runs), cols = lead times.
    Each panel shows pred-vs-true density heatmap (viridis, log counts) pooled over variables,
    with 1:1 line, best-fit line, and Pearson correlation annotation.
    """

    assert len(time_indices) == len(
        time_labels
    ), "time_indices and time_labels must match length."
    if features is None:
        features = [
            "Bx",
            "Ex",
            "vx",
            "rho",
            "P",
            "T",
            "By",
            "Bz",
            "Ey",
            "Ez",
            "vy",
            "vz",
        ]

    rng = np.random.default_rng(seed)

    def load_ds(run_id):
        path = os.path.join(input_dir, run_id, file_name)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return xr.open_zarr(path).fillna(0)

    dss = [load_ds(rid) for rid in efm_run_ids]

    def _get_target(ds, var_name, t_idx):
        arr = ds["target"].sel(state_feature=var_name).isel(time=t_idx)
        if set(["x", "z"]).issubset(arr.dims):
            arr = arr.transpose("z", "x")
        return arr.values

    def _get_efm_mean(ds, var_name, t_idx):
        key = "ens_mean" if "ens_mean" in ds.data_vars else "prediction"
        arr = ds[key].sel(state_feature=var_name).isel(time=t_idx)
        if set(["x", "z"]).issubset(arr.dims):
            arr = arr.transpose("z", "x")
        return arr.values

    def _standardize(x, mean, std):
        std = std if std > 0 else 1.0
        return (x - mean) / std

    fig, axes = plt.subplots(
        nrows=len(efm_run_ids),
        ncols=len(time_indices),
        figsize=(10.5, 10.0),
        constrained_layout=True,
        sharex=False,
        sharey=False,
    )

    im_for_cbar = None

    for row_idx, ds in enumerate(dss):
        run_label = f"Run {row_idx + 1}"

        for col_idx, (t_idx, t_val) in enumerate(zip(time_indices, time_labels)):
            ax = axes[row_idx, col_idx]

            yt_list, yp_list = [], []

            # Pool across variables within this run
            for var in features:
                yt = _get_target(ds, var, t_idx).ravel()
                yp = _get_efm_mean(ds, var, t_idx).ravel()

                mask = np.isfinite(yt) & np.isfinite(yp)
                idx = np.where(mask)[0]
                if sample_max_per_var is not None and idx.size > sample_max_per_var:
                    idx = rng.choice(idx, size=sample_max_per_var, replace=False)

                yt = yt[idx]
                yp = yp[idx]

                if standardize_per_var:
                    m = np.nanmean(yt)
                    s = np.nanstd(yt)
                    yt = _standardize(yt, m, s)
                    yp = _standardize(yp, m, s)

                yt_list.append(yt)
                yp_list.append(yp)

            yt = np.concatenate(yt_list) if yt_list else np.array([])
            yp = np.concatenate(yp_list) if yp_list else np.array([])

            if yt.size == 0:
                ax.set_axis_off()
                continue

            # Panel bounds
            pooled_vals = np.concatenate([yt, yp])
            lo = np.nanpercentile(pooled_vals, 0.5)
            hi = np.nanpercentile(pooled_vals, 99.5)
            if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                lo, hi = np.nanmin(pooled_vals), np.nanmax(pooled_vals)
                if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                    lo, hi = -1.0, 1.0

            # Heatmap (log counts)
            H, _, _ = np.histogram2d(yt, yp, bins=bins, range=[[lo, hi], [lo, hi]])
            im = ax.imshow(
                H.T,
                origin="lower",
                extent=[lo, hi, lo, hi],
                aspect="equal",
                cmap="viridis",
                norm=LogNorm(vmin=1, vmax=max(2, float(np.nanmax(H)))),
            )
            im_for_cbar = im

            # 1:1 and best-fit lines
            ax.plot([lo, hi], [lo, hi], color="black", ls="--", lw=1.0)
            a, b = _best_fit_line(yt, yp)
            if np.isfinite(a) and np.isfinite(b):
                xs = np.array([lo, hi])
                ax.plot(xs, a * xs + b, color="white", lw=1.2, alpha=0.9)

            # Pearson annotation
            r = _pearson(yt, yp)
            ax.text(
                0.05,
                0.95,
                rf"r$={r:.3f}$",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                color="white",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="black",
                    alpha=0.75,
                    edgecolor="none",
                ),
            )

            # Titles / labels
            if row_idx == 0:
                ax.set_title(rf"$t={t_val}\,\mathrm{{s}}$", fontsize=12)
            if col_idx == 0:
                ax.set_ylabel(run_label, fontsize=12)

            if row_idx == len(efm_run_ids) - 1:
                ax.set_xlabel("Vlasiator", fontsize=11)
            ax.tick_params(axis="both", labelsize=8)

    # Shared colorbar
    if im_for_cbar is not None:
        cbar = fig.colorbar(
            im_for_cbar,
            ax=axes.ravel().tolist(),
            fraction=0.025,
            aspect=45,
            pad=0.02,
            shrink=0.98,
        )
        cbar.set_label("Count (log)", fontsize=11)
        cbar.ax.tick_params(labelsize=9)

    for ext in ["png", "pdf"]:
        out_path = os.path.join(output_dir, f"efm_pred_vs_true.{ext}")
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved EFM per-run pred-vs-true correlation heatmap grid to {out_path}")


def plot_pred_vs_true_heatmaps_per_run(
    efm_run_ids,
    fm_run_ids,
    input_dir,
    output_dir,
    prefix,
    **kwargs,
):
    """
    Calls plot_pred_vs_true_heatmaps_avg_runs() once per run.
    Produces files like:
      <prefix>_run_1_pred_vs_true.(png|pdf)
      ...
    """
    assert len(efm_run_ids) == len(fm_run_ids)

    for efm_id, fm_id in zip(efm_run_ids, fm_run_ids):
        # Derive run tag (e.g. run_1 from run_1_efm)
        run_tag = efm_id.replace("_efm", "")
        plot_pred_vs_true_heatmaps_avg_runs(
            efm_run_ids=[efm_id],
            fm_run_ids=[fm_id],
            input_dir=input_dir,
            output_dir=output_dir,
            prefix="run_tag",
            **kwargs,
        )


def plot_efm_pred_vs_true_heatmaps_by_var_avg_runs(
    efm_run_ids,
    input_dir,
    output_dir,
    file_name="example_1.zarr",
    features=None,
    t_idx=49,
    t_val=50,
    bins=120,
    sample_max_per_run=None,  # e.g. 300_000 if you need speed
    seed=0,
    percentiles=(0.5, 99.5),
):
    """
    3x4 grid of pred-vs-true density heatmaps for Graph-EFM, one panel per variable.
    - Compile grid points across runs, no standardization.
    - Overlays: 1:1 line, best-fit line, Pearson r.
    """
    features = ["Bx", "By", "Bz", "Ex", "Ey", "Ez", "vx", "vy", "vz", "rho", "P", "T"]

    rng = np.random.default_rng(seed)

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

    dss = [load_ds(rid) for rid in efm_run_ids]

    def _get_target(ds, var_name, t_idx):
        arr = ds["target"].sel(state_feature=var_name).isel(time=t_idx)
        if set(["x", "z"]).issubset(arr.dims):
            arr = arr.transpose("z", "x")
        return arr.values

    def _get_pred(ds, var_name, t_idx):
        key = "ens_mean" if "ens_mean" in ds.data_vars else "prediction"
        arr = ds[key].sel(state_feature=var_name).isel(time=t_idx)
        if set(["x", "z"]).issubset(arr.dims):
            arr = arr.transpose("z", "x")
        return arr.values

    fig, axes = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(7, 9),
        constrained_layout=True,
        sharex=False,
        sharey=False,
    )
    axes = axes.reshape(4, 3)

    for j, var in enumerate(features):
        r0, c0 = divmod(j, 3)
        ax = axes[r0, c0]

        # Pool across runs
        yts, yps = [], []
        for ds in dss:
            yt = _get_target(ds, var, t_idx).ravel()
            yp = _get_pred(ds, var, t_idx).ravel()

            mask = np.isfinite(yt) & np.isfinite(yp)
            idx = np.where(mask)[0]

            if sample_max_per_run is not None and idx.size > sample_max_per_run:
                idx = rng.choice(idx, size=sample_max_per_run, replace=False)

            yts.append(yt[idx])
            yps.append(yp[idx])

        yt = np.concatenate(yts) if yts else np.array([])
        yp = np.concatenate(yps) if yps else np.array([])

        # Panel bounds
        lo = np.nanpercentile(yt, percentiles[0])
        hi = np.nanpercentile(yt, percentiles[1])

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        # 2D histogram heatmap (log counts)
        H, _, _ = np.histogram2d(yt, yp, bins=bins, range=[[lo, hi], [lo, hi]])
        im = ax.imshow(
            H.T,
            origin="lower",
            extent=[lo, hi, lo, hi],
            aspect="equal",
            cmap="viridis",
            norm=LogNorm(vmin=1, vmax=max(2, float(np.nanmax(H)))),
        )

        # 1:1 line
        ax.plot([lo, hi], [lo, hi], color="black", ls="--", lw=1.0)

        # Best-fit line
        a, b = _best_fit_line(yt, yp)
        if np.isfinite(a) and np.isfinite(b):
            xs = np.array([lo, hi])
            ax.plot(xs, a * xs + b, color="white", lw=1.2, alpha=0.9)

        # Pearson annotation
        r = _pearson(yt, yp)
        ax.text(
            0.05,
            0.95,
            rf"r$={r:.3f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="black",
                alpha=0.75,
                edgecolor="none",
            ),
        )

        ax.set_title(latex_vars.get(var, var), fontsize=12)
        ax.tick_params(axis="both", labelsize=10)

        # Labels only on outer axes
        if r0 == 3:
            ax.set_xlabel("Vlasiator", fontsize=11)
        if c0 == 0:
            ax.set_ylabel("Graph-EFM", fontsize=11)

    # Shared colorbar
    cbar = fig.colorbar(
        im,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        fraction=0.05,
        pad=0.03,
        aspect=40,
        shrink=0.6,
    )
    cbar.set_label("Count (log)", fontsize=10)
    cbar.ax.tick_params(labelsize=10)

    for ext in ["png", "pdf"]:
        out_path = os.path.join(output_dir, f"pred_vs_true_by_var_t{t_val:02d}.{ext}")
        plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved EFM by-variable pred-vs-true grid to {out_path}")


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    plot_fm_efm_comparison(args.metrics_dir, args.output_dir)
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

    efm_runs = ["run_1_efm", "run_2_efm", "run_3_efm", "run_4_efm"]
    fm_runs = ["run_1_fm", "run_2_fm", "run_3_fm", "run_4_fm"]

    plot_power_spectra_grid_avg_runs(
        efm_runs, fm_runs, args.forecasts_dir, args.output_dir, prefix="run_avg"
    )

    plot_pred_vs_true_heatmaps_avg_runs(
        efm_runs,
        fm_runs,
        args.forecasts_dir,
        args.output_dir,
        prefix="all_runs",
        standardize_per_var=True,
    )

    plot_pred_vs_true_heatmaps_per_run(
        efm_runs,
        fm_runs,
        args.forecasts_dir,
        args.output_dir,
        standardize_per_var=True,
    )

    plot_efm_pred_vs_true_heatmaps_corr_per_run(
        efm_runs,
        args.forecasts_dir,
        args.output_dir,
        standardize_per_var=True,
    )

    time_indices = (0, 9, 29, 49)
    time_labels = (1, 10, 30, 50)
    for t_idx, t_val in zip(time_indices, time_labels):
        plot_efm_pred_vs_true_heatmaps_by_var_avg_runs(
            efm_runs,
            args.forecasts_dir,
            args.output_dir,
            t_idx=t_idx,
            t_val=t_val,
            sample_max_per_run=None,
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
