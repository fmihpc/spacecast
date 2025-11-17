# Standard library
from typing import List

# Third-party
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

# Local
from . import utils
from .datastore.base import BaseRegularGridDatastore


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_error_map(errors, datastore: BaseRegularGridDatastore, title=None):
    """
    Plot a heatmap of errors of different variables at different
    predictions horizons
    errors: (pred_steps, d_f)
    """
    errors_np = errors.T.cpu().numpy()  # (d_f, pred_steps)
    d_f, pred_steps = errors_np.shape
    step_length = datastore.step_length

    # Normalize all errors to [0,1] for color map
    max_errors = errors_np.max(axis=1)  # d_f
    errors_norm = errors_np / np.expand_dims(max_errors, axis=1)

    if d_f == 1:
        fig, ax = plt.subplots(figsize=(15, 3))
    else:
        fig, ax = plt.subplots(figsize=(15, 10))

    ax.imshow(
        errors_norm,
        cmap="OrRd",
        vmin=0,
        vmax=1.0,
        interpolation="none",
        aspect="auto",
        alpha=0.8,
    )

    # ax and labels
    for (j, i), error in np.ndenumerate(errors_np):
        # Numbers > 9999 will be too large to fit
        formatted_error = f"{error:.3f}" if error < 9999 else f"{error:.2E}"
        ax.text(i, j, formatted_error, ha="center", va="center", usetex=False)

    # Ticks and labels
    label_size = 15
    ax.set_xticks(np.arange(pred_steps))
    pred_hor_i = np.arange(pred_steps) + 1  # Prediction horiz. in index
    pred_hor_s = step_length * pred_hor_i  # Prediction horiz. in seconds
    ax.set_xticklabels(pred_hor_s, size=label_size)
    ax.set_xlabel("Lead time (s)", size=label_size)

    ax.set_yticks(np.arange(d_f))
    var_names = datastore.get_vars_names(category="state")
    var_units = datastore.get_vars_units(category="state")
    if d_f == 1:
        # Magnetic divergence
        y_ticklabels = ["Div B (nT / Re)"]
    else:
        y_ticklabels = [f"{name} ({unit})" for name, unit in zip(var_names, var_units)]
    ax.set_yticklabels(y_ticklabels, rotation=30, size=label_size)

    if title:
        ax.set_title(title, size=15)

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_prediction(
    datastore: BaseRegularGridDatastore,
    da_prediction: xr.DataArray = None,
    da_target: xr.DataArray = None,
    title=None,
    vrange=None,
):
    """
    Plot example prediction and grond truth.

    Each has shape (N_grid,)

    """
    # Get common scale for values
    if vrange is None:
        vmin = min(da_prediction.min(), da_target.min())
        vmax = max(da_prediction.max(), da_target.max())
    else:
        vmin, vmax = vrange

    extent = datastore.get_xz_extent("state")

    # Set up masking of border region
    da_mask = (
        datastore.unstack_grid_coords(datastore.boundary_mask).isel(mask_feature=0).T
    )
    mask_values = np.invert(da_mask.values.astype(bool)).astype(float)
    pixel_alpha = mask_values.clip(0.7, 1)  # Faded border region

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6),
    )

    # Plot pred and target
    for ax, da in zip(axes, (da_target, da_prediction)):
        da.plot.imshow(
            ax=ax,
            origin="lower",
            x="x",
            y="z",
            extent=extent,
            alpha=pixel_alpha,
            vmin=vmin,
            vmax=vmax,
            cmap="plasma",
        )
        ax.grid(False)

    # Ticks and labels
    axes[0].set_title("Ground Truth", size=15)
    axes[1].set_title("Prediction", size=15)

    if title:
        fig.suptitle(title, size=20)

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_ensemble_prediction(
    samples: List[xr.DataArray],
    target: xr.DataArray,
    ens_mean: xr.DataArray,
    ens_std: xr.DataArray,
    datastore: BaseRegularGridDatastore,
    title=None,
    var_name=None,
    var_unit=None,
    vrange=None,
):
    """
    Plot example predictions, ground truth, mean and std.-dev.
    from ensemble forecast

    samples: (S, N_grid,)
    target: (N_grid,)
    ens_mean: (N_grid,)
    ens_std: (N_grid,)
    datastore: grid information
    (optional) title: title of plot
    (optional) var_name: name of variable
    (optional) var_unit: unit of variable
    (optional) vrange: tuple of length with common min and max of values
        (not for std.)
    """

    # Define variables that should use a diverging colormap
    diverging_vars = [
        f"{prefix}{suffix}" for prefix in ("E", "B", "v") for suffix in ("x", "y", "z")
    ]

    if var_name in diverging_vars:
        cmap = "RdBu_r"
        if vrange is None:
            vrange_vals = samples + [target, ens_mean]
            # Create symmetric color range
            max_abs = max(abs(vals).max().values for vals in vrange_vals)
            vmin, vmax = -max_abs, max_abs
        else:
            max_abs = max(abs(vals) for vals in vrange)
            vmin, vmax = -max_abs, max_abs
    else:
        # Default to viridis for all other features
        cmap = "viridis"
        if vrange is None:
            vrange_vals = samples + [target, ens_mean]
            vmin = min(vals.min().values for vals in vrange_vals)
            vmax = max(vals.max().values for vals in vrange_vals)
        else:
            vmin, vmax = vrange

    # Set up masking of border region
    da_mask = (
        datastore.unstack_grid_coords(datastore.boundary_mask).isel(mask_feature=0).T
    )
    mask_values = np.invert(da_mask.values.astype(bool)).astype(float)
    pixel_alpha = mask_values.clip(0.7, 1)  # Faded border region

    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10.5, 7), sharex=True, sharey=True)
    axes = axes.flatten()

    # Plot target, ensemble mean and std.
    plot_on_axis(
        axes[0],
        target,
        datastore,
        alpha=pixel_alpha,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        ax_title="Ground Truth",
    )
    plot_on_axis(
        axes[1],
        ens_mean,
        datastore,
        alpha=pixel_alpha,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        ax_title="Ens. Mean",
    )
    member_im = plot_on_axis(
        axes[2],
        samples[0],
        datastore,
        alpha=pixel_alpha,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        ax_title="Ens. Member",
    )
    std_im = plot_on_axis(
        axes[3],
        ens_std,
        datastore,
        alpha=pixel_alpha,
        ax_title="Ens. Std.",
        cmap="OrRd",
    )  # Own vrange

    # Add colorbars
    values_cbar = fig.colorbar(
        member_im, ax=axes[1], orientation="vertical", aspect=30, pad=0.02
    )
    values_cbar.ax.tick_params(labelsize=10)
    if var_unit:
        values_cbar.set_label(var_unit, size=10)

    std_cbar = fig.colorbar(
        std_im, ax=axes[3], orientation="vertical", aspect=30, pad=0.02
    )
    std_cbar.ax.tick_params(labelsize=10)
    if var_unit:
        std_cbar.set_label(var_unit, size=10)

    for i, ax in enumerate(axes):
        if i % ncols == 0:
            ax.set_ylabel(r"z ($R_E$)", fontsize=12)
        if i >= (nrows - 1) * ncols:
            ax.set_xlabel(r"x ($R_E$)", fontsize=12)

    if title:
        fig.suptitle(title, size=20)

    return fig


def plot_on_axis(
    ax,
    data: xr.DataArray,
    datastore: BaseRegularGridDatastore,
    alpha=None,
    vmin=None,
    vmax=None,
    ax_title=None,
    cmap="plasma",
):
    """
    Plot weather state on given axis
    """
    extent = datastore.get_xz_extent("state")

    im = data.plot.imshow(
        ax=ax,
        origin="lower",
        x="x",
        y="z",
        extent=extent,
        alpha=alpha,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        add_colorbar=False,
        add_labels=False,
    )
    ax.grid(False)

    # May change/remove hardcoded ticks and spacing
    ax.set_xticks(np.arange(-60, 31, 10))
    ax.set_yticks(np.arange(-30, 31, 10))
    ax.tick_params(axis="both", labelsize=10)

    if ax_title:
        ax.set_title(ax_title, size=12)
    return im


def plot_spatial_error(
    error, datastore: BaseRegularGridDatastore, title=None, vrange=None
):
    """
    Plot errors over spatial map
    Error and obs_mask has shape (N_grid,)
    """
    # Get common scale for values
    if vrange is None:
        vmin = error.min().cpu().item()
        vmax = error.max().cpu().item()
    else:
        vmin, vmax = vrange

    extent = datastore.get_xz_extent("state")

    # Set up masking of border region
    da_mask = (
        datastore.unstack_grid_coords(datastore.boundary_mask).isel(mask_feature=0).T
    )
    mask_reshaped = da_mask.values
    pixel_alpha = mask_reshaped.clip(0.7, 1)  # Faded border region

    space_mask = datastore.get_mask(stacked=True, invert=False)
    full_error = np.full_like(space_mask, np.nan)
    full_error[space_mask] = error.cpu().numpy()

    fig, ax = plt.subplots(
        figsize=(5, 4.8),
    )

    error_grid = full_error.reshape(
        [datastore.grid_shape_state.x, datastore.grid_shape_state.z]
    ).T

    im = ax.imshow(
        error_grid,
        origin="lower",
        extent=extent,
        alpha=pixel_alpha,
        vmin=vmin,
        vmax=vmax,
        cmap="OrRd",
    )

    # Ticks and labels
    cbar = fig.colorbar(im, aspect=30)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.get_offset_text().set_fontsize(10)
    cbar.formatter.set_powerlimits((-3, 3))

    if title:
        fig.suptitle(title, size=10)

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_latent_samples(prior_samples, vi_samples, title=None):
    """
    Plot samples of latent variable drawn from prior and
    variational distribution

    prior_samples: (samples, N_mesh, d_latent)
    vi_samples: (samples, N_mesh, d_latent)

    Returns:
    fig: the plot figure
    """
    num_samples, num_mesh_nodes, latent_dim = prior_samples.shape
    plot_dims = min(latent_dim, 3)  # Plot first 3 dimensions
    img_side_size = int(np.sqrt(num_mesh_nodes))

    # Check if number of nodes is a square
    if img_side_size**2 != num_mesh_nodes:
        # Number of mesh nodes is not a square number, can not directly plot
        # latent samples as images"
        # Fix this by not plotting all nodes (choose amount to work as image)
        num_mesh_subset = img_side_size**2
        prior_samples = prior_samples[:, :num_mesh_subset]
        vi_samples = vi_samples[:, :num_mesh_subset]

    # Get common scale for values
    vmin = min(
        vals[..., :plot_dims].min().cpu().item() for vals in (prior_samples, vi_samples)
    )
    vmax = max(
        vals[..., :plot_dims].max().cpu().item() for vals in (prior_samples, vi_samples)
    )

    # Create figure
    fig, axes = plt.subplots(num_samples, 2 * plot_dims, figsize=(20, 16))

    # Plot samples
    for row_i, (axes_row, prior_sample, vi_sample) in enumerate(
        zip(axes, prior_samples, vi_samples)
    ):

        for dim_i in range(plot_dims):
            prior_sample_reshaped = (
                prior_sample[:, dim_i]
                .reshape(img_side_size, img_side_size)
                .cpu()
                .to(torch.float32)
                .numpy()
            )
            vi_sample_reshaped = (
                vi_sample[:, dim_i]
                .reshape(img_side_size, img_side_size)
                .cpu()
                .to(torch.float32)
                .numpy()
            )
            # Plot every other as prior and vi
            prior_ax = axes_row[2 * dim_i]
            vi_ax = axes_row[2 * dim_i + 1]
            prior_ax.imshow(prior_sample_reshaped, vmin=vmin, vmax=vmax)
            vi_im = vi_ax.imshow(vi_sample_reshaped, vmin=vmin, vmax=vmax)

            if row_i == 0:
                # Add titles at top of columns
                prior_ax.set_title(f"d{dim_i} (prior)", size=15)
                vi_ax.set_title(f"d{dim_i} (vi)", size=15)

    # Remove ticks from all axes
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    # Add colorbar
    cbar = fig.colorbar(vi_im, ax=axes, aspect=60, location="bottom")
    cbar.ax.tick_params(labelsize=15)

    if title:
        fig.suptitle(title, size=20)

    return fig
