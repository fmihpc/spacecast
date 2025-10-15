# Third-party
import torch


def get_metric(metric_name):
    """
    Get a defined metric with given name

    metric_name: str, name of the metric

    Returns:
    metric: function implementing the metric
    """
    metric_name_lower = metric_name.lower()
    assert metric_name_lower in DEFINED_METRICS, f"Unknown metric: {metric_name}"
    return DEFINED_METRICS[metric_name_lower]


def mask_and_reduce_metric(metric_entry_vals, mask, average_grid, sum_vars):
    """
    Masks and (optionally) reduces entry-wise metric values

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    metric_entry_vals: (..., N, d_state), prediction
    mask: (N, d_state), mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """

    metric_entry_vals *= mask

    # Optionally reduce last two dimensions
    if average_grid:  # Reduce grid first
        metric_entry_vals = torch.sum(metric_entry_vals, dim=-2) / torch.sum(
            mask, dim=-2
        )
        # (..., d_state)
    if sum_vars:  # Reduce vars second
        metric_entry_vals = torch.sum(metric_entry_vals, dim=-1)  # (..., N) or (...,)

    return metric_entry_vals


def wmse(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True):
    """
    Weighted Mean Squared Error

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
    mask: (N,), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    entry_mse = torch.nn.functional.mse_loss(
        pred, target, reduction="none"
    )  # (..., N, d_state)
    entry_mse_weighted = entry_mse / (pred_std**2)  # (..., N, d_state)

    return mask_and_reduce_metric(
        entry_mse_weighted,
        mask=mask,
        average_grid=average_grid,
        sum_vars=sum_vars,
    )


# Allow for unused pred_std for consistent signature
# pylint: disable-next=unused-argument
def mse(pred, target, pred_std=None, mask=None, average_grid=True, sum_vars=True):
    """
    (Unweighted) Mean Squared Error

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev. (unused)
    mask: (N,), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    # Replace pred_std with constant ones
    return wmse(pred, target, torch.ones_like(pred), mask, average_grid, sum_vars)


def wmae(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True):
    """
    Weighted Mean Absolute Error

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
    mask: (N,), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    entry_mae = torch.nn.functional.l1_loss(
        pred, target, reduction="none"
    )  # (..., N, d_state)
    entry_mae_weighted = entry_mae / pred_std  # (..., N, d_state)

    return mask_and_reduce_metric(
        entry_mae_weighted,
        mask=mask,
        average_grid=average_grid,
        sum_vars=sum_vars,
    )


# Allow for unused pred_std for consistent signature
# pylint: disable-next=unused-argument
def mae(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True):
    """
    (Unweighted) Mean Absolute Error

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev. (unused)
    mask: (N,), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    # Replace pred_std with constant ones
    return wmae(pred, target, torch.ones_like(pred), mask, average_grid, sum_vars)


def nll(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True):
    """
    Negative Log Likelihood loss, for isotropic Gaussian likelihood

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
    mask: (N,), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    # Broadcast pred_std if shaped (d_state,), done internally in Normal class
    dist = torch.distributions.Normal(pred, pred_std)  # (..., N, d_state)
    entry_nll = -dist.log_prob(target)  # (..., N, d_state)

    return mask_and_reduce_metric(
        entry_nll, mask=mask, average_grid=average_grid, sum_vars=sum_vars
    )


def div_b(pred, mask, bx_idx, bz_idx, nx, nz, dx, dz, average_grid=True, sum_vars=True):
    """
    Compute divergence penalty (∂Bx/∂x + ∂Bz/∂z)² for the magnetic field.

    pred: (..., N, d_state), prediction
    mask: (N,), boolean mask describing which grid nodes to use in metric
    bx_idx, bz_idx: int, variable indices for Bx, Bz
    nx, nz: int, grid shape
    dx, dz: float, grid spacings
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    divergence_loss: scalar (or batch-dim tensor), mean squared divergence
    """

    # Reshape flattened Bx, Bz fields back to 2D (nx, nz) grid
    bx = pred[..., bx_idx].reshape(*pred.shape[:-2], nx, nz)
    bz = pred[..., bz_idx].reshape(*pred.shape[:-2], nx, nz)

    # Central differences in x and z (skip 1 layer boundaries)
    dbx_dx = (bx[..., 2:, 1:-1] - bx[..., :-2, 1:-1]) / (2 * dx)  # (..., nx-2, nz-2)
    dbz_dz = (bz[..., 1:-1, 2:] - bz[..., 1:-1, :-2]) / (2 * dz)  # (..., nx-2, nz-2)

    # Compute divergence on the intersection interior
    div_b_squared = (dbx_dx + dbz_dz) ** 2  # (..., nx-2, nz-2)

    # Flatten interior divergence field to (..., N_inner, 1)
    div_b_flat = div_b_squared.reshape(*div_b_squared.shape[:-2], -1, 1)

    return mask_and_reduce_metric(
        div_b_flat,
        mask=mask,
        average_grid=average_grid,
        sum_vars=sum_vars,
    )


def crps_gauss(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True):
    """
    (Negative) Continuous Ranked Probability Score (CRPS)
    Closed-form expression based on Gaussian predictive distribution

    (...,) is any number of batch dimensions, potentially different
            but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
    mask: (N,), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    std_normal = torch.distributions.Normal(
        torch.zeros((), device=pred.device), torch.ones((), device=pred.device)
    )
    target_standard = (target - pred) / pred_std  # (..., N, d_state)

    entry_crps = -pred_std * (
        torch.pi ** (-0.5)
        - 2 * torch.exp(std_normal.log_prob(target_standard))
        - target_standard * (2 * std_normal.cdf(target_standard) - 1)
    )  # (..., N, d_state)

    return mask_and_reduce_metric(
        entry_crps, mask=mask, average_grid=average_grid, sum_vars=sum_vars
    )


# Allow for unused pred_std for consistent signature
def crps_ens(
    pred,
    target,
    pred_std,  # pylint: disable=unused-argument
    mask=None,
    average_grid=True,
    sum_vars=True,
    ens_dim=1,
):
    """
    (Negative) Continuous Ranked Probability Score (CRPS)
    Unbiased estimator from samples. See e.g. Weatherbench 2.

    (..., M, ...,) is any number of batch dimensions, including ensemble
        dimension M
    pred: (..., M, ..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., M, ..., N, d_state) or (d_state,), predicted std.-dev.
    mask: (N,), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced
        (sum over d_state)
    ens_dim: batch dimension where ensemble members are laid out, to reduce over

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    num_ens = pred.shape[ens_dim]  # Number of ensemble members
    if num_ens == 1:
        # With one sample CRPS reduces to MAE
        return mae(
            pred.squeeze(ens_dim),
            target,
            None,
            mask=mask,
            average_grid=average_grid,
        )

    if num_ens == 2:
        mean_mae = torch.mean(
            torch.abs(pred - target.unsqueeze(ens_dim)), dim=ens_dim
        )  # (..., N, d_state)

        # Use simpler estimator
        pair_diffs_term = -0.5 * torch.abs(
            pred.select(ens_dim, 0) - pred.select(ens_dim, 1)
        )  # (..., N, d_state)

        crps_estimator = mean_mae + pair_diffs_term  # (..., N, d_state)
    elif num_ens < 10:
        # This is the rank-based implementation with O(M*log(M)) compute and
        # O(M) memory. See Zamo and Naveau and WB2 for explanation.
        # For smaller ensemble we can compute all of this directly in memory.
        mean_mae = torch.mean(
            torch.abs(pred - target.unsqueeze(ens_dim)), dim=ens_dim
        )  # (..., N, d_state)

        # Ranks start at 1, two argsorts will compute entry ranks
        ranks = pred.argsort(dim=ens_dim).argsort(ens_dim) + 1

        pair_diffs_term = (1 / (num_ens - 1)) * torch.mean(
            (num_ens + 1 - 2 * ranks) * pred,
            dim=ens_dim,
        )  # (..., N, d_state)

        crps_estimator = mean_mae + pair_diffs_term  # (..., N, d_state)
    else:
        # For large ensembles we batch this over the variable dimension
        crps_res = []
        for var_i in range(pred.shape[-1]):
            pred_var = pred[..., var_i]
            target_var = target[..., var_i]

            mean_mae = torch.mean(
                torch.abs(pred_var - target_var.unsqueeze(ens_dim)), dim=ens_dim
            )  # (..., N)

            # Ranks start at 1, two argsorts will compute entry ranks
            ranks = pred_var.argsort(dim=ens_dim).argsort(ens_dim) + 1
            # (..., M, ..., N)

            pair_diffs_term = (1 / (num_ens - 1)) * torch.mean(
                (num_ens + 1 - 2 * ranks) * pred_var,
                dim=ens_dim,
            )  # (..., N)
            crps_res.append(mean_mae + pair_diffs_term)

        crps_estimator = torch.stack(crps_res, dim=-1)

    return mask_and_reduce_metric(crps_estimator, mask, average_grid, sum_vars)


def spread_squared(
    pred,
    target,  # pylint: disable=unused-argument
    pred_std,  # pylint: disable=unused-argument
    mask=None,
    average_grid=True,
    sum_vars=True,
    ens_dim=1,
):
    """
    (Squared) spread of ensemble.
    Similarly to RMSE, we want to take sqrt after spatial and sample averaging,
    so we need to average the squared spread.

    (..., M, ...,) is any number of batch dimensions, including ensemble
        dimension M
    pred: (..., M, ..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., M, ..., N, d_state) or (d_state,), predicted std.-dev.
    mask: (N,), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced
        (sum over d_state)
    ens_dim: batch dimension where ensemble members are laid out, to reduce over

    Returns:
    metric_val: One of (...,), (..., d_state) depending on reduction arguments.
    """
    entry_var = torch.var(pred, dim=ens_dim)  # (..., N, d_state)
    return mask_and_reduce_metric(entry_var, mask, average_grid, sum_vars)


DEFINED_METRICS = {
    "mse": mse,
    "mae": mae,
    "wmse": wmse,
    "wmae": wmae,
    "nll": nll,
    "crps_gauss": crps_gauss,
    "crps_ens": crps_ens,
    "spread_squared": spread_squared,
}
