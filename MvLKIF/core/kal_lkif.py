"""Core time-varying LKIF routines based on Kalman covariance dynamics."""

from statistics import NormalDist

import numpy as np

from ..kalman.square_root_filter import square_root_kalman_filter2
from .causality import (
    _build_estimator_covariance,
    _compute_residual,
    _moving_block_bootstrap_indices,
)


_DEFAULT_LEVELS = (0.90, 0.95, 0.99)


def _validate_levels(levels):
    vals = tuple(float(v) for v in levels)
    if len(vals) == 0:
        raise ValueError("levels must be non-empty")
    for v in vals:
        if not (0.0 < v < 1.0):
            raise ValueError(f"invalid level {v}; must satisfy 0 < level < 1")
    return vals


def _level_key(level):
    return f"{int(round(100 * level))}"


def _z_score(level):
    """Two-sided normal quantile z for confidence level."""
    level = float(level)
    if not (0.0 < level < 1.0):
        raise ValueError(f"invalid level {level}; must satisfy 0 < level < 1")
    return NormalDist().inv_cdf(0.5 * (1.0 + level))


def KalLKIF(
    series_list,
    MAlen,
    NN,
    MA,
    np_step,
    reverse=False,
    return_tau=False,
    eps=1e-12,
    err_levels=_DEFAULT_LEVELS,
    return_var_t21=False,
    return_all=False,
):
    """Core directional time-varying LKIF pass.

    Parameters
    ----------
    series_list : sequence
        Aligned 1D series. Column 0 is target X1 and column 1 is source X2.
    MAlen, NN, MA, np_step : int / str
        Kalman moving-average and differencing parameters.
    reverse : bool
        Reverse each input series before estimation (MATLAB edge block parity).
    return_tau : bool
        If True, also return normalized tau(t) following Toy Model equations.
    eps : float
        Numerical stabilizer.
    err_levels : tuple[float, float, float]
        Three confidence levels used for the three returned T21 error series.

    Returns
    -------
    T, err1, err2, err3
        If ``return_tau=False``.
    T, tau, err1, err2, err3
        If ``return_tau=True``.
    If ``return_var_t21=True``, append ``var_t21`` as last output.
    """
    arrs = [np.asarray(v, dtype=float).reshape(-1) for v in series_list]
    if len(arrs) < 2:
        raise ValueError("KalLKIF requires at least two series")

    if reverse:
        arrs = [arr[::-1] for arr in arrs]

    variable_num = len(arrs)
    variable_length = arrs[0].shape[0]

    for arr in arrs[1:]:
        if arr.shape[0] != variable_length:
            raise ValueError("all series must have the same length")

    np_step = int(np_step)
    if np_step < 1:
        raise ValueError("np_step must be >= 1")

    err_levels = _validate_levels(err_levels)
    if len(err_levels) != 3:
        raise ValueError("err_levels must contain exactly 3 confidence levels")
    z_err = tuple(_z_score(lv) for lv in err_levels)

    m = variable_num
    nm = variable_length
    n = variable_length
    dt = 1.0

    xx = np.column_stack(arrs)
    xx2 = xx.T
    nx = xx2[:, : n - 1]

    fx1 = xx2[0, 1:n] - xx2[0, : n - 1]
    y = nx.copy()

    c = np.zeros((2, n - 1, variable_num), dtype=float)
    for k in range(variable_num):
        c[:, :, k] = np.vstack([nx[k, :], fx1])

    _, z = square_root_kalman_filter2(y, MAlen, NN, MA)

    r = np.zeros((2, 2, z.shape[2], variable_num), dtype=float)
    for k in range(variable_num):
        _, r[:, :, :, k] = square_root_kalman_filter2(c[:, :, k], MAlen, NN, MA)

    dz = np.zeros((variable_num, 1, z.shape[2]), dtype=float)
    for k in range(variable_num):
        dz[k, 0, :] = r[0, 1, :, k]

    x = xx[: nm - np_step, :]
    n2 = nm - np_step
    if n2 < 2:
        raise ValueError("time length must be larger than np_step")

    dx1 = (xx[np_step:nm, 0] - xx[: nm - np_step, 0]) / (np_step * dt)
    vardot = np.sum((dx1 - np.mean(dx1)) ** 2) / max(1, len(dx1) - 1)

    t = np.zeros((n, 1), dtype=float)
    tau = np.zeros((n, 1), dtype=float) if return_tau else None
    err1 = np.zeros((n, 1), dtype=float)
    err2 = np.zeros((n, 1), dtype=float)
    err3 = np.zeros((n, 1), dtype=float)
    var_t21_series = np.zeros((n, 1), dtype=float) if return_var_t21 else None
    all_outputs = None
    if return_all:
        nsrc = m - 1
        all_outputs = {
            "source_indices": np.arange(1, m, dtype=int),
            "t_all": np.full((n, nsrc), np.nan, dtype=float),
            "var_t_all": np.full((n, nsrc), np.nan, dtype=float),
            "err1_all": np.full((n, nsrc), np.nan, dtype=float),
            "err2_all": np.full((n, nsrc), np.nan, dtype=float),
            "err3_all": np.full((n, nsrc), np.nan, dtype=float),
            "err_levels": err_levels,
        }
        if return_tau:
            all_outputs["tau_all"] = np.full((n, nsrc), np.nan, dtype=float)
            all_outputs["tau_self"] = np.full(n, np.nan, dtype=float)
            all_outputs["h_noise"] = np.full(n, np.nan, dtype=float)
            all_outputs["z_norm"] = np.full(n, np.nan, dtype=float)

    for idx in range(1, n - 1):
        try:
            a1n = np.linalg.solve(z[:, :, idx], dz[:, :, idx])
        except np.linalg.LinAlgError:
            continue

        z00 = z[0, 0, idx]
        if abs(z00) < eps:
            continue

        t[idx, 0] = (z[0, 1, idx] / z00) * a1n[1, 0]

        a_row = a1n[:, 0]
        r1 = _compute_residual(dx1, x, a_row)
        q1 = np.sum(r1 * r1)
        b1 = np.sqrt(q1 * dt / n2)

        ni = _build_estimator_covariance(x, r1, b1, n2, dt)
        try:
            inv_ni = np.linalg.inv(ni)
            var_a12 = inv_ni[2, 2]
        except np.linalg.LinAlgError:
            continue

        var_t21 = float((z[0, 1, idx] / z00) ** 2 * var_a12)
        var_t21 = max(var_t21, 0.0)
        sqrt_var = np.sqrt(var_t21)
        err1[idx, 0] = sqrt_var * z_err[0]
        err2[idx, 0] = sqrt_var * z_err[1]
        err3[idx, 0] = sqrt_var * z_err[2]
        if return_var_t21:
            var_t21_series[idx, 0] = var_t21

        if return_all:
            for col in range(1, m):
                pos = col - 1
                t_col = float((z[0, col, idx] / z00) * a_row[col])
                var_a1j = float(inv_ni[col + 1, col + 1])
                var_t_col = float((z[0, col, idx] / z00) ** 2 * var_a1j)
                var_t_col = max(var_t_col, 0.0)
                sqrt_var_col = np.sqrt(var_t_col)
                all_outputs["t_all"][idx, pos] = t_col
                all_outputs["var_t_all"][idx, pos] = var_t_col
                all_outputs["err1_all"][idx, pos] = sqrt_var_col * z_err[0]
                all_outputs["err2_all"][idx, pos] = sqrt_var_col * z_err[1]
                all_outputs["err3_all"][idx, pos] = sqrt_var_col * z_err[2]

        if return_tau:
            c_now = z[:, :, idx]
            cxdot_now = dz[:, 0, idx]
            t_all = np.zeros(m, dtype=float)
            z00_eps = z00 + eps
            for col in range(1, m):
                t_all[col] = a_row[col] * (c_now[0, col] / z00_eps)

            var_resid = vardot - 2.0 * float(a_row @ cxdot_now) + float(a_row.T @ c_now @ a_row)
            g = max(var_resid / dt, 0.0)
            h_self = float(a_row[0])
            h_noise = g / (2.0 * z00_eps)
            z_norm = max(abs(h_self) + np.sum(np.abs(t_all)) + abs(h_noise), eps)
            tau[idx, 0] = t[idx, 0] / z_norm
            if return_all:
                all_outputs["tau_all"][idx, :] = t_all[1:] / z_norm
                all_outputs["tau_self"][idx] = h_self / z_norm
                all_outputs["h_noise"][idx] = h_noise
                all_outputs["z_norm"][idx] = z_norm

    if return_tau:
        if return_var_t21:
            if return_all:
                return t, tau, err1, err2, err3, var_t21_series, all_outputs
            return t, tau, err1, err2, err3, var_t21_series
        if return_all:
            return t, tau, err1, err2, err3, all_outputs
        return t, tau, err1, err2, err3
    if return_var_t21:
        if return_all:
            return t, err1, err2, err3, var_t21_series, all_outputs
        return t, err1, err2, err3, var_t21_series
    if return_all:
        return t, err1, err2, err3, all_outputs
    return t, err1, err2, err3


def _edge_adjust_errors(err1, err2, err3, cal):
    """Apply MATLAB confidence-bound edge overwrite."""
    lhs_len = max(0, cal - 2)
    rhs_start = max(0, cal - 2)
    rhs_end = min(err1.shape[0], 2 * (cal - 2))
    if lhs_len > 0 and rhs_end - rhs_start == lhs_len and lhs_len <= err1.shape[0]:
        err1[:lhs_len] = err1[rhs_start:rhs_end]
        err2[:lhs_len] = err2[rhs_start:rhs_end]
        err3[:lhs_len] = err3[rhs_start:rhs_end]


def _tau_ci_analytic_series(t21, tau, var_t21, levels=(0.90, 0.95, 0.99), eps=1e-12):
    """Delta-method tau CI for each time index under local-constant denominator."""
    levels = _validate_levels(levels)
    t21 = np.asarray(t21, dtype=float).reshape(-1)
    tau = np.asarray(tau, dtype=float).reshape(-1)
    var_t21 = np.asarray(var_t21, dtype=float).reshape(-1)
    se_t = np.sqrt(np.maximum(var_t21, 0.0))

    # tau = T / Z  =>  Z = |T / tau|. Under local-constant Z assumption:
    # se_tau ~= se_T / Z.
    z_norm = np.maximum(np.abs(t21) / np.maximum(np.abs(tau), eps), eps)
    se_tau = np.zeros_like(se_t)
    finite = np.isfinite(se_t) & np.isfinite(tau)
    se_tau[finite] = se_t[finite] / z_norm[finite]

    out = {}
    for lv in levels:
        zq = _z_score(lv)
        half = zq * se_tau
        out[_level_key(lv)] = np.column_stack([tau - half, tau + half])
    return out


def _tau_ci_bootstrap_series(
    MAlen,
    NN,
    MA,
    np_step,
    series,
    *,
    n_boot=300,
    block_size=None,
    random_state=None,
    eps=1e-12,
    levels=(0.90, 0.95, 0.99),
    err_levels=_DEFAULT_LEVELS,
):
    """Percentile bootstrap CI for time-varying tau(t)."""
    levels = _validate_levels(levels)
    arrs = [np.asarray(v, dtype=float).reshape(-1) for v in series]
    n = arrs[0].shape[0]
    if block_size is None:
        block_size = max(4, int(round(np.sqrt(n))))

    rng = np.random.default_rng(random_state)
    tau_samples = []
    for _ in range(int(n_boot)):
        idx = _moving_block_bootstrap_indices(n, block_size, rng)
        resampled = [arr[idx] for arr in arrs]
        try:
            _, tau_b, _, _, _ = kal_lkif(
                MAlen,
                NN,
                MA,
                np_step,
                *resampled,
                return_tau=True,
                eps=eps,
                tau_ci_method=None,
                err_levels=err_levels,
            )
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            continue
        tau_samples.append(np.asarray(tau_b, dtype=float).reshape(-1))

    if not tau_samples:
        raise RuntimeError("bootstrap failed: no valid tau paths")

    stack = np.stack(tau_samples, axis=0)  # (B, T)
    out = {}
    for lv in levels:
        alpha = 1.0 - lv
        lo = np.percentile(stack, 100.0 * (alpha / 2.0), axis=0)
        hi = np.percentile(stack, 100.0 * (1.0 - alpha / 2.0), axis=0)
        out[_level_key(lv)] = np.column_stack([lo, hi])
    return out


def kal_lkif(
    MAlen_shuru,
    NN_shuru,
    MA_shuru,
    np_shuru,
    *series,
    return_tau=True,
    eps=1e-12,
    tau_ci_method=None,
    levels=_DEFAULT_LEVELS,
    err_levels=_DEFAULT_LEVELS,
    n_boot=300,
    block_size=None,
    random_state=None,
    return_all=False,
):
    """Time-varying Kalman LKIF with MATLAB-compatible edge treatment.
    Parameters
    ----------
    series_list : sequence
        Aligned 1D series. Column 0 is target X1 and column 1 is source X2.
    MAlen, NN, MA, np_step : int / str
        Kalman moving-average and differencing parameters.
    return_tau : bool
        If True, also return normalized tau(t) following Toy Model equations.
    eps : float
        Numerical stabilizer.
    err_levels : tuple[float, float, float]
        Three confidence levels used for the three returned T21 error series.
    
    Returns
    -------
    T21, tau21, err1, err2, err3
        Default output when ``return_tau=True``.
        The three error outputs correspond to ``err_levels``.
    T21, err1, err2, err3
        Output when ``return_tau=False``.
    If ``tau_ci_method`` is set and ``return_tau=True``, appends ``tau_ci`` dict.
    ``levels`` controls confidence levels for tau CI.
    """
    if len(series) < 2:
        raise ValueError("kal_lkif requires at least two series")

    levels = _validate_levels(levels)
    err_levels = _validate_levels(err_levels)
    if len(err_levels) != 3:
        raise ValueError("err_levels must contain exactly 3 confidence levels")

    method = None if tau_ci_method is None else str(tau_ci_method).strip().lower()
    if method not in (None, "analytic", "bootstrap"):
        raise ValueError("tau_ci_method must be one of: None, 'analytic', 'bootstrap'")

    MAlen = int(MAlen_shuru)
    NN = int(NN_shuru)
    MA = str(MA_shuru)
    np_step = int(np_shuru)
    need_var_t21 = bool(return_tau and method == "analytic")

    if return_tau:
        if need_var_t21:
            if return_all:
                t21, tau21, err90, err95, err99, var_t21, all_outputs = KalLKIF(
                    series,
                    MAlen,
                    NN,
                    MA,
                    np_step,
                    reverse=False,
                    return_tau=True,
                    eps=eps,
                    err_levels=err_levels,
                    return_var_t21=True,
                    return_all=True,
                )
            else:
                t21, tau21, err90, err95, err99, var_t21 = KalLKIF(
                    series,
                    MAlen,
                    NN,
                    MA,
                    np_step,
                    reverse=False,
                    return_tau=True,
                    eps=eps,
                    err_levels=err_levels,
                    return_var_t21=True,
                )
        else:
            if return_all:
                t21, tau21, err90, err95, err99, all_outputs = KalLKIF(
                    series,
                    MAlen,
                    NN,
                    MA,
                    np_step,
                    reverse=False,
                    return_tau=True,
                    eps=eps,
                    err_levels=err_levels,
                    return_all=True,
                )
            else:
                t21, tau21, err90, err95, err99 = KalLKIF(
                    series,
                    MAlen,
                    NN,
                    MA,
                    np_step,
                    reverse=False,
                    return_tau=True,
                    eps=eps,
                    err_levels=err_levels,
                )
            var_t21 = None
    else:
        if return_all:
            t21, err90, err95, err99, all_outputs = KalLKIF(
                series,
                MAlen,
                NN,
                MA,
                np_step,
                reverse=False,
                return_tau=False,
                err_levels=err_levels,
                return_all=True,
            )
        else:
            t21, err90, err95, err99 = KalLKIF(
                series,
                MAlen,
                NN,
                MA,
                np_step,
                reverse=False,
                return_tau=False,
                err_levels=err_levels,
            )
        var_t21 = None

    swapped = list(series)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    t12_rev, _, _, _ = KalLKIF(
        swapped,
        MAlen,
        NN,
        MA,
        np_step,
        reverse=True,
        return_tau=False,
        err_levels=err_levels,
    )

    t12fan = t12_rev[::-1]
    cal = MAlen + NN
    _edge_adjust_errors(err90, err95, err99, cal)

    # MATLAB computes this series for early-time overwrite parity.
    _dd = t21.copy()
    _dd[: min(cal, _dd.shape[0])] = t12fan[: min(cal, t12fan.shape[0])]

    if not return_tau:
        if tau_ci_method is not None:
            raise ValueError("tau_ci_method requires return_tau=True")
        if return_all:
            return t21, err90, err95, err99, all_outputs
        return t21, err90, err95, err99

    if tau_ci_method is None:
        if return_all:
            return t21, tau21, err90, err95, err99, all_outputs
        return t21, tau21, err90, err95, err99

    if method == "analytic":
        tau_ci = _tau_ci_analytic_series(t21, tau21, var_t21, levels=levels)
    else:
        tau_ci = _tau_ci_bootstrap_series(
            MAlen,
            NN,
            MA,
            np_step,
            series,
            n_boot=n_boot,
            block_size=block_size,
            random_state=random_state,
            eps=eps,
            levels=levels,
            err_levels=err_levels,
        )

    if return_all:
        return t21, tau21, err90, err95, err99, tau_ci, all_outputs
    return t21, tau21, err90, err95, err99, tau_ci


def _progress_iter(iterable, show_progress=False, desc=None):
    """Optionally wrap iterable with a tqdm progress bar."""
    if not show_progress:
        return iterable
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, desc=desc, dynamic_ncols=True)
    except Exception:
        return iterable


def kal_lkif_target_all(
    MAlen_shuru,
    NN_shuru,
    MA_shuru,
    np_shuru,
    xx,
    target_index,
    *,
    return_tau=True,
    eps=1e-12,
    err_levels=_DEFAULT_LEVELS,
):
    """Compute all source->target time-varying LKIF for one fixed target.

    Returns a dict with arrays shaped (time, n_sources) for the given target.
    """
    xx = np.asarray(xx, dtype=float)
    if xx.ndim != 2:
        raise ValueError("xx must be 2D with columns as variables")
    n, m = xx.shape
    if m < 2:
        raise ValueError("xx must have at least two variables")

    i_tgt = int(target_index)
    if i_tgt < 0 or i_tgt >= m:
        raise ValueError("target_index out of range")

    order = [i_tgt] + [j for j in range(m) if j != i_tgt]
    sub = xx[:, order]

    out = kal_lkif(
        MAlen_shuru,
        NN_shuru,
        MA_shuru,
        np_shuru,
        *[sub[:, k] for k in range(sub.shape[1])],
        return_tau=return_tau,
        eps=eps,
        tau_ci_method=None,
        err_levels=err_levels,
        return_all=True,
    )

    if return_tau:
        t21, tau21, err1, err2, err3, all_out = out
        _ = t21, tau21, err1, err2, err3
    else:
        t21, err1, err2, err3, all_out = out
        _ = t21, err1, err2, err3

    src_orig = np.asarray(order[1:], dtype=int)
    result = {
        "target_index": i_tgt,
        "source_indices": src_orig,
        "t_all": np.asarray(all_out["t_all"], dtype=float),
        "err1_all": np.asarray(all_out["err1_all"], dtype=float),
        "err2_all": np.asarray(all_out["err2_all"], dtype=float),
        "err3_all": np.asarray(all_out["err3_all"], dtype=float),
        "err_levels": all_out["err_levels"],
    }
    if return_tau:
        result["tau_all"] = np.asarray(all_out["tau_all"], dtype=float)
        result["tau_self"] = np.asarray(all_out["tau_self"], dtype=float)
        result["h_noise"] = np.asarray(all_out["h_noise"], dtype=float)
        result["z_norm"] = np.asarray(all_out["z_norm"], dtype=float)
    return result


def kal_lkif_all(
    MAlen_shuru,
    NN_shuru,
    MA_shuru,
    np_shuru,
    xx,
    *,
    return_tau=True,
    eps=1e-12,
    tau_ci_method=None,
    levels=_DEFAULT_LEVELS,
    err_levels=_DEFAULT_LEVELS,
    show_progress=False,
    progress_desc="kal_lkif_all",
):
    """Compute all directed time-varying LKIF with source->target indexing.

    Returns
    -------
    T, err1, err2, err3 : ndarray, shape (m, m, t)
        If ``return_tau=False``.
    T, tau, err1, err2, err3 : ndarray, shape (m, m, t)
        If ``return_tau=True``. tau diagonal is tau_self(t).
    If ``tau_ci_method='analytic'`` and ``return_tau=True``, appends tau_ci_all:
        tau_ci_all[level_key] -> ndarray shape (m, m, t, 2).
    """
    xx = np.asarray(xx, dtype=float)
    if xx.ndim != 2:
        raise ValueError("xx must be 2D with columns as variables")
    n, m = xx.shape
    if m < 2:
        raise ValueError("xx must have at least two variables")

    method = None if tau_ci_method is None else str(tau_ci_method).strip().lower()
    if method not in (None, "analytic"):
        raise ValueError("kal_lkif_all currently supports tau_ci_method=None or 'analytic'")
    levels = _validate_levels(levels)

    t_cube = np.full((m, m, n), np.nan, dtype=float)
    e1_cube = np.full((m, m, n), np.nan, dtype=float)
    e2_cube = np.full((m, m, n), np.nan, dtype=float)
    e3_cube = np.full((m, m, n), np.nan, dtype=float)
    tau_cube = np.full((m, m, n), np.nan, dtype=float) if return_tau else None
    tau_ci_all = None
    if return_tau and method == "analytic":
        tau_ci_all = {_level_key(lv): np.full((m, m, n, 2), np.nan, dtype=float) for lv in levels}

    for i_tgt in _progress_iter(range(m), show_progress=show_progress, desc=progress_desc):
        order = [i_tgt] + [j for j in range(m) if j != i_tgt]
        sub = xx[:, order]
        out = kal_lkif(
            MAlen_shuru,
            NN_shuru,
            MA_shuru,
            np_shuru,
            *[sub[:, k] for k in range(sub.shape[1])],
            return_tau=return_tau,
            eps=eps,
            tau_ci_method=None,
            err_levels=err_levels,
            return_all=True,
        )
        if return_tau:
            t21, tau21, e1, e2, e3, all_out = out
            _ = t21, tau21, e1, e2, e3
        else:
            t21, e1, e2, e3, all_out = out
            _ = t21, e1, e2, e3

        src_orig = np.asarray(order[1:], dtype=int)
        t_vals = np.asarray(all_out["t_all"], dtype=float)  # (n, m-1)
        e1_vals = np.asarray(all_out["err1_all"], dtype=float)
        e2_vals = np.asarray(all_out["err2_all"], dtype=float)
        e3_vals = np.asarray(all_out["err3_all"], dtype=float)
        t_cube[src_orig, i_tgt, :] = t_vals.T
        e1_cube[src_orig, i_tgt, :] = e1_vals.T
        e2_cube[src_orig, i_tgt, :] = e2_vals.T
        e3_cube[src_orig, i_tgt, :] = e3_vals.T

        if return_tau:
            tau_vals = np.asarray(all_out["tau_all"], dtype=float)
            tau_cube[src_orig, i_tgt, :] = tau_vals.T
            tau_cube[i_tgt, i_tgt, :] = np.asarray(all_out["tau_self"], dtype=float)
            if method == "analytic":
                var_t_vals = np.asarray(all_out["var_t_all"], dtype=float)
                z_norm_vals = np.asarray(all_out["z_norm"], dtype=float)
                for p, j_src in enumerate(src_orig):
                    t_series = t_vals[:, p]
                    tau_series = tau_vals[:, p]
                    var_series = var_t_vals[:, p]
                    ci_j = _tau_ci_analytic_series(t_series, tau_series, var_series, levels=levels)
                    for lv in levels:
                        key = _level_key(lv)
                        tau_ci_all[key][j_src, i_tgt, :, 0] = ci_j[key][:, 0]
                        tau_ci_all[key][j_src, i_tgt, :, 1] = ci_j[key][:, 1]
                # diagonal tau_self CI from local z_norm and var(a11) is not provided here.
                _ = z_norm_vals

    if not return_tau:
        return t_cube, e1_cube, e2_cube, e3_cube
    if method is None:
        return t_cube, tau_cube, e1_cube, e2_cube, e3_cube
    return t_cube, tau_cube, e1_cube, e2_cube, e3_cube, tau_ci_all


def normalized_kal_lkif(
    MAlen_shuru,
    NN_shuru,
    MA_shuru,
    np_shuru,
    *series,
    eps=1e-12,
    tau_ci_method=None,
    levels=_DEFAULT_LEVELS,
    err_levels=_DEFAULT_LEVELS,
    n_boot=300,
    block_size=None,
    random_state=None,
):
    """Compatibility wrapper that always returns normalized tau21.

    ``levels`` controls which confidence levels are returned for tau CI.
    ``err_levels`` controls which confidence levels map to err1/err2/err3.
    """
    return kal_lkif(
        MAlen_shuru,
        NN_shuru,
        MA_shuru,
        np_shuru,
        *series,
        return_tau=True,
        eps=eps,
        tau_ci_method=tau_ci_method,
        levels=levels,
        err_levels=err_levels,
        n_boot=n_boot,
        block_size=block_size,
        random_state=random_state,
    )
