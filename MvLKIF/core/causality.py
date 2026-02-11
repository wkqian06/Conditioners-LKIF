"""Liang-Kleeman information-flow estimators (bivariate and multivariate)."""

from statistics import NormalDist

import numpy as np


_DEFAULT_LEVELS = (0.90, 0.95, 0.99)


def _validate_np_step(np_step):
    np_step = int(np_step)
    if np_step < 1:
        raise ValueError("np_step must be >= 1")
    return np_step


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


def _moving_block_bootstrap_indices(n, block_size, rng):
    """Generate moving-block bootstrap indices preserving local autocorrelation."""
    if n <= 0:
        return np.array([], dtype=int)
    b = int(block_size)
    if b < 1:
        raise ValueError("block_size must be >= 1")
    b = min(b, n)
    n_blocks = int(np.ceil(n / b))
    starts = rng.integers(0, n - b + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + b, dtype=int) for s in starts])
    return idx[:n]


def _prepare_multivariate_inputs(xx, np_step, dt=1.0):
    """Build aligned state matrix and target derivative for Liang row-1 fit."""
    xx = np.asarray(xx, dtype=float)
    if xx.ndim != 2:
        raise ValueError("xx must be 2D with columns as variables")

    nm, m = xx.shape
    if m < 2:
        raise ValueError("xx must have at least two variables (target + source)")

    np_step = _validate_np_step(np_step)
    if nm <= np_step:
        raise ValueError("time length must be larger than np_step")
    if (nm - np_step) < 2:
        raise ValueError("effective sample size (n - np_step) must be >= 2")

    x = xx[: nm - np_step, :]
    dx1 = (xx[np_step:nm, 0] - xx[: nm - np_step, 0]) / (np_step * dt)
    n = nm - np_step
    return x, dx1, n, dt


def _compute_cov_blocks(x, dx1):
    """Return C and Cxdot for the target equation xdot_1 = f + a^T x + noise."""
    c = np.cov(x, rowvar=False, bias=False)
    m = x.shape[1]
    cxdot = np.zeros(m, dtype=float)
    dx1c = dx1 - np.mean(dx1)
    for k in range(m):
        cxdot[k] = np.sum((x[:, k] - np.mean(x[:, k])) * dx1c)
    cxdot = cxdot / (x.shape[0] - 1)
    return c, cxdot


def _solve_row_coefficients(c, cxdot):
    """Solve Liang row coefficients a from C a = Cxdot."""
    return np.linalg.solve(c, cxdot)


def _compute_residual(dx1, x, a):
    """Residual of target row regression used by Liang asymptotic errors."""
    f1 = np.mean(dx1) - float(a @ np.mean(x, axis=0))
    return dx1 - (f1 + x @ a)


def _build_estimator_covariance(x, r1, b1, n, dt):
    """Build NI covariance matrix of estimator (f1, a11..a1M, b1)."""
    m = x.shape[1]
    ni = np.zeros((m + 2, m + 2), dtype=float)
    ni[0, 0] = n * dt / (b1 * b1)
    ni[m + 1, m + 1] = 3.0 * dt / (b1**4) * np.sum(r1 * r1) - n / (b1 * b1)

    for k in range(m):
        ni[0, k + 1] = dt / (b1 * b1) * np.sum(x[:, k])
    ni[0, m + 1] = 2.0 * dt / (b1**3) * np.sum(r1)

    for k in range(m):
        for j in range(m):
            ni[j + 1, k + 1] = dt / (b1 * b1) * np.sum(x[:, j] * x[:, k])

    for k in range(m):
        ni[k + 1, m + 1] = 2.0 * dt / (b1**3) * np.sum(r1 * x[:, k])

    ni = (ni + ni.T) / 2.0
    return ni


def _estimate_transfer(
    xx,
    np_step,
    *,
    return_tau=False,
    eps=1e-12,
    stabilized_ratio=False,
    return_details=False,
    return_all=False,
    err_levels=_DEFAULT_LEVELS,
):
    """Unified Liang estimator for bivariate/multivariate and tau variants."""
    x, dx1, n, dt = _prepare_multivariate_inputs(xx, np_step)
    c, cxdot = _compute_cov_blocks(x, dx1)
    a = _solve_row_coefficients(c, cxdot)

    c11 = c[0, 0]
    denom = c11 + eps if stabilized_ratio else c11

    m = x.shape[1]
    t_all = np.zeros(m, dtype=float)
    for idx in range(1, m):
        t_all[idx] = a[idx] * (c[0, idx] / denom)
    t21 = float(t_all[1])

    r1 = _compute_residual(dx1, x, a)
    q1 = np.sum(r1 * r1)
    b1 = np.sqrt(q1 * dt / n)

    ni = _build_estimator_covariance(x, r1, b1, n, dt)
    var_a12 = np.linalg.inv(ni)[2, 2]
    var_t21 = float((c[0, 1] / denom) ** 2 * var_a12)
    var_t21 = max(var_t21, 0.0)

    err_levels = _validate_levels(err_levels)
    if len(err_levels) != 3:
        raise ValueError("err_levels must contain exactly 3 confidence levels")
    err_values = [float(np.sqrt(var_t21) * _z_score(level)) for level in err_levels]
    err1, err2, err3 = err_values

    details = {
        "var_t21": var_t21,
        "z_norm": None,
        "n_eff": int(n),
        "err_levels": err_levels,
    }

    all_outputs = None
    if return_all:
        inv_ni = np.linalg.inv(ni)
        src_idx = np.arange(1, m, dtype=int)
        t_src = t_all[1:].astype(float, copy=True)
        var_t_src = np.zeros_like(t_src, dtype=float)
        err_src = [np.zeros_like(t_src, dtype=float) for _ in range(3)]
        for pos, j in enumerate(src_idx):
            var_a1j = float(inv_ni[j + 1, j + 1])
            var_tj = float((c[0, j] / denom) ** 2 * var_a1j)
            var_tj = max(var_tj, 0.0)
            var_t_src[pos] = var_tj
            sqrt_var_j = np.sqrt(var_tj)
            for k, lv in enumerate(err_levels):
                err_src[k][pos] = float(sqrt_var_j * _z_score(lv))
        all_outputs = {
            "source_indices": src_idx,
            "t_all": t_src,
            "var_t_all": var_t_src,
            "err1_all": err_src[0],
            "err2_all": err_src[1],
            "err3_all": err_src[2],
            "err_levels": err_levels,
        }

    if not return_tau:
        if return_details:
            if return_all:
                return t21, err1, err2, err3, details, all_outputs
            return t21, err1, err2, err3, details
        if return_all:
            return t21, err1, err2, err3, all_outputs
        return t21, err1, err2, err3

    # Tau definition follows Toy Model.ipynb equations.
    vardot = np.sum((dx1 - np.mean(dx1)) ** 2) / max(1, len(dx1) - 1)
    var_resid = vardot - 2.0 * float(a @ cxdot) + float(a.T @ c @ a)
    g = max(var_resid / dt, 0.0)
    h_self = float(a[0])
    h_noise = g / (2.0 * (c11 + eps))

    if m == 2:
        z = abs(h_self) + abs(h_noise) + abs(t21)
    else:
        z = abs(h_self) + np.sum(np.abs(t_all)) + abs(h_noise)
    z = max(z, eps)
    tau21 = float(t21 / z)

    details["z_norm"] = float(z)
    if return_all:
        all_outputs["z_norm"] = float(z)
        all_outputs["tau_all"] = (all_outputs["t_all"] / z).astype(float, copy=False)
        all_outputs["h_self"] = float(h_self)
        all_outputs["h_noise"] = float(h_noise)
        all_outputs["tau_self"] = float(h_self / z)

    if return_details:
        if return_all:
            return t21, tau21, err1, err2, err3, details, all_outputs
        return t21, tau21, err1, err2, err3, details
    if return_all:
        return t21, tau21, err1, err2, err3, all_outputs
    return t21, tau21, err1, err2, err3

def _tau_ci_analytic(tau21, var_t21, z_norm, levels=(0.90, 0.95, 0.99), eps=1e-12):
    """Delta-method CI for tau under local-constant denominator assumption."""
    levels = _validate_levels(levels)
    scale = max(float(z_norm), eps)
    se_tau = np.sqrt(max(float(var_t21), 0.0)) / scale
    out = {}
    for lv in levels:
        zq = _z_score(lv)
        half = zq * se_tau
        out[_level_key(lv)] = (float(tau21 - half), float(tau21 + half))
    return out


def _tau_ci_bootstrap(
    xx,
    np_step,
    *,
    n_boot=500,
    block_size=None,
    random_state=None,
    eps=1e-12,
    stabilized_ratio=True,
    levels=(0.90, 0.95, 0.99),
):
    """Percentile CI for tau via moving-block bootstrap."""
    levels = _validate_levels(levels)
    xx = np.asarray(xx, dtype=float)
    n = xx.shape[0]
    if block_size is None:
        block_size = max(2, int(round(np.sqrt(n))))

    rng = np.random.default_rng(random_state)
    samples = []
    for _ in range(int(n_boot)):
        idx = _moving_block_bootstrap_indices(n, block_size, rng)
        x_boot = xx[idx, :]
        try:
            _, tau_b, _, _, _ = _estimate_transfer(
                x_boot,
                np_step,
                return_tau=True,
                eps=eps,
                stabilized_ratio=stabilized_ratio,
                return_details=False,
            )
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            continue
        if np.isfinite(tau_b):
            samples.append(float(tau_b))

    if not samples:
        raise RuntimeError("bootstrap failed: no valid tau samples")

    arr = np.asarray(samples, dtype=float)
    out = {}
    for lv in levels:
        alpha = 1.0 - lv
        lo = np.percentile(arr, 100.0 * (alpha / 2.0))
        hi = np.percentile(arr, 100.0 * (1.0 - alpha / 2.0))
        out[_level_key(lv)] = (float(lo), float(hi))

    return out


def causality_est(xx1, xx2, np_step, err_levels=_DEFAULT_LEVELS, return_all=False):
    """Estimate bivariate Liang-Kleeman transfer T21 (X2 -> X1)."""
    xx = np.column_stack([
        np.asarray(xx1, dtype=float).reshape(-1),
        np.asarray(xx2, dtype=float).reshape(-1),
    ])
    return _estimate_transfer(
        xx,
        np_step,
        return_tau=False,
        stabilized_ratio=False,
        return_all=return_all,
        err_levels=err_levels,
    )


def normalized_causality_est(
    xx1,
    xx2,
    np_step,
    eps=1e-12,
    tau_ci_method=None,
    levels=_DEFAULT_LEVELS,
    err_levels=_DEFAULT_LEVELS,
    n_boot=500,
    block_size=None,
    random_state=None,
    return_all=False,
):
    """Estimate bivariate T21 and normalized tau21 (Toy Model formula).

    Parameters
    ----------
    tau_ci_method : {None, "analytic", "bootstrap"}
        If provided, appends tau CI dict as a sixth return value.
    """
    xx = np.column_stack([
        np.asarray(xx1, dtype=float).reshape(-1),
        np.asarray(xx2, dtype=float).reshape(-1),
    ])
    est = _estimate_transfer(
        xx,
        np_step,
        return_tau=True,
        eps=eps,
        stabilized_ratio=True,
        return_details=True,
        return_all=return_all,
        err_levels=err_levels,
    )
    if return_all:
        t21, tau21, err90, err95, err99, details, all_outputs = est
    else:
        t21, tau21, err90, err95, err99, details = est

    if tau_ci_method is None:
        if return_all:
            return t21, tau21, err90, err95, err99, all_outputs
        return t21, tau21, err90, err95, err99

    method = str(tau_ci_method).strip().lower()
    if method == "analytic":
        tau_ci = _tau_ci_analytic(
            tau21,
            details["var_t21"],
            details["z_norm"],
            levels=levels,
            eps=eps,
        )
    elif method == "bootstrap":
        tau_ci = _tau_ci_bootstrap(
            xx,
            np_step,
            n_boot=n_boot,
            block_size=block_size,
            random_state=random_state,
            eps=eps,
            stabilized_ratio=True,
            levels=levels,
        )
    else:
        raise ValueError("tau_ci_method must be one of: None, 'analytic', 'bootstrap'")

    if return_all:
        return t21, tau21, err90, err95, err99, tau_ci, all_outputs
    return t21, tau21, err90, err95, err99, tau_ci


def multi_causality_est(xx, np_step, err_levels=_DEFAULT_LEVELS, return_all=False):
    """Estimate multivariate conditional transfer T21 (X2 -> X1)."""
    return _estimate_transfer(
        xx,
        np_step,
        return_tau=False,
        stabilized_ratio=False,
        return_all=return_all,
        err_levels=err_levels,
    )


def normalized_multi_causality_est(
    xx,
    np_step,
    eps=1e-12,
    tau_ci_method=None,
    levels=_DEFAULT_LEVELS,
    err_levels=_DEFAULT_LEVELS,
    n_boot=500,
    block_size=None,
    random_state=None,
    return_all=False,
):
    """Estimate multivariate conditional T21 and normalized tau21.

    Parameters
    ----------
    tau_ci_method : {None, "analytic", "bootstrap"}
        If provided, appends tau CI dict as a sixth return value.
    """
    xx = np.asarray(xx, dtype=float)
    est = _estimate_transfer(
        xx,
        np_step,
        return_tau=True,
        eps=eps,
        stabilized_ratio=True,
        return_details=True,
        return_all=return_all,
        err_levels=err_levels,
    )
    if return_all:
        t21, tau21, err90, err95, err99, details, all_outputs = est
    else:
        t21, tau21, err90, err95, err99, details = est

    if tau_ci_method is None:
        if return_all:
            return t21, tau21, err90, err95, err99, all_outputs
        return t21, tau21, err90, err95, err99

    method = str(tau_ci_method).strip().lower()
    if method == "analytic":
        tau_ci = _tau_ci_analytic(
            tau21,
            details["var_t21"],
            details["z_norm"],
            levels=levels,
            eps=eps,
        )
    elif method == "bootstrap":
        tau_ci = _tau_ci_bootstrap(
            xx,
            np_step,
            n_boot=n_boot,
            block_size=block_size,
            random_state=random_state,
            eps=eps,
            stabilized_ratio=True,
            levels=levels,
        )
    else:
        raise ValueError("tau_ci_method must be one of: None, 'analytic', 'bootstrap'")

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


def all_causality_est(
    xx,
    np_step,
    err_levels=_DEFAULT_LEVELS,
    show_progress=False,
    progress_desc="all_causality_est",
):
    """Estimate all directed transfers T_{j->i} and errors in one API.

    Parameters
    ----------
    xx : array_like, shape (n, m)
        Multivariate time series matrix; columns are variables.
    np_step : int
        Forward-difference lag.
    err_levels : tuple[float, float, float]
        Confidence levels mapped to err1/err2/err3.
    show_progress : bool
        If True, show dynamic progress bar (tqdm if available).
    progress_desc : str
        Description text for the progress bar.

    Returns
    -------
    T, err1, err2, err3 : ndarray, shape (m, m)
        Directional matrices with source->target indexing:
        entry [j, i] is j->i. Diagonal entries are NaN.
    """
    xx = np.asarray(xx, dtype=float)
    if xx.ndim != 2:
        raise ValueError("xx must be 2D with columns as variables")
    n, m = xx.shape
    if m < 2:
        raise ValueError("xx must have at least two variables")

    t_mat = np.full((m, m), np.nan, dtype=float)
    e1_mat = np.full((m, m), np.nan, dtype=float)
    e2_mat = np.full((m, m), np.nan, dtype=float)
    e3_mat = np.full((m, m), np.nan, dtype=float)

    for i_tgt in _progress_iter(range(m), show_progress=show_progress, desc=progress_desc):
        order = [i_tgt] + [j for j in range(m) if j != i_tgt]
        sub = xx[:, order]
        _, _, _, _, all_out = multi_causality_est(
            sub,
            np_step,
            err_levels=err_levels,
            return_all=True,
        )

        src_orig = np.asarray(order[1:], dtype=int)
        t_vals = np.asarray(all_out["t_all"], dtype=float)
        e1_vals = np.asarray(all_out["err1_all"], dtype=float)
        e2_vals = np.asarray(all_out["err2_all"], dtype=float)
        e3_vals = np.asarray(all_out["err3_all"], dtype=float)
        t_mat[src_orig, i_tgt] = t_vals
        e1_mat[src_orig, i_tgt] = e1_vals
        e2_mat[src_orig, i_tgt] = e2_vals
        e3_mat[src_orig, i_tgt] = e3_vals

    return t_mat, e1_mat, e2_mat, e3_mat


def normalized_all_causality_est(
    xx,
    np_step,
    eps=1e-12,
    tau_ci_method=None,
    levels=_DEFAULT_LEVELS,
    err_levels=_DEFAULT_LEVELS,
    n_boot=500,
    block_size=None,
    random_state=None,
    show_progress=False,
    progress_desc="normalized_all_causality_est",
    return_h_noise=False,
):
    """Estimate all directed normalized flows tau_{j->i} and T_{j->i}.

    Returns
    -------
    T, tau, err1, err2, err3 : ndarray, shape (m, m)
        Directional matrices with source->target indexing:
        entry [j, i] is j->i.
        For tau, diagonal is filled as H_self / Z.
        T and error diagonals are NaN.
    If ``return_h_noise=True``, appends ``h_noise`` vector (shape (m,)).
    If ``tau_ci_method`` is set, appends ``tau_ci_all`` dict:
        tau_ci_all[level_key] -> ndarray shape (m, m, 2) for (lower, upper).
    """
    xx = np.asarray(xx, dtype=float)
    if xx.ndim != 2:
        raise ValueError("xx must be 2D with columns as variables")
    n, m = xx.shape
    if m < 2:
        raise ValueError("xx must have at least two variables")

    method = None if tau_ci_method is None else str(tau_ci_method).strip().lower()
    if method not in (None, "analytic", "bootstrap"):
        raise ValueError("tau_ci_method must be one of: None, 'analytic', 'bootstrap'")

    levels = _validate_levels(levels)
    t_mat = np.full((m, m), np.nan, dtype=float)
    tau_mat = np.full((m, m), np.nan, dtype=float)
    e1_mat = np.full((m, m), np.nan, dtype=float)
    e2_mat = np.full((m, m), np.nan, dtype=float)
    e3_mat = np.full((m, m), np.nan, dtype=float)

    tau_ci_all = None
    if method is not None:
        tau_ci_all = {_level_key(lv): np.full((m, m, 2), np.nan, dtype=float) for lv in levels}
    h_noise_vec = None
    if return_h_noise:
        h_noise_vec = np.full(m, np.nan, dtype=float)

    outer_iter = _progress_iter(range(m), show_progress=show_progress, desc=progress_desc)
    for i_tgt in outer_iter:
        order = [i_tgt] + [j for j in range(m) if j != i_tgt]
        sub = xx[:, order]
        _, _, _, _, _, all_out = normalized_multi_causality_est(
            sub,
            np_step,
            eps=eps,
            tau_ci_method=None,
            err_levels=err_levels,
            return_all=True,
        )

        src_orig = np.asarray(order[1:], dtype=int)
        t_vals = np.asarray(all_out["t_all"], dtype=float)
        tau_vals = np.asarray(all_out["tau_all"], dtype=float)
        e1_vals = np.asarray(all_out["err1_all"], dtype=float)
        e2_vals = np.asarray(all_out["err2_all"], dtype=float)
        e3_vals = np.asarray(all_out["err3_all"], dtype=float)

        t_mat[src_orig, i_tgt] = t_vals
        tau_mat[src_orig, i_tgt] = tau_vals
        tau_mat[i_tgt, i_tgt] = float(all_out["tau_self"])
        if return_h_noise:
            h_noise_vec[i_tgt] = float(all_out["h_noise"])
        e1_mat[src_orig, i_tgt] = e1_vals
        e2_mat[src_orig, i_tgt] = e2_vals
        e3_mat[src_orig, i_tgt] = e3_vals

        if method == "analytic":
            z_norm = float(all_out["z_norm"])
            var_t_all = np.asarray(all_out["var_t_all"], dtype=float)
            for pos, j_src in enumerate(src_orig):
                tau_ci_j = _tau_ci_analytic(
                    tau_vals[pos],
                    var_t_all[pos],
                    z_norm,
                    levels=levels,
                    eps=eps,
                )
                for lv in levels:
                    key = _level_key(lv)
                    lo, hi = tau_ci_j[key]
                    tau_ci_all[key][j_src, i_tgt, 0] = lo
                    tau_ci_all[key][j_src, i_tgt, 1] = hi
        elif method == "bootstrap":
            # Bootstrap CI is pair-specific (target i_tgt, source j_src) with other vars as conditioners.
            inner_desc = f"{progress_desc}:bootstrap(target={i_tgt})"
            inner_iter = _progress_iter(range(m), show_progress=show_progress, desc=inner_desc)
            for j_src in inner_iter:
                if j_src == i_tgt:
                    continue
                order_pair = [i_tgt, j_src] + [k for k in range(m) if k not in (i_tgt, j_src)]
                sub_pair = xx[:, order_pair]
                out_pair = normalized_multi_causality_est(
                    sub_pair,
                    np_step,
                    eps=eps,
                    tau_ci_method="bootstrap",
                    levels=levels,
                    err_levels=err_levels,
                    n_boot=n_boot,
                    block_size=block_size,
                    random_state=random_state,
                    return_all=False,
                )
                tau_ci_pair = out_pair[-1]
                for lv in levels:
                    key = _level_key(lv)
                    lo, hi = tau_ci_pair[key]
                    tau_ci_all[key][j_src, i_tgt, 0] = lo
                    tau_ci_all[key][j_src, i_tgt, 1] = hi

    if method is None:
        if return_h_noise:
            return t_mat, tau_mat, e1_mat, e2_mat, e3_mat, h_noise_vec
        return t_mat, tau_mat, e1_mat, e2_mat, e3_mat
    if return_h_noise:
        return t_mat, tau_mat, e1_mat, e2_mat, e3_mat, tau_ci_all, h_noise_vec
    return t_mat, tau_mat, e1_mat, e2_mat, e3_mat, tau_ci_all


def z_norm_cal(tau_mat, h_noise, eps=1e-12):
    """Infer target-wise Z from tau matrix and h_noise.

    Assumes tau indexing follows tau_{j->i} (source j to target i), i.e.
    target i corresponds to column i in tau_mat.

    Uses:
        Z_i = sum_j |tau_{j->i}| + |h_noise_i|
    where the sum is taken over column i and includes tau_self if present.
    """
    tau = np.asarray(tau_mat, dtype=float)
    hn = np.asarray(h_noise, dtype=float).reshape(-1)
    if tau.ndim != 2 or tau.shape[0] != tau.shape[1]:
        raise ValueError("tau_mat must be square 2D matrix")
    m = tau.shape[0]
    if hn.shape[0] != m:
        raise ValueError("h_noise length must match tau_mat size")

    # source->target convention: column i aggregates contributions to target i.
    s = np.sum(np.abs(tau), axis=0)
    z = s + np.abs(hn)
    return z
