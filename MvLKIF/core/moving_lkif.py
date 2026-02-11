"""Sliding-window LKIF estimators built on static causality routines."""

import numpy as np

from .causality import (
    causality_est,
    multi_causality_est,
    normalized_causality_est,
    normalized_multi_causality_est,
)


def _as_matrix(series):
    """Normalize input into (n, m) matrix form."""
    if len(series) == 1:
        xx = np.asarray(series[0], dtype=float)
        if xx.ndim == 1:
            raise ValueError("single input must be a 2D matrix with columns as variables")
        if xx.ndim != 2:
            raise ValueError("input must be 2D")
        return xx

    cols = [np.asarray(v, dtype=float).reshape(-1) for v in series]
    n = cols[0].shape[0]
    for col in cols[1:]:
        if col.shape[0] != n:
            raise ValueError("all series must have equal length")
    return np.column_stack(cols)


def moving_lkif(window, step, np_step, *series, return_tau=True, eps=1e-12):
    """Compute time-varying LKIF by sliding window over static estimators.

    Parameters
    ----------
    window : int
        Window length.
    step : int
        Step size between consecutive windows.
    np_step : int
        Forward-difference lag passed to static estimators.
    *series :
        Either one 2D matrix ``(n, m)`` or multiple aligned 1D series.
    return_tau : bool
        Whether to compute normalized tau.

    Returns
    -------
    centers, T21, err90, err95, err99
        If ``return_tau=False``.
    centers, T21, tau21, err90, err95, err99
        If ``return_tau=True``.
    """
    xx = _as_matrix(series)
    n, m = xx.shape

    window = int(window)
    step = int(step)
    np_step = int(np_step)

    if m < 2:
        raise ValueError("MovingLKIF requires at least two variables")
    if window < 3:
        raise ValueError("window must be >= 3")
    if step < 1:
        raise ValueError("step must be >= 1")
    if n < window:
        raise ValueError("time length must be >= window")

    starts = list(range(0, n - window + 1, step))
    centers = np.array([s + window // 2 for s in starts], dtype=int)

    t21_vals = []
    tau_vals = []
    e90_vals = []
    e95_vals = []
    e99_vals = []

    for s in starts:
        sub = xx[s : s + window, :]
        if m == 2:
            if return_tau:
                t21, tau21, e90, e95, e99 = normalized_causality_est(sub[:, 0], sub[:, 1], np_step, eps=eps)
                tau_vals.append(tau21)
            else:
                t21, e90, e95, e99 = causality_est(sub[:, 0], sub[:, 1], np_step)
        else:
            if return_tau:
                t21, tau21, e90, e95, e99 = normalized_multi_causality_est(sub, np_step, eps=eps)
                tau_vals.append(tau21)
            else:
                t21, e90, e95, e99 = multi_causality_est(sub, np_step)

        t21_vals.append(t21)
        e90_vals.append(e90)
        e95_vals.append(e95)
        e99_vals.append(e99)

    t21_arr = np.asarray(t21_vals, dtype=float)
    e90_arr = np.asarray(e90_vals, dtype=float)
    e95_arr = np.asarray(e95_vals, dtype=float)
    e99_arr = np.asarray(e99_vals, dtype=float)

    if return_tau:
        tau_arr = np.asarray(tau_vals, dtype=float)
        return centers, t21_arr, tau_arr, e90_arr, e95_arr, e99_arr
    return centers, t21_arr, e90_arr, e95_arr, e99_arr


# Requested public alias.
MovingLKIF = moving_lkif
