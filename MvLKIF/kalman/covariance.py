"""Covariance estimators for square-root Kalman filtering in LKIF workflows."""

import numpy as np



def standard_cov_est(z, smoothed_z, i, N):
    """Estimate process covariance ``Q`` and noise covariance ``R``.

    Syntax:
        R, Q = standard_cov_est(z, smoothed_z, i, N)

    Inputs:
        z:
            Noisy measurements, shape ``(m, n)`` or ``(n,)``.
        smoothed_z:
            Smoothed proxy for latent process state, same layout as ``z``.
        i:
            Current sample index (0-based in Python).
            MATLAB equivalent is 1-based index ``i+1``.
        N:
            Lookback window length used for covariance estimation.

    Outputs:
        R:
            Measurement-noise covariance estimate (``m x m``).
        Q:
            Process covariance estimate (``m x m``).

    Description:
        Direct translation of ``StandardCovEst.m``. Uses the last ``N`` samples
        ending at ``i`` and computes sample covariance for:
        - process proxy ``smoothed_z`` -> ``Q``
        - residual ``z - smoothed_z`` -> ``R``

    Notes:
        - For scalar data, returns ``1x1`` matrices for compatibility.
        - Raises ``ValueError`` when ``i < N-1``.
    """
    z = np.asarray(z, dtype=float)
    smoothed_z = np.asarray(smoothed_z, dtype=float)

    if z.ndim == 1:
        z = z.reshape(1, -1)
    if smoothed_z.ndim == 1:
        smoothed_z = smoothed_z.reshape(1, -1)

    dim = z.shape[0]
    if i < N - 1:
        raise ValueError("i must be >= N-1 for zero-based indexing")

    if dim == 1:
        z_win = z[0, i - (N - 1): i + 1]
        s_win = smoothed_z[0, i - (N - 1): i + 1]
        diffs = z_win - s_win

        q = np.sum((s_win - np.mean(s_win)) ** 2) / (N - 1)
        r = np.sum((diffs - np.mean(diffs)) ** 2) / (N - 1)
        return np.array([[r]], dtype=float), np.array([[q]], dtype=float)

    z_win = z[:, i - (N - 1): i + 1]
    s_win = smoothed_z[:, i - (N - 1): i + 1]
    diffs = z_win - s_win

    R = np.zeros((dim, dim), dtype=float)
    Q = np.zeros((dim, dim), dtype=float)

    for ii in range(dim):
        mean1i = np.mean(s_win[ii, :])
        mean2i = np.mean(diffs[ii, :])
        for jj in range(dim):
            mean1j = np.mean(s_win[jj, :])
            mean2j = np.mean(diffs[jj, :])
            Q[ii, jj] = np.sum((s_win[ii, :] - mean1i) * (s_win[jj, :] - mean1j)) / (N - 1)
            R[ii, jj] = np.sum((diffs[ii, :] - mean2i) * (diffs[jj, :] - mean2j)) / (N - 1)

    return R, Q



def ewma_cov_est(z, smoothed_z, i, N, Rold, Qold):
    """Recursively update covariance estimates with EWMA weighting.

    Syntax:
        R, Q = ewma_cov_est(z, smoothed_z, i, N, Rold, Qold)

    Inputs:
        z:
            Noisy measurements (``m x n`` or ``n,``).
        smoothed_z:
            Smoothed process surrogate.
        i:
            Current sample index (0-based).
        N:
            Memory length controlling EWMA weight:
            ``lambda = 1 - 2/(N+1)``.
        Rold, Qold:
            Previous covariance estimates at sample ``i-1``.

    Outputs:
        R:
            Updated measurement covariance.
        Q:
            Updated process covariance.

    Description:
        Translation of ``EWMACovEst.m``:
        - At the initialization point (MATLAB ``i == 2*N-1``), it delegates
          to the standard covariance estimator.
        - Otherwise it forms innovation vectors ``Rnew`` and ``Qnew`` and
          performs recursive updates:
            ``Q = (1-lambda) Qnew Qnew' + lambda Qold``
            ``R = (1-lambda) Rnew Rnew' + lambda Rold``

    Notes:
        - Initialization condition uses Python index ``i == 2*N-2``.
        - Returned objects are always 2D covariance matrices.
    """
    z = np.asarray(z, dtype=float)
    smoothed_z = np.asarray(smoothed_z, dtype=float)
    Rold = np.asarray(Rold, dtype=float)
    Qold = np.asarray(Qold, dtype=float)

    if z.ndim == 1:
        z = z.reshape(1, -1)
    if smoothed_z.ndim == 1:
        smoothed_z = smoothed_z.reshape(1, -1)

    if i == (2 * N - 2):
        return standard_cov_est(z, smoothed_z, i, N)

    dim = z.shape[0]
    lam = 1.0 - 2.0 / (N + 1.0)

    if dim == 1:
        qnew = smoothed_z[0, i] - np.mean(smoothed_z[0, i - (N - 1): i + 1])
        rnew = z[0, i] - smoothed_z[0, i]
        qnew = np.array([[qnew]], dtype=float)
        rnew = np.array([[rnew]], dtype=float)
    else:
        qnew = smoothed_z[:, i] - np.mean(smoothed_z[:, i - (N - 1): i + 1], axis=1)
        rnew = z[:, i] - smoothed_z[:, i]
        qnew = qnew.reshape(-1, 1)
        rnew = rnew.reshape(-1, 1)

    Q = (1.0 - lam) * (qnew @ qnew.T) + lam * Qold
    R = (1.0 - lam) * (rnew @ rnew.T) + lam * Rold
    return R, Q
