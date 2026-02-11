"""Moving-average utilities used by covariance estimation in LKIF filtering."""

import numpy as np



def ewma(z, L):
    """Compute exponentially weighted moving average (EWMA).

    Syntax:
        smoothed_z = ewma(z, L)

    Inputs:
        z:
            1D array of shape (n,) or 2D array of shape (m, n), where
            each row is one signal and n is the number of samples.
        L:
            EWMA memory length. MATLAB-equivalent smoothing factor is:
            lambda = 1 - 2 / (L + 1)

    Outputs:
        smoothed_z:
            Array with same shape as ``z`` (after 1D promotion to row form),
            containing the recursively smoothed sequence for each row.

    Description:
        This is a direct Python translation of ``EWMA.m``. For each signal
        row, the first sample is copied directly, and each subsequent sample
        is computed recursively:
            s[j] = lambda * s[j-1] + (1-lambda) * z[j]

    Notes:
        - Larger ``L`` increases smoothing and memory.
        - ``z`` is converted to ``float`` for numeric stability.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z.reshape(1, -1)

    lam = 1.0 - 2.0 / (L + 1.0)
    smoothed = np.zeros_like(z, dtype=float)
    for i in range(z.shape[0]):
        smoothed[i, 0] = z[i, 0]
        for j in range(1, z.shape[1]):
            smoothed[i, j] = lam * smoothed[i, j - 1] + (1.0 - lam) * z[i, j]
    return smoothed



def uwma(z, L):
    """Compute unweighted moving average (UWMA).

    Syntax:
        smoothed_z = uwma(z, L)

    Inputs:
        z:
            1D array of shape (n,) or 2D array of shape (m, n), where each
            row is one signal.
        L:
            Lookback window length for arithmetic moving average.

    Outputs:
        smoothed_z:
            Array with same shape as ``z`` (after 1D promotion to row form).
            Entries before index ``L-1`` remain zero to match MATLAB behavior.

    Description:
        Translation of ``UWMA.m``. For each row and sample index ``j >= L-1``:
            smoothed_z[j] = mean(z[j-(L-1):j+1])

    Notes:
        - Early samples are intentionally zero, consistent with MATLAB code.
        - Use this function when the Kalman workflow selects ``MAType='UWMA'``.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z.reshape(1, -1)

    smoothed = np.zeros_like(z, dtype=float)
    for i in range(z.shape[0]):
        for j in range(L - 1, z.shape[1]):
            smoothed[i, j] = np.mean(z[i, j - (L - 1): j + 1])
    return smoothed
