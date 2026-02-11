"""Square-root Kalman filter driver translated from MATLAB implementation."""

import numpy as np

from .moving_average import ewma, uwma
from .covariance import ewma_cov_est, standard_cov_est
from .linear_algebra import my_ud, my_unit_tri_sys_sol
from .updates import bierman, thornton



def square_root_kalman_filter2(z, MAlen, *args):
    """Run square-root Kalman filtering on multivariate measurements.

    Syntax:
        x_kf, R = square_root_kalman_filter2(z, MAlen, N)
        x_kf, R = square_root_kalman_filter2(z, MAlen, 'EWMA')
        x_kf, R = square_root_kalman_filter2(z, MAlen, N, 'EWMA')
        x_kf, R = square_root_kalman_filter2(z, MAlen, N, 'UWMA')

    Inputs:
        z:
            Measurement matrix ``(m, n)`` where ``m`` is signal dimension and
            ``n`` is number of samples. A 1D sequence is promoted to ``(1, n)``.
        MAlen:
            Length of moving-average smoothing used during covariance
            estimation.
        *args:
            MATLAB-compatible optional arguments:
            - ``(N,)`` where ``N`` is covariance lookback and EWMA smoothing
              is used for state proxy; covariance update method is standard.
            - ``('EWMA',)`` to use ``N = MAlen`` and recursive EWMA
              covariance update.
            - ``(N, MAType)`` where ``MAType`` is ``'EWMA'`` or ``'UWMA'``;
              covariance update method is standard.

    Outputs:
        x_kf:
            A-posteriori filtered state estimates, shape ``(m, n)``.
        R:
            Time-varying measurement covariance estimates, shape
            ``(m, m, n)``.

    Description:
        Translation of ``SquareRootKalmanFilter2.m``. The model assumes:
            ``z = x + noise``
        Covariance estimates are inferred online from smoothed measurements.
        Filtering is performed in UD factorized form for numerical stability:
        1. Estimate ``Q`` and ``R``.
        2. UD-factorize ``Q`` and ``R``.
        3. Time update via Thornton routine.
        4. Decorrelate measurement channels.
        5. Scalar measurement updates via Bierman recursion.

    Notes:
        - If update fails at any step, this implementation falls back to the
          MATLAB behavior: current state equals measurement with identity
          covariance factors.
        - Index handling preserves MATLAB semantics under Python 0-based
          indexing.
    """
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z.reshape(1, -1)

    if len(args) == 1:
        if isinstance(args[0], str):
            N = MAlen
            ma_type = "EWMA"
            cov_method = "EWMA"
        else:
            N = int(args[0])
            ma_type = "EWMA"
            cov_method = "Standard"
    elif len(args) == 2:
        N = int(args[0])
        ma_type = str(args[1])
        cov_method = "Standard"
    else:
        raise ValueError("Input syntax error for square_root_kalman_filter2")

    dim, n = z.shape

    x_apriori = np.zeros((dim, n), dtype=float)
    x_aposteriori = np.zeros((dim, n), dtype=float)

    UP_apriori = np.zeros((dim, dim, n), dtype=float)
    DP_apriori = np.zeros((dim, dim, n), dtype=float)
    UP_aposteriori = np.zeros((dim, dim, n), dtype=float)
    DP_aposteriori = np.zeros((dim, dim, n), dtype=float)

    Q = np.zeros((dim, dim, n), dtype=float)
    UQ = np.zeros((dim, dim, n), dtype=float)
    DQ = np.zeros((dim, dim, n), dtype=float)

    R = np.zeros((dim, dim, n), dtype=float)
    UR = np.zeros((dim, dim, n), dtype=float)
    DR = np.zeros((dim, dim, n), dtype=float)

    if ma_type.upper() == "UWMA":
        smoothed_z = uwma(z, MAlen)
        start_index = N + MAlen - 1
    else:
        smoothed_z = ewma(z, MAlen)
        start_index = N

    init_idx = max(0, start_index - 2)
    x_aposteriori[:, init_idx] = smoothed_z[:, init_idx]
    UP_aposteriori[:, :, init_idx] = np.eye(dim)
    DP_aposteriori[:, :, init_idx] = np.eye(dim)

    for i in range(start_index - 1, n):
        if cov_method.lower() == "standard":
            R[:, :, i], Q[:, :, i] = standard_cov_est(z, smoothed_z, i, N)
        else:
            R[:, :, i], Q[:, :, i] = ewma_cov_est(z, smoothed_z, i, N, R[:, :, i - 1], Q[:, :, i - 1])

        try:
            UQ[:, :, i], DQ[:, :, i] = my_ud(Q[:, :, i])
            UR[:, :, i], DR[:, :, i] = my_ud(R[:, :, i])

            x_apriori[:, i], UP_apriori[:, :, i], DP_apriori[:, :, i] = thornton(
                x_aposteriori[:, i - 1],
                UP_aposteriori[:, :, i - 1],
                DP_aposteriori[:, :, i - 1],
                UQ[:, :, i],
                DQ[:, :, i],
            )

            z_ind = my_unit_tri_sys_sol(UR[:, :, i], z[:, i], "upper")
            H_ind = my_unit_tri_sys_sol(UR[:, :, i], np.eye(dim), "upper")

            x_aposteriori[:, i] = x_apriori[:, i]
            UP_aposteriori[:, :, i] = UP_apriori[:, :, i]
            DP_aposteriori[:, :, i] = DP_apriori[:, :, i]

            for j in range(dim):
                x_aposteriori[:, i], UP_aposteriori[:, :, i], DP_aposteriori[:, :, i] = bierman(
                    z_ind[j],
                    DR[j, j, i],
                    H_ind[j, :],
                    x_aposteriori[:, i],
                    UP_aposteriori[:, :, i],
                    DP_aposteriori[:, :, i],
                )
        except Exception:
            x_aposteriori[:, i] = z[:, i]
            UP_aposteriori[:, :, i] = np.eye(dim)
            DP_aposteriori[:, :, i] = np.eye(dim)

    return x_aposteriori, R
