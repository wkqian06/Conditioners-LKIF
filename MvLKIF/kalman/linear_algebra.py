"""Linear-algebra helpers for UD-based square-root Kalman filtering."""

import numpy as np


UD_TOL = 1e-15



def my_ud(mat, noerror=False):
    """Compute UD decomposition of a symmetric positive-definite matrix.

    Syntax:
        U, D = my_ud(mat)
        U, D = my_ud(mat, noerror=True)

    Inputs:
        mat:
            Square matrix assumed symmetric (upper triangle is effectively
            used by the recursion, matching MATLAB behavior).
        noerror:
            If ``False`` (default), raises if matrix is not positive definite.
            If ``True``, suppresses error and inserts fallback diagonal terms,
            preserving MATLAB's ``'noerror'`` mode.

    Outputs:
        U:
            Unit upper-triangular matrix.
        D:
            Diagonal matrix.

    Description:
        Direct translation of ``myUD.m``. Produces factors such that:
            ``mat ~= U @ D @ U.T``
        This UD form is the square-root representation used throughout the
        Thornton/Bierman Kalman update steps.
    """
    mat = np.asarray(mat, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("Input matrix must be square")

    n = mat.shape[0]
    U = np.zeros((n, n), dtype=float)
    D = np.zeros((n, n), dtype=float)

    for j in range(n - 1, -1, -1):
        for i in range(j, -1, -1):
            s = mat[i, j]
            for k in range(j + 1, n):
                s -= U[i, k] * D[k, k] * U[j, k]
            if i == j:
                if s <= UD_TOL:
                    if not noerror:
                        raise np.linalg.LinAlgError("Input matrix is not positive definite")
                    D[j, j] = 1.0
                    U[j, j] = 0.0
                else:
                    D[j, j] = s
                    U[j, j] = 1.0
            else:
                U[i, j] = s / D[j, j]

    return U, D



def my_unit_tri_sys_sol(U, B, triangle):
    """Solve a unit-triangular linear system by substitution.

    Syntax:
        X = my_unit_tri_sys_sol(U, B, 'upper')
        X = my_unit_tri_sys_sol(U, B, 'lower')

    Inputs:
        U:
            Unit triangular coefficient matrix.
        B:
            Right-hand side vector or matrix.
        triangle:
            ``'upper'`` for backward substitution, ``'lower'`` for forward
            substitution.

    Outputs:
        X:
            Solution of ``U @ X = B`` with shape matching ``B``.

    Description:
        MATLAB code references ``myUnitTriSysSol`` but does not include the
        source in this repository. This implementation follows the expected
        behavior required by ``KalmanGainCalc`` and
        ``SquareRootKalmanFilter2``.
    """
    U = np.asarray(U, dtype=float)
    B = np.asarray(B, dtype=float)
    if B.ndim == 1:
        B = B.reshape(-1, 1)

    n = U.shape[0]
    X = np.zeros_like(B, dtype=float)

    tri = triangle.lower()
    if tri == "upper":
        for col in range(B.shape[1]):
            for i in range(n - 1, -1, -1):
                s = B[i, col]
                for j in range(i + 1, n):
                    s -= U[i, j] * X[j, col]
                X[i, col] = s
    elif tri == "lower":
        for col in range(B.shape[1]):
            for i in range(n):
                s = B[i, col]
                for j in range(i):
                    s -= U[i, j] * X[j, col]
                X[i, col] = s
    else:
        raise ValueError("triangle must be 'upper' or 'lower'")

    return X if B.shape[1] > 1 else X[:, 0]



def kalman_gain_calc(P, R, H=None):
    """Compute Kalman gain without explicit matrix inversion.

    Syntax:
        K, is_singular = kalman_gain_calc(P, R)
        K, is_singular = kalman_gain_calc(P, R, H)

    Inputs:
        P:
            Prior covariance matrix.
        R:
            Measurement covariance matrix.
        H:
            Optional measurement matrix. If omitted, identity observation
            model is assumed for gain computation.

    Outputs:
        K:
            Kalman gain matrix.
        is_singular:
            Boolean flag indicating decomposition/solve failure.

    Description:
        Translation of ``KalmanGainCalc.m``. It computes gain from:
            ``K = P H' (H P H' + R)^(-1)``
        but avoids direct inversion by:
        1. UD decomposition of innovation covariance.
        2. Unit-triangular substitutions.
        3. Diagonal scaling by UD factors.

    Notes:
        - On decomposition failure, returns zero gain and ``True``.
    """
    P = np.asarray(P, dtype=float)
    R = np.asarray(R, dtype=float)

    try:
        if H is None:
            U, D = my_ud(P + R)
            X1 = my_unit_tri_sys_sol(U, P.T, "upper")
        else:
            H = np.asarray(H, dtype=float)
            U, D = my_ud(H @ P @ H.T + R)
            X1 = my_unit_tri_sys_sol(U, H @ P.T, "upper")

        X2 = np.array(X1, dtype=float)
        for i in range(X2.shape[0]):
            X2[i, :] = X2[i, :] / D[i, i]

        X3 = my_unit_tri_sys_sol(U.T, X2, "lower")
        K = np.asarray(X3).T
        return K, False
    except Exception:
        return np.zeros((P.shape[0], R.shape[1]), dtype=float), True
