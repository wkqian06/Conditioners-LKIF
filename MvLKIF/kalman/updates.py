"""Thornton and Bierman square-root Kalman update routines."""

import numpy as np



def thornton(x_post, U_post, D_post, Uq, Dq, A=None):
    """Perform Thornton square-root Kalman time update.

    Syntax:
        x_prior, U_prior, D_prior = thornton(x_post, U_post, D_post, Uq, Dq)
        x_prior, U_prior, D_prior = thornton(x_post, U_post, D_post, Uq, Dq, A)

    Inputs:
        x_post:
            A-posteriori state estimate at time k-1.
        U_post, D_post:
            UD factors for a-posteriori covariance ``P_post``.
        Uq, Dq:
            UD factors for process covariance ``Q``.
        A:
            Optional state transition matrix. If omitted, identity transition
            is assumed.

    Outputs:
        x_prior:
            A-priori state estimate at time k.
        U_prior, D_prior:
            UD factors for a-priori covariance ``P_prior``.

    Description:
        Translation of ``thornton.m`` implementing the modified weighted QR
        update in UD space. Equivalent covariance update target is:
            ``P_prior = A P_post A' + Q``
        but computed in factorized form for numerical robustness.
    """
    x_post = np.asarray(x_post, dtype=float).reshape(-1)
    U_post = np.asarray(U_post, dtype=float)
    D_post = np.asarray(D_post, dtype=float)
    Uq = np.asarray(Uq, dtype=float).copy()
    Dq = np.asarray(Dq, dtype=float)

    tol = 1e-15
    n = x_post.size

    a1 = np.zeros(n, dtype=float)
    a2 = np.zeros(n, dtype=float)
    v1 = np.zeros(n, dtype=float)
    v2 = np.zeros(n, dtype=float)
    D_prior = np.zeros((n, n), dtype=float)

    if A is not None:
        A = np.asarray(A, dtype=float)
        x_prior = A @ x_post

        U_prior = A.copy()
        for i in range(n):
            for j in range(n - 1, -1, -1):
                s = U_prior[i, j]
                for k in range(j):
                    s += U_prior[i, k] * U_post[k, j]
                U_prior[i, j] = s
    else:
        x_prior = x_post.copy()
        U_prior = U_post.copy()

    for j in range(n - 1, -1, -1):
        s = 0.0
        for k in range(n):
            v1[k] = U_prior[j, k]
            a1[k] = D_post[k, k] * v1[k]
            s += v1[k] * a1[k]
        for k in range(n):
            v2[k] = Uq[j, k]
            a2[k] = Dq[k, k] * v2[k]
            s += v2[k] * a2[k]

        U_prior[j, j] = s
        if s < tol:
            raise np.linalg.LinAlgError("New error covariance matrix is not positive definite")

        dinv = 1.0 / s
        for k in range(j):
            s2 = 0.0
            for i in range(n):
                s2 += U_prior[k, i] * a1[i]
            for i in range(n):
                s2 += Uq[k, i] * a2[i]
            s2 *= dinv

            for i in range(n):
                U_prior[k, i] = U_prior[k, i] - s2 * v1[i]
            for i in range(n):
                Uq[k, i] = Uq[k, i] - s2 * v2[i]

            U_prior[j, k] = s2

    for j in range(n):
        D_prior[j, j] = U_prior[j, j]
        U_prior[j, j] = 1.0
        for i in range(j):
            U_prior[i, j] = U_prior[j, i]
            U_prior[j, i] = 0.0

    return x_prior, U_prior, D_prior



def bierman(z, R, H, x_prior, U_prior, D_prior):
    """Perform Bierman scalar-measurement square-root update.

    Syntax:
        x_post, U_post, D_post = bierman(z, R, H, x_prior, U_prior, D_prior)

    Inputs:
        z:
            Scalar measurement.
        R:
            Measurement variance for ``z``.
        H:
            Row measurement vector mapping state -> scalar observation.
        x_prior:
            A-priori state estimate.
        U_prior, D_prior:
            UD factors of a-priori covariance.

    Outputs:
        x_post:
            A-posteriori state estimate.
        U_post, D_post:
            UD factors of a-posteriori covariance.

    Description:
        Translation of ``bierman.m``. Sequentially updates UD factors for one
        scalar measurement and then applies the innovation correction to the
        state estimate.
    """
    H = np.asarray(H, dtype=float).reshape(1, -1)
    x_prior = np.asarray(x_prior, dtype=float).reshape(-1)
    U_post = np.asarray(U_prior, dtype=float).copy()
    D_post = np.asarray(D_prior, dtype=float).copy()

    x_post = x_prior.copy()
    a = U_post.T @ H.T
    b = D_post @ a
    dz = float(z) - float(H @ x_prior)
    alpha = float(R)
    gamma = 1.0 / alpha

    n = x_prior.size
    for j in range(n):
        beta = alpha
        alpha = alpha + a[j, 0] * b[j, 0]
        lam = -a[j, 0] * gamma
        gamma = 1.0 / alpha
        D_post[j, j] = beta * gamma * D_post[j, j]
        for i in range(j):
            beta2 = U_post[i, j]
            U_post[i, j] = beta2 + b[i, 0] * lam
            b[i, 0] = b[i, 0] + b[j, 0] * beta2

    x_post = x_post + gamma * dz * b[:, 0]
    return x_post, U_post, D_post
