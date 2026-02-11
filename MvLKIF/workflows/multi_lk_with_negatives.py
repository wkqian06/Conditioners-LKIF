"""Workflow conversion for MultiLK_with_negatives MATLAB script."""

import numpy as np

from ..core.kal_lkif import kal_lkif



def run_multilk_with_negatives(
    mat_path,
    target="LAI",
    driver="SM",
    conditioners=("VPD", "T", "SSR"),
    ma_len=24,
    n_window=90,
    ma_type="UWMA",
    np_step=1,
):
    """Run the MultiLK-with-negatives workflow on a MAT dataset.

    Syntax:
        out = run_multilk_with_negatives(mat_path, ...)

    Inputs:
        mat_path:
            Path to ``.mat`` file containing realization matrices.
        target:
            Name of target variable in MAT file (default ``'LAI'``).
        driver:
            Name of source/driver variable (default ``'SM'``).
        conditioners:
            Tuple/list of conditioning variable names.
        ma_len, n_window, ma_type, np_step:
            Parameters passed to ``kal_lkif``.

    Outputs:
        out:
            Dictionary containing realization-level arrays and mean curves:
            ``T21``, ``err90``, ``err95``, ``err99``, and their ``nanmean``
            counterparts.

    Description:
        Functionalized translation of ``MultiLK_with_negatives.m``.
        For each realization column:
        1. Extract target/driver/conditioner series.
        2. Call ``kal_lkif``.
        3. Aggregate transfer and significance outputs.

    Notes:
        - Plotting from the MATLAB script is intentionally not embedded here;
          this function returns raw arrays for flexible downstream plotting.
        - Requires ``scipy.io.loadmat``.
    """
    try:
        from scipy.io import loadmat
    except Exception as exc:
        raise ImportError("scipy is required for loading .mat files") from exc

    data = loadmat(mat_path)

    series_names = [target, driver] + list(conditioners)
    missing = [name for name in series_names if name not in data]
    if missing:
        raise KeyError(f"Missing variables in MAT file: {missing}")

    mats = [np.asarray(data[name], dtype=float) for name in series_names]
    variable_realization = mats[0].shape[1]

    T21 = []
    tau21 = []
    err90 = []
    err95 = []
    err99 = []

    for r in range(variable_realization):
        args = [m[:, r] for m in mats]
        t, tau, e90, e95, e99 = kal_lkif(ma_len, n_window, ma_type, np_step, *args)
        T21.append(t[:, 0])
        tau21.append(tau[:, 0])
        err90.append(np.real(e90[:, 0]))
        err95.append(np.real(e95[:, 0]))
        err99.append(np.real(e99[:, 0]))

    T21 = np.column_stack(T21)
    tau21 = np.column_stack(tau21)
    err90 = np.column_stack(err90)
    err95 = np.column_stack(err95)
    err99 = np.column_stack(err99)

    out = {
        "T21": T21,
        "tau21": tau21,
        "err90": err90,
        "err95": err95,
        "err99": err99,
        "T21mean": np.nanmean(T21, axis=1),
        "tau21mean": np.nanmean(tau21, axis=1),
        "err90mean": np.nanmean(err90, axis=1),
        "err95mean": np.nanmean(err95, axis=1),
        "err99mean": np.nanmean(err99, axis=1),
    }
    return out
