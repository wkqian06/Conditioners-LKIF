"""Refactored LKIF package with modular static and time-varying APIs."""

from .kalman.moving_average import ewma, uwma
from .kalman.covariance import standard_cov_est, ewma_cov_est
from .kalman.linear_algebra import my_ud, my_unit_tri_sys_sol, kalman_gain_calc
from .kalman.updates import thornton, bierman
from .kalman.square_root_filter import square_root_kalman_filter2
from .core.causality import (
    causality_est,
    multi_causality_est,
    normalized_causality_est,
    normalized_multi_causality_est,
    all_causality_est,
    normalized_all_causality_est,
    z_norm_cal,
)
from .core.kal_lkif import (
    KalLKIF,
    kal_lkif,
    normalized_kal_lkif,
    kal_lkif_target_all,
    kal_lkif_all,
)
from .core.moving_lkif import MovingLKIF, moving_lkif
from .workflows.multi_lk_with_negatives import run_multilk_with_negatives
