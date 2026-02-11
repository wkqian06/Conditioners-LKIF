# Causality And Flow Summary

Related note:
- `MvLKIF/BIVARIATE_VS_MULTIVARIATE.md`: detailed differences between bivariate and multivariate calculations.

## What `causality.py` is for
`MvLKIF/core/causality.py` is the static covariance-based LKIF layer.

Functions:
- `causality_est(xx1, xx2, np_step)`
- `multi_causality_est(xx, np_step)`
- `normalized_causality_est(..., tau_ci_method=None|"analytic"|"bootstrap")`
- `normalized_multi_causality_est(..., tau_ci_method=None|"analytic"|"bootstrap")`
- `all_causality_est(xx, np_step, ...)` (all directed pairs)
- `normalized_all_causality_est(xx, np_step, ...)` (all directed pairs + tau)
- `z_norm_cal(tau_mat, h_noise)` (target-wise `Z` post-processing)

`tau_ci_method="analytic"` uses delta-method approximation.
`tau_ci_method="bootstrap"` uses moving-block bootstrap percentile CI.

Batch matrix orientation in `all_*` APIs:
- source is row index `j`
- target is column index `i`
- entry `[j, i]` means `j -> i`

## What `kal_lkif.py` is for
`MvLKIF/core/kal_lkif.py` is the time-varying Kalman LKIF engine.
It reproduces MATLAB `multiLK_code.m` behavior and returns full trajectories.

Functions:
- `kal_lkif(MAlen, NN, MA, np_step, *series, return_tau=True, tau_ci_method=None|"analytic"|"bootstrap")`
- `normalized_kal_lkif(...)`
- `KalLKIF(...)` (directional core routine)
- `kal_lkif_target_all(...)` (fixed target, all sources)
- `kal_lkif_all(...)` (all directed pairs as `(m, m, t)` arrays)

`kal_lkif_all` matrix orientation:
- source is row `j`
- target is column `i`
- entry `[j, i, t]` is time-varying `j -> i`

## Calling Flow
### Static path
1. `MvLKIF/__init__.py`
2. `MvLKIF/core/causality.py`

### Time-varying path
1. `MvLKIF/__init__.py`
2. `MvLKIF/core/kal_lkif.py`
3. `MvLKIF/kalman/square_root_filter.py`
4. `MvLKIF/kalman/moving_average.py`
5. `MvLKIF/kalman/covariance.py`
6. `MvLKIF/kalman/linear_algebra.py`
7. `MvLKIF/kalman/updates.py`

### Workflow path
1. `MvLKIF/workflows/multi_lk_with_negatives.py`
2. loops realizations and calls `kal_lkif(...)`

## Practical rule
- Choose `causality.py` for single-value causality and tau CI.
- Choose `kal_lkif.py` for time-resolved causality and tau CI trajectories.
