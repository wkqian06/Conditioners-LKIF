# Bivariate vs Multivariate in MvLKIF

This note explains how `bivariate` and `multivariate` estimators differ in `MvLKIF`, and where they are exactly the same.

## Scope
- Static estimators: `MvLKIF/core/causality.py`
- Time-varying estimators: `MvLKIF/core/moving_lkif.py`, `MvLKIF/core/kal_lkif.py`

## 1) Static Layer (`causality.py`)

### APIs
- Bivariate:
  - `causality_est(xx1, xx2, np_step, ...)`
  - `normalized_causality_est(xx1, xx2, np_step, ...)`
- Multivariate:
  - `multi_causality_est(xx, np_step, ...)`
  - `normalized_multi_causality_est(xx, np_step, ...)`

### Core implementation relationship
All four functions use the same core solver: `_estimate_transfer(...)`.

- Bivariate builds a 2-column matrix `[x1, x2]` and calls `_estimate_transfer`.
- Multivariate passes full matrix `[x1, x2, x3, ..., xm]` to `_estimate_transfer`.

So the estimator logic is unified; the key difference is dimensionality of the regression/covariance system.

### T21 difference
`T21` is always computed from row-1 coefficients and covariance ratio:
- `T21 = a12 * C12 / denom`

In multivariate mode, `a12` is estimated conditionally with all extra variables in the system, so `T21` is conditional transfer.

### tau difference (most important)
`tau = T21 / Z`, but `Z` differs by dimensionality:
- Bivariate (`m == 2`):
  - `Z = |H_self| + |H_noise| + |T21|`
- Multivariate (`m > 2`):
  - `Z = |H_self| + sum(|T_all_sources|) + |H_noise|`

Therefore, multivariate `tau` includes normalization by total source contributions, not only the selected source.

### Normalized vs non-normalized API detail
- Non-normalized (`causality_est`, `multi_causality_est`) uses unstabilized ratio denominator (`c11`).
- Normalized (`normalized_*`) uses stabilized denominator (`c11 + eps`) for better numerical robustness.

This is an implementation detail; it can cause tiny numerical differences near singular/low-variance cases.

## 2) Moving-Window Layer (`moving_lkif.py`)

`moving_lkif(window, step, np_step, *series, ...)` chooses backend by dimension:
- If `m == 2`: calls bivariate static estimators per window.
- If `m > 2`: calls multivariate static estimators per window.

So moving-window behavior directly inherits the static-layer differences above.

## 3) Kalman Time-Varying Layer (`kal_lkif.py`)

`kal_lkif(...)` is implemented as a unified `m >= 2` estimator; there is no separate bivariate API.

- For `m == 2`, formulas naturally reduce to bivariate behavior.
- For `m > 2`, normalization uses total directional-source sum (`sum(|T_all|)`), matching multivariate logic.

## 4) Confidence Intervals (CI)

### Static (`causality.py`)
Both bivariate and multivariate normalized APIs support:
- `tau_ci_method=None | "analytic" | "bootstrap"`
- same return shape and level-key format (`"90"`, `"95"`, `"99"`, ...)

### Time-varying (`kal_lkif.py`)
`kal_lkif(..., tau_ci_method=...)` supports:
- analytic series CI via `_tau_ci_analytic_series`
- bootstrap series CI via `_tau_ci_bootstrap_series`

CI interfaces are consistent across dimensions; numerical values differ because `Z` and fitted coefficients differ.

## 5) Batch APIs and matrix orientation
The package now also exposes batch APIs:
- Static all-directions: `all_causality_est`, `normalized_all_causality_est`
- Time-varying all-directions: `kal_lkif_target_all`, `kal_lkif_all`

For all batch matrix outputs, orientation is:
- row = source `j`
- column = target `i`
- entry `[j, i]` (or `[j, i, t]`) means `j -> i`

In `normalized_all_causality_est`, `tau` diagonal is filled as `tau_self = H_self / Z`.
`h_noise` is returned as a target-wise vector.

`z_norm_cal(tau_mat, h_noise)` computes target-wise `Z` from:
- `Z_i = sum_j |tau_{j->i}| + |h_noise_i|`
- the summation is over column `i`.

## 6) When should bivariate and multivariate match?
They should match exactly (up to floating-point roundoff) if multivariate input contains only two columns `[x1, x2]`.

They generally differ when extra variables are included, because multivariate estimates conditional transfer and uses broader normalization in `tau`.

## 7) Practical guidance
- Use bivariate when you only trust/observe a pair and want pairwise transfer.
- Use multivariate when confounders/conditioners are available and conditional effects matter.
- Compare both when diagnosing suppression/confounding.

## 8) Code locations
- Unified static core: `MvLKIF/core/causality.py` (`_estimate_transfer`)
- Moving-window switch: `MvLKIF/core/moving_lkif.py`
- Time-varying Kalman core: `MvLKIF/core/kal_lkif.py` (`KalLKIF`, `kal_lkif`, `kal_lkif_target_all`, `kal_lkif_all`)
