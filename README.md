# Conditioners-LKIF
Benchmarking Conditioners in Liang–Kleeman Information Flow: Application to Land–Atmosphere Interactions

This repository provides the MATLAB and Python code used in the analysis presented in:

Benchmarking Conditioners in Liang–Kleeman Information Flow: Application to Land–Atmosphere Interactions
Siddique et al., 2026 (submitted to Earth System Dynamics)

Repository Structure

- `Multi_LKIF/`: MATLAB functions and scripts for multivariate LKIF, including time-varying estimation with Kalman filter.
- `MvLKIF/`: refactored Python package for static and time-varying LKIF APIs.
- `ANOVA.ipynb`: regime-based ANOVA of ΔIF.
- `Toy Model.ipynb`: original synthetic VAR toy-model notebook.
- `Toy Model_python.ipynb`: updated Python toy-model workflow using the new `MvLKIF` package.

Key Python API entry points:
- Static single-direction: `causality_est`, `multi_causality_est`, `normalized_*`.
- Static all-directions: `all_causality_est`, `normalized_all_causality_est`.
- Time-varying single-direction: `kal_lkif`, `normalized_kal_lkif`.
- Time-varying all-directions: `kal_lkif_target_all`, `kal_lkif_all`.

Batch output orientation:
- row = source `j`
- column = target `i`
- entry `[j, i]` means `j -> i`
