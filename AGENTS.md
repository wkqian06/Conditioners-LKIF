# Repository Guidelines

## Project Structure & Module Organization
This repository combines MATLAB LKIF methods and Python analysis notebooks.

- `Multi_LKIF/`: core MATLAB toolbox and runnable scripts.
- `Multi_LKIF/multiLK_code.m`: main multivariate time-varying LKIF function.
- `Multi_LKIF/multiLK_toymodel*.m`: synthetic experiment entry points.
- `Multi_LKIF/*realtest.m`: regional application scripts.
- `MvLKIF/`: refactored Python package for static and time-varying LKIF.
- `MvLKIF/core/causality.py`: static estimators (`causality_est`, `multi_causality_est`) and normalized variants with optional `tau` confidence intervals.
- `MvLKIF/core/kal_lkif.py`: time-varying Kalman LKIF engine (`kal_lkif`, `normalized_kal_lkif`, `KalLKIF`).
- `MvLKIF/core/moving_lkif.py`: sliding-window estimator (`MovingLKIF` / `moving_lkif`).
- Root notebooks: `ANOVA.ipynb`, `Toy Model.ipynb`, `rIF_deltaIF.ipynb`, `Conditioner_Based_Couplings_Analysis.ipynb`.
- Documentation: `README.md`, `readme.txt`, `Multi_LKIF/readme.txt`.

Keep reusable MATLAB functions in `Multi_LKIF/`; keep one-off exploration in notebooks.

## Build, Test, and Development Commands
No single build system is configured; use direct MATLAB and notebook execution.

- `matlab -batch "run('Multi_LKIF/multiLK_toymodel1.m')"`: run synthetic MATLAB workflow.
- `matlab -batch "run('Multi_LKIF/multiLK_huabeirealtest.m')"`: run regional MATLAB case study.
- `jupyter nbconvert --execute --to notebook --inplace "ANOVA.ipynb"`: execute a notebook end-to-end.
- `jupyter nbconvert --execute --to notebook --inplace "Toy Model.ipynb"`: validate synthetic Python analysis.
- `python -m compileall MvLKIF`: quick syntax validation of Python package.
- `python -c "import numpy as np; from MvLKIF import kal_lkif"`: import smoke test for renamed time-varying API.
- `python -c "import numpy as np; from MvLKIF import normalized_causality_est; x=np.random.randn(300); y=np.random.randn(300); print(normalized_causality_est(x,y,1,tau_ci_method='analytic')[-1])"`: static analytic `tau` CI check.
- `python -c "import numpy as np; from MvLKIF import normalized_multi_causality_est; X=np.random.randn(300,3); print(normalized_multi_causality_est(X,1,tau_ci_method='bootstrap',n_boot=20,random_state=0)[-1])"`: static bootstrap `tau` CI check.

Run commands from repository root so relative file loads (for `.mat` inputs) resolve correctly.

## Coding Style & Naming Conventions
- MATLAB: follow existing style in `Multi_LKIF/` (procedural scripts, descriptive snake/camel mix, `.m` filenames aligned with function names where possible).
- Python notebooks: use clear, vectorized `numpy`/`pandas` workflows and readable plotting sections.
- Indentation: 4 spaces in new MATLAB/Python code cells; avoid tab characters.
- Naming: keep script names task-specific (for example `multiLK_toymodel2.m`), and keep variable names consistent with LKIF terminology (`T21`, `err95`, `MAlen`).
- Python API naming: use `kal_lkif` (not `multi_lk_code`) for time-varying LKIF in new code.
- For normalized LKIF calls, prefer explicit `tau_ci_method` (`None`, `analytic`, `bootstrap`) when uncertainty reporting is needed.

## Testing Guidelines
There is no formal unit-test suite yet. Validate by deterministic re-runs:

- Re-run modified MATLAB scripts and confirm outputs/figures regenerate without errors.
- Execute touched notebooks with `nbconvert --execute`.
- When adding methods, include a minimal reproducible script or notebook cell that demonstrates expected behavior.

## Commit & Pull Request Guidelines
Recent history uses short, imperative commit subjects (`Add ...`, `Rename ...`, `Delete ...`).

- Commit format: imperative subject line, <= 72 chars, focused scope.
- PRs should include: purpose, affected files, how to reproduce, and key output changes (figures/tables).
- If notebook outputs changed, state whether changes are expected from re-execution.
