"""Application-style demos for the Python MvLKIF package.

This script mirrors the MATLAB application scripts under ``Multi_LKIF/``:
- multiLK_toymodel1.m
- multiLK_toymodel2.m
- multiLK_toymodel3.m
- multiLK_huabeirealtest.m
- multiLK_huananrealtest.m
- MultiLK_with_negatives.m

Because this repository currently does not contain the referenced ``.mat`` data
files, the demo uses synthetic realizations while keeping the same function
calling patterns, parameter settings, and variable ordering.
"""

from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from MvLKIF import kal_lkif


@dataclass(frozen=True)
class AppConfig:
    """Configuration matching one MATLAB application script."""

    name: str
    ma_len: int
    nn: int
    ma_type: str
    np_step: int
    variable_order: Sequence[str]
    trim_start: int = 0


APP_CONFIGS: Dict[str, AppConfig] = {
    "toy1": AppConfig(
        name="toy1",
        ma_len=10,
        nn=290,
        ma_type="UWMA",
        np_step=1,
        variable_order=("x3", "x2", "x1"),
        trim_start=0,
    ),
    "toy2": AppConfig(
        name="toy2",
        ma_len=10,
        nn=290,
        ma_type="UWMA",
        np_step=1,
        variable_order=("x3", "x2", "x1"),
        trim_start=0,
    ),
    "toy3": AppConfig(
        name="toy3",
        ma_len=10,
        nn=240,
        ma_type="UWMA",
        np_step=1,
        variable_order=("x3", "x2", "x1"),
        trim_start=0,
    ),
    "huabei": AppConfig(
        name="huabei",
        ma_len=10,
        nn=80,
        ma_type="UWMA",
        np_step=1,
        variable_order=("huabei_GPP", "huabei_SM", "huabei_VPD"),
        trim_start=96,  # MATLAB 97:end
    ),
    "huanan": AppConfig(
        name="huanan",
        ma_len=10,
        nn=80,
        ma_type="UWMA",
        np_step=1,
        variable_order=("huanan_VPD", "huanan_SM", "huanan_GPP"),
        trim_start=96,  # MATLAB 97:end
    ),
    "amazon_negatives": AppConfig(
        name="amazon_negatives",
        ma_len=24,
        nn=90,
        ma_type="UWMA",
        np_step=1,
        variable_order=("LAI", "SM", "VPD", "T", "SSR"),
        trim_start=275,  # MATLAB 276:end
    ),
}



def _generate_synthetic_matrix(length: int, realizations: int, seed: int) -> np.ndarray:
    """Generate smooth, weakly autocorrelated synthetic realizations."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, size=(length, realizations))
    for t in range(1, length):
        x[t, :] = 0.88 * x[t - 1, :] + 0.18 * x[t, :]
    x = (x - np.mean(x, axis=0, keepdims=True)) / (np.std(x, axis=0, keepdims=True) + 1e-8)
    return x



def build_synthetic_inputs(variable_names: Sequence[str], length: int, realizations: int, seed: int) -> Dict[str, np.ndarray]:
    """Build synthetic variables with cross-coupling for demo purposes."""
    data: Dict[str, np.ndarray] = {}
    for i, name in enumerate(variable_names):
        base = _generate_synthetic_matrix(length, realizations, seed + i)
        data[name] = base

    # Add mild coupling so directional flow is non-trivial.
    names = list(variable_names)
    if len(names) >= 3:
        data[names[0]] = 0.55 * data[names[0]] + 0.30 * data[names[1]] - 0.10 * data[names[2]]
    if len(names) >= 5:
        data[names[0]] = data[names[0]] + 0.10 * data[names[3]] - 0.08 * data[names[4]]

    return data



def run_application(config: AppConfig, source_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Run one application using MATLAB-equivalent loop structure."""
    mats: List[np.ndarray] = [source_data[name] for name in config.variable_order]
    n_realizations = mats[0].shape[1]

    t_all, tau_all, e90_all, e95_all, e99_all = [], [], [], [], []
    for r in range(n_realizations):
        series = [m[:, r] for m in mats]
        t, tau, e90, e95, e99 = kal_lkif(
            config.ma_len,
            config.nn,
            config.ma_type,
            config.np_step,
            *series,
        )
        t_all.append(t[:, 0])
        tau_all.append(tau[:, 0])
        e90_all.append(np.real(e90[:, 0]))
        e95_all.append(np.real(e95[:, 0]))
        e99_all.append(np.real(e99[:, 0]))

    T = np.column_stack(t_all)
    Tau = np.column_stack(tau_all)
    E90 = np.column_stack(e90_all)
    E95 = np.column_stack(e95_all)
    E99 = np.column_stack(e99_all)

    result = {
        "T21": T,
        "tau21": Tau,
        "err90": E90,
        "err95": E95,
        "err99": E99,
        "T21mean": np.nanmean(T, axis=1),
        "tau21mean": np.nanmean(Tau, axis=1),
        "err90mean": np.nanmean(E90, axis=1),
        "err95mean": np.nanmean(E95, axis=1),
        "err99mean": np.nanmean(E99, axis=1),
    }

    if config.trim_start > 0:
        s = config.trim_start
        result["T21mean_trimmed"] = result["T21mean"][s:]
        result["err95mean_trimmed"] = result["err95mean"][s:]

    return result



def run_all_demos(
    length: int = 400,
    realizations: int = 4,
    seed: int = 42,
    apps: Sequence[str] | None = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Run MATLAB-mapped application demos on synthetic data.

    Parameters are intentionally smaller than MATLAB scripts by default so the
    demo is quick to run on CPU-only development environments.
    """
    out: Dict[str, Dict[str, np.ndarray]] = {}
    selected = list(APP_CONFIGS.keys()) if apps is None else list(apps)
    for name in selected:
        if name not in APP_CONFIGS:
            raise KeyError(f"Unknown app '{name}'. Available: {sorted(APP_CONFIGS)}")
        cfg = APP_CONFIGS[name]
        data = build_synthetic_inputs(cfg.variable_order, length, realizations, seed)
        out[name] = run_application(cfg, data)
    return out



def print_summary(results: Dict[str, Dict[str, np.ndarray]]) -> None:
    """Print compact summary similar to quick MATLAB inspection."""
    for name, res in results.items():
        tmean = res["T21mean"]
        e95 = res["err95mean"]
        print(f"[{name}] len={len(tmean)} mean(T21)={np.nanmean(tmean): .6f} mean(err95)={np.nanmean(e95): .6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Python MvLKIF demos mapped from MATLAB applications.")
    parser.add_argument("--length", type=int, default=400, help="Length of synthetic time series.")
    parser.add_argument("--realizations", type=int, default=4, help="Number of synthetic realizations.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed for synthetic data.")
    parser.add_argument(
        "--apps",
        nargs="*",
        default=None,
        help=f"Subset of apps to run. Choices: {', '.join(APP_CONFIGS.keys())}",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    results = run_all_demos(
        length=args.length,
        realizations=args.realizations,
        seed=args.seed,
        apps=args.apps,
    )
    print_summary(results)
