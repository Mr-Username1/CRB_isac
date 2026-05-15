"""Shared utilities for simulation scripts."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from config_factory import build_default_configs


RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def get_configs(
    scenario_name: str = "paper_baseline",
    mu: int = 5,
    max_sca_iter: int = 50,
    step_size: float = 0.6,
):
    return build_default_configs(
        scenario_name=scenario_name,
        mu=mu,
        max_sca_iter=max_sca_iter,
        step_size=step_size,
    )


def save_sim_results(filepath: str | Path, **kwargs) -> None:
    ensure_dirs()
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert lists of arrays for np.savez compatibility
    save_dict = {}
    for k, v in kwargs.items():
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.ndarray):
            save_dict[k] = np.array(v, dtype=object)
        else:
            save_dict[k] = v
    np.savez(path, **save_dict)
    print(f"  Saved: {path}")


def load_sim_results(filepath: str | Path) -> dict:
    data = np.load(filepath, allow_pickle=True)
    result = {}
    for key in data.files:
        result[key] = data[key]
    return result


# Default experiment parameters shared across simulations
DEFAULT_USER_XY = np.array([300.0, 400.0], dtype=float)
DEFAULT_TARGET_XYZ = np.array([1842.8, 1709.2, 18.3], dtype=float)
DEFAULT_NSTG = 20
DEFAULT_ETOT = 55e3
DEFAULT_ETA = 0.7
