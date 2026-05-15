"""
S5: Energy Budget Impact Analysis
=================================
Vary total energy budget Etot and observe impact on final CRB, rate,
and localisation accuracy.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from simulation.common import (
    DEFAULT_ETA, DEFAULT_NSTG, DEFAULT_TARGET_XYZ, DEFAULT_USER_XY,
    FIGURES_DIR, RESULTS_DIR, ensure_dirs, get_configs, save_sim_results, load_sim_results,
)
from simulation_pipeline import run_method_case

ETOT_LIST = [15e3, 25e3, 35e3, 50e3, 65e3, 80e3]
NPZ_PATH = RESULTS_DIR / "s5_energy_impact.npz"


def run() -> None:
    cfg, e_cfg, scfg = get_configs(scenario_name="paper_baseline", mu=5, max_sca_iter=20, step_size=0.6)

    etot_values = []
    final_crb = []
    final_rate = []
    final_pos_err = []
    num_stages = []
    energy_used = []

    for etot in ETOT_LIST:
        print(f"\n=== S5: Etot = {etot / 1e3:.0f} kJ ===")
        case = run_method_case(
            method_name=f"etot_{etot}", eta=DEFAULT_ETA,
            user_xy=DEFAULT_USER_XY, true_target_xyz=DEFAULT_TARGET_XYZ,
            nstg=DEFAULT_NSTG, etot=etot, random_seed=1,
            cfg=cfg, e_cfg=e_cfg, scfg=scfg, localizer="ekf",
        )
        etot_values.append(etot)
        num_stages.append(case["num_stages"])
        energy_used.append(float(etot - case["energy_left"]))
        if case["stage_logs"]:
            final_crb.append(float(case["stage_logs"][-1]["crb_final"]))
            final_rate.append(float(case["stage_logs"][-1]["rate_final"]))
        else:
            final_crb.append(np.nan)
            final_rate.append(np.nan)
        final_pos_err.append(float(case["final_position_error_m"]))
        print(f"  stages={case['num_stages']}, CRB={final_crb[-1]:.3e}, "
              f"Rate={final_rate[-1]:.2f}, PosErr={final_pos_err[-1]:.2f}m")

    save_sim_results(
        NPZ_PATH,
        etot_list=np.array(etot_values, dtype=float),
        final_crb=np.array(final_crb, dtype=float),
        final_rate=np.array(final_rate, dtype=float),
        final_pos_err=np.array(final_pos_err, dtype=float),
        num_stages=np.array(num_stages, dtype=int),
        energy_used=np.array(energy_used, dtype=float),
    )
    print(f"S5 done.")


def plot() -> None:
    ensure_dirs()
    data = load_sim_results(NPZ_PATH)
    etot = np.array(data["etot_list"], dtype=float) / 1e3  # kJ
    crb = np.array(data["final_crb"], dtype=float)
    rate = np.array(data["final_rate"], dtype=float)
    pos = np.array(data["final_pos_err"], dtype=float)
    nstg = np.array(data["num_stages"], dtype=int)

    fig, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
    series = [
        (axes[0], "CRB trace", crb, "tab:red", "o"),
        (axes[1], "Avg Rate (bps)", rate, "tab:blue", "^"),
        (axes[2], "Pos Error (m)", pos, "tab:orange", "s"),
        (axes[3], "Num Stages", nstg, "tab:green", "D"),
    ]
    for ax, name, vals, color, marker in series:
        valid = np.isfinite(vals)
        ax.plot(etot[valid], vals[valid], marker=marker, color=color, lw=2, ms=6)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Total Energy Budget (kJ)")
    axes[0].set_title("Impact of Energy Budget on System Performance")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s5_energy_impact.png", dpi=200)
    print("S5 plot saved.")


def main() -> None:
    run()
    plot()
    plt.show()


if __name__ == "__main__":
    main()
