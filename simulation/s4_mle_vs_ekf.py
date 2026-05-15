"""
S4: MLE vs EKF Comparison
=========================
Compare Maximum Likelihood Estimation (MLE) and Extended Kalman Filter (EKF)
in terms of localisation accuracy and computational speed.
Uses multiple random seeds for statistical robustness.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from simulation.common import (
    DEFAULT_ETA, DEFAULT_ETOT, DEFAULT_NSTG, DEFAULT_TARGET_XYZ, DEFAULT_USER_XY,
    FIGURES_DIR, RESULTS_DIR, ensure_dirs, get_configs, save_sim_results, load_sim_results,
)
from simulation_pipeline import run_method_case

SEEDS = [1, 2, 3]
NPZ_PATH = RESULTS_DIR / "s4_mle_vs_ekf.npz"


def _pad_lists(list_of_lists, max_len):
    """Pad each sub-list to max_len with NaN."""
    out = np.full((len(list_of_lists), max_len), np.nan)
    for i, row in enumerate(list_of_lists):
        out[i, :len(row)] = row
    return out


def run() -> None:
    cfg, e_cfg, scfg = get_configs(scenario_name="paper_baseline", mu=5, max_sca_iter=20, step_size=0.6)

    mle_pos_err_per_stage = []
    ekf_pos_err_per_stage = []
    mle_t_sca = []
    ekf_t_sca = []
    mle_t_loc = []
    ekf_t_loc = []

    for seed in SEEDS:
        print(f"\n=== S4: seed={seed} ===")

        print("  Running MLE...")
        case_mle = run_method_case(
            method_name="mle", eta=DEFAULT_ETA,
            user_xy=DEFAULT_USER_XY, true_target_xyz=DEFAULT_TARGET_XYZ,
            nstg=DEFAULT_NSTG, etot=DEFAULT_ETOT, random_seed=seed,
            cfg=cfg, e_cfg=e_cfg, scfg=scfg, localizer="mle",
        )
        mle_errs = [float(log["position_error_m"]) for log in case_mle["stage_logs"]]
        mle_pos_err_per_stage.append(mle_errs)
        mle_t_sca.append([float(log.get("t_sca_s", np.nan)) for log in case_mle["stage_logs"]])
        mle_t_loc.append([float(log.get("t_loc_s", np.nan)) for log in case_mle["stage_logs"]])

        print("  Running EKF...")
        case_ekf = run_method_case(
            method_name="ekf", eta=DEFAULT_ETA,
            user_xy=DEFAULT_USER_XY, true_target_xyz=DEFAULT_TARGET_XYZ,
            nstg=DEFAULT_NSTG, etot=DEFAULT_ETOT, random_seed=seed,
            cfg=cfg, e_cfg=e_cfg, scfg=scfg, localizer="ekf",
        )
        ekf_errs = [float(log["position_error_m"]) for log in case_ekf["stage_logs"]]
        ekf_pos_err_per_stage.append(ekf_errs)
        ekf_t_sca.append([float(log.get("t_sca_s", np.nan)) for log in case_ekf["stage_logs"]])
        ekf_t_loc.append([float(log.get("t_loc_s", np.nan)) for log in case_ekf["stage_logs"]])

    # Pad to uniform length
    max_stages = max(
        max(len(r) for r in mle_pos_err_per_stage),
        max(len(r) for r in ekf_pos_err_per_stage),
    )
    mle_pos = _pad_lists(mle_pos_err_per_stage, max_stages)
    ekf_pos = _pad_lists(ekf_pos_err_per_stage, max_stages)
    mle_sca = _pad_lists(mle_t_sca, max_stages)
    ekf_sca = _pad_lists(ekf_t_sca, max_stages)
    mle_loc_t = _pad_lists(mle_t_loc, max_stages)
    ekf_loc_t = _pad_lists(ekf_t_loc, max_stages)

    save_sim_results(
        NPZ_PATH,
        mle_pos_err=mle_pos, ekf_pos_err=ekf_pos,
        mle_t_sca=mle_sca, ekf_t_sca=ekf_sca,
        mle_t_loc=mle_loc_t, ekf_t_loc=ekf_loc_t,
        seeds=np.array(SEEDS, dtype=int),
    )
    print(f"S4 done.")


def plot() -> None:
    ensure_dirs()
    data = load_sim_results(NPZ_PATH)
    mle_pos = data["mle_pos_err"]
    ekf_pos = data["ekf_pos_err"]
    mle_sca = data["mle_t_sca"]
    ekf_sca = data["ekf_t_sca"]
    mle_loc_t = data["mle_t_loc"]
    ekf_loc_t = data["ekf_t_loc"]

    n_stages = mle_pos.shape[1]

    # --- Position error vs stage ---
    fig1, ax = plt.subplots(figsize=(9, 5))
    stages = np.arange(1, n_stages + 1)
    mle_mean = np.nanmean(mle_pos, axis=0)
    mle_std = np.nanstd(mle_pos, axis=0)
    ekf_mean = np.nanmean(ekf_pos, axis=0)
    ekf_std = np.nanstd(ekf_pos, axis=0)
    ax.plot(stages, mle_mean, "-o", color="tab:blue", lw=2, ms=5, label="MLE")
    ax.fill_between(stages, mle_mean - mle_std, mle_mean + mle_std, alpha=0.2, color="tab:blue")
    ax.plot(stages, ekf_mean, "-^", color="tab:red", lw=2, ms=5, label="EKF")
    ax.fill_between(stages, ekf_mean - ekf_std, ekf_mean + ekf_std, alpha=0.2, color="tab:red")
    ax.set_xlabel("Stage")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Localisation Accuracy: MLE vs EKF")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(FIGURES_DIR / "s4_position_error.png", dpi=200)

    # --- Runtime comparison ---
    fig2, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    mle_total = np.nansum(mle_sca + mle_loc_t, axis=1)
    ekf_total = np.nansum(ekf_sca + ekf_loc_t, axis=1)
    mle_loc_sum = np.nansum(mle_loc_t, axis=1)
    ekf_loc_sum = np.nansum(ekf_loc_t, axis=1)

    ax = axes[0]
    x = np.arange(2)
    w = 0.35
    bars_sca = ax.bar(x - w / 2, [np.mean(mle_total - mle_loc_sum), np.mean(ekf_total - ekf_loc_sum)],
                      width=w, color="tab:purple", alpha=0.8, label="SCA solver")
    bars_loc = ax.bar(x - w / 2, [np.mean(mle_loc_sum), np.mean(ekf_loc_sum)],
                      width=w, bottom=[np.mean(mle_total - mle_loc_sum), np.mean(ekf_total - ekf_loc_sum)],
                      color="tab:orange", alpha=0.8, label="Localizer")
    ax.set_xticks(x)
    ax.set_xticklabels(["MLE", "EKF"])
    ax.set_ylabel("Time (s)")
    ax.set_title("Per-stage Runtime (average)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(["MLE", "EKF"], [np.mean(mle_total), np.mean(ekf_total)], color=["tab:blue", "tab:red"], alpha=0.7)
    ax.set_ylabel("Total Time (s)")
    ax.set_title("Total Runtime")
    ax.grid(True, axis="y", alpha=0.3)

    fig2.suptitle("Computational Efficiency: MLE vs EKF")
    fig2.tight_layout()
    fig2.savefig(FIGURES_DIR / "s4_runtime_comparison.png", dpi=200)
    print("S4 plots saved.")


def main() -> None:
    run()
    plot()
    plt.show()


if __name__ == "__main__":
    main()
