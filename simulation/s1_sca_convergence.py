"""
S1: SCA Convergence Analysis
============================
Analyse per-iteration convergence of objective, CRB, and rate within each SCA stage.
Generates two figures:
  - s1_sca_convergence.png       Per-stage subplot grid
  - s1_global_convergence.png    Combined single-chart view (overall trend)
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

NPZ_PATH = RESULTS_DIR / "s1_sca_convergence.npz"


def run() -> None:
    cfg, e_cfg, scfg = get_configs(scenario_name="paper_baseline", mu=5, max_sca_iter=50, step_size=0.6)

    case = run_method_case(
        method_name="tradeoff",
        eta=DEFAULT_ETA,
        user_xy=DEFAULT_USER_XY,
        true_target_xyz=DEFAULT_TARGET_XYZ,
        nstg=DEFAULT_NSTG,
        etot=DEFAULT_ETOT,
        random_seed=1,
        cfg=cfg, e_cfg=e_cfg, scfg=scfg,
        localizer="ekf",
    )

    stage_histories = [np.asarray(h, dtype=object) for h in case["stage_histories"]]
    stage_nm = [int(log["Nm"]) for log in case["stage_logs"]]

    save_sim_results(
        NPZ_PATH,
        stage_histories=np.array(stage_histories, dtype=object),
        stage_nm=np.array(stage_nm, dtype=int),
    )
    print(f"S1 done: {len(stage_histories)} stages saved.")


def plot() -> None:
    """Per-stage subplot grid (original)."""
    ensure_dirs()
    data = load_sim_results(NPZ_PATH)
    stage_histories = list(data["stage_histories"])

    if not stage_histories:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No stage history data", ha="center", va="center")
        fig.savefig(FIGURES_DIR / "s1_sca_convergence.png", dpi=200)
        return

    n_stage = len(stage_histories)
    fig, axes = plt.subplots(
        n_stage, 3, figsize=(15.0, max(3.0 * n_stage, 4.5)),
        squeeze=False,
    )

    metric_specs = [
        ("Objective", 0, "tab:purple"),
        ("CRB", 1, "tab:red"),
        ("Rate", 2, "tab:blue"),
    ]

    for s_idx, history in enumerate(stage_histories):
        for m_idx, (metric_name, col, color) in enumerate(metric_specs):
            ax = axes[s_idx, m_idx]
            if history.size > 0:
                it = np.arange(1, history.shape[0] + 1)
                y = history[:, col].astype(float)
                ax.plot(it, y, marker="o", ms=2.5, lw=1.2, color=color)
                ax.set_xlabel("Iteration")
                ax.set_ylabel(metric_name)
                ax.grid(True, alpha=0.3)
                ax.set_title(f"Stage {s_idx + 1} — {metric_name}")
            else:
                ax.text(0.5, 0.5, "Empty", ha="center", va="center")
                ax.set_title(f"Stage {s_idx + 1} — {metric_name}")

    fig.suptitle("SCA Convergence per Stage", y=1.01, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s1_sca_convergence.png", dpi=200, bbox_inches="tight")
    print("S1 per-stage plot saved.")


def plot_combined() -> None:
    """Single-chart combined view: global iteration index across all stages."""
    ensure_dirs()
    data = load_sim_results(NPZ_PATH)
    stage_histories = list(data["stage_histories"])

    if not stage_histories:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No stage history data", ha="center", va="center")
        fig.savefig(FIGURES_DIR / "s1_global_convergence.png", dpi=200)
        return

    # Concatenate all stages into continuous arrays
    obj_all = []
    crb_all = []
    rate_all = []
    stage_boundaries = [0]  # cumulative iteration indices at stage transitions

    for history in stage_histories:
        if history.size > 0:
            obj_all.extend(history[:, 0].astype(float))
            crb_all.extend(history[:, 1].astype(float))
            rate_all.extend(history[:, 2].astype(float))
            stage_boundaries.append(stage_boundaries[-1] + history.shape[0])

    obj_all = np.array(obj_all)
    crb_all = np.array(crb_all)
    rate_all = np.array(rate_all)
    total_iters = len(obj_all)
    x_global = np.arange(1, total_iters + 1)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    series = [
        (axes[0], "Objective $J$", obj_all, "tab:purple"),
        (axes[1], "CRB (xyz trace)", crb_all, "tab:red"),
        (axes[2], "Avg Rate (bps)", rate_all, "tab:blue"),
    ]

    for ax, name, y, color in series:
        ax.plot(x_global, y, color=color, lw=1.8)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)
        for b in stage_boundaries[1:-1]:
            ax.axvline(b, color="tab:gray", linestyle="--", lw=1.2, alpha=0.7)

    # Stage labels at top of each panel
    for ax in axes:
        y_lim = ax.get_ylim()
        y_top = y_lim[1]
        for j in range(len(stage_boundaries) - 1):
            mid = 0.5 * (stage_boundaries[j] + stage_boundaries[j + 1])
            ax.annotate(
                f"Stage {j + 1}",
                xy=(mid, y_top),
                xytext=(0, 5), textcoords="offset points",
                fontsize=10, color="dimgray", ha="center", va="bottom",
            )

    # CRB on log scale for better visualisation of multi-order magnitude change
    if crb_all.max() / (crb_all.min() + 1e-30) > 100:
        axes[1].set_yscale("log")
        axes[1].set_ylabel("CRB (log scale)")

    axes[-1].set_xlabel("Cumulative SCA Iterations")
    axes[0].set_title("SCA Convergence — Overall Trend Across Stages")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s1_global_convergence.png", dpi=200, bbox_inches="tight")
    print("S1 global convergence plot saved.")


def main() -> None:
    run()
    plot()
    plot_combined()
    plt.show()


if __name__ == "__main__":
    main()
