"""
S2: Trade-off Analysis (Communication vs Sensing)
==================================================
Sweep eta from 0 (communication-only) to 1 (sensing-only) to reveal the Pareto frontier.
The ETA_LIST is the extensible interface — add/remove values as needed.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from simulation.common import (
    DEFAULT_ETOT, DEFAULT_NSTG, DEFAULT_TARGET_XYZ, DEFAULT_USER_XY,
    FIGURES_DIR, RESULTS_DIR, ensure_dirs, get_configs, save_sim_results, load_sim_results,
)
from simulation_pipeline import run_method_case

# ---- Extensible eta list ----
ETA_LIST = [0.0, 0.3, 0.5, 0.8, 1.0]

NPZ_PATH = RESULTS_DIR / "s2_tradeoff.npz"


def run() -> None:
    cfg, e_cfg, scfg = get_configs(scenario_name="paper_baseline", mu=5, max_sca_iter=50, step_size=0.6)

    eta_values = []
    final_crb = []
    final_rate = []
    final_pos_err = []
    all_results = []

    for eta in ETA_LIST:
        method_name = f"eta_{eta}"
        print(f"\n=== S2: eta = {eta} ===")
        case = run_method_case(
            method_name=method_name,
            eta=eta,
            user_xy=DEFAULT_USER_XY,
            true_target_xyz=DEFAULT_TARGET_XYZ,
            nstg=DEFAULT_NSTG,
            etot=DEFAULT_ETOT,
            random_seed=1,
            cfg=cfg, e_cfg=e_cfg, scfg=scfg,
            localizer="ekf",
        )
        all_results.append(case)
        eta_values.append(eta)
        if case["stage_logs"]:
            final_crb.append(float(case["stage_logs"][-1]["crb_final"]))
            final_rate.append(float(case["stage_logs"][-1]["rate_final"]))
        else:
            final_crb.append(np.nan)
            final_rate.append(np.nan)
        final_pos_err.append(float(case["final_position_error_m"]))
        print(f"  CRB={final_crb[-1]:.3e}, Rate={final_rate[-1]:.2f}, PosErr={final_pos_err[-1]:.2f}m")

    save_sim_results(
        NPZ_PATH,
        eta_list=np.array(eta_values, dtype=float),
        final_crb=np.array(final_crb, dtype=float),
        final_rate=np.array(final_rate, dtype=float),
        final_pos_err=np.array(final_pos_err, dtype=float),
        results=np.array(all_results, dtype=object),
    )
    print(f"S2 done: {len(eta_values)} eta values.")


def _concat_paths(path_list):
    if not path_list:
        return np.zeros((0, 3))
    return np.vstack(path_list)


def plot() -> None:
    ensure_dirs()
    data = load_sim_results(NPZ_PATH)
    eta_list = list(data["eta_list"])
    final_crb = list(data["final_crb"])
    final_rate = list(data["final_rate"])
    final_pos_err = list(data["final_pos_err"])
    all_results = list(data["results"])

    x = np.arange(len(eta_list))
    eta_labels = [f"$\\eta$={e:.1f}" for e in eta_list]
    colors = ["#1b6e3a", "#2a8c5c", "#1a6b9e", "#6b2f8e", "#a82828"][:len(eta_list)]

    # ---- Figure 1: Grouped bar chart (CRB + Rate + Pos Error) ----
    fig1, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    metric_specs = [
        (axes[0], "CRB (xyz trace)", final_crb, "tab:red"),
        (axes[1], "Avg Rate (bps)", final_rate, "tab:blue"),
        (axes[2], "Position Error (m)", final_pos_err, "tab:orange"),
    ]
    for ax, title, vals, color in metric_specs:
        bars = ax.bar(x, vals, 0.55, color=color, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(eta_labels)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        for bar in bars:
            h = bar.get_height()
            if np.isfinite(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h,
                        f"{h:.1f}" if abs(h) >= 1 else f"{h:.2e}",
                        ha="center", va="bottom", fontsize=8)

    fig1.suptitle("Communication–Sensing Trade-off Performance Comparison")
    fig1.tight_layout()
    fig1.savefig(FIGURES_DIR / "s2_tradeoff_bars.png", dpi=200)

    # ---- Figure 2: Pareto frontier (Rate vs CRB) ----
    fig2, ax = plt.subplots(figsize=(6, 5))
    valid = [i for i, (c, r) in enumerate(zip(final_crb, final_rate)) if np.isfinite(c) and np.isfinite(r)]
    crb_arr = np.array([final_crb[i] for i in valid])
    rate_arr = np.array([final_rate[i] for i in valid])
    labels = [f"$\\eta$={eta_list[i]:.1f}" for i in valid]
    ax.plot(crb_arr, rate_arr, "-o", color="tab:green", lw=2, ms=8)
    for i, lab in enumerate(labels):
        ax.annotate(lab, (crb_arr[i], rate_arr[i]), textcoords="offset points",
                    xytext=(10, 6), fontsize=10)
    ax.set_xlabel("CRB (xyz trace)")
    ax.set_ylabel("Average Rate (bps)")
    ax.set_title("Pareto Frontier: Rate vs CRB")
    ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(FIGURES_DIR / "s2_tradeoff_pareto.png", dpi=200)

    # ---- Figure 3: Trajectory comparison (2-D top view) ----
    fig3, ax = plt.subplots(figsize=(10, 8))

    method_labels = {
        "eta_0.0": "η=0.0 (comm. only)",
        "eta_0.5": "η=0.5 (balanced)",
        "eta_0.8": "η=0.8 (sensing-leaning)",
        "eta_1.0": "η=1.0 (sensing only)",
    }

    for i, (case, eta, c) in enumerate(zip(all_results, eta_list, colors)):
        traj = _concat_paths(case.get("all_paths", []))
        hover = np.asarray(case.get("all_hover_xyz", []), dtype=float)
        coarse = np.asarray(case.get("coarse_hover_xyz", []), dtype=float)
        key = f"eta_{eta:.1f}"
        label = method_labels.get(key, f"η={eta:.1f}")

        if traj.size > 0:
            ax.plot(traj[:, 0], traj[:, 1], "-", color=c, lw=2, alpha=0.85, label=label)
        if hover.size > 0:
            ax.scatter(hover[:, 0], hover[:, 1], s=20, marker="^",
                       color=c, edgecolors="black", linewidths=0.3, alpha=0.9)
        # Show coarse scan only once
        if i == 0 and coarse.size > 0:
            ax.plot(coarse[:, 0], coarse[:, 1], "--", lw=1.2, color="gray", alpha=0.7, label="coarse scan")
        # Annotate η near last waypoint
        if traj.size > 0:
            ax.annotate(f"η={eta:.1f}", (traj[-1, 0], traj[-1, 1]),
                        xytext=(8, -8), textcoords="offset points", fontsize=9,
                        color=c, fontweight="bold")

    ax.scatter(DEFAULT_USER_XY[0], DEFAULT_USER_XY[1], marker="*", s=200,
               c="green", edgecolors="black", linewidths=0.5, zorder=5, label="User")
    ax.scatter(DEFAULT_TARGET_XYZ[0], DEFAULT_TARGET_XYZ[1], marker="x", s=120,
               c="red", linewidths=2.5, zorder=5, label="Target")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("UAV Trajectories for Different η Values")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=7.5, ncol=1)
    fig3.tight_layout()
    fig3.savefig(FIGURES_DIR / "s2_trajectories.png", dpi=200)

    print("S2 plots saved.")


def main() -> None:
    # run()
    plot()
    plt.show()


if __name__ == "__main__":
    main()
