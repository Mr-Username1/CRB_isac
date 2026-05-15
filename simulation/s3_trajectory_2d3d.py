"""
S3: 2D vs 3D Trajectory Comparison
===================================
Compare fixed-altitude (2-D, z=H) vs variable-altitude (3-D) trajectory optimisation.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from simulation.common import (
    DEFAULT_ETA, DEFAULT_ETOT, DEFAULT_NSTG, DEFAULT_TARGET_XYZ, DEFAULT_USER_XY,
    FIGURES_DIR, RESULTS_DIR, ensure_dirs, get_configs, save_sim_results, load_sim_results,
)
from simulation_pipeline import run_method_case

NPZ_PATH = RESULTS_DIR / "s3_trajectory_2d3d.npz"


def _concat_paths(path_list):
    if not path_list:
        return np.zeros((0, 3))
    return np.vstack(path_list)


def run() -> None:
    cfg_3d, e_cfg, scfg = get_configs(scenario_name="paper_baseline", mu=5, max_sca_iter=20, step_size=0.6)

    # 2-D config: fix altitude to H
    cfg_2d = deepcopy(cfg_3d)
    cfg_2d.z_min = float(cfg_3d.H)
    cfg_2d.z_max = float(cfg_3d.H)

    print("\n=== S3: 3-D trajectory ===")
    case_3d = run_method_case(
        method_name="tradeoff_3d", eta=DEFAULT_ETA,
        user_xy=DEFAULT_USER_XY, true_target_xyz=DEFAULT_TARGET_XYZ,
        nstg=DEFAULT_NSTG, etot=DEFAULT_ETOT, random_seed=1,
        cfg=cfg_3d, e_cfg=e_cfg, scfg=scfg, localizer="ekf",
    )

    print("\n=== S3: 2-D trajectory (z fixed) ===")
    case_2d = run_method_case(
        method_name="tradeoff_2d", eta=DEFAULT_ETA,
        user_xy=DEFAULT_USER_XY, true_target_xyz=DEFAULT_TARGET_XYZ,
        nstg=DEFAULT_NSTG, etot=DEFAULT_ETOT, random_seed=1,
        cfg=cfg_2d, e_cfg=e_cfg, scfg=scfg, localizer="ekf",
    )

    crb_3d = float(case_3d["stage_logs"][-1]["crb_final"]) if case_3d["stage_logs"] else np.nan
    crb_2d = float(case_2d["stage_logs"][-1]["crb_final"]) if case_2d["stage_logs"] else np.nan
    rate_3d = float(case_3d["stage_logs"][-1]["rate_final"]) if case_3d["stage_logs"] else np.nan
    rate_2d = float(case_2d["stage_logs"][-1]["rate_final"]) if case_2d["stage_logs"] else np.nan

    save_sim_results(
        NPZ_PATH,
        traj_3d=_concat_paths(case_3d["all_paths"]),
        traj_2d=_concat_paths(case_2d["all_paths"]),
        hover_3d=np.asarray(case_3d["all_hover_xyz"], dtype=float),
        hover_2d=np.asarray(case_2d["all_hover_xyz"], dtype=float),
        crb_3d=np.array([crb_3d]), crb_2d=np.array([crb_2d]),
        rate_3d=np.array([rate_3d]), rate_2d=np.array([rate_2d]),
        pos_err_3d=np.array([float(case_3d["final_position_error_m"])]),
        pos_err_2d=np.array([float(case_2d["final_position_error_m"])]),
    )
    print(f"S3 done: 3D CRB={crb_3d:.3e}, 2D CRB={crb_2d:.3e}")


def plot() -> None:
    ensure_dirs()
    data = load_sim_results(NPZ_PATH)
    traj_3d = data["traj_3d"]
    traj_2d = data["traj_2d"]
    hover_3d = data["hover_3d"]
    hover_2d = data["hover_2d"]
    crb_3d = float(data["crb_3d"][0])
    crb_2d = float(data["crb_2d"][0])
    rate_3d = float(data["rate_3d"][0])
    rate_2d = float(data["rate_2d"][0])
    pos_3d = float(data["pos_err_3d"][0])
    pos_2d = float(data["pos_err_2d"][0])

    # --- 3-D trajectory comparison ---
    fig1 = plt.figure(figsize=(14, 6))
    for idx, (traj, hov, title) in enumerate([
        (traj_3d, hover_3d, "3-D Trajectory (variable altitude)"),
        (traj_2d, hover_2d, "2-D Trajectory (fixed altitude)"),
    ]):
        ax = fig1.add_subplot(1, 2, idx + 1, projection="3d")
        if traj.size > 0:
            sc = ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], c=traj[:, 2], cmap="viridis",
                            s=8, alpha=0.8)
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], lw=1, alpha=0.5, color="gray")
            fig1.colorbar(sc, ax=ax, shrink=0.6, label="Altitude (m)")
        if hov.size > 0:
            ax.scatter(hov[:, 0], hov[:, 1], hov[:, 2], s=15, marker="^", color="red", label="Hover")
        ax.scatter(DEFAULT_USER_XY[0], DEFAULT_USER_XY[1], 0, marker="*", s=100, c="green", label="User")
        ax.scatter(DEFAULT_TARGET_XYZ[0], DEFAULT_TARGET_XYZ[1], DEFAULT_TARGET_XYZ[2],
                   marker="x", s=80, c="red", label="Target")
        ax.set_title(title)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
        ax.legend(loc="best", fontsize=7)
    fig1.tight_layout()
    fig1.savefig(FIGURES_DIR / "s3_trajectory_comparison.png", dpi=200)

    # --- Performance comparison bars ---
    fig2, axes = plt.subplots(1, 3, figsize=(12, 4))
    metrics = [
        ("CRB trace", [crb_3d, crb_2d], "tab:red"),
        ("Avg Rate (bps)", [rate_3d, rate_2d], "tab:blue"),
        ("Pos Error (m)", [pos_3d, pos_2d], "tab:orange"),
    ]
    for ax, (name, vals, color) in zip(axes, metrics):
        bars = ax.bar(["3-D", "2-D"], vals, color=color, alpha=0.8, width=0.4)
        ax.set_title(name)
        ax.grid(True, axis="y", alpha=0.3)
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.2e}" if abs(h) < 0.01 else f"{h:.2f}",
                        ha="center", va="bottom", fontsize=8)
    fig2.suptitle("2-D vs 3-D Performance Comparison")
    fig2.tight_layout()
    fig2.savefig(FIGURES_DIR / "s3_crb_comparison.png", dpi=200)
    print("S3 plots saved.")


def main() -> None:
    run()
    plot()
    plt.show()


if __name__ == "__main__":
    main()
