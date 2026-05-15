"""
S6: CRB Spatial Heatmap
=======================
Compute CRB on a 2-D grid from the final hover points of an optimised trajectory,
showing how trajectory design shapes the spatial distribution of localisation accuracy.
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
import system_model as sm

GRID_RES = 50  # grid resolution
NPZ_PATH = RESULTS_DIR / "s6_crb_heatmap.npz"


def _concat_paths(path_list):
    if not path_list:
        return np.zeros((0, 3))
    return np.vstack(path_list)


def run() -> None:
    cfg, e_cfg, scfg = get_configs(scenario_name="paper_baseline", mu=5, max_sca_iter=20, step_size=0.6)

    # Run tradeoff simulation to get optimised hover points
    print("\n=== S6: Running tradeoff simulation ===")
    case = run_method_case(
        method_name="tradeoff", eta=DEFAULT_ETA,
        user_xy=DEFAULT_USER_XY, true_target_xyz=DEFAULT_TARGET_XYZ,
        nstg=DEFAULT_NSTG, etot=DEFAULT_ETOT, random_seed=1,
        cfg=cfg, e_cfg=e_cfg, scfg=scfg, localizer="ekf",
    )

    hover_xyz = np.asarray(case["all_hover_xyz"], dtype=float)
    trajectory = _concat_paths(case["all_paths"])
    coarse_hover = np.asarray(case["coarse_hover_xyz"], dtype=float)

    # Build 2-D grid at target height
    x_grid = np.linspace(0, cfg.Lx, GRID_RES)
    y_grid = np.linspace(0, cfg.Ly, GRID_RES)
    crb_map = np.full((GRID_RES, GRID_RES), np.nan)

    print(f"  Computing CRB on {GRID_RES}x{GRID_RES} grid...")
    for i, y in enumerate(y_grid):
        for j, x in enumerate(x_grid):
            target = np.array([x, y, float(DEFAULT_TARGET_XYZ[2])], dtype=float)
            v = sm.crb_xyz_sum(hover_xyz, target, cfg)
            crb_map[i, j] = v if np.isfinite(v) else np.nan

    save_sim_results(
        NPZ_PATH,
        crb_grid=crb_map,
        x_grid=x_grid, y_grid=y_grid,
        hover_xyz=hover_xyz,
        trajectory=trajectory,
        coarse_hover_xyz=coarse_hover,
        user_xy=DEFAULT_USER_XY,
        true_target_xyz=DEFAULT_TARGET_XYZ,
    )
    print("S6 done.")


def plot() -> None:
    ensure_dirs()
    data = load_sim_results(NPZ_PATH)
    crb_map = data["crb_grid"]
    x_grid = data["x_grid"]
    y_grid = data["y_grid"]
    hover_xyz = data["hover_xyz"]
    trajectory = data["trajectory"]
    coarse_hover = data["coarse_hover_xyz"]
    user_xy = data["user_xy"]
    target_xyz = data["true_target_xyz"]

    fig, ax = plt.subplots(figsize=(9, 7))

    # Log-scale for better visualisation
    crb_log = np.log10(np.maximum(crb_map, 1e-30))
    vmin = np.nanmin(crb_log)
    vmax = np.nanmax(crb_log)
    pcm = ax.pcolormesh(x_grid, y_grid, crb_log, cmap="hot_r", shading="auto",
                        vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(pcm, ax=ax, shrink=0.8, label="log10(CRB)")

    # Overlay trajectory and hover points
    if trajectory.size > 0:
        ax.plot(trajectory[:, 0], trajectory[:, 1], "-", color="cyan", lw=1.5, label="UAV trajectory")
    if hover_xyz.size > 0:
        ax.scatter(hover_xyz[:, 0], hover_xyz[:, 1], s=18, marker="^",
                   color="lime", edgecolors="black", linewidths=0.3, label="Hover points")
    if coarse_hover.size > 0:
        ax.plot(coarse_hover[:, 0], coarse_hover[:, 1], "--", lw=1, color="white", alpha=0.6, label="Coarse scan")

    ax.scatter(user_xy[0], user_xy[1], marker="*", s=150, c="white", edgecolors="black", label="User")
    ax.scatter(target_xyz[0], target_xyz[1], marker="x", s=100, c="cyan", linewidths=2, label="Target (true)")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("CRB Spatial Distribution (log10 scale)")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", fontsize=7, facecolor="white")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s6_crb_heatmap.png", dpi=200)
    print("S6 plot saved.")


def main() -> None:
    run()
    plot()
    plt.show()


if __name__ == "__main__":
    main()
