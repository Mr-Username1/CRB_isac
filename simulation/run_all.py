"""
run_all.py — one-click batch execution of all 6 simulation experiments.
Each module can also be run independently via `python simulation/sN_*.py`.
"""
from __future__ import annotations

import time
from pathlib import Path
import sys

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simulation.common import ensure_dirs


def main() -> None:
    ensure_dirs()
    modules = [
        ("S1: SCA Convergence",         "simulation.s1_sca_convergence"),
        ("S2: Trade-off Analysis",      "simulation.s2_tradeoff"),
        ("S3: 2D vs 3D Trajectory",     "simulation.s3_trajectory_2d3d"),
        ("S4: MLE vs EKF",              "simulation.s4_mle_vs_ekf"),
        ("S5: Energy Impact",           "simulation.s5_energy_impact"),
        ("S6: CRB Heatmap",             "simulation.s6_crb_heatmap"),
    ]

    t0 = time.perf_counter()
    for name, mod_path in modules:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        t_start = time.perf_counter()
        try:
            mod = __import__(mod_path, fromlist=["main"])
            mod.main()
            elapsed = time.perf_counter() - t_start
            print(f"  {name} — OK ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.perf_counter() - t_start
            print(f"  {name} — FAILED ({elapsed:.1f}s): {e}")
    total = time.perf_counter() - t0
    print(f"\n{'='*60}")
    print(f"  All simulations finished in {total:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
