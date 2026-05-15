"""
S7: NSTG (Waypoints-per-Stage) Impact Analysis
===============================================
Compare different NSTG values across multiple random seeds to reveal how
stage granularity affects trajectory geometry, localisation accuracy,
and convergence efficiency with statistical robustness.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np

from simulation.common import (
    DEFAULT_ETA, DEFAULT_ETOT, DEFAULT_TARGET_XYZ, DEFAULT_USER_XY,
    FIGURES_DIR, RESULTS_DIR, ensure_dirs, get_configs, save_sim_results, load_sim_results,
)
from simulation_pipeline import run_method_case

# ---- Extensible parameter lists ----
NSTG_LIST = [15, 20, 25, 30]
SEEDS = [1, 2, 3]

NPZ_PATH = RESULTS_DIR / "s7_nstg_impact.npz"


def _concat_paths(path_list):
    if not path_list:
        return np.zeros((0, 3))
    return np.vstack(path_list)


def _pad_to_len(arr, target_len):
    """Pad 1-D array to target_len with NaN."""
    out = np.full(target_len, np.nan)
    out[:len(arr)] = arr
    return out


def run() -> None:
    cfg, e_cfg, scfg = get_configs(scenario_name="paper_baseline", mu=5, max_sca_iter=50, step_size=0.6)

    # Structure: all_records[nstg_idx][seed_idx] = record_dict
    all_records: list[list[dict]] = []

    for nstg in NSTG_LIST:
        seed_records = []
        for seed in SEEDS:
            print(f"\n=== S7: NSTG={nstg}, seed={seed} ===")
            case = run_method_case(
                method_name=f"nstg_{nstg}_s{seed}", eta=DEFAULT_ETA,
                user_xy=DEFAULT_USER_XY, true_target_xyz=DEFAULT_TARGET_XYZ,
                nstg=nstg, etot=DEFAULT_ETOT, random_seed=seed,
                cfg=cfg, e_cfg=e_cfg, scfg=scfg, localizer="ekf",
            )

            pos_errs = [float(log["position_error_m"]) for log in case["stage_logs"]]
            crbs = [float(log["crb_final"]) for log in case["stage_logs"]]
            rates = [float(log["rate_final"]) for log in case["stage_logs"]]
            iters = [int(log["iters"]) for log in case["stage_logs"]]
            nm_per_stage = [int(log["Nm"]) for log in case["stage_logs"]]
            t_sca = [float(log.get("t_sca_s", np.nan)) for log in case["stage_logs"]]
            t_loc = [float(log.get("t_loc_s", np.nan)) for log in case["stage_logs"]]

            seed_records.append({
                "seed": seed,
                "num_stages": case["num_stages"],
                "final_pos_err": float(case["final_position_error_m"]),
                "trajectory": _concat_paths(case["all_paths"]),
                "hover_xyz": np.asarray(case["all_hover_xyz"], dtype=float),
                "coarse_hover": np.asarray(case["coarse_hover_xyz"], dtype=float),
                "pos_errs": np.array(pos_errs, dtype=float),
                "crbs": np.array(crbs, dtype=float),
                "rates": np.array(rates, dtype=float),
                "iters": np.array(iters, dtype=int),
                "nm_per_stage": np.array(nm_per_stage, dtype=int),
                "t_sca": np.array(t_sca, dtype=float),
                "t_loc": np.array(t_loc, dtype=float),
            })
            final_info = (f"  stages={case['num_stages']}, pos_err={case['final_position_error_m']:.2f}m, "
                          f"CRB={crbs[-1]:.2f}, Rate={rates[-1]:.0f}" if crbs else
                          f"  stages={case['num_stages']} (no stages)")
            print(final_info)
        all_records.append(seed_records)

    # ---- Aggregate across seeds for each NSTG ----
    summary = []
    for nstg, seed_recs in zip(NSTG_LIST, all_records):
        valid = [r for r in seed_recs if r["num_stages"] > 0]

        # Per-seed scalar metrics
        f_pe = np.array([r["final_pos_err"] for r in valid], dtype=float)
        f_crb = np.array([r["crbs"][-1] if len(r["crbs"]) > 0 else np.nan for r in valid], dtype=float)
        f_rate = np.array([r["rates"][-1] if len(r["rates"]) > 0 else np.nan for r in valid], dtype=float)
        n_stages_arr = np.array([r["num_stages"] for r in valid], dtype=int)
        total_iters_arr = np.array([r["iters"].sum() for r in valid], dtype=float)
        total_sca_time_arr = np.array([np.nansum(r["t_sca"]) for r in valid], dtype=float)

        # Per-stage error evolution: pad to max stages and average
        max_s = max((r["num_stages"] for r in valid), default=0)
        pe_matrix = np.array([_pad_to_len(r["pos_errs"], max_s) for r in valid])
        crb_matrix = np.array([_pad_to_len(r["crbs"], max_s) for r in valid])
        rate_matrix = np.array([_pad_to_len(r["rates"], max_s) for r in valid])

        summary.append({
            "nstg": nstg,
            # means
            "final_pos_err_mean": np.nanmean(f_pe), "final_pos_err_std": np.nanstd(f_pe),
            "final_crb_mean": np.nanmean(f_crb), "final_crb_std": np.nanstd(f_crb),
            "final_rate_mean": np.nanmean(f_rate), "final_rate_std": np.nanstd(f_rate),
            "num_stages_mean": np.nanmean(n_stages_arr),
            "total_iters_mean": np.nanmean(total_iters_arr),
            "total_sca_time_mean": np.nanmean(total_sca_time_arr),
            # per-stage evolution (mean only; std for position error)
            "pos_err_evol_mean": np.nanmean(pe_matrix, axis=0),
            "pos_err_evol_std": np.nanstd(pe_matrix, axis=0),
            "crb_evol_mean": np.nanmean(crb_matrix, axis=0),
            "rate_evol_mean": np.nanmean(rate_matrix, axis=0),
            "max_stages": max_s,
            # representative trajectory from seed=1
            "trajectory": seed_recs[0]["trajectory"],
            "hover_xyz": seed_recs[0]["hover_xyz"],
            "coarse_hover": seed_recs[0]["coarse_hover"],
            "nm_per_stage_ref": seed_recs[0]["nm_per_stage"],
        })
        print(f"\nS7 summary NSTG={nstg}: "
              f"pos_err={summary[-1]['final_pos_err_mean']:.1f}±{summary[-1]['final_pos_err_std']:.1f}m, "
              f"CRB={summary[-1]['final_crb_mean']:.1f}±{summary[-1]['final_crb_std']:.1f}, "
              f"Rate={summary[-1]['final_rate_mean']:.0f}±{summary[-1]['final_rate_std']:.0f}")

    save_dict = {
        "nstg_list": np.array(NSTG_LIST, dtype=int),
        "seeds": np.array(SEEDS, dtype=int),
        "summary": np.array(summary, dtype=object),
        "all_records": np.array(all_records, dtype=object),
    }
    save_sim_results(NPZ_PATH, **save_dict)
    print(f"\nS7 done: {len(NSTG_LIST)} NSTG × {len(SEEDS)} seeds = {len(NSTG_LIST)*len(SEEDS)} runs.")


def plot() -> None:
    ensure_dirs()
    data = load_sim_results(NPZ_PATH)
    summary = list(data["summary"])
    nstg_list = [s["nstg"] for s in summary]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(summary)))

    # ---- Figure 1: Trajectory comparison (seed=1 representative) ----
    fig1, ax = plt.subplots(figsize=(10, 8))

    for s, nstg, c in zip(summary, nstg_list, colors):
        traj = s["trajectory"]
        hover = s["hover_xyz"]
        if traj.size > 0:
            ax.plot(traj[:, 0], traj[:, 1], "-", color=c, lw=1.8, alpha=0.85,
                    label=f"NSTG={nstg} ({s['max_stages']} stages)")
        if hover.size > 0:
            ax.scatter(hover[:, 0], hover[:, 1], s=18, marker="^",
                       color=c, edgecolors="black", linewidths=0.3, alpha=0.9)

    s0 = summary[0]
    if s0["coarse_hover"].size > 0:
        ax.plot(s0["coarse_hover"][:, 0], s0["coarse_hover"][:, 1], "--",
                lw=1.2, color="gray", alpha=0.7, label="coarse scan")

    ax.scatter(DEFAULT_USER_XY[0], DEFAULT_USER_XY[1], marker="*", s=200,
               c="green", edgecolors="black", linewidths=0.5, zorder=5, label="User")
    ax.scatter(DEFAULT_TARGET_XYZ[0], DEFAULT_TARGET_XYZ[1], marker="x", s=120,
               c="red", linewidths=2.5, zorder=5, label="Target")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"UAV Trajectories for Different NSTG "
                 f"($E_{{\\text{{tot}}}}$={DEFAULT_ETOT/1e3:.0f} kJ, $\\eta$={DEFAULT_ETA}, "
                 f"{len(SEEDS)} seeds avg)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    fig1.tight_layout()
    fig1.savefig(FIGURES_DIR / "s7_trajectories.png", dpi=200)

    # ---- Figure 2: Position error evolution (mean ± std) ----
    fig2, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    for s, nstg, c in zip(summary, nstg_list, colors):
        max_s = s["max_stages"]
        if max_s == 0:
            continue
        stages = np.arange(1, max_s + 1)
        mu = s["pos_err_evol_mean"]
        sd = s["pos_err_evol_std"]
        ax.plot(stages, mu, "-o", color=c, lw=2, ms=5, label=f"NSTG={nstg}")
        ax.fill_between(stages, mu - sd, mu + sd, alpha=0.15, color=c)
    ax.set_xlabel("Stage")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Error vs Stage (mean ± std)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for s, nstg, c in zip(summary, nstg_list, colors):
        if s["max_stages"] == 0:
            continue
        cum_wp = np.cumsum(s["nm_per_stage_ref"])
        mu = s["pos_err_evol_mean"]
        sd = s["pos_err_evol_std"]
        ax.plot(cum_wp, mu, "-s", color=c, lw=2, ms=5, label=f"NSTG={nstg}")
        ax.fill_between(cum_wp, mu - sd, mu + sd, alpha=0.15, color=c)
    ax.set_xlabel("Cumulative Waypoints")
    ax.set_ylabel("Position Error (m)")
    ax.set_title("Error vs Cumulative Waypoints (mean ± std)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig2.suptitle(f"Localisation Accuracy Across NSTG Values ({len(SEEDS)} seeds averaged)")
    fig2.tight_layout()
    fig2.savefig(FIGURES_DIR / "s7_pos_error.png", dpi=200)

    # ---- Figure 3: Performance summary bars with error bars ----
    fig3, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = np.arange(len(summary))
    labels = [f"NSTG={n}" for n in nstg_list]
    w = 0.55

    metrics = [
        (axes[0, 0], "Final CRB",
         [s["final_crb_mean"] for s in summary],
         [s["final_crb_std"] for s in summary], "tab:red"),
        (axes[0, 1], "Final Rate (bps)",
         [s["final_rate_mean"] for s in summary],
         [s["final_rate_std"] for s in summary], "tab:blue"),
        (axes[1, 0], "Final Pos Error (m)",
         [s["final_pos_err_mean"] for s in summary],
         [s["final_pos_err_std"] for s in summary], "tab:orange"),
        (axes[1, 1], "Num Stages",
         [s["num_stages_mean"] for s in summary],
         None, "tab:green"),
    ]
    for ax, title, vals, errs, color in metrics:
        err = errs if errs is not None else [0] * len(vals)
        bars = ax.bar(x, vals, w, color=color, alpha=0.85,
                      yerr=err, capsize=5, error_kw={"elinewidth": 1.2})
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)
        for bar, val, std in zip(bars, vals, err):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (std or 0),
                        f"{val:.1f}" if abs(val) >= 1 else f"{val:.2e}",
                        ha="center", va="bottom", fontsize=8)

    fig3.suptitle(f"Performance Summary vs NSTG "
                  f"($E_{{\\text{{tot}}}}$={DEFAULT_ETOT/1e3:.0f} kJ, {len(SEEDS)} seeds)")
    fig3.tight_layout()
    fig3.savefig(FIGURES_DIR / "s7_performance.png", dpi=200)

    # ---- Figure 4: Computational efficiency ----
    fig4, axes = plt.subplots(3, 1, figsize=(11, 9))

    ax = axes[0]
    ax.bar(labels, [s["total_iters_mean"] for s in summary], color="tab:purple", alpha=0.8)
    ax.set_ylabel("Total SCA Iters")
    ax.set_title("Total SCA Iterations (mean)")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[1]
    avg_iters = [s["total_iters_mean"] / max(s["max_stages"], 1) for s in summary]
    ax.bar(labels, avg_iters, color="tab:cyan", alpha=0.8)
    ax.set_ylabel("Avg Iters / Stage")
    ax.set_title("Average SCA Iterations per Stage")
    ax.grid(True, axis="y", alpha=0.3)

    ax = axes[2]
    ax.bar(labels, [s["total_sca_time_mean"] for s in summary], color="tab:pink", alpha=0.8)
    ax.set_ylabel("Total SCA Time (s)")
    ax.set_title("Total SCA Solver Time (mean)")
    ax.grid(True, axis="y", alpha=0.3)

    fig4.suptitle(f"Computational Efficiency vs NSTG ({len(SEEDS)} seeds averaged)")
    fig4.tight_layout()
    fig4.savefig(FIGURES_DIR / "s7_efficiency.png", dpi=200)

    print("S7 plots saved.")


def main() -> None:
    run()
    plot()
    plt.show()


if __name__ == "__main__":
    main()
