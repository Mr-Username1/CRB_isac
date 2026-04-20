from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _concat_paths(path_list):
    if not path_list:
        return np.zeros((0, 2))
    return np.vstack(path_list)


def main():
    results_file = Path("results/isac_results.npz")
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results_file.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_file}. Run generate_results.py first."
        )

    data = np.load(results_file, allow_pickle=True)
    results = list(data["results"])
    user_xy = data["user_xy"]
    true_target_xy = data["true_target_xy"]

    # ---- Figure 1: Performance comparison ----
    method_names = [r["method_name"] for r in results]
    final_crb = []
    final_rate = []
    for r in results:
        if r["stage_logs"]:
            final_crb.append(float(r["stage_logs"][-1]["crb_final"]))
            final_rate.append(float(r["stage_logs"][-1]["rate_final"]))
        else:
            final_crb.append(np.nan)
            final_rate.append(np.nan)

    x = np.arange(len(method_names))
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()
    w = 0.35
    ax1.bar(x - w / 2, final_crb, width=w, color="tab:red", alpha=0.8, label="Final CRB")
    ax2.bar(x + w / 2, final_rate, width=w, color="tab:blue", alpha=0.7, label="Final Rate")
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_names, rotation=15)
    ax1.set_ylabel("CRB")
    ax2.set_ylabel("Rate (bps)")
    ax1.set_title("Performance Comparison Across Methods")
    ax1.grid(True, axis="y", alpha=0.3)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    fig1.tight_layout()
    fig1.savefig(out_dir / "performance_comparison.png", dpi=200)

    # ---- Figure 2: Convergence (tradeoff first stage) ----
    tradeoff = None
    for r in results:
        if r["method_name"] == "tradeoff":
            tradeoff = r
            break
    if tradeoff is None and results:
        tradeoff = results[0]

    hist0 = None
    if tradeoff is not None and tradeoff["stage_histories"]:
        hist_arr = np.asarray(tradeoff["stage_histories"][0], dtype=object)
        if hist_arr.size > 0:
            hist0 = np.column_stack(
                [
                    hist_arr[:, 0].astype(float),
                    hist_arr[:, 1].astype(float),
                    hist_arr[:, 2].astype(float),
                ]
            )

    fig2, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    if hist0 is not None and hist0.size > 0:
        axes[0].plot(hist0[:, 0], marker="o", ms=3, label="objective")
        axes[0].plot(hist0[:, 1], marker="s", ms=3, label="crb")
        axes[0].plot(hist0[:, 2], marker="^", ms=3, label="rate")
        axes[0].set_title("SCA Convergence (Stage 1)")
        axes[0].set_xlabel("iteration")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "No convergence history", ha="center", va="center")
        axes[0].set_title("SCA Convergence (Stage 1)")

    # ---- Figure 3: Optimized trajectory with true/estimated targets ----
    if tradeoff is not None:
        path_xy = _concat_paths(tradeoff["all_paths"])
        hover_xy = np.asarray(tradeoff["all_hover_xy"], dtype=float)
        coarse_hover_xy = np.asarray(tradeoff["coarse_hover_xy"], dtype=float)
        est_hist = np.asarray(tradeoff["target_hat_history"], dtype=float)
        est_init = np.asarray(tradeoff["target_hat_init_xy"], dtype=float)
        est_final = np.asarray(tradeoff["target_hat_final_xy"], dtype=float)
        if path_xy.size > 0:
            axes[1].plot(path_xy[:, 0], path_xy[:, 1], "-o", ms=2.5, label="UAV trajectory")
        if hover_xy.size > 0:
            axes[1].scatter(hover_xy[:, 0], hover_xy[:, 1], s=25, marker="^", label="hover points")
        if coarse_hover_xy.size > 0:
            axes[1].plot(
                coarse_hover_xy[:, 0],
                coarse_hover_xy[:, 1],
                "--",
                lw=1.2,
                color="tab:gray",
                label="coarse scan path",
            )
        axes[1].scatter(user_xy[0], user_xy[1], marker="*", s=120, c="tab:green", label="user")
        axes[1].scatter(true_target_xy[0], true_target_xy[1], marker="x", s=90, c="tab:red", label="target true")
        axes[1].scatter(est_init[0], est_init[1], marker="o", s=60, c="tab:orange", label="target est init")
        axes[1].scatter(est_final[0], est_final[1], marker="D", s=55, c="tab:purple", label="target est final")
        if est_hist.size > 0:
            axes[1].plot(
                est_hist[:, 0],
                est_hist[:, 1],
                "-.",
                lw=1.5,
                color="tab:purple",
                alpha=0.8,
                label="target est updates",
            )
        axes[1].set_title("Optimized Trajectory (Tradeoff)")
        axes[1].set_xlabel("x (m)")
        axes[1].set_ylabel("y (m)")
        axes[1].set_aspect("equal", adjustable="box")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="best")
    else:
        axes[1].text(0.5, 0.5, "No trajectory data", ha="center", va="center")
        axes[1].set_title("Optimized Trajectory")

    fig2.tight_layout()
    fig2.savefig(out_dir / "convergence_and_trajectory.png", dpi=200)

    print(f"Saved figures to: {out_dir}")
    plt.show()


if __name__ == "__main__":
    main()
