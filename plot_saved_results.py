from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _concat_paths(path_list):
    if not path_list:
        return np.zeros((0, 2))
    return np.vstack(path_list)


def load_results(results_file: Path):
    if not results_file.exists():
        raise FileNotFoundError(
            f"Results file not found: {results_file}. Run generate_results.py first."
        )

    data = np.load(results_file, allow_pickle=True)
    return list(data["results"]), data["user_xy"], data["true_target_xy"]


def get_tradeoff_result(results):
    for result in results:
        if result["method_name"] == "tradeoff":
            return result
    return results[0] if results else None


def annotate_bars(ax, bars, fmt="{:.3f}"):
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2.0, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_performance_comparison(results, out_dir: Path):
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
    bars_crb = ax1.bar(
        x - w / 2, final_crb, width=w, color="tab:red", alpha=0.8, label="Final CRB"
    )
    bars_rate = ax2.bar(
        x + w / 2, final_rate, width=w, color="tab:blue", alpha=0.7, label="Final Rate"
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(method_names, rotation=15)
    ax1.set_ylabel("CRB")
    ax2.set_ylabel("Rate (bps)")
    ax1.set_title("Performance Comparison Across Methods")
    ax1.grid(True, axis="y", alpha=0.3)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")
    annotate_bars(ax1, bars_crb, fmt="{:.3e}")
    annotate_bars(ax2, bars_rate, fmt="{:.2f}")
    fig1.tight_layout()
    fig1.savefig(out_dir / "performance_comparison.png", dpi=200)


def extract_all_stage_histories(tradeoff):
    histories = []
    if tradeoff is None:
        return histories
    for stage_hist in tradeoff.get("stage_histories", []):
        hist_arr = np.asarray(stage_hist, dtype=object)
        if hist_arr.size == 0:
            histories.append(np.zeros((0, 3)))
            continue
        histories.append(
            np.column_stack(
                [
                    hist_arr[:, 0].astype(float),  # objective
                    hist_arr[:, 1].astype(float),  # crb
                    hist_arr[:, 2].astype(float),  # rate
                ]
            )
        )
    return histories


def plot_stagewise_convergence(stage_histories, out_dir: Path):
    if not stage_histories:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.text(0.5, 0.5, "No stage convergence history", ha="center", va="center")
        ax.set_title("Stage-wise Convergence")
        fig.tight_layout()
        fig.savefig(out_dir / "stagewise_convergence.png", dpi=200)
        return

    n_stage = len(stage_histories)
    fig, axes = plt.subplots(
        n_stage,
        3,
        figsize=(15.0, max(3.2 * n_stage, 4.5)),
        squeeze=False,
        sharex=False,
    )
    metric_specs = [
        ("Objective", 0, "tab:purple", "o"),
        ("Rate", 2, "tab:blue", "^"),
        ("CRB", 1, "tab:red", "s"),
    ]

    for s_idx, history in enumerate(stage_histories):
        for m_idx, (metric_name, col, color, marker) in enumerate(metric_specs):
            ax = axes[s_idx, m_idx]
            if history.size > 0:
                it = np.arange(1, history.shape[0] + 1)
                y = history[:, col]
                ax.plot(it, y, marker=marker, ms=2.8, lw=1.2, color=color)
                if metric_name == "CRB":
                    positive = y[y > 0]
                    if positive.size > 1 and positive.max() / positive.min() >= 100:
                        ax.set_yscale("log")
                ax.set_xlabel("Iteration")
                ax.set_ylabel(metric_name)
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, "Empty history", ha="center", va="center")
                ax.set_ylabel(metric_name)
                ax.set_xlabel("Iteration")
            ax.set_title(f"Stage {s_idx + 1} - {metric_name}")

    fig.suptitle("Stage-wise Convergence", y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "stagewise_convergence.png", dpi=200, bbox_inches="tight")


def plot_global_evolution_by_waypoints(tradeoff, out_dir: Path):
    if tradeoff is None or not tradeoff.get("stage_logs"):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.text(0.5, 0.5, "No stage logs", ha="center", va="center")
        ax.set_title("Global Evolution by Cumulative Waypoints")
        fig.tight_layout()
        fig.savefig(out_dir / "global_evolution_by_waypoints.png", dpi=200)
        return

    stage_logs = tradeoff["stage_logs"]
    stage_nm = np.array([int(log["Nm"]) for log in stage_logs], dtype=int)
    x_cum_wp = np.cumsum(stage_nm)
    y_obj = np.array([float(log["obj_final"]) for log in stage_logs], dtype=float)
    y_rate = np.array([float(log["rate_final"]) for log in stage_logs], dtype=float)
    y_crb = np.array([float(log["crb_final"]) for log in stage_logs], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8.4), sharex=True)
    series = [
        ("Objective", y_obj, "tab:purple", "o"),
        ("Rate", y_rate, "tab:blue", "^"),
        ("CRB", y_crb, "tab:red", "s"),
    ]

    for ax, (name, y, color, marker) in zip(axes, series):
        ax.plot(x_cum_wp, y, color=color, marker=marker, lw=1.6, ms=4)
        for boundary in x_cum_wp[:-1]:
            ax.axvline(boundary, color="tab:gray", ls="--", lw=1.0, alpha=0.6)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

    axes[0].set_title("Global Evolution Across Stages")
    axes[-1].set_xlabel("Cumulative waypoint count")
    fig.tight_layout()
    fig.savefig(out_dir / "global_evolution_by_waypoints.png", dpi=200)


def plot_trajectory(tradeoff, user_xy, true_target_xy, out_dir: Path):
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    if tradeoff is None:
        ax.text(0.5, 0.5, "No trajectory data", ha="center", va="center")
        ax.set_title("Optimized Trajectory")
        fig.tight_layout()
        fig.savefig(out_dir / "trajectory_tradeoff.png", dpi=200)
        return

    path_xy = _concat_paths(tradeoff["all_paths"])
    hover_xy = np.asarray(tradeoff["all_hover_xy"], dtype=float)
    coarse_hover_xy = np.asarray(tradeoff["coarse_hover_xy"], dtype=float)
    est_hist = np.asarray(tradeoff["target_hat_history"], dtype=float)
    est_init = np.asarray(tradeoff["target_hat_init_xy"], dtype=float)
    est_final = np.asarray(tradeoff["target_hat_final_xy"], dtype=float)
    if path_xy.size > 0:
        ax.plot(path_xy[:, 0], path_xy[:, 1], "-o", ms=2.5, label="UAV trajectory")
    if hover_xy.size > 0:
        ax.scatter(hover_xy[:, 0], hover_xy[:, 1], s=25, marker="^", color="red", label="hover points")
    if coarse_hover_xy.size > 0:
        ax.plot(
            coarse_hover_xy[:, 0],
            coarse_hover_xy[:, 1],
            "--",
            lw=1.2,
            color="tab:gray",
            label="coarse scan path",
        )
    ax.scatter(user_xy[0], user_xy[1], marker="*", s=120, c="tab:green", label="user")
    ax.scatter(
        true_target_xy[0], true_target_xy[1], marker="x", s=90, c="tab:red", label="target true"
    )
    ax.scatter(est_init[0], est_init[1], marker="o", s=60, c="tab:orange", label="target est init")
    ax.scatter(
        est_final[0], est_final[1], marker="D", s=55, c="tab:purple", label="target est final"
    )
    if est_hist.size > 0:
        ax.plot(
            est_hist[:, 0],
            est_hist[:, 1],
            "-.",
            lw=1.5,
            color="tab:purple",
            alpha=0.8,
            label="target est updates",
        )
    # Add UAV initial position
    ax.scatter(100.0, 100.0, marker="s", s=60, c="tab:cyan", label="UAV start")
    ax.set_title("Optimized Trajectory (Tradeoff)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_tradeoff.png", dpi=200)


def plot_all_trajectories(results, user_xy, true_target_xy, out_dir: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'communication_only': 'blue', 'tradeoff': 'green', 'sensing_only': 'red'}
    markers = {'communication_only': 'o', 'tradeoff': 's', 'sensing_only': '^'}

    for result in results:
        method = result['method_name']
        if method not in colors:
            continue
        path_xy = _concat_paths(result.get("all_paths", []))
        hover_xy = np.asarray(result.get("all_hover_xy", []), dtype=float)
        coarse_hover_xy = np.asarray(result.get("coarse_hover_xy", []), dtype=float)
        if path_xy.size > 0:
            ax.plot(path_xy[:, 0], path_xy[:, 1], "-", marker=markers[method], ms=3, color=colors[method], label=f"{method} trajectory")
        if hover_xy.size > 0:
            ax.scatter(hover_xy[:, 0], hover_xy[:, 1], s=30, marker="^", color=colors[method], alpha=0.7, label=f"{method} hover")
        if coarse_hover_xy.size > 0:
            ax.plot(coarse_hover_xy[:, 0], coarse_hover_xy[:, 1], "--", lw=1.2, color=colors[method], alpha=0.5, label=f"{method} coarse")

    # Common elements
    ax.scatter(user_xy[0], user_xy[1], marker="*", s=120, c="tab:green", label="user")
    ax.scatter(true_target_xy[0], true_target_xy[1], marker="x", s=90, c="tab:red", label="target true")
    ax.scatter(100.0, 100.0, marker="s", s=60, c="tab:cyan", label="UAV start")
    ax.set_title("Trajectories for All Methods")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "all_trajectories.png", dpi=200)


def main():
    results_file = Path("results/isac_results.npz")
    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    results, user_xy, true_target_xy = load_results(results_file)
    tradeoff = get_tradeoff_result(results)
    stage_histories = extract_all_stage_histories(tradeoff)

    plot_performance_comparison(results, out_dir)
    plot_stagewise_convergence(stage_histories, out_dir)
    plot_global_evolution_by_waypoints(tradeoff, out_dir)
    plot_trajectory(tradeoff, user_xy, true_target_xy, out_dir)
    plot_all_trajectories(results, user_xy, true_target_xy, out_dir)
    print(f"Saved figures to: {out_dir}")
    plt.show()


if __name__ == "__main__":
    main()
