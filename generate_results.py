from pathlib import Path
import numpy as np

from config_factory import build_default_configs
from simulation_pipeline import run_method_case
from io_utils import save_results_bundle


def main():
    scenario_name = "high_noise_realistic"  # options: paper_baseline / high_noise_realistic / extreme_noise
    cfg, e_cfg, scfg = build_default_configs(
        scenario_name=scenario_name,
        mu=5,
        max_sca_iter=20,
        step_size=0.6,
    )
    print(f"Active scenario: {cfg.scenario_name}")

    output_dir = Path("results")
    npz_path = output_dir / "isac_results.npz"
    meta_path = output_dir / "isac_results_meta.json"

    user_xy = np.array([300.0, 400.0], dtype = float)
    true_target_xy = np.array([1350.0, 1150.0], dtype = float)
    nstg = 25
    etot = 35e3
    seed = 1

    methods = [
        ("communication_only", 0.0),
        ("tradeoff", 0.8),
        ("sensing_only", 1.0),
    ]

    results = []
    for method_name, eta in methods:
        print(f"\n=== Running {method_name} (eta={eta}) ===")
        case = run_method_case(
            method_name=method_name,
            eta=eta,
            user_xy=user_xy,
            true_target_xy=true_target_xy,
            nstg=nstg,
            etot=etot,
            random_seed=seed,
            cfg=cfg,
            e_cfg=e_cfg,
            scfg=scfg,
        )
        results.append(case)
        print(
            f"{method_name}: stages={case['num_stages']}, "
            f"E_left={case['energy_left']:.2f}, "
            f"target_hat_final={case['target_hat_final_xy']}"
        )

    save_results_bundle(
        output_npz_path=npz_path,
        meta_json_path=meta_path,
        results=results,
        user_xy=user_xy,
        true_target_xy=true_target_xy,
        cfg=cfg,
        save_full_trajectories=True,
    )
    print(f"\nSaved results to: {npz_path}")
    print(f"Saved metadata to: {meta_path}")


if __name__ == "__main__":
    main()
