import numpy as np

from config_factory import build_default_configs
from simulation_pipeline import run_method_case


def main() -> None:
    """Lightweight single-case demo entry."""
    cfg, e_cfg, scfg = build_default_configs(
        scenario_name="paper_baseline",
        mu=5,
        max_sca_iter=20,
        step_size=0.6,
    )

    user_xy = np.array([300.0, 400.0])
    true_target_xy = np.array([1350.0, 1150.0])
    case = run_method_case(
        method_name="tradeoff",
        eta=0.7,
        user_xy=user_xy,
        true_target_xy=true_target_xy,
        nstg=25,
        etot=35e3,
        random_seed=1,
        cfg=cfg,
        e_cfg=e_cfg,
        scfg=scfg,
    )
    print("method:", case["method_name"])
    print("num_stages:", case["num_stages"])
    print("final_energy_left:", case["energy_left"])
    print("scan_energy_used:", case["scan_energy_used"])
    print("target_hat_init_xy:", case["target_hat_init_xy"])
    print("target_hat_final_xy:", case["target_hat_final_xy"])
    if case["stage_logs"]:
        print("last_stage:", case["stage_logs"][-1])


if __name__ == "__main__":
    main()

