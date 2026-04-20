from __future__ import annotations

import system_model as sm
import problem as pb
from p2_solver import SolverCfg


def build_sim_config(scenario_name: str = "paper_baseline", mu: int = 5) -> sm.SimConfig:
    """Build and finalize simulation config."""
    return sm.finalize_config(sm.SimConfig(mu=mu, scenario_name=scenario_name))


def build_energy_config() -> pb.EnergyConfig:
    """Build default propulsion-energy config."""
    return pb.EnergyConfig()


def build_solver_config(max_sca_iter: int = 20, step_size: float = 0.6) -> SolverCfg:
    """Build default SCA solver config."""
    return SolverCfg(max_sca_iter=max_sca_iter, step_size=step_size)


def build_default_configs(
    scenario_name: str = "paper_baseline",
    mu: int = 5,
    max_sca_iter: int = 20,
    step_size: float = 0.6,
) -> tuple[sm.SimConfig, pb.EnergyConfig, SolverCfg]:
    """Build cfg/e_cfg/scfg for experiment entry scripts."""
    cfg = build_sim_config(scenario_name=scenario_name, mu=mu)
    e_cfg = build_energy_config()
    scfg = build_solver_config(max_sca_iter=max_sca_iter, step_size=step_size)
    return cfg, e_cfg, scfg
