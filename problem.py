from dataclasses import dataclass
import numpy as np
import system_model as sm


@dataclass
class EnergyConfig:
    Etot: float = 35e3
    P0: float = 80.0
    PI: float = 88.6
    Utip: float = 120.0
    v0: float = 4.03
    D0: float = 0.6
    rho: float = 1.225
    s: float = 0.05
    A: float = 0.503


def propulsion_power(v_norm: np.ndarray, e_cfg: EnergyConfig) -> np.ndarray:
    P0, PI = e_cfg.P0, e_cfg.PI
    Utip, v0 = e_cfg.Utip, e_cfg.v0
    D0, rho, s, A = e_cfg.D0, e_cfg.rho, e_cfg.s, e_cfg.A

    term1 = P0 * (1.0 + 3.0 * (v_norm ** 2) / (Utip ** 2))
    term2 = PI * np.sqrt(np.sqrt(1.0 + (v_norm ** 4) / (4.0 * v0 ** 4)) - (v_norm ** 2) / (2.0 * v0 ** 2))
    term3 = 0.5 * D0 * rho * s * A * (v_norm ** 3)
    return term1 + term2 + term3


def evaluate_p1(
    S: np.ndarray,
    user_xy: np.ndarray,
    target_hat_xy: np.ndarray,
    eta: float,
    cfg: sm.SimConfig,      # 通信+感知参数
    e_cfg: EnergyConfig     # 能耗参数
) -> dict:
    """
    对应 Section III-A 的 P1 数值评估（不是优化器）
    """
    # ---- 轨迹/速度 ----
    start_xy = np.array([cfg.xB, cfg.yB], dtype=float)
    V = sm.compute_velocities(S, start_xy, cfg.Tf)
    v_norm = np.linalg.norm(V, axis=1)

    # ---- 通信项 R_bar ----
    dc = sm.dc_uav_to_user(S, user_xy, cfg.H)
    h = sm.channel_gain_comm(dc, cfg.alpha0)
    Rn = sm.rate_per_waypoint(h, cfg.P_w, cfg.sigma0_sq, cfg.B)
    R_bar = sm.average_rate(Rn)

    # ---- 感知项 CRB ----
    Hov = sm.extract_hover_points(S, cfg.mu)
    CRB = sm.crb_xy_sum(Hov, target_hat_xy, cfg)

    # ---- 目标函数 ----
    obj = eta * CRB - (1.0 - eta) * R_bar

    # ---- 约束 (30a)(30b)(30c) ----
    speed_violation = np.maximum(v_norm - cfg.Vmax, 0.0).sum()

    x = S[:, 0]
    y = S[:, 1]
    area_violation = (
        np.maximum(-x, 0.0).sum()
        + np.maximum(x - cfg.Lx, 0.0).sum()
        + np.maximum(-y, 0.0).sum()
        + np.maximum(y - cfg.Ly, 0.0).sum()
    )

    P_fly = propulsion_power(v_norm, e_cfg)              # sum over n
    P_hover0 = propulsion_power(np.array([0.0]), e_cfg)[0]  # P(0)
    Ktot = Hov.shape[0]
    E_used = cfg.Tf * P_fly.sum() + cfg.Th * Ktot * P_hover0
    energy_violation = max(E_used - e_cfg.Etot, 0.0)

    feasible = (speed_violation == 0.0) and (area_violation == 0.0) and (energy_violation == 0.0)

    return {
        "objective": float(obj),
        "CRB_xy_sum": float(CRB),
        "R_bar": float(R_bar),
        "E_used": float(E_used),
        "Ktot": int(Ktot),
        "constraints": {
            "speed_violation": float(speed_violation),
            "area_violation": float(area_violation),
            "energy_violation": float(energy_violation),
            "feasible": bool(feasible),
        }
    }


def stage_energy_used(
    S: np.ndarray,
    start_xy: np.ndarray,
    cfg: sm.SimConfig,
    e_cfg: EnergyConfig,
) -> tuple[float, int]:
    """
    Compute propulsion energy for one stage trajectory.
    Returns:
      - E_used: total stage energy
      - K: number of hover points in the stage
    """
    start_xy = np.asarray(start_xy, dtype=float).reshape(2,)
    V = sm.compute_velocities(S, start_xy, cfg.Tf)
    v_norm = np.linalg.norm(V, axis=1)
    P_fly = propulsion_power(v_norm, e_cfg)
    K = int(sm.extract_hover_points(S, cfg.mu).shape[0])
    P_hover0 = propulsion_power(np.array([0.0]), e_cfg)[0]
    E_used = cfg.Tf * P_fly.sum() + cfg.Th * K * P_hover0
    return float(E_used), K