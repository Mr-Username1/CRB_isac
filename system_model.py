from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# =========================
# Config
# =========================

@dataclass
class SimConfig:
    # Geometry
    Lx: float = 2000.0
    Ly: float = 2000.0
    xB: float = 0.0
    yB: float = 0.0
    H: float = 150.0          # nominal UAV altitude (initial path, coarse scan)
    z_min: float = 50.0       # min UAV altitude
    z_max: float = 200.0      # max UAV altitude
    z_t_min: float = 0.0      # min target height (ground)
    z_t_max: float = 25.0    # max target height

    # Time / trajectory
    Tf: float = 1.5
    Th: float = 1.0
    mu: int = 3
    Vmax: float = 30.0
    # Paper Table II / Eq.(60): straight-line initial guess, step i along start -> midpoint
    Vstr: float = 20.0

    # Communication (raw)
    B: float = 1e6
    P_dbm: float = 20.0
    N0_dbm_per_hz: float = -170.0
    alpha0_db: float = -50.0

    # Sensing (raw)
    beta0_db: float = -47.0
    a: float = 10.0  # measurement-noise scaling factor
    c: float = 3e8   # speed of light
    gp_scale: float = 0.1  # Gp = gp_scale * B

    # Scenario control
    scenario_name: str = "paper_baseline"
    nlos_bias_mean: float = 0.0
    nlos_bias_std: float = 0.0
    outlier_prob: float = 0.0
    outlier_std: float = 0.0
    model_mismatch_h: float = 0.0
    model_mismatch_beta0_db: float = 0.0

    # Derived (filled by finalize_config)
    P_w: float | None = None
    N0_w_per_hz: float | None = None
    sigma0_sq: float | None = None
    alpha0: float | None = None
    beta0: float | None = None
    Gp: float | None = None


def db_to_linear(db: float) -> float:
    return 10.0 ** (db / 10.0)


def dbm_to_watt(dbm: float) -> float:
    # 0 dBm = 1 mW = 1e-3 W
    return 1e-3 * (10.0 ** (dbm / 10.0))


def finalize_config(cfg: SimConfig) -> SimConfig:
    cfg = apply_scenario_preset(cfg)
    cfg.P_w = dbm_to_watt(cfg.P_dbm)
    cfg.N0_w_per_hz = dbm_to_watt(cfg.N0_dbm_per_hz)
    cfg.sigma0_sq = cfg.N0_w_per_hz * cfg.B
    cfg.alpha0 = db_to_linear(cfg.alpha0_db)
    cfg.beta0 = db_to_linear(cfg.beta0_db)
    cfg.Gp = cfg.gp_scale * cfg.B
    return cfg


def apply_scenario_preset(cfg: SimConfig) -> SimConfig:
    """
    Fill scenario-dependent parameters.
    - paper_baseline: close to the paper setup (low-noise ideal model)
    - high_noise_realistic: higher noise + mild model mismatch + outliers
    - extreme_noise: stress-test setting
    """
    name = cfg.scenario_name.lower()
    if name == "paper_baseline":
        cfg.a = 10.0
        cfg.gp_scale = 0.1
        cfg.nlos_bias_mean = 0.0
        cfg.nlos_bias_std = 0.0
        cfg.outlier_prob = 0.0
        cfg.outlier_std = 0.0
        cfg.model_mismatch_h = 0.0
        cfg.model_mismatch_beta0_db = 0.0
    elif name == "high_noise_realistic":
        cfg.a = 100.0
        cfg.gp_scale = 0.04
        cfg.nlos_bias_mean = 12.0
        cfg.nlos_bias_std = 4.0
        cfg.outlier_prob = 0.08
        cfg.outlier_std = 50.0
        cfg.model_mismatch_h = 10.0
        cfg.model_mismatch_beta0_db = 2.0
    elif name == "extreme_noise":
        cfg.a = 220.0
        cfg.gp_scale = 0.02
        cfg.nlos_bias_mean = 20.0
        cfg.nlos_bias_std = 8.0
        cfg.outlier_prob = 0.15
        cfg.outlier_std = 90.0
        cfg.model_mismatch_h = 20.0
        cfg.model_mismatch_beta0_db = 4.0
    else:
        raise ValueError(f"Unknown scenario_name: {cfg.scenario_name}")
    return cfg


# =========================
# II-A UAV trajectory model
# =========================

def compute_velocities(S: np.ndarray, start_xyz: np.ndarray, Tf: float) -> np.ndarray:
    """
    Eq. (1)
    S: (N, D), start_xyz: (D,)
    return V: (N, D)
    """
    S = np.asarray(S, dtype=float)
    start_xyz = np.asarray(start_xyz, dtype=float).reshape(S.shape[1])
    N = S.shape[0]
    V = np.zeros_like(S, dtype=float)
    V[0] = (S[0] - start_xyz) / Tf
    if N > 1:
        V[1:] = (S[1:] - S[:-1]) / Tf
    return V


def extract_hover_points(S: np.ndarray, mu: int) -> np.ndarray:
    """
    Eq. (2): hover at indices mu, 2mu, ... (1-based)
    Python 0-based -> mu-1, 2mu-1, ...
    return Hov: (K, D)
    """
    S = np.asarray(S, dtype=float)
    idx = np.arange(mu - 1, S.shape[0], mu, dtype=int)
    return S[idx]


# =========================
# II-B communication model
# =========================

def dc_uav_to_user(S: np.ndarray, user_xy: np.ndarray) -> np.ndarray:
    """
    Eq. (3): 3-D slant distance from each UAV waypoint to ground user.
    S: (N, 3)  — (x, y, z)
    user_xy: (2,) — ground user (x_u, y_u)
    return dc: (N,)
    """
    S = np.asarray(S, dtype=float)
    user_xy = np.asarray(user_xy, dtype=float).reshape(2,)
    diff_xy = S[:, :2] - user_xy
    return np.sqrt(S[:, 2]**2 + np.sum(diff_xy**2, axis=1))


def channel_gain_comm(dc: np.ndarray, alpha0: float) -> np.ndarray:
    """
    Eq. (4): h(n) = alpha0 / dc(n)^2
    """
    dc = np.asarray(dc, dtype=float)
    return alpha0 / (dc**2)


def rate_per_waypoint(h: np.ndarray, P_w: float, sigma0_sq: float, B: float) -> np.ndarray:
    """
    Eq. (6)
    """
    h = np.asarray(h, dtype=float)
    snr = (P_w * h) / sigma0_sq
    return B * np.log2(1.0 + snr)


def average_rate(Rn: np.ndarray) -> float:
    """
    Eq. (7)
    """
    return float(np.mean(Rn))


# =========================
# II-C sensing / CRB model
# =========================

def ds_uav_to_target(Hov: np.ndarray, target_xyz: np.ndarray) -> np.ndarray:
    """
    Eq. (8): 3-D slant distance from each hover point to 3-D target.
    Hov: (K, 3)        — hover point (x, y, z)
    target_xyz: (3,)    — target (x_t, y_t, z_t)
    return ds: (K,)
    """
    Hov = np.asarray(Hov, dtype=float)
    target_xyz = np.asarray(target_xyz, dtype=float).reshape(3,)
    diff = Hov - target_xyz
    return np.sqrt(np.sum(diff**2, axis=1))


def channel_gain_sensing(ds: np.ndarray, beta0: float) -> np.ndarray:
    """
    Eq. (11): g(k) = beta0 / ds(k)^4
    """
    ds = np.asarray(ds, dtype=float)
    return beta0 / (ds**4)


def sensing_snr(g: np.ndarray, P_w: float, Gp: float, sigma0_sq: float) -> np.ndarray:
    """
    Eq. (13)
    """
    g = np.asarray(g, dtype=float)
    return (P_w * Gp * g) / sigma0_sq


def sigma2_measurement_from_g(g: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """
    Eq. (14): sigma^2(k) = a*sigma0_sq / (P_w * Gp * g(k))
    """
    g = np.asarray(g, dtype=float)
    return (cfg.a * cfg.sigma0_sq) / (cfg.P_w * cfg.Gp * g)


def crb_xyz_sum(Hov: np.ndarray, target_xyz: np.ndarray, cfg: SimConfig) -> float:
    """
    CRB trace for 3-D target [x_t, y_t, z_t] from range-only measurements.

    FIM (3x3):  F = Σ_k w_k * M_k
      w_k = c1 / d_k^6 + 8 / d_k^4,   c1 = P_w * G_p * beta0 / (a * sigma0^2)
      M_k = [[dx^2,  dx*dy, dx*dz],
             [dx*dy, dy^2,  dy*dz],
             [dx*dz, dy*dz, dz^2 ]]
    CRB = trace(F^{-1}), regularised with 1e-6*I.
    """
    Hov = np.asarray(Hov, dtype=float)
    target_xyz = np.asarray(target_xyz, dtype=float).reshape(3,)

    dx = Hov[:, 0] - target_xyz[0]
    dy = Hov[:, 1] - target_xyz[1]
    dz = Hov[:, 2] - target_xyz[2]
    ds2 = dx*dx + dy*dy + dz*dz
    ds = np.sqrt(ds2)
    ds = np.maximum(ds, 1e-6)

    c1 = (cfg.P_w * cfg.Gp * cfg.beta0) / (cfg.a * cfg.sigma0_sq)
    w = c1 / ds**6 + 8.0 / ds**4

    F = np.zeros((3, 3), dtype=float)
    F[0, 0] = np.sum(w * dx * dx)
    F[1, 1] = np.sum(w * dy * dy)
    F[2, 2] = np.sum(w * dz * dz)
    F[0, 1] = F[1, 0] = np.sum(w * dx * dy)
    F[0, 2] = F[2, 0] = np.sum(w * dx * dz)
    F[1, 2] = F[2, 1] = np.sum(w * dy * dz)

    F += np.eye(3, dtype=float) * 1e-6

    try:
        return float(np.trace(np.linalg.inv(F)))
    except np.linalg.LinAlgError:
        return float("inf")


# =========================
# Optional: one-shot evaluator
# =========================

def evaluate_stage_metrics(
    S: np.ndarray,
    user_xy: np.ndarray,
    target_hat_xyz: np.ndarray,
    cfg: SimConfig,
    start_xyz: np.ndarray | None = None,
) -> dict:
    """
    返回当前轨迹的通信+感知指标（Section II）

    start_xyz: 轨迹起点用于速度计算；默认与仿真一致，为 (xB, yB, 0)。
    """
    if start_xyz is None:
        start_xyz = np.array([cfg.xB, cfg.yB, 0.0], dtype=float)
    else:
        start_xyz = np.asarray(start_xyz, dtype=float).reshape(-1)
    V = compute_velocities(S, start_xyz, cfg.Tf)
    Hov = extract_hover_points(S, cfg.mu)

    dc = dc_uav_to_user(S, user_xy)
    h = channel_gain_comm(dc, cfg.alpha0)
    Rn = rate_per_waypoint(h, cfg.P_w, cfg.sigma0_sq, cfg.B)
    R_bar = average_rate(Rn)

    crb = crb_xyz_sum(Hov, target_hat_xyz, cfg)

    return {
        "N": int(S.shape[0]),
        "K": int(Hov.shape[0]),
        "V": V,
        "R_bar": R_bar,
        "CRB_xyz_sum": crb,
    }