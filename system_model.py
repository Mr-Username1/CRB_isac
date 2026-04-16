from __future__ import annotations
from dataclasses import dataclass
import numpy as np


# =========================
# Config
# =========================

@dataclass
class SimConfig:
    # Geometry
    Lx: float = 1500.0
    Ly: float = 1500.0
    xB: float = 100.0
    yB: float = 100.0
    H: float = 200.0

    # Time / trajectory
    Tf: float = 1.5
    Th: float = 1.0
    mu: int = 5
    Vmax: float = 30.0

    # Communication (raw)
    B: float = 1e6
    P_dbm: float = 20.0
    N0_dbm_per_hz: float = -170.0
    alpha0_db: float = -50.0

    # Sensing (raw)
    beta0_db: float = -47.0
    a: float = 10.0  # measurement-noise scaling factor
    c: float = 3e8   # speed of light

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
    cfg.P_w = dbm_to_watt(cfg.P_dbm)
    cfg.N0_w_per_hz = dbm_to_watt(cfg.N0_dbm_per_hz)
    cfg.sigma0_sq = cfg.N0_w_per_hz * cfg.B
    cfg.alpha0 = db_to_linear(cfg.alpha0_db)
    cfg.beta0 = db_to_linear(cfg.beta0_db)
    cfg.Gp = 0.1 * cfg.B
    return cfg


# =========================
# II-A UAV trajectory model
# =========================

def compute_velocities(S: np.ndarray, start_xy: np.ndarray, Tf: float) -> np.ndarray:
    """
    Eq. (1)
    S: (N,2), start_xy: (2,)
    return V: (N,2)
    """
    S = np.asarray(S, dtype=float)
    start_xy = np.asarray(start_xy, dtype=float).reshape(2,)
    N = S.shape[0]
    V = np.zeros_like(S, dtype=float)
    V[0] = (S[0] - start_xy) / Tf
    if N > 1:
        V[1:] = (S[1:] - S[:-1]) / Tf
    return V


def extract_hover_points(S: np.ndarray, mu: int) -> np.ndarray:
    """
    Eq. (2): hover at indices mu, 2mu, ... (1-based)
    Python 0-based -> mu-1, 2mu-1, ...
    return Hov: (K,2)
    """
    S = np.asarray(S, dtype=float)
    idx = np.arange(mu - 1, S.shape[0], mu, dtype=int)
    return S[idx]


# =========================
# II-B communication model
# =========================

def dc_uav_to_user(S: np.ndarray, user_xy: np.ndarray, H: float) -> np.ndarray:
    """
    Eq. (3)
    return dc: (N,)
    """
    S = np.asarray(S, dtype=float)
    user_xy = np.asarray(user_xy, dtype=float).reshape(2,)
    diff = S - user_xy
    return np.sqrt(H**2 + np.sum(diff**2, axis=1))


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

def ds_uav_to_target(Hov: np.ndarray, target_xy: np.ndarray, H: float) -> np.ndarray:
    """
    Eq. (8)
    return ds: (K,)
    """
    Hov = np.asarray(Hov, dtype=float)
    target_xy = np.asarray(target_xy, dtype=float).reshape(2,)
    diff = Hov - target_xy
    return np.sqrt(H**2 + np.sum(diff**2, axis=1))


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


def theta_abc(Hov: np.ndarray, target_hat_xy: np.ndarray, cfg: SimConfig) -> tuple[float, float, float]:
    """
    Eqs. (23)-(25), using estimated target (x_hat_t, y_hat_t)
    """
    Hov = np.asarray(Hov, dtype=float)
    target_hat_xy = np.asarray(target_hat_xy, dtype=float).reshape(2,)

    dx = Hov[:, 0] - target_hat_xy[0]
    dy = Hov[:, 1] - target_hat_xy[1]
    ds = np.sqrt(cfg.H**2 + dx**2 + dy**2)

    c1 = (cfg.P_w * cfg.Gp * cfg.beta0) / (cfg.a * cfg.sigma0_sq)

    theta_a = np.sum(c1 * (dx**2) / (ds**6) + 8.0 * (dx**2) / (ds**4))
    theta_b = np.sum(c1 * (dy**2) / (ds**6) + 8.0 * (dy**2) / (ds**4))
    theta_c = np.sum(c1 * (dx * dy) / (ds**6) + 8.0 * (dx * dy) / (ds**4))

    return float(theta_a), float(theta_b), float(theta_c)


def crb_xy_sum(Hov: np.ndarray, target_hat_xy: np.ndarray, cfg: SimConfig) -> float:
    """
    Eq. (28): CRB_xt + CRB_yt
    """
    theta_a, theta_b, theta_c = theta_abc(Hov, target_hat_xy, cfg)
    denom = theta_a * theta_b - theta_c**2
    eps = 1e-12
    if denom <= eps:
        return np.inf
    return float((theta_a + theta_b) / denom)


# =========================
# Optional: one-shot evaluator
# =========================

def evaluate_stage_metrics(
    S: np.ndarray,
    user_xy: np.ndarray,
    target_hat_xy: np.ndarray,
    cfg: SimConfig
) -> dict:
    """
    返回当前轨迹的通信+感知指标（Section II）
    """
    V = compute_velocities(S, np.array([cfg.xB, cfg.yB]), cfg.Tf)
    Hov = extract_hover_points(S, cfg.mu)

    dc = dc_uav_to_user(S, user_xy, cfg.H)
    h = channel_gain_comm(dc, cfg.alpha0)
    Rn = rate_per_waypoint(h, cfg.P_w, cfg.sigma0_sq, cfg.B)
    R_bar = average_rate(Rn)

    crb = crb_xy_sum(Hov, target_hat_xy, cfg)

    return {
        "N": int(S.shape[0]),
        "K": int(Hov.shape[0]),
        "V": V,
        "R_bar": R_bar,
        "CRB_xy_sum": crb,
    }