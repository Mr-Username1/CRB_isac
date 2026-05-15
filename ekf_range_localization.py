"""
Static 3-D target + monostatic range measurements: sequential EKF.

State x = [x_t, y_t, z_t]^T.  Constant-position dynamics (F = I)
with small diagonal process noise Q for numerical stability.

For each new scalar range z_k at known UAV position u_k = [x_k^u, y_k^u, z_k^u]^T:

Prediction (static):
    x_{k|k-1} = x_{k-1|k-1}
    P_{k|k-1} = P_{k-1|k-1} + Q

Predicted measurement (evaluated at x_{k|k-1}):
    d_s = sqrt(||u_k - x_{k|k-1}||^2)
    h(x_{k|k-1}) = d_s

Jacobian (1 x 3), evaluated at x_{k|k-1}:
    H_k = [ -(x_k^u - x) / d_s ,  -(y_k^u - y) / d_s ,  -(z_k^u - z) / d_s ]

Measurement variance (heteroskedastic, same structure as simulation / MLE):
    g = beta0_est / d_s^4 ,   R_k = max( a * sigma0^2 / (P_w * Gp * g) , eps )

Kalman update (Joseph covariance for stability):
    S_k = H_k P_{k|k-1} H_k^T + R_k
    K_k = P_{k|k-1} H_k^T S_k^{-1}
    x_{k|k} = x_{k|k-1} + K_k ( z_k - h(x_{k|k-1}) )
    P_{k|k} = (I - K_k H_k) P_{k|k-1} (I - K_k H_k)^T + K_k R_k K_k^T
"""

from __future__ import annotations

import numpy as np

import system_model as sm


def default_prior_variance(cfg: sm.SimConfig, frac_xy: float = 0.33, frac_z: float = 0.50) -> np.ndarray:
    """
    Per-axis prior variance [sigma_x^2, sigma_y^2, sigma_z^2] for initial diagonal P0.
    Z-direction prior based on target height range, not map size.
    """
    side = min(float(cfg.Lx), float(cfg.Ly))
    s_xy = frac_xy * side
    z_range = float(cfg.z_t_max - cfg.z_t_min)
    s_z = frac_z * z_range
    return np.array([s_xy * s_xy, s_xy * s_xy, s_z * s_z], dtype=float)


def default_process_variance(cfg: sm.SimConfig) -> np.ndarray:
    """Per-axis process noise [q_xy, q_xy, q_z] added to P each predict step."""
    side = min(float(cfg.Lx), float(cfg.Ly))
    q_xy = (1.5e-4 * side) ** 2
    z_range = float(cfg.z_t_max - cfg.z_t_min)
    q_z = max((1.5e-4 * z_range) ** 2, 1e-8)
    return np.array([max(q_xy, 1e-4), max(q_xy, 1e-4), max(q_z, 1e-8)], dtype=float)


class StaticRangeEKF3D:
    """
    Sequential EKF for 3-D target position from range-only measurements.
    Uses the same H_est / beta0_est as ``mle_grid_search`` for R_k (estimator-side).
    """

    def __init__(self, cfg: sm.SimConfig, prior_frac_xy: float = 0.33, prior_frac_z: float = 0.50) -> None:
        self.cfg = cfg
        self.prior_frac_xy = float(prior_frac_xy)
        self.prior_frac_z = float(prior_frac_z)
        self._q_per_axis = default_process_variance(cfg)  # (3,)
        self.x_hat: np.ndarray = np.zeros(3, dtype=float)
        self.P: np.ndarray = np.eye(3, dtype=float)

    def reset(self, x0: np.ndarray, P0_diag: np.ndarray | None = None) -> None:
        """Posterior after coarse scan: mean = coarse MLE, covariance = diag(P0_diag)."""
        self.x_hat = np.asarray(x0, dtype=float).reshape(3).copy()
        self.x_hat[2] = float(np.clip(self.x_hat[2], self.cfg.z_t_min, self.cfg.z_t_max))
        if P0_diag is None:
            v = default_prior_variance(self.cfg, self.prior_frac_xy, self.prior_frac_z)
            self.P = np.diag(v)
        elif P0_diag.ndim == 2:
            self.P = np.asarray(P0_diag, dtype=float).reshape(3, 3).copy()
        else:
            self.P = np.diag(np.asarray(P0_diag, dtype=float).reshape(3))

    def predict_static(self) -> None:
        """F = I, Q = diag(q_per_axis): x unchanged, P <- P + Q."""
        self.P = self.P + np.diag(self._q_per_axis)

    def _predicted_range_and_H_and_R(self, uav_xyz: np.ndarray, x_prior: np.ndarray):
        h_est = float(self.cfg.H + self.cfg.model_mismatch_h)
        beta0_est = float(self.cfg.beta0 * (10.0 ** (self.cfg.model_mismatch_beta0_db / 10.0)))
        u = np.asarray(uav_xyz, dtype=float).reshape(3)
        x = np.asarray(x_prior, dtype=float).reshape(3)
        dx = float(u[0] - x[0])
        dy = float(u[1] - x[1])
        dz = float(u[2] - x[2])
        ds = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        ds = max(ds, 1e-6)
        g = float(sm.channel_gain_sensing(np.array([ds], dtype=float), beta0_est)[0])
        R = float(np.maximum(sm.sigma2_measurement_from_g(np.array([g]), self.cfg)[0], 1e-12))
        H = np.array([[-dx / ds, -dy / ds, -dz / ds]], dtype=float)
        return ds, R, H

    def update_one(self, uav_xyz: np.ndarray, z: float) -> None:
        """
        Ingest one range measurement z at UAV position uav_xyz.
        Prediction uses posterior from previous step; H,h,R evaluated at prior x^-.
        """
        self.predict_static()
        x_minus = self.x_hat.copy()
        P_minus = self.P.copy()
        h, R, H = self._predicted_range_and_H_and_R(uav_xyz, x_minus)
        innovation = float(z) - h
        S_mat = H @ P_minus @ H.T
        S = float(np.asarray(S_mat, dtype=float).reshape(-1)[0] + R)
        S = max(S, 1e-18)
        K = (P_minus @ H.T) / S
        self.x_hat = x_minus + (K.flatten() * innovation)
        self.x_hat[2] = float(np.clip(self.x_hat[2], self.cfg.z_t_min, self.cfg.z_t_max))
        I3 = np.eye(3, dtype=float)
        KH = K @ H
        joseph = (I3 - KH) @ P_minus @ (I3 - KH).T + R * (K @ K.T)
        self.P = 0.5 * (joseph + joseph.T)

    def ingest_sequence(self, hover_xyz: np.ndarray, z_vec: np.ndarray) -> None:
        """Apply ``update_one`` in row order (sequential)."""
        hover_xyz = np.asarray(hover_xyz, dtype=float)
        z_vec = np.asarray(z_vec, dtype=float).reshape(-1)
        n = int(z_vec.shape[0])
        if hover_xyz.shape[0] != n:
            raise ValueError("hover_xyz rows must match length of z_vec")
        for j in range(n):
            self.update_one(hover_xyz[j, :], float(z_vec[j]))
