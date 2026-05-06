"""
Static 2-D ground target + monostatic range measurements: sequential EKF.

State x = [x_t, y_t]^T. Constant-velocity dynamics are not used; we take F = I
with small diagonal process noise Q for numerical stability.

For each new scalar range z_k at known UAV horizontal position u_k = [x_k^u, y_k^u]^T:

  Prediction (static):
      x_{k|k-1} = x_{k-1|k-1}
      P_{k|k-1} = P_{k-1|k-1} + Q

  Predicted measurement (evaluated at prior x_{k|k-1}):
      d_s = sqrt(H_est^2 + ||u_k - x_{k|k-1}||^2)
      h(x_{k|k-1}) = d_s

  Jacobian (1 x 2), evaluated at x_{k|k-1}:
      H_k = [ - (x_k^u - x^-) / d_s ,  - (y_k^u - y^-) / d_s ]

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


def default_prior_variance_xy(cfg: sm.SimConfig, frac: float = 0.32) -> float:
    """
    Per-axis prior variance sigma^2 for initial diagonal P0.
    sigma = frac * min(Lx, Ly) (large, map-relative).
    """
    side = min(float(cfg.Lx), float(cfg.Ly))
    sigma = frac * side
    return float(sigma * sigma)


def default_process_variance(cfg: sm.SimConfig) -> float:
    """Per-axis process noise q added to P each predict step (static target, tiny)."""
    side = min(float(cfg.Lx), float(cfg.Ly))
    q = (1.5e-4 * side) ** 2
    return float(max(q, 1e-4))


class StaticRangeEKF2D:
    """
    Sequential EKF for horizontal target position from range-only measurements.
    Uses the same H_est / beta0_est as ``mle_grid_search`` for R_k (estimator-side).
    """

    def __init__(self, cfg: sm.SimConfig, prior_frac: float = 0.32) -> None:
        self.cfg = cfg
        self.prior_frac = float(prior_frac)
        self._q_per_axis = default_process_variance(cfg)
        self.x_hat: np.ndarray = np.zeros(2, dtype=float)
        self.P: np.ndarray = np.eye(2, dtype=float)

    def reset(self, x0: np.ndarray, P0: np.ndarray | None = None) -> None:
        """Posterior after coarse scan: mean = coarse MLE (or other), covariance = P0."""
        self.x_hat = np.asarray(x0, dtype=float).reshape(2).copy()
        if P0 is None:
            v = default_prior_variance_xy(self.cfg, self.prior_frac)
            self.P = np.eye(2, dtype=float) * v
        else:
            self.P = np.asarray(P0, dtype=float).reshape(2, 2).copy()

    def predict_static(self) -> None:
        """F = I, Q = q I: x unchanged, P <- P + Q."""
        self.P = self.P + np.eye(2, dtype=float) * self._q_per_axis

    def _predicted_range_and_H_and_R(self, uav_xy: np.ndarray, x_prior: np.ndarray):
        h_est = float(self.cfg.H + self.cfg.model_mismatch_h)
        beta0_est = float(self.cfg.beta0 * (10.0 ** (self.cfg.model_mismatch_beta0_db / 10.0)))
        u = np.asarray(uav_xy, dtype=float).reshape(2)
        x = np.asarray(x_prior, dtype=float).reshape(2)
        dx = float(u[0] - x[0])
        dy = float(u[1] - x[1])
        ds = float(np.sqrt(h_est * h_est + dx * dx + dy * dy))
        ds = max(ds, 1e-6)
        g = float(sm.channel_gain_sensing(np.array([ds], dtype=float), beta0_est)[0])
        R = float(np.maximum(sm.sigma2_measurement_from_g(np.array([g]), self.cfg)[0], 1e-12))
        H = np.array([[-dx / ds, -dy / ds]], dtype=float)
        return ds, R, H

    def update_one(self, uav_xy: np.ndarray, z: float) -> None:
        """
        Ingest one range measurement z at horizontal UAV position uav_xy.
        Prediction uses posterior from previous step; H,h,R evaluated at prior x^-.
        """
        self.predict_static()
        x_minus = self.x_hat.copy()
        P_minus = self.P.copy()
        h, R, H = self._predicted_range_and_H_and_R(uav_xy, x_minus)
        innovation = float(z) - h
        # H @ P @ H.T is shape (1,1); must reduce to scalar for float() / division
        S_mat = H @ P_minus @ H.T
        S = float(np.asarray(S_mat, dtype=float).reshape(-1)[0] + R)
        S = max(S, 1e-18)
        K = (P_minus @ H.T) / S
        self.x_hat = x_minus + (K.flatten() * innovation)
        I2 = np.eye(2, dtype=float)
        KH = K @ H
        joseph = (I2 - KH) @ P_minus @ (I2 - KH).T + R * (K @ K.T)
        self.P = 0.5 * (joseph + joseph.T)

    def ingest_sequence(self, hover_xy: np.ndarray, z_vec: np.ndarray) -> None:
        """Apply ``update_one`` in row order (sequential)."""
        hover_xy = np.asarray(hover_xy, dtype=float)
        z_vec = np.asarray(z_vec, dtype=float).reshape(-1)
        n = int(z_vec.shape[0])
        if hover_xy.shape[0] != n:
            raise ValueError("hover_xy rows must match length of z_vec")
        for j in range(n):
            self.update_one(hover_xy[j, :], float(z_vec[j]))
