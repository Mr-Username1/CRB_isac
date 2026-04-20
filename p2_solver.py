from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import cvxpy as cp

import system_model as sm
import problem as pb


@dataclass
class StageData:
    m: int
    Nm: int
    Km: int
    Em: float
    eta: float
    user_xy: np.ndarray          # (2,)
    target_hat_xy: np.ndarray    # (2,)
    start_xy: np.ndarray         # (2,)
    prev_hover_xy: np.ndarray    # (K_prev,2)
    N_prev_total: int = 0        # Eq.(33): previous waypoint count
    R_prev_sum: float = 0.0      # Eq.(33): previous sum-rate accumulator


@dataclass
class SolverCfg:
    max_sca_iter: int = 1000
    tol_obj: float = 1e-3
    step_size: float = 0.6
    delta_eps: float = 1e-3
    min_step_size: float = 1e-3
    line_search_candidates: tuple[float, ...] = (1.0, 0.8, 0.6, 0.4, 0.2, 0.1, 0.05)


def _extract_hover_expr(S: cp.Variable, mu: int, Km: int) -> cp.Expression:
    idx = [mu * (k + 1) - 1 for k in range(Km)]  # 0-based
    return cp.vstack([S[i, :] for i in idx])     # (Km,2)


def _init_path(stage: StageData, cfg: sm.SimConfig) -> np.ndarray:
    mid = 0.5 * (stage.user_xy + stage.target_hat_xy)
    d = mid - stage.start_xy
    d = d / (np.linalg.norm(d) + 1e-12)
    v0 = min(20.0, 0.8 * cfg.Vmax)
    S0 = np.vstack([stage.start_xy + (i + 1) * cfg.Tf * v0 * d for i in range(stage.Nm)])
    S0[:, 0] = np.clip(S0[:, 0], 0.0, cfg.Lx)
    S0[:, 1] = np.clip(S0[:, 1], 0.0, cfg.Ly)
    return S0


def _rate_linearized(S: cp.Variable, S_ref: np.ndarray, stage: StageData, cfg: sm.SimConfig):
    # 与你现有实现基本一致：返回 affine 的 R_lin
    N = stage.Nm
    C = cfg.P_w * cfg.alpha0 / cfg.sigma0_sq
    H2 = cfg.H**2
    ln2 = np.log(2.0)

    R0 = 0.0
    G = np.zeros((N, 2))
    c = stage.user_xy

    for i in range(N):
        d = S_ref[i] - c
        q = H2 + d @ d
        z = 1.0 + C / q
        R0 += cfg.B * np.log2(z)
        dr_dq = cfg.B / ln2 * (1.0 / z) * (-C / (q**2))
        G[i] = dr_dq * 2.0 * d

    R0 /= N
    G /= N

    R_lin = R0
    for i in range(N):
        R_lin += cp.sum(cp.multiply(G[i], (S[i, :] - S_ref[i, :])))
    return R_lin


def _rate_value(S: np.ndarray, stage: StageData, cfg: sm.SimConfig) -> float:
    """Evaluate Eq.(33)-style cumulative average rate."""
    dc = sm.dc_uav_to_user(S, stage.user_xy, cfg.H)
    h = sm.channel_gain_comm(dc, cfg.alpha0)
    R_cur_sum = float(np.sum(sm.rate_per_waypoint(h, cfg.P_w, cfg.sigma0_sq, cfg.B)))
    denom = float(stage.N_prev_total + stage.Nm)
    if denom <= 0.0:
        return 0.0
    return float((stage.R_prev_sum + R_cur_sum) / denom)


def _crb_linearized(S: cp.Variable, S_ref: np.ndarray, stage: StageData, cfg: sm.SimConfig):
    # 对“当前阶段悬停点”做有限差分梯度，历史悬停点固定
    H_cur_ref = sm.extract_hover_points(S_ref, cfg.mu)             # (Km,2)
    H_all_ref = np.vstack([stage.prev_hover_xy, H_cur_ref]) if stage.prev_hover_xy.size else H_cur_ref

    def f(H):
        return sm.crb_xy_sum(H, stage.target_hat_xy, cfg)

    crb0 = f(H_all_ref)
    eps = 1e-3
    G_all = np.zeros_like(H_all_ref)

    for k in range(H_all_ref.shape[0]):
        for d in range(2):
            Hp = H_all_ref.copy(); Hm = H_all_ref.copy()
            Hp[k, d] += eps; Hm[k, d] -= eps
            G_all[k, d] = (f(Hp) - f(Hm)) / (2.0 * eps)

    G_cur = G_all[-stage.Km:, :]          # (Km,2)
    H_cur_var = _extract_hover_expr(S, cfg.mu, stage.Km)  # (Km,2)

    F = crb0 + cp.sum(cp.multiply(G_cur, H_cur_var - H_cur_ref))   # affine
    return F


def solve_p2m_sca(stage: StageData, cfg: sm.SimConfig, e_cfg: pb.EnergyConfig, scfg: SolverCfg = SolverCfg()):
    if stage.Km != stage.Nm // cfg.mu:
        raise ValueError("StageData.Km must equal floor(Nm/mu).")

    S_ref = _init_path(stage, cfg)
    V_ref = sm.compute_velocities(S_ref, stage.start_xy, cfg.Tf)
    delta_ref = np.maximum(np.linalg.norm(V_ref, axis=1) / e_cfg.v0, scfg.delta_eps)

    last_obj = None
    hist = []

    for it in range(scfg.max_sca_iter):
        S = cp.Variable((stage.Nm, 2))
        V = cp.Variable((stage.Nm, 2))
        delta = cp.Variable(stage.Nm)
        xi = cp.Variable(stage.Nm)

        cons = []
        # kinematics
        cons += [V[0, :] == (S[0, :] - stage.start_xy) / cfg.Tf]
        if stage.Nm > 1:
            cons += [V[1:, :] == (S[1:, :] - S[:-1, :]) / cfg.Tf]

        # (40a)(40b)
        cons += [cp.norm(V, axis=1) <= cfg.Vmax]
        cons += [S[:, 0] >= 0, S[:, 0] <= cfg.Lx, S[:, 1] >= 0, S[:, 1] <= cfg.Ly]

        # (42)(48)
        cons += [delta >= scfg.delta_eps, xi >= 0]

        # (44) energy
        energy_terms = []
        for i in range(stage.Nm):
            v2 = cp.sum_squares(V[i, :])
            v3 = cp.power(cp.norm(V[i, :], 2), 3)
            energy_terms.append(
                e_cfg.P0 * (1 + 3 * v2 / (e_cfg.Utip**2))
                + 0.5 * e_cfg.D0 * e_cfg.rho * e_cfg.s * e_cfg.A * v3
                + e_cfg.PI * delta[i]
            )
        P_hover0 = pb.propulsion_power(np.array([0.0]), e_cfg)[0]
        E_expr = cfg.Tf * cp.sum(cp.hstack(energy_terms)) + cfg.Th * stage.Km * (e_cfg.P0 + e_cfg.PI)
        cons += [E_expr <= stage.Em]

        # (51a)(51b)
        for i in range(stage.Nm):
            v_prev = V_ref[i]
            lhs_51a = (np.dot(v_prev, v_prev) / (e_cfg.v0**2)) + (2.0 / (e_cfg.v0**2)) * cp.sum(cp.multiply(v_prev, (V[i, :] - v_prev)))
            rhs_51a = cp.square(cp.inv_pos(delta[i])) - xi[i]
            cons += [rhs_51a <= lhs_51a]

            d_prev = delta_ref[i]
            lhs_51b = (d_prev**2) + 2.0 * d_prev * (delta[i] - d_prev)
            cons += [lhs_51b >= xi[i]]

        # objective
        F = _crb_linearized(S, S_ref, stage, cfg)
        R_lin_stage = _rate_linearized(S, S_ref, stage, cfg)
        denom = float(stage.N_prev_total + stage.Nm)
        if denom <= 0.0:
            raise ValueError("Invalid cumulative denominator in Eq.(33).")
        R_lin = (stage.R_prev_sum + stage.Nm * R_lin_stage) / denom
        obj = cp.Minimize(stage.eta * F - (1 - stage.eta) * R_lin)

        prob = cp.Problem(obj, cons)
        prob.solve(verbose=False)
        solver_name = prob.solver_stats.solver_name if prob.solver_stats is not None else "UNKNOWN" 

        if S.value is None or delta.value is None:
            raise RuntimeError(f"iter={it}, status={prob.status}")

        S_new = np.asarray(S.value)
        delta_new = np.maximum(np.asarray(delta.value).reshape(-1), scfg.delta_eps)

        # Eq.(54)-(55)-style line search along descent direction.
        S_cand = S_ref.copy()
        delta_cand = delta_ref.copy()
        obj_cand = np.inf
        crb_cand = np.inf
        r_cand = 0.0
        accepted = False
        for step in scfg.line_search_candidates:
            S_try = (1 - step) * S_ref + step * S_new
            delta_try = (1 - step) * delta_ref + step * delta_new
            H_cur_try = sm.extract_hover_points(S_try, cfg.mu)
            H_all_try = np.vstack([stage.prev_hover_xy, H_cur_try]) if stage.prev_hover_xy.size else H_cur_try
            crb_try = sm.crb_xy_sum(H_all_try, stage.target_hat_xy, cfg)
            r_try = _rate_value(S_try, stage, cfg)
            obj_try = stage.eta * crb_try - (1 - stage.eta) * r_try
            if np.isfinite(obj_try):
                if obj_try < obj_cand:
                    S_cand = S_try
                    delta_cand = np.maximum(delta_try, scfg.delta_eps)
                    obj_cand = obj_try
                    crb_cand = crb_try
                    r_cand = r_try
                    accepted = True

        if not accepted:
            # Fallback to direct update if no descent candidate is found.
            S_cand = S_new
            delta_cand = delta_new
            H_cur_try = sm.extract_hover_points(S_cand, cfg.mu)
            H_all_try = np.vstack([stage.prev_hover_xy, H_cur_try]) if stage.prev_hover_xy.size else H_cur_try
            crb_cand = sm.crb_xy_sum(H_all_try, stage.target_hat_xy, cfg)
            r_cand = _rate_value(S_cand, stage, cfg)
            obj_cand = stage.eta * crb_cand - (1 - stage.eta) * r_cand

        S_ref = S_cand
        V_ref = sm.compute_velocities(S_ref, stage.start_xy, cfg.Tf)
        delta_ref = delta_cand

        hist.append((obj_cand, crb_cand, r_cand, prob.status))

        if last_obj is not None and abs(last_obj - obj_cand) <= scfg.tol_obj:
            break
        last_obj = obj_cand

    return {
        "S_opt": S_ref,
        "V_opt": sm.compute_velocities(S_ref, stage.start_xy, cfg.Tf),
        "Hov_cur": sm.extract_hover_points(S_ref, cfg.mu),
        "history": hist,
        "obj_final": hist[-1][0] if hist else None,
        "crb_final": hist[-1][1] if hist else None,
        "rate_final": hist[-1][2] if hist else None,
        "status_final": hist[-1][3] if hist else None,
        "solver_final": solver_name if hist else None,
    }