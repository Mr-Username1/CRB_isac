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
    user_xy: np.ndarray          # (2,)   — ground user, 2-D
    target_hat_xyz: np.ndarray   # (3,)   — estimated target [x_t, y_t, z_t]
    start_xyz: np.ndarray        # (3,)   — stage start position
    prev_hover_xyz: np.ndarray   # (K_prev, 3) — historical hover points
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
    return cp.vstack([S[i, :] for i in idx])     # (Km, D)


def _init_path(stage: StageData, cfg: sm.SimConfig) -> np.ndarray:
    """
    Paper Sec. IV-C, Eq.(60): connect communication user with \\hat x_t, take midpoint
    [x_mdl, y_mdl]; initial waypoints lie on the ray from stage start toward that midpoint
    at distances V_str, 2 V_str, ... along the unit direction.
    Z coordinate initialised at nominal altitude cfg.H.
    """
    mdl_xy = 0.5 * (stage.user_xy + stage.target_hat_xyz[:2])
    vec_xy = mdl_xy - stage.start_xyz[:2]
    u_xy = vec_xy / (np.linalg.norm(vec_xy) + 1e-12)
    vstr = float(cfg.Vstr)
    z_init = float(cfg.H)

    S0 = np.zeros((stage.Nm, 3), dtype=float)
    for i in range(stage.Nm):
        S0[i, 0] = stage.start_xyz[0] + vstr * float(i + 1) * u_xy[0]
        S0[i, 1] = stage.start_xyz[1] + vstr * float(i + 1) * u_xy[1]
        S0[i, 2] = z_init

    S0[:, 0] = np.clip(S0[:, 0], 0.0, cfg.Lx)
    S0[:, 1] = np.clip(S0[:, 1], 0.0, cfg.Ly)
    S0[:, 2] = np.clip(S0[:, 2], cfg.z_min, cfg.z_max)
    return S0


def _rate_linearized(S: cp.Variable, S_ref: np.ndarray, stage: StageData, cfg: sm.SimConfig):
    """Return affine R_lin for the current stage."""
    N = stage.Nm
    C = cfg.P_w * cfg.alpha0 / cfg.sigma0_sq
    ln2 = np.log(2.0)

    R0 = 0.0
    G = np.zeros((N, 3))
    c = stage.user_xy  # (2,)

    for i in range(N):
        d_xy = S_ref[i, :2] - c
        z_i = S_ref[i, 2]
        q = z_i * z_i + float(d_xy @ d_xy)
        z_ratio = 1.0 + C / q
        R0 += cfg.B * np.log2(z_ratio)
        dr_dq = cfg.B / ln2 * (1.0 / z_ratio) * (-C / (q * q))
        G[i, :2] = dr_dq * 2.0 * d_xy
        G[i, 2] = dr_dq * 2.0 * z_i

    R0 /= N
    G /= N

    R_lin = R0
    for i in range(N):
        R_lin += cp.sum(cp.multiply(G[i], (S[i, :] - S_ref[i, :])))
    return R_lin


def _rate_value(S: np.ndarray, stage: StageData, cfg: sm.SimConfig) -> float:
    """Evaluate Eq.(33)-style cumulative average rate."""
    dc = sm.dc_uav_to_user(S, stage.user_xy)
    h = sm.channel_gain_comm(dc, cfg.alpha0)
    R_cur_sum = float(np.sum(sm.rate_per_waypoint(h, cfg.P_w, cfg.sigma0_sq, cfg.B)))
    denom = float(stage.N_prev_total + stage.Nm)
    if denom <= 0.0:
        return 0.0
    return float((stage.R_prev_sum + R_cur_sum) / denom)


def _crb_linearized(S: cp.Variable, S_ref: np.ndarray, stage: StageData, cfg: sm.SimConfig):
    """Affine CRB linearisation for 3-D target.  Historical hover points are fixed."""
    H_cur_ref = sm.extract_hover_points(S_ref, cfg.mu)                          # (Km, 3)
    H_all_ref = np.vstack([stage.prev_hover_xyz, H_cur_ref]) if stage.prev_hover_xyz.size else H_cur_ref

    def f(H):
        return sm.crb_xyz_sum(H, stage.target_hat_xyz, cfg)

    crb0 = f(H_all_ref)
    eps = 1e-3
    G_all = np.zeros_like(H_all_ref)  # (K_all, 3)

    for k in range(H_all_ref.shape[0]):
        for d in range(3):
            Hp = H_all_ref.copy(); Hm = H_all_ref.copy()
            Hp[k, d] += eps; Hm[k, d] -= eps
            G_all[k, d] = (f(Hp) - f(Hm)) / (2.0 * eps)

    G_cur = G_all[-stage.Km:, :]          # (Km, 3)
    H_cur_var = _extract_hover_expr(S, cfg.mu, stage.Km)  # (Km, 3)

    F = crb0 + cp.sum(cp.multiply(G_cur, H_cur_var - H_cur_ref))
    return F


_SOLVER_CANDIDATES = ("MOSEK","CLARABEL", "SCS", "OSQP")


def _solve_with_fallback(prob: cp.Problem, stats: dict) -> str:
    """Try solvers in order; return the name of the first one that succeeds.
    Updates ``stats`` in-place with per-solver counts.
    """
    last_err = None
    for solver_name in _SOLVER_CANDIDATES:
        try:
            prob.solve(verbose=False, solver=solver_name)
        except Exception as e:
            last_err = e
            continue
        if prob.status not in ("infeasible", "unbounded", "infeasible_or_unbounded"):
            stats.setdefault(solver_name, 0)
            stats[solver_name] += 1
            return solver_name
    raise RuntimeError(
        f"All solvers {_SOLVER_CANDIDATES} failed. "
        f"Last error: {last_err}"
    )


def solve_p2m_sca(
    stage: StageData,
    cfg: sm.SimConfig,
    e_cfg: pb.EnergyConfig,
    scfg: SolverCfg = SolverCfg(),
    kappa: float = 1000.0,
):
    if kappa == 0.0:
        raise ValueError("kappa must be non-zero.")
    if stage.Km != stage.Nm // cfg.mu:
        raise ValueError("StageData.Km must equal floor(Nm/mu).")
    
    S_ref = _init_path(stage, cfg)
    V_ref = sm.compute_velocities(S_ref, stage.start_xyz, cfg.Tf)
    delta_ref = np.maximum(np.linalg.norm(V_ref, axis=1) / e_cfg.v0, scfg.delta_eps)

    last_obj = None
    hist = []
    solver_name = None
    solver_stats: dict = {}

    for it in range(scfg.max_sca_iter):
        S = cp.Variable((stage.Nm, 3))
        V = cp.Variable((stage.Nm, 3))
        delta = cp.Variable(stage.Nm)
        xi = cp.Variable(stage.Nm)

        cons = []
        # kinematics
        cons += [V[0, :] == (S[0, :] - stage.start_xyz) / cfg.Tf]
        if stage.Nm > 1:
            cons += [V[1:, :] == (S[1:, :] - S[:-1, :]) / cfg.Tf]

        # (40a)(40b) — speed & area bounds
        cons += [cp.norm(V, axis=1) <= cfg.Vmax]
        cons += [S[:, 0] >= 0, S[:, 0] <= cfg.Lx]
        cons += [S[:, 1] >= 0, S[:, 1] <= cfg.Ly]
        cons += [S[:, 2] >= cfg.z_min, S[:, 2] <= cfg.z_max]

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
        E_expr = cfg.Tf * cp.sum(cp.hstack(energy_terms)) + cfg.Th * stage.Km * P_hover0
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
        obj = cp.Minimize(stage.eta * F - (1 - stage.eta) * R_lin / kappa)

        prob = cp.Problem(obj, cons)
        solver_name = _solve_with_fallback(prob, solver_stats)

        if prob.solver_stats is not None:
            solver_name = prob.solver_stats.solver_name

        if S.value is None or delta.value is None:
            raise RuntimeError(f"iter={it}, status={prob.status}")

        S_new = np.asarray(S.value)
        delta_new = np.maximum(np.asarray(delta.value).reshape(-1), scfg.delta_eps)

        # line search along descent direction
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
            H_all_try = np.vstack([stage.prev_hover_xyz, H_cur_try]) if stage.prev_hover_xyz.size else H_cur_try
            crb_try = sm.crb_xyz_sum(H_all_try, stage.target_hat_xyz, cfg)
            r_try = _rate_value(S_try, stage, cfg)
            obj_try = stage.eta * crb_try - (1 - stage.eta) * r_try / kappa
            if np.isfinite(obj_try):
                if obj_try < obj_cand:
                    S_cand = S_try
                    delta_cand = np.maximum(delta_try, scfg.delta_eps)
                    obj_cand = obj_try
                    crb_cand = crb_try
                    r_cand = r_try
                    accepted = True

        if not accepted:
            S_cand = S_new
            delta_cand = delta_new
            H_cur_try = sm.extract_hover_points(S_cand, cfg.mu)
            H_all_try = np.vstack([stage.prev_hover_xyz, H_cur_try]) if stage.prev_hover_xyz.size else H_cur_try
            crb_cand = sm.crb_xyz_sum(H_all_try, stage.target_hat_xyz, cfg)
            r_cand = _rate_value(S_cand, stage, cfg)
            obj_cand = stage.eta * crb_cand - (1 - stage.eta) * r_cand / kappa

        S_ref = S_cand
        V_ref = sm.compute_velocities(S_ref, stage.start_xyz, cfg.Tf)
        delta_ref = delta_cand

        hist.append((obj_cand, crb_cand, r_cand, prob.status))

        if last_obj is not None and abs(last_obj - obj_cand) <= scfg.tol_obj:
            break
        last_obj = obj_cand

    # one-line solver stats summary
    stats_str = "  ".join(f"{s}={c}" for s, c in sorted(solver_stats.items()))
    iters = len(hist)
    print(f"  [stage {stage.m}] {iters} iters  |  {stats_str}  |  final {hist[-1][3]}")

    return {
        "S_opt": S_ref,
        "V_opt": sm.compute_velocities(S_ref, stage.start_xyz, cfg.Tf),
        "Hov_cur": sm.extract_hover_points(S_ref, cfg.mu),
        "history": hist,
        "obj_final": hist[-1][0] if hist else None,
        "crb_final": hist[-1][1] if hist else None,
        "rate_final": hist[-1][2] if hist else None,
        "status_final": hist[-1][3] if hist else None,
        "solver_final": solver_name if hist else None,
    }
