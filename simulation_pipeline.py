from __future__ import annotations

from typing import Literal

import numpy as np
import system_model as sm
import problem as pb
from ekf_range_localization import StaticRangeEKF3D
from p2_solver import StageData, SolverCfg, solve_p2m_sca


def simulate_range_measurements(
    hover_xyz: np.ndarray,
    true_target_xyz: np.ndarray,
    cfg: sm.SimConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate noisy distance measurements d_hat(k)=d(k)+w(k)."""
    ds_true = sm.ds_uav_to_target(hover_xyz, true_target_xyz)
    g_true = sm.channel_gain_sensing(ds_true, cfg.beta0)
    sigma2 = sm.sigma2_measurement_from_g(g_true, cfg)
    noise = rng.normal(loc=0.0, scale=np.sqrt(sigma2), size=ds_true.shape[0])
    nlos_bias = rng.normal(
        loc=cfg.nlos_bias_mean,
        scale=max(cfg.nlos_bias_std, 0.0),
        size=ds_true.shape[0],
    )
    nlos_bias = np.maximum(nlos_bias, 0.0)
    outlier_mask = rng.random(ds_true.shape[0]) < cfg.outlier_prob
    outlier_noise = np.zeros_like(ds_true)
    if np.any(outlier_mask):
        outlier_noise[outlier_mask] = rng.normal(
            loc=0.0,
            scale=cfg.outlier_std,
            size=int(np.sum(outlier_mask)),
        )
    return ds_true + noise + nlos_bias + outlier_noise


def _crb_xyz_sum_finite(hover_xyz: np.ndarray, ref_xyz: np.ndarray, cfg: sm.SimConfig) -> float:
    """CRB trace at reference point ``ref_xyz``; nan if geometry is singular."""
    v = sm.crb_xyz_sum(
        np.asarray(hover_xyz, dtype=float),
        np.asarray(ref_xyz, dtype=float).reshape(3),
        cfg,
    )
    return float(v) if np.isfinite(v) else float("nan")


def _near_map_corner(xy: np.ndarray, cfg: sm.SimConfig, tol: float = 5.0) -> bool:
    """True if xy lies near a rectangle corner (common spurious discrete-MLE artifact)."""
    x, y = float(xy[0]), float(xy[1])
    on_v = x <= tol or x >= cfg.Lx - tol
    on_h = y <= tol or y >= cfg.Ly - tol
    return on_v and on_h


def _two_level_grid_min_3d(
    neg_log_like,
    x_lo: float, x_hi: float,
    y_lo: float, y_hi: float,
    z_lo: float, z_hi: float,
    coarse_step: float,
    fine_step: float,
    fine_radius: float,
    z_coarse_step: float = 50.0,
    z_fine_step: float = 10.0,
    z_fine_radius: float = 100.0,
) -> tuple[float, float, float, float]:
    """Return (best_nll, best_x, best_y, best_z) over coarse+fine 3-D grid."""
    best = (np.inf, 0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi), 0.5 * (z_lo + z_hi))
    x_grid = np.arange(x_lo, x_hi + 1e-9, coarse_step)
    y_grid = np.arange(y_lo, y_hi + 1e-9, coarse_step)
    z_grid = np.arange(z_lo, z_hi + 1e-9, z_coarse_step)
    for x in x_grid:
        for y in y_grid:
            for z_val in z_grid:
                val = neg_log_like(x, y, z_val)
                if val < best[0]:
                    best = (val, float(x), float(y), float(z_val))
    x0, y0, z0 = best[1], best[2], best[3]
    xf_l = max(x_lo, x0 - fine_radius)
    xf_r = min(x_hi, x0 + fine_radius)
    yf_l = max(y_lo, y0 - fine_radius)
    yf_r = min(y_hi, y0 + fine_radius)
    zf_l = max(z_lo, z0 - z_fine_radius)
    zf_r = min(z_hi, z0 + z_fine_radius)
    x_fine = np.arange(xf_l, xf_r + 1e-9, fine_step)
    y_fine = np.arange(yf_l, yf_r + 1e-9, fine_step)
    z_fine = np.arange(zf_l, zf_r + 1e-9, z_fine_step)
    for x in x_fine:
        for y in y_fine:
            for z_val in z_fine:
                val = neg_log_like(x, y, z_val)
                if val < best[0]:
                    best = (val, float(x), float(y), float(z_val))
    return best


def _two_level_grid_min(
    neg_log_like,
    x_lo: float, x_hi: float,
    y_lo: float, y_hi: float,
    coarse_step: float,
    fine_step: float,
    fine_radius: float,
    fixed_z: float | None = None,
) -> tuple[float, float, float]:
    """Return (best_nll, best_x, best_y) over coarse+fine 2-D grid."""
    best = (np.inf, 0.5 * (x_lo + x_hi), 0.5 * (y_lo + y_hi))
    x_grid = np.arange(x_lo, x_hi + 1e-9, coarse_step)
    y_grid = np.arange(y_lo, y_hi + 1e-9, coarse_step)
    for x in x_grid:
        for y in y_grid:
            val = neg_log_like(x, y, fixed_z) if fixed_z is not None else neg_log_like(x, y)
            if val < best[0]:
                best = (val, float(x), float(y))
    x0, y0 = best[1], best[2]
    xf_l = max(x_lo, x0 - fine_radius)
    xf_r = min(x_hi, x0 + fine_radius)
    yf_l = max(y_lo, y0 - fine_radius)
    yf_r = min(y_hi, y0 + fine_radius)
    x_fine = np.arange(xf_l, xf_r + 1e-9, fine_step)
    y_fine = np.arange(yf_l, yf_r + 1e-9, fine_step)
    for x in x_fine:
        for y in y_fine:
            val = neg_log_like(x, y, fixed_z) if fixed_z is not None else neg_log_like(x, y)
            if val < best[0]:
                best = (val, float(x), float(y))
    return best


def mle_grid_search(
    measured_ds: np.ndarray,
    hover_xyz: np.ndarray,
    cfg: sm.SimConfig,
    coarse_step: float = 200.0,
    fine_step: float = 1.0,
    fine_radius: float = 200.0,
    ref_xyz: np.ndarray | None = None,
    ref_trust_radius: float = 450.0,
    ref_coarse_step: float = 35.0,
    ref_fine_radius: float = 130.0,
    corner_nll_slack: float | None = None,
    jump_ref_factor: float = 2.5,
    fixed_z: float | None = None,
    z_coarse_step: float = 50.0,
    z_fine_step: float = 10.0,
) -> np.ndarray:
    """
    MLE target estimate via two-level grid search.

    When ``fixed_z`` is given, a 2-D grid (x, y) is used with z clamped at that value
    (suitable for the initial coarse scan where z is unobservable).

    When ``fixed_z`` is None, a full 3-D grid (x, y, z) is used.
    """
    measured_ds = np.asarray(measured_ds, dtype=float).reshape(-1)
    hover_xyz = np.asarray(hover_xyz, dtype=float)
    k_meas = int(measured_ds.shape[0])
    if corner_nll_slack is None:
        corner_nll_slack = max(250.0, 18.0 * float(k_meas))

    h_est = cfg.H + cfg.model_mismatch_h
    beta0_est = cfg.beta0 * (10.0 ** (cfg.model_mismatch_beta0_db / 10.0))

    if fixed_z is not None:
        # ---- 2-D search (x, y) with fixed z ----
        def neg_log_like(x: float, y: float, z_fixed: float | None = None) -> float:
            target = np.array([x, y, float(fixed_z)], dtype=float)
            ds_model = sm.ds_uav_to_target(hover_xyz, target)
            g_model = sm.channel_gain_sensing(ds_model, beta0_est)
            sigma2 = np.maximum(sm.sigma2_measurement_from_g(g_model, cfg), 1e-12)
            resid2 = (measured_ds - ds_model) ** 2
            return float(0.5 * np.sum(np.log(sigma2) + resid2 / sigma2))

        nll_g, xg, yg = _two_level_grid_min(
            neg_log_like, 0.0, cfg.Lx, 0.0, cfg.Ly, coarse_step, fine_step, fine_radius
        )
        xyz_global = np.array([xg, yg, float(fixed_z)], dtype=float)
    else:
        # ---- 3-D search ----
        def neg_log_like(x: float, y: float, z_val: float) -> float:
            target = np.array([x, y, z_val], dtype=float)
            ds_model = sm.ds_uav_to_target(hover_xyz, target)
            g_model = sm.channel_gain_sensing(ds_model, beta0_est)
            sigma2 = np.maximum(sm.sigma2_measurement_from_g(g_model, cfg), 1e-12)
            resid2 = (measured_ds - ds_model) ** 2
            return float(0.5 * np.sum(np.log(sigma2) + resid2 / sigma2))

        if ref_xyz is not None:
            z_center = float(np.asarray(ref_xyz).reshape(3)[2])
        else:
            z_center = float(cfg.H) * 0.5
        z_lo = max(cfg.z_t_min, z_center - 150.0)
        z_hi = min(cfg.z_t_max, z_center + 150.0)

        nll_g, xg, yg, zg = _two_level_grid_min_3d(
            neg_log_like, 0.0, cfg.Lx, 0.0, cfg.Ly, z_lo, z_hi,
            coarse_step, fine_step, fine_radius,
            z_coarse_step=z_coarse_step, z_fine_step=z_fine_step,
        )
        xyz_global = np.array([xg, yg, zg], dtype=float)

    if ref_xyz is None:
        return xyz_global

    ref_xyz = np.asarray(ref_xyz, dtype=float).reshape(3,)
    rx, ry, rz = float(ref_xyz[0]), float(ref_xyz[1]), float(ref_xyz[2])
    x_lo = max(0.0, rx - ref_trust_radius)
    x_hi = min(cfg.Lx, rx + ref_trust_radius)
    y_lo = max(0.0, ry - ref_trust_radius)
    y_hi = min(cfg.Ly, ry + ref_trust_radius)

    if fixed_z is not None:
        def neg_log_like_ref(x: float, y: float, z_fixed: float | None = None) -> float:
            target = np.array([x, y, float(fixed_z)], dtype=float)
            ds_model = sm.ds_uav_to_target(hover_xyz, target)
            g_model = sm.channel_gain_sensing(ds_model, beta0_est)
            sigma2 = np.maximum(sm.sigma2_measurement_from_g(g_model, cfg), 1e-12)
            resid2 = (measured_ds - ds_model) ** 2
            return float(0.5 * np.sum(np.log(sigma2) + resid2 / sigma2))

        nll_r, xr, yr = _two_level_grid_min(
            neg_log_like_ref, x_lo, x_hi, y_lo, y_hi,
            ref_coarse_step, fine_step, ref_fine_radius,
        )
        xyz_ref = np.array([xr, yr, float(fixed_z)], dtype=float)
    else:
        z_lo_r = max(cfg.z_t_min, rz - 150.0)
        z_hi_r = min(cfg.z_t_max, rz + 150.0)

        def neg_log_like_ref(x: float, y: float, z_val: float) -> float:
            target = np.array([x, y, z_val], dtype=float)
            ds_model = sm.ds_uav_to_target(hover_xyz, target)
            g_model = sm.channel_gain_sensing(ds_model, beta0_est)
            sigma2 = np.maximum(sm.sigma2_measurement_from_g(g_model, cfg), 1e-12)
            resid2 = (measured_ds - ds_model) ** 2
            return float(0.5 * np.sum(np.log(sigma2) + resid2 / sigma2))

        nll_r, xr, yr, zr = _two_level_grid_min_3d(
            neg_log_like_ref, x_lo, x_hi, y_lo, y_hi, z_lo_r, z_hi_r,
            ref_coarse_step, fine_step, ref_fine_radius,
            z_coarse_step=z_coarse_step, z_fine_step=z_fine_step,
        )
        xyz_ref = np.array([xr, yr, zr], dtype=float)

    prefer_ref = False
    if _near_map_corner(xyz_global[:2], cfg) and (nll_g >= nll_r - corner_nll_slack):
        prefer_ref = True
    dist_jump = float(np.linalg.norm(xyz_global[:2] - ref_xyz[:2]))
    if (not prefer_ref) and dist_jump > jump_ref_factor * ref_trust_radius:
        if nll_g >= nll_r - corner_nll_slack:
            prefer_ref = True

    return xyz_ref.copy() if prefer_ref else xyz_global.copy()


def build_coarse_scan_hover_points(
    cfg: sm.SimConfig,
    nx: int = 2,
    ny: int = 2,
    center_xy: np.ndarray | None = None,
    span_xy: tuple[float, float] = (40.0, 40.0),
) -> np.ndarray:
    """Build a local coarse scan grid of hover points (3-D with z=cfg.H)."""
    if center_xy is None:
        center_xy = np.array([cfg.xB, cfg.yB], dtype=float)
    center_xy = np.asarray(center_xy, dtype=float).reshape(2,)
    half_x = 0.5 * span_xy[0]
    half_y = 0.5 * span_xy[1]
    x_l = max(0.0, center_xy[0] - half_x)
    x_r = min(cfg.Lx, center_xy[0] + half_x)
    y_l = max(0.0, center_xy[1] - half_y)
    y_r = min(cfg.Ly, center_xy[1] + half_y)
    xs = np.linspace(x_l, x_r, nx)
    ys = np.linspace(y_l, y_r, ny)
    points = []
    for i, y in enumerate(ys):
        x_row = xs if i % 2 == 0 else xs[::-1]
        for x in x_row:
            points.append([x, y, float(cfg.H)])
    return np.asarray(points, dtype=float)


def compute_scan_energy(
    hover_xyz: np.ndarray,
    start_xyz: np.ndarray,
    cfg: sm.SimConfig,
    e_cfg: pb.EnergyConfig,
) -> float:
    """Compute flying + hovering energy for a coarse scan path."""
    hover_xyz = np.asarray(hover_xyz, dtype=float)
    start_xyz = np.asarray(start_xyz, dtype=float).reshape(-1)
    if hover_xyz.size == 0:
        return 0.0
    V = sm.compute_velocities(hover_xyz, start_xyz, cfg.Tf)
    v_norm = np.linalg.norm(V, axis=1)
    P_fly = pb.propulsion_power(v_norm, e_cfg)
    P_hover0 = pb.propulsion_power(np.array([0.0]), e_cfg)[0]
    E = cfg.Tf * float(P_fly.sum()) + cfg.Th * hover_xyz.shape[0] * float(P_hover0)
    return float(E)


def run_initial_coarse_scan(
    cfg: sm.SimConfig,
    e_cfg: pb.EnergyConfig,
    true_target_xyz: np.ndarray,
    rng: np.random.Generator,
    start_xyz: np.ndarray | None = None,
    nx: int = 2,
    ny: int = 2,
) -> dict:
    """Run coarse scan and return initial target estimate."""
    if start_xyz is None:
        start_xyz = np.array([cfg.xB, cfg.yB, 0.0], dtype=float)
    start_xyz = np.asarray(start_xyz, dtype=float).reshape(-1)
    coarse_hover_xyz = build_coarse_scan_hover_points(cfg, nx=nx, ny=ny, center_xy=start_xyz[:2])
    measured_ds = simulate_range_measurements(coarse_hover_xyz, true_target_xyz, cfg, rng)
    # 2-D MLE with fixed z = z_t_min (z unobservable from 4 co-altitude points)
    target_hat_xyz = mle_grid_search(
        measured_ds=measured_ds,
        hover_xyz=coarse_hover_xyz,
        cfg=cfg,
        coarse_step=80.0,
        fine_step=20.0,
        fine_radius=150.0,
        fixed_z=0.5 * float(cfg.z_t_min + cfg.z_t_max),
    )
    E_scan = compute_scan_energy(coarse_hover_xyz, start_xyz, cfg, e_cfg)
    return {
        "coarse_hover_xyz": coarse_hover_xyz,
        "measured_ds": measured_ds,
        "target_hat_init_xyz": target_hat_xyz,
        "scan_energy_used": E_scan,
    }


def run_multistage_with_mle(
    cfg: sm.SimConfig,
    e_cfg: pb.EnergyConfig,
    scfg: SolverCfg,
    user_xy: np.ndarray,
    true_target_xyz: np.ndarray,
    eta: float,
    nstg: int,
    etot: float,
    random_seed: int = 1,
    coarse_scan_nx: int = 2,
    coarse_scan_ny: int = 2,
    localizer: Literal["mle", "ekf"] = "ekf",
) -> dict:
    """Run multi-stage trajectory design and stage-wise target localization (MLE or EKF)."""
    rng = np.random.default_rng(random_seed)
    start_xyz0 = np.array([cfg.xB, cfg.yB, 0.0], dtype=float)
    coarse = run_initial_coarse_scan(
        cfg=cfg, e_cfg=e_cfg, true_target_xyz=true_target_xyz,
        rng=rng, start_xyz=start_xyz0, nx=coarse_scan_nx, ny=coarse_scan_ny,
    )
    coarse_hover_xyz = coarse["coarse_hover_xyz"]
    measured_ds_all = np.asarray(coarse["measured_ds"], dtype=float).copy()
    target_hat_xyz = np.asarray(coarse["target_hat_init_xyz"], dtype=float).copy()
    scan_energy_used = float(coarse["scan_energy_used"])

    m = 1
    start_xyz = coarse_hover_xyz[-1, :].copy()
    prev_hover_xyz = coarse_hover_xyz.copy()
    energy_left = float(etot - scan_energy_used)
    if energy_left < 0.0:
        energy_left = 0.0

    stage_logs = []
    all_paths = []
    all_hovers = []
    stage_histories = []
    target_hat_history = [target_hat_xyz.copy()]
    n_prev_total = 0
    r_prev_sum = 0.0

    loc = str(localizer).lower()
    if loc not in ("mle", "ekf"):
        raise ValueError(f"localizer must be 'mle' or 'ekf', got {localizer!r}")
    ekf: StaticRangeEKF3D | None = None
    if loc == "ekf":
        ekf = StaticRangeEKF3D(cfg)
        ekf.reset(target_hat_xyz)

    while True:
        if energy_left <= 1e-6:
            break

        out = None
        nm_used = None
        km_used = None
        min_nm = 3 * cfg.mu
        for nm_try in range(int(nstg), min_nm - 1, -cfg.mu):
            km_try = nm_try // cfg.mu
            stage = StageData(
                m=m, Nm=nm_try, Km=km_try, Em=energy_left, eta=eta,
                user_xy=user_xy, target_hat_xyz=target_hat_xyz,
                start_xyz=start_xyz, prev_hover_xyz=prev_hover_xyz,
                N_prev_total=n_prev_total, R_prev_sum=r_prev_sum,
            )
            try:
                out_try = solve_p2m_sca(stage, cfg, e_cfg, scfg)
            except RuntimeError:
                continue

            if out_try["status_final"] not in ("optimal", "optimal_inaccurate"):
                continue
            s_try = out_try["S_opt"]
            e_used_try, _ = pb.stage_energy_used(s_try, start_xyz, cfg, e_cfg)
            if e_used_try <= energy_left + 1e-6:
                out = out_try
                nm_used = nm_try
                km_used = km_try
                break

        if out is None:
            break

        s_opt = out["S_opt"]
        hov = out["Hov_cur"]
        e_used, _ = pb.stage_energy_used(s_opt, start_xyz, cfg, e_cfg)
        energy_left -= e_used
        prev_hover_xyz = np.vstack([prev_hover_xyz, hov]) if prev_hover_xyz.size else hov.copy()
        start_xyz = s_opt[-1, :].copy()
        dc_stage = sm.dc_uav_to_user(s_opt, user_xy)
        h_stage = sm.channel_gain_comm(dc_stage, cfg.alpha0)
        r_prev_sum += float(np.sum(sm.rate_per_waypoint(h_stage, cfg.P_w, cfg.sigma0_sq, cfg.B)))
        n_prev_total += int(s_opt.shape[0])
        measured_ds_stage = simulate_range_measurements(hov, true_target_xyz, cfg, rng)
        measured_ds_all = np.hstack([measured_ds_all, measured_ds_stage])
        target_hat_prev = target_hat_xyz.copy()
        if ekf is not None:
            for j in range(int(hov.shape[0])):
                ekf.update_one(hov[j, :], float(measured_ds_stage[j]))
            target_hat_xyz = ekf.x_hat.copy()
        else:
            target_hat_xyz = mle_grid_search(
                measured_ds_all, prev_hover_xyz, cfg,
                ref_xyz=target_hat_prev,
                fixed_z=None,  # full 3-D after coarse scan
            )
        target_hat_history.append(target_hat_xyz.copy())

        pos_err_m = float(np.linalg.norm(target_hat_xyz - true_target_xyz))
        crb_at_true = _crb_xyz_sum_finite(prev_hover_xyz, true_target_xyz, cfg)
        crb_at_hat = _crb_xyz_sum_finite(prev_hover_xyz, target_hat_xyz, cfg)

        all_paths.append(s_opt)
        all_hovers.append(hov)
        stage_histories.append(np.array(out["history"], dtype=object))
        stage_logs.append(
            {
                "stage": m, "Nm": nm_used, "Km": km_used,
                "E_used": e_used, "E_left": energy_left,
                "obj_final": out["obj_final"],
                "crb_final": out["crb_final"],
                "rate_final": out["rate_final"],
                "iters": len(out["history"]),
                "status": out["status_final"],
                "solver": out["solver_final"],
                "target_hat_prev_xyz": target_hat_prev,
                "target_hat_xyz": target_hat_xyz.copy(),
                "position_error_m": pos_err_m,
                "crb_xyz_sum_at_true": crb_at_true,
                "crb_xyz_sum_at_hat": crb_at_hat,
                "localizer": loc,
            }
        )
        print(
            f"[stage {m}] Nm={nm_used}, Km={km_used}, loc={loc} E_used={e_used:.2f}, "
            f"E_left={energy_left:.2f}, status={out['status_final']}, solver={out['solver_final']}, "
            f"target_hat=({target_hat_xyz[0]:.1f},{target_hat_xyz[1]:.1f},{target_hat_xyz[2]:.1f}) "
            f"crb_plan={out['crb_final']:.2f} "
            f"pos_err={pos_err_m:.2f}m crb@true={crb_at_true:.2f} crb@hat={crb_at_hat:.2f}"
        )
        m += 1

    return {
        "num_stages": len(stage_logs),
        "stage_logs": stage_logs,
        "all_paths": all_paths,
        "all_hovers": all_hovers,
        "stage_histories": stage_histories,
        "all_hover_xyz": prev_hover_xyz,
        "coarse_hover_xyz": coarse_hover_xyz,
        "scan_energy_used": scan_energy_used,
        "target_hat_init_xyz": np.asarray(coarse["target_hat_init_xyz"], dtype=float),
        "target_hat_history": np.asarray(target_hat_history, dtype=float),
        "measured_ds_all": measured_ds_all,
        "target_hat_final_xyz": target_hat_xyz,
        "energy_left": energy_left,
        "localizer": loc,
    }


def run_method_case(
    method_name: str,
    eta: float,
    user_xy: np.ndarray,
    true_target_xyz: np.ndarray,
    nstg: int,
    etot: float,
    random_seed: int,
    cfg: sm.SimConfig,
    e_cfg: pb.EnergyConfig,
    scfg: SolverCfg,
    localizer: Literal["mle", "ekf"] = "ekf",
) -> dict:
    """Run one method case and return serializable data."""
    res = run_multistage_with_mle(
        cfg=cfg, e_cfg=e_cfg, scfg=scfg,
        user_xy=user_xy, true_target_xyz=true_target_xyz,
        eta=eta, nstg=nstg, etot=etot,
        random_seed=random_seed, localizer=localizer,
    )
    th_f = np.asarray(res["target_hat_final_xyz"], dtype=float)
    tt = np.asarray(true_target_xyz, dtype=float).reshape(3,)
    final_position_error_m = float(np.linalg.norm(th_f - tt))
    return {
        "method_name": method_name,
        "eta": float(eta),
        "localizer": str(res.get("localizer", localizer)),
        "num_stages": int(res["num_stages"]),
        "energy_left": float(res["energy_left"]),
        "target_hat_final_xyz": th_f,
        "target_hat_init_xyz": np.asarray(res["target_hat_init_xyz"], dtype=float),
        "target_hat_history": np.asarray(res["target_hat_history"], dtype=float),
        "scan_energy_used": float(res["scan_energy_used"]),
        "final_position_error_m": final_position_error_m,
        "stage_logs": res["stage_logs"],
        "all_paths": [np.asarray(p, dtype=float) for p in res["all_paths"]],
        "all_hovers": [np.asarray(h, dtype=float) for h in res["all_hovers"]],
        "all_hover_xyz": np.asarray(res["all_hover_xyz"], dtype=float),
        "coarse_hover_xyz": np.asarray(res["coarse_hover_xyz"], dtype=float),
        "stage_histories": [np.asarray(his, dtype=object) for his in res["stage_histories"]],
    }
