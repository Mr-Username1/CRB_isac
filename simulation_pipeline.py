from __future__ import annotations

import numpy as np
import system_model as sm
import problem as pb
from p2_solver import StageData, SolverCfg, solve_p2m_sca


def simulate_range_measurements(
    hover_xy: np.ndarray,
    true_target_xy: np.ndarray,
    cfg: sm.SimConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate noisy distance measurements d_hat(k)=d(k)+w(k)."""
    ds_true = sm.ds_uav_to_target(hover_xy, true_target_xy, cfg.H)
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


def mle_grid_search(
    measured_ds: np.ndarray,
    hover_xy: np.ndarray,
    cfg: sm.SimConfig,
    coarse_step: float = 200.0,
    fine_step: float = 1.0,
    fine_radius: float = 200.0,
) -> np.ndarray:
    """MLE target estimate via two-level grid search."""
    measured_ds = np.asarray(measured_ds, dtype=float).reshape(-1)
    hover_xy = np.asarray(hover_xy, dtype=float)

    def neg_log_like(x: float, y: float) -> float:
        target = np.array([x, y], dtype=float)
        h_est = cfg.H + cfg.model_mismatch_h
        beta0_est = cfg.beta0 * (10.0 ** (cfg.model_mismatch_beta0_db / 10.0))
        ds_model = sm.ds_uav_to_target(hover_xy, target, h_est)
        g_model = sm.channel_gain_sensing(ds_model, beta0_est)
        sigma2 = np.maximum(sm.sigma2_measurement_from_g(g_model, cfg), 1e-12)
        resid2 = (measured_ds - ds_model) ** 2
        return float(0.5 * np.sum(np.log(sigma2) + resid2 / sigma2))

    x_grid = np.arange(0.0, cfg.Lx + 1e-9, coarse_step)
    y_grid = np.arange(0.0, cfg.Ly + 1e-9, coarse_step)
    best = (np.inf, 0.0, 0.0)
    for x in x_grid:
        for y in y_grid:
            val = neg_log_like(x, y)
            if val < best[0]:
                best = (val, x, y)

    x0, y0 = best[1], best[2]
    xf_l = max(0.0, x0 - fine_radius)
    xf_r = min(cfg.Lx, x0 + fine_radius)
    yf_l = max(0.0, y0 - fine_radius)
    yf_r = min(cfg.Ly, y0 + fine_radius)
    x_fine = np.arange(xf_l, xf_r + 1e-9, fine_step)
    y_fine = np.arange(yf_l, yf_r + 1e-9, fine_step)
    for x in x_fine:
        for y in y_fine:
            val = neg_log_like(x, y)
            if val < best[0]:
                best = (val, x, y)
    return np.array([best[1], best[2]], dtype=float)


def build_coarse_scan_hover_points(
    cfg: sm.SimConfig,
    nx: int = 2,
    ny: int = 2,
    center_xy: np.ndarray | None = None,
    span_xy: tuple[float, float] = (40.0, 40.0),
) -> np.ndarray:
    """Build a local coarse scan grid of hover points."""
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
            points.append([x, y])
    return np.asarray(points, dtype=float)


def compute_scan_energy(
    hover_xy: np.ndarray,
    start_xy: np.ndarray,
    cfg: sm.SimConfig,
    e_cfg: pb.EnergyConfig,
) -> float:
    """Compute flying + hovering energy for a coarse scan path."""
    hover_xy = np.asarray(hover_xy, dtype=float)
    start_xy = np.asarray(start_xy, dtype=float).reshape(2,)
    if hover_xy.size == 0:
        return 0.0
    V = sm.compute_velocities(hover_xy, start_xy, cfg.Tf)
    v_norm = np.linalg.norm(V, axis=1)
    P_fly = pb.propulsion_power(v_norm, e_cfg)
    P_hover0 = pb.propulsion_power(np.array([0.0]), e_cfg)[0]
    E = cfg.Tf * float(P_fly.sum()) + cfg.Th * hover_xy.shape[0] * float(P_hover0)
    return float(E)


def run_initial_coarse_scan(
    cfg: sm.SimConfig,
    e_cfg: pb.EnergyConfig,
    true_target_xy: np.ndarray,
    rng: np.random.Generator,
    start_xy: np.ndarray | None = None,
    nx: int = 2,
    ny: int = 2,
) -> dict:
    """Run coarse scan and return initial target estimate."""
    if start_xy is None:
        start_xy = np.array([cfg.xB, cfg.yB], dtype=float)
    start_xy = np.asarray(start_xy, dtype=float).reshape(2,)
    coarse_hover_xy = build_coarse_scan_hover_points(cfg, nx=nx, ny=ny, center_xy=start_xy)
    measured_ds = simulate_range_measurements(coarse_hover_xy, true_target_xy, cfg, rng)
    target_hat_init_xy = mle_grid_search(
        measured_ds=measured_ds,
        hover_xy=coarse_hover_xy,
        cfg=cfg,
        coarse_step=80.0,
        fine_step=20.0,
        fine_radius=150.0,
    )
    E_scan = compute_scan_energy(coarse_hover_xy, start_xy, cfg, e_cfg)
    return {
        "coarse_hover_xy": coarse_hover_xy,
        "measured_ds": measured_ds,
        "target_hat_init_xy": target_hat_init_xy,
        "scan_energy_used": E_scan,
    }


def run_multistage_with_mle(
    cfg: sm.SimConfig,
    e_cfg: pb.EnergyConfig,
    scfg: SolverCfg,
    user_xy: np.ndarray,
    true_target_xy: np.ndarray,
    eta: float,
    nstg: int,
    etot: float,
    random_seed: int = 1,
    coarse_scan_nx: int = 2,
    coarse_scan_ny: int = 2,
) -> dict:
    """Run multi-stage trajectory design and stage-wise target MLE updates."""
    rng = np.random.default_rng(random_seed)
    start_xy0 = np.array([cfg.xB, cfg.yB], dtype=float)
    coarse = run_initial_coarse_scan(
        cfg=cfg,
        e_cfg=e_cfg,
        true_target_xy=true_target_xy,
        rng=rng,
        start_xy=start_xy0,
        nx=coarse_scan_nx,
        ny=coarse_scan_ny,
    )
    coarse_hover_xy = coarse["coarse_hover_xy"]
    measured_ds_all = np.asarray(coarse["measured_ds"], dtype=float).copy()
    target_hat_xy = np.asarray(coarse["target_hat_init_xy"], dtype=float).copy()
    scan_energy_used = float(coarse["scan_energy_used"])

    m = 1
    start_xy = coarse_hover_xy[-1, :].copy()
    prev_hover_xy = coarse_hover_xy.copy()
    energy_left = float(etot - scan_energy_used)
    if energy_left < 0.0:
        energy_left = 0.0

    stage_logs = []
    all_paths = []
    all_hovers = []
    stage_histories = []
    target_hat_history = [target_hat_xy.copy()]
    n_prev_total = 0
    r_prev_sum = 0.0

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
                m=m,
                Nm=nm_try,
                Km=km_try,
                Em=energy_left,
                eta=eta,
                user_xy=user_xy,
                target_hat_xy=target_hat_xy,
                start_xy=start_xy,
                prev_hover_xy=prev_hover_xy,
                N_prev_total=n_prev_total,
                R_prev_sum=r_prev_sum,
            )
            try:
                out_try = solve_p2m_sca(stage, cfg, e_cfg, scfg)
            except RuntimeError:
                continue

            if out_try["status_final"] not in ("optimal", "optimal_inaccurate"):
                continue
            s_try = out_try["S_opt"]
            e_used_try, _ = pb.stage_energy_used(s_try, start_xy, cfg, e_cfg)
            if e_used_try <= energy_left + 1e-6:
                out = out_try
                nm_used = nm_try
                km_used = km_try
                break

        if out is None:
            break

        s_opt = out["S_opt"]
        hov = out["Hov_cur"]
        e_used, _ = pb.stage_energy_used(s_opt, start_xy, cfg, e_cfg)
        energy_left -= e_used
        prev_hover_xy = np.vstack([prev_hover_xy, hov]) if prev_hover_xy.size else hov.copy()
        start_xy = s_opt[-1, :].copy()
        dc_stage = sm.dc_uav_to_user(s_opt, user_xy, cfg.H)
        h_stage = sm.channel_gain_comm(dc_stage, cfg.alpha0)
        r_prev_sum += float(np.sum(sm.rate_per_waypoint(h_stage, cfg.P_w, cfg.sigma0_sq, cfg.B)))
        n_prev_total += int(s_opt.shape[0])
        measured_ds_stage = simulate_range_measurements(hov, true_target_xy, cfg, rng)
        measured_ds_all = np.hstack([measured_ds_all, measured_ds_stage])
        target_hat_prev = target_hat_xy.copy()
        target_hat_xy = mle_grid_search(measured_ds_all, prev_hover_xy, cfg)
        target_hat_history.append(target_hat_xy.copy())

        all_paths.append(s_opt)
        all_hovers.append(hov)
        stage_histories.append(np.array(out["history"], dtype=object))
        stage_logs.append(
            {
                "stage": m,
                "Nm": nm_used,
                "Km": km_used,
                "E_used": e_used,
                "E_left": energy_left,
                "obj_final": out["obj_final"],
                "crb_final": out["crb_final"],
                "rate_final": out["rate_final"],
                "iters": len(out["history"]),
                "status": out["status_final"],
                "solver": out["solver_final"],
                "target_hat_prev_xy": target_hat_prev,
                "target_hat_xy": target_hat_xy.copy(),
            }
        )
        print(
            f"[stage {m}] Nm={nm_used}, Km={km_used}, E_used={e_used:.2f}, "
            f"E_left={energy_left:.2f}, status={out['status_final']}, solver={out['solver_final']}, "
            f"target_hat=({target_hat_xy[0]:.1f},{target_hat_xy[1]:.1f})"
            f"crb_final={out['crb_final']:.2f}"
        )
        m += 1

    return {
        "num_stages": len(stage_logs),
        "stage_logs": stage_logs,
        "all_paths": all_paths,
        "all_hovers": all_hovers,
        "stage_histories": stage_histories,
        "all_hover_xy": prev_hover_xy,
        "coarse_hover_xy": coarse_hover_xy,
        "scan_energy_used": scan_energy_used,
        "target_hat_init_xy": np.asarray(coarse["target_hat_init_xy"], dtype=float),
        "target_hat_history": np.asarray(target_hat_history, dtype=float),
        "measured_ds_all": measured_ds_all,
        "target_hat_final_xy": target_hat_xy,
        "energy_left": energy_left,
    }


def run_method_case(
    method_name: str,
    eta: float,
    user_xy: np.ndarray,
    true_target_xy: np.ndarray,
    nstg: int,
    etot: float,
    random_seed: int,
    cfg: sm.SimConfig,
    e_cfg: pb.EnergyConfig,
    scfg: SolverCfg,
) -> dict:
    """Run one method case and return serializable data."""
    res = run_multistage_with_mle(
        cfg=cfg,
        e_cfg=e_cfg,
        scfg=scfg,
        user_xy=user_xy,
        true_target_xy=true_target_xy,
        eta=eta,
        nstg=nstg,
        etot=etot,
        random_seed=random_seed,
    )
    return {
        "method_name": method_name,
        "eta": float(eta),
        "num_stages": int(res["num_stages"]),
        "energy_left": float(res["energy_left"]),
        "target_hat_final_xy": np.asarray(res["target_hat_final_xy"], dtype=float),
        "target_hat_init_xy": np.asarray(res["target_hat_init_xy"], dtype=float),
        "target_hat_history": np.asarray(res["target_hat_history"], dtype=float),
        "scan_energy_used": float(res["scan_energy_used"]),
        "stage_logs": res["stage_logs"],
        "all_paths": [np.asarray(p, dtype=float) for p in res["all_paths"]],
        "all_hovers": [np.asarray(h, dtype=float) for h in res["all_hovers"]],
        "all_hover_xy": np.asarray(res["all_hover_xy"], dtype=float),
        "coarse_hover_xy": np.asarray(res["coarse_hover_xy"], dtype=float),
        "stage_histories": [np.asarray(his, dtype=object) for his in res["stage_histories"]],
    }
