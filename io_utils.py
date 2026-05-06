from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import system_model as sm


def save_results_bundle(
    output_npz_path: str | Path,
    meta_json_path: str | Path,
    results: list[dict],
    user_xy: np.ndarray,
    true_target_xy: np.ndarray,
    cfg: sm.SimConfig,
    save_full_trajectories: bool = True,
) -> None:
    """Save experiment results and metadata."""
    output_npz_path = Path(output_npz_path)
    meta_json_path = Path(meta_json_path)
    output_npz_path.parent.mkdir(parents=True, exist_ok=True)
    meta_json_path.parent.mkdir(parents=True, exist_ok=True)

    results_to_save = []
    for r in results:
        rc = dict(r)
        if not save_full_trajectories:
            rc.pop("all_paths", None)
            rc.pop("all_hovers", None)
        results_to_save.append(rc)

    np.savez(
        output_npz_path,
        results=np.array(results_to_save, dtype=object),
        user_xy=np.asarray(user_xy, dtype=float),
        true_target_xy=np.asarray(true_target_xy, dtype=float),
    )

    localizer_meta = None
    if results_to_save:
        localizer_meta = results_to_save[0].get("localizer")

    meta = {
        "npz_file": str(output_npz_path),
        "num_methods": len(results_to_save),
        "methods": [r["method_name"] for r in results_to_save],
        "localizer": localizer_meta,
        "scenario_name": cfg.scenario_name,
        "noise_factor_a": cfg.a,
        "gp_scale": cfg.gp_scale,
        "save_full_trajectories": bool(save_full_trajectories),
        # Geometry / trajectory (for plotting and downstream tools; matches simulation)
        "xB": float(cfg.xB),
        "yB": float(cfg.yB),
        "Lx": float(cfg.Lx),
        "Ly": float(cfg.Ly),
        "H": float(cfg.H),
        "mu": int(cfg.mu),
        "Tf": float(cfg.Tf),
        "Th": float(cfg.Th),
        "Vmax": float(cfg.Vmax),
        "Vstr": float(cfg.Vstr),
        # Communication / sensing raw knobs (rebuild cfg without guessing)
        "P_dbm": float(cfg.P_dbm),
        "B": float(cfg.B),
        "N0_dbm_per_hz": float(cfg.N0_dbm_per_hz),
        "alpha0_db": float(cfg.alpha0_db),
        "beta0_db": float(cfg.beta0_db),
    }
    meta_json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
