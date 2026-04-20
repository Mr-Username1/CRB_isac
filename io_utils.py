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

    meta = {
        "npz_file": str(output_npz_path),
        "num_methods": len(results_to_save),
        "methods": [r["method_name"] for r in results_to_save],
        "scenario_name": cfg.scenario_name,
        "noise_factor_a": cfg.a,
        "gp_scale": cfg.gp_scale,
        "save_full_trajectories": bool(save_full_trajectories),
    }
    meta_json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
