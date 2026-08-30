"""Vectorized open-loop tests and parameter search for the tail-wave controller."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from rlmm_common import add_env_package_to_path, find_project_root


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
add_env_package_to_path(PROJECT_ROOT)
from metamaterial_envs.env import metamaterial  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = itertools.product(
        (1.5, 2.0, 2.5),          # amplitude
        (0.70, 0.95, 1.10),       # centre end
        (0.08, 0.14),             # width
        (0.65, 0.90),             # hold
        (8.0, 12.0),              # kp
        (0.4, 1.0),               # kd
    )
    candidates = [
        {
            "amplitude": a,
            "center_start": 0.05,
            "center_end": c,
            "width": w,
            "hold": h,
            "kp": kp,
            "kd": kd,
            "travel_steps": args.steps,
        }
        for a, c, w, h, kp, kd in values
    ]
    env = metamaterial.env(
        num_envs=len(candidates),
        material_shape="crawler",
        num_particles=10,
        max_steps=args.steps,
        terrain_type="flat",
        observation_func="dth_tot",
        reward_func="tail_roll_curriculum",
        control_mode="tail_wave",
        tail_roll_observation=True,
        tail_side="left",
        tail_curl_sign="auto",
        tail_roll_stage=0,
        tail_roll_potential_gamma=1.0,
        init_pos_randomness=0.0,
    )
    td = env.reset()
    peak = {
        name: np.zeros(len(candidates), dtype=np.float32)
        for name in ("tail_lift_score", "tail_forward_score", "curl_prefix_progress", "tail_stage_success")
    }
    mean_contact = np.zeros(len(candidates), dtype=np.float64)
    min_closure = np.ones(len(candidates), dtype=np.float32)
    amplitudes = np.asarray([p["amplitude"] for p in candidates], dtype=np.float32)
    center_start = np.asarray([p["center_start"] for p in candidates], dtype=np.float32)
    center_end = np.asarray([p["center_end"] for p in candidates], dtype=np.float32)
    widths = np.asarray([p["width"] for p in candidates], dtype=np.float32)
    holds = np.asarray([p["hold"] for p in candidates], dtype=np.float32)
    kps = np.asarray([p["kp"] for p in candidates], dtype=np.float32)
    kds = np.asarray([p["kd"] for p in candidates], dtype=np.float32)

    for step in range(args.steps):
        progress = np.float32(step / max(1, args.steps - 1))
        smooth = progress * progress * (np.float32(3.0) - np.float32(2.0) * progress)
        centers = center_start + smooth * (center_end - center_start)
        action = np.stack((amplitudes, centers, widths, holds, kps, kds), axis=1)[:, None, :]
        td.set(env.action_key, torch.as_tensor(action, dtype=torch.float32))
        td = env.step(td)["next"]
        metrics = env._compute_tail_roll_metrics()
        for name in peak:
            peak[name] = np.maximum(peak[name], metrics[name][:, 0])
        mean_contact += metrics["head_contact_score"][:, 0]
        min_closure = np.minimum(min_closure, metrics["closure_ratio"][:, 0])

    mean_contact /= args.steps
    results = []
    for index, params in enumerate(candidates):
        score = (
            0.45 * peak["tail_lift_score"][index]
            + 0.35 * peak["tail_forward_score"][index]
            + 0.15 * peak["curl_prefix_progress"][index]
            + 0.05 * mean_contact[index]
        )
        results.append(
            {
                **params,
                "score": float(score),
                "peak_tail_lift_score": float(peak["tail_lift_score"][index]),
                "peak_tail_forward_score": float(peak["tail_forward_score"][index]),
                "peak_curl_prefix_progress": float(peak["curl_prefix_progress"][index]),
                "mean_head_contact_score": float(mean_contact[index]),
                "min_closure_ratio": float(min_closure[index]),
                "stage0_success": bool(peak["tail_stage_success"][index] > 0.5),
            }
        )
    results.sort(key=lambda row: (row["stage0_success"], row["score"]), reverse=True)
    best = results[0]
    checks = {
        "single_global_six_dimensional_action": tuple(env.action_spec[env.action_key].shape[-2:]) == (1, 6),
        "startup_from_exactly_straight": best["peak_tail_lift_score"] > 0.10,
        "tail_to_head_propagation": best["peak_curl_prefix_progress"] > 0.08,
        "tail_moves_forward": best["peak_tail_forward_score"] > 0.10,
        "parameter_search_found_candidate": best["score"] > 0.25,
    }
    report = {
        "method": "vectorized_tail_wave_open_loop_search",
        "steps": args.steps,
        "candidate_count": len(candidates),
        "teacher": {key: best[key] for key in (
            "amplitude", "center_start", "center_end", "width", "hold", "kp", "kd", "travel_steps"
        )},
        "best_result": best,
        "top_10": results[:10],
        "checks": checks,
        "all_passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if hasattr(env, "close"):
        env.close()
    if not report["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
