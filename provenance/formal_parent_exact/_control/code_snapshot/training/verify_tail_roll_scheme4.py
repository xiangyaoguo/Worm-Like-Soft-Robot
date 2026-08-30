"""Four deterministic pre-training checks for the tail-first rolling reward."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rlmm_common import add_env_package_to_path, find_project_root


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
add_env_package_to_path(PROJECT_ROOT)
from metamaterial_envs.env import metamaterial  # noqa: E402


def polyline_from_angles(angles: np.ndarray, radius: float) -> np.ndarray:
    points = [0.0 + 0.0j]
    for angle in angles:
        points.append(points[-1] + np.exp(1j * angle))
    points = np.asarray(points, dtype=np.complex64)
    points += 1j * (radius - np.imag(points[-1]))
    return points


def make_env(stage: int = 0):
    return metamaterial.env(
        num_envs=1,
        material_shape="crawler",
        num_particles=10,
        max_steps=200,
        terrain_type="flat",
        observation_func="dth_tot",
        reward_func="tail_roll_curriculum",
        control_mode="formula",
        k_action_scale=100.0,
        rolling_direction="right",
        tail_roll_observation=True,
        tail_side="left",
        tail_curl_sign="auto",
        tail_roll_stage=stage,
        init_pos_randomness=0.0,
        init_angle_range_degrees=0.0,
        init_height_jitter=0.0,
    )


def install_pose(env, points: np.ndarray) -> None:
    env.pos = np.ascontiguousarray(points[None, :], dtype=np.complex64)
    env.vel = np.zeros_like(env.pos, dtype=np.complex64)
    env.thdot = np.zeros(env.pos.shape, dtype=np.float32)
    env.mean_speed = np.zeros((1, 1), dtype=np.float32)
    env._rolling_metrics_cache = None
    env._tail_roll_metrics_cache = None
    env._tail_previous_potential = None
    env._tail_stage_success_latched[...] = False
    env._tail_cumulative_rotation[...] = 0.0
    centered = env.pos - np.mean(env.pos, axis=1, keepdims=True)
    env._tail_previous_centered_shape = centered.copy()
    env._tail_previous_support_x = None


def selected(metrics: dict[str, np.ndarray]) -> dict[str, float]:
    keys = (
        "tail_lift_score",
        "tail_forward_score",
        "head_contact_score",
        "curl_prefix_progress",
        "curl_order_penalty",
        "total_signed_curvature",
        "closure_ratio",
        "support_margin",
        "loop_gate",
    )
    result = {key: float(np.asarray(metrics[key]).reshape(-1)[0]) for key in keys}
    result["stage0_potential"] = float(metrics["tail_stage_potentials"][0, 0])
    result["stage1_potential"] = float(metrics["tail_stage_potentials"][0, 1])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("scheme4_small_tests.json"))
    args = parser.parse_args()

    env = make_env(stage=0)
    env.reset()
    straight_points = np.arange(10, dtype=np.float32).astype(np.complex64)
    straight_points += 1j * env.particle_radius
    install_pose(env, straight_points)
    straight = selected(env._compute_tail_roll_metrics())

    # Tail-first: the rear segments begin steeply downward and progressively
    # flatten toward the grounded head.  The positive curvature is therefore
    # concentrated at the configured tail and advances toward the head.
    tail_first_points = polyline_from_angles(
        np.asarray([-1.45, -1.15, -0.85, -0.55, -0.25, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        env.particle_radius,
    )
    install_pose(env, tail_first_points)
    tail_first = selected(env._compute_tail_roll_metrics())

    # Head-first: the tail stays flat while curvature appears only near the
    # head.  It may be curved, but it must score below the ordered tail-first
    # posture.
    head_first_points = polyline_from_angles(
        np.asarray([0.0, 0.0, 0.0, 0.0, 0.25, 0.55, 0.85, 1.15, 1.45], dtype=np.float32),
        env.particle_radius,
    )
    install_pose(env, head_first_points)
    head_first = selected(env._compute_tail_roll_metrics())

    # Static nearly closed loop: hold exactly the same pose for 100 reward
    # evaluations.  Potential-difference shaping should not pay a persistent
    # positive reward merely for remaining closed.
    env.set_tail_roll_stage(2)
    loop_angles = np.linspace(-0.90 * np.pi, 0.90 * np.pi, 10, dtype=np.float32)
    loop_points = (1.5 * np.exp(1j * loop_angles)).astype(np.complex64)
    loop_points += 1j * (env.particle_radius - np.min(np.imag(loop_points)))
    install_pose(env, loop_points)
    loop_metrics = selected(env._compute_tail_roll_metrics())
    rewards = []
    for _ in range(100):
        env._rolling_metrics_cache = None
        env._tail_roll_metrics_cache = None
        reward = env._reward_func().detach().cpu().numpy().reshape(-1)[0]
        rewards.append(float(reward))

    checks = {
        "straight_is_neutral": (
            straight["tail_lift_score"] < 0.05
            and straight["curl_prefix_progress"] < 0.05
            and straight["closure_ratio"] > 0.90
        ),
        "tail_first_is_rewarded": (
            tail_first["tail_lift_score"] > 0.30
            and tail_first["curl_prefix_progress"] > 0.15
            and tail_first["stage0_potential"] > straight["stage0_potential"] + 0.10
        ),
        "head_first_is_rejected": (
            tail_first["stage0_potential"] > head_first["stage0_potential"] + 0.10
            and tail_first["stage1_potential"] > head_first["stage1_potential"] + 0.10
            and head_first["curl_order_penalty"] > tail_first["curl_order_penalty"]
        ),
        "static_loop_cannot_farm_reward": (
            loop_metrics["closure_ratio"] < 0.25
            and max(rewards[1:]) <= 1e-5
            and sum(rewards[1:]) <= 1e-4
        ),
    }
    report = {
        "method": {
            "tail_side": "left",
            "rolling_direction": "right",
            "curl_sign": "auto",
            "static_loop_steps": 100,
        },
        "straight": straight,
        "tail_first": tail_first,
        "head_first": head_first,
        "static_loop": {
            **loop_metrics,
            "first_reward": rewards[0],
            "max_reward_after_first": max(rewards[1:]),
            "sum_reward_after_first": sum(rewards[1:]),
            "mean_reward_after_first": float(np.mean(rewards[1:])),
        },
        "checks": checks,
        "all_passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
