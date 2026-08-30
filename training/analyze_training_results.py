r"""Automatic thesis-style analysis for metamaterial training runs.

The script can be used in two ways:

1. Standalone after a run finishes:
       python .\training\analyze_training_results.py --run-dir .\results\my_run --terrains all

2. Programmatically from train_metamaterial.py after a checkpoint is saved.

It creates a compact analysis folder containing:
  - training_speed_curve.png from training_log.csv
  - policy heatmaps by delegating to analyze_policy_heatmaps.py
  - k1_k2_evolution.png for action/formula-channel coefficient evolution
  - equivalent_k_evolution.png for obs-channel equivalent K = tau / dtheta_tot
  - motion_frames_*.png showing rendered rollout snapshots
  - cross_terrain_evaluation.csv / .png for flat, stairs and tunnel tests
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from tensordict import TensorDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rlmm_common import (
    add_env_package_to_path,
    find_project_root,
    load_checkpoint,
    terrain_config,
    terrain_contact_mode_from_metadata,
    terrain_label,
    to_plain,
)

PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
add_env_package_to_path(PROJECT_ROOT)
from metamaterial_envs.env import metamaterial  # noqa: E402

# Reuse the policy-construction code that is already known to match the checkpoints.
from demo_metamaterial import build_policy, choose_action, enable_follow_camera  # noqa: E402
from analyze_policy_heatmaps import analyze_checkpoint, observation_from_grid  # noqa: E402


def finite_bound_or_none(value: Any) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    return parsed if np.isfinite(parsed) else None


@dataclass(frozen=True)
class TerrainArgs:
    start_stairs: float = 5.0
    step_width: float = 5.0
    step_height: float = 0.2
    steps: int = 10
    tunnel_start: float = 10.0
    tunnel_slope: float = 5.0
    tunnel_slope_height: float = 1.0
    tunnel_length: float = 10.0
    tunnel_height: float = 5.0

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "TerrainArgs":
        return cls(
            start_stairs=float(args.start_stairs),
            step_width=float(args.step_width),
            step_height=float(args.step_height),
            steps=int(args.steps),
            tunnel_start=float(args.tunnel_start),
            tunnel_slope=float(args.tunnel_slope),
            tunnel_slope_height=float(args.tunnel_slope_height),
            tunnel_length=float(args.tunnel_length),
            tunnel_height=float(args.tunnel_height),
        )

    def to_terrain_config_kwargs(self) -> dict[str, Any]:
        return {
            "start_stairs": self.start_stairs,
            "step_width": self.step_width,
            "step_height": self.step_height,
            "steps": self.steps,
            "tunnel_start": self.tunnel_start,
            "tunnel_slope": self.tunnel_slope,
            "tunnel_slope_height": self.tunnel_slope_height,
            "tunnel_length": self.tunnel_length,
            "tunnel_height": self.tunnel_height,
        }


def checkpoint_episode(path: Path) -> int:
    match = re.search(r"checkpoint_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def latest_checkpoint_in_dir(run_dir: Path) -> Path:
    candidates = sorted(run_dir.rglob("checkpoint_*.pt"), key=lambda p: (checkpoint_episode(p), p.stat().st_mtime))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint_*.pt found in {run_dir}")
    return candidates[-1]


def resolve_checkpoint(checkpoint: str | Path | None, run_dir: str | Path | None) -> Path:
    if checkpoint is not None and str(checkpoint).lower() not in {"", "latest", "none"}:
        return Path(checkpoint).resolve()
    if run_dir is not None:
        return latest_checkpoint_in_dir(Path(run_dir).resolve())
    raise ValueError("Pass either --checkpoint or --run-dir.")


def metadata_from_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    ckpt = load_checkpoint(checkpoint_path)
    return to_plain(ckpt.get("metadata", {}))


def training_terrain_name(metadata: dict[str, Any]) -> str:
    return terrain_label(str(metadata.get("terrain_type", "flat")), to_plain(metadata.get("terrain_settings", None)))


def expand_terrains(terrains: Iterable[str], metadata: dict[str, Any]) -> list[str]:
    requested = [str(t).strip().lower() for t in terrains if str(t).strip()]
    if not requested:
        requested = ["training"]
    out: list[str] = []
    for item in requested:
        if item == "all":
            out.extend(["flat", "stairs", "tunnel"])
        elif item in {"training", "checkpoint"}:
            # Preserve the sentinel until terrain_spec_for_demo so checkpoint
            # type and exact settings are used.  Converting it to only a label
            # here would rebuild stairs/tunnel with TerrainArgs defaults.
            out.append(item)
        elif item in {"flat", "stairs", "tunnel"}:
            out.append(item)
        else:
            raise ValueError(f"Unsupported terrain {item!r}. Use flat, stairs, tunnel, training, checkpoint, or all.")
    # Preserve order but remove duplicates.
    deduped: list[str] = []
    for item in out:
        if item not in deduped:
            deduped.append(item)
    return deduped


def terrain_spec_for_demo(terrain: str, metadata: dict[str, Any], terrain_args: TerrainArgs):
    terrain = terrain.lower()
    if terrain in {"training", "checkpoint"}:
        return metadata.get("terrain_type", "flat"), to_plain(metadata.get("terrain_settings", None)), training_terrain_name(metadata)
    terrain_type, terrain_settings = terrain_config(terrain, **terrain_args.to_terrain_config_kwargs())
    return terrain_type, terrain_settings, terrain


def build_demo_env(metadata: dict[str, Any], terrain: str, terrain_args: TerrainArgs, *, max_steps: int, render_mode: str = "rgb_array", num_envs: int = 1, window_width: int = 1000, window_height: int = 420):
    terrain_type, terrain_settings, resolved_name = terrain_spec_for_demo(terrain, metadata, terrain_args)
    # This is checkpoint-controlled for evaluation parity.  Historical
    # checkpoints without the field deliberately retain legacy_flat behavior.
    terrain_contact_mode = terrain_contact_mode_from_metadata(metadata)
    material_shape = metadata.get("scenario", metadata.get("robot", "crawler"))
    num_particles = int(metadata.get("n_particles", metadata.get("num_particles", 13)))
    observation_func = str(metadata.get("observation_func", "dth_tot"))
    control_mode = str(metadata.get("control_mode", metadata.get("control_channel", "direct")))
    feedback_gain = 1.0
    max_control_gain = float(metadata.get("max_control_gain", metadata.get("coefficient_limit", 9.0)))
    k1_min = finite_bound_or_none(metadata.get("k1_min", None))
    k1_max = finite_bound_or_none(metadata.get("k1_max", None))
    k2_min = finite_bound_or_none(metadata.get("k2_min", None))
    k2_max = finite_bound_or_none(metadata.get("k2_max", None))
    fix_k1 = bool(metadata.get("fix_k1", metadata.get("formula_fix_k1", False)))
    fixed_k1 = float(metadata.get("fixed_k1", -5.0))
    fix_k2 = bool(metadata.get("fix_k2", metadata.get("formula_fix_k2", False)))
    fixed_k2 = float(metadata.get("fixed_k2", 0.0))
    k_action_scale = float(metadata.get("k_action_scale", metadata.get("formula_action_scale", 1.0)))
    min_k2_magnitude = float(metadata.get("min_k2_magnitude", 1e-3))
    passive_kappa = float(metadata.get("passive_kappa", 4.0))
    # Non-scratch checkpoints explicitly store both alpha fields as ``None``.
    # Treat that as the disabled value instead of calling ``float(None)``.
    scratch_wr_alpha_value = metadata.get("scratch_wr_current_alpha")
    if scratch_wr_alpha_value is None:
        scratch_wr_alpha_value = metadata.get("scratch_wr_initial_alpha")
    scratch_wr_alpha = 0.0 if scratch_wr_alpha_value is None else float(scratch_wr_alpha_value)
    reward_func = str(metadata.get("reward_func", "horizontal_speed"))
    rolling_observation = bool(metadata.get("rolling_observation", False))
    rolling_direction = str(metadata.get("rolling_direction", "right"))
    rolling_curl_episodes = int(metadata.get("rolling_curl_episodes", 500))
    rolling_transition_episodes = int(metadata.get("rolling_transition_episodes", 300))
    rolling_speed_ref_x100 = float(metadata.get("rolling_speed_ref_x100", 2.0))
    rolling_omega_ref = float(metadata.get("rolling_omega_ref", 1.0))
    rolling_reward_scale = float(metadata.get("rolling_reward_scale", 3.0))
    action_smoothness_weight = float(metadata.get("action_smoothness_weight", 0.0))
    tail_roll_observation = bool(metadata.get("tail_roll_observation", False))
    tail_side = str(metadata.get("tail_side", "left"))
    tail_curl_sign = metadata.get("tail_curl_sign", "auto")
    tail_roll_stage = int(metadata.get("tail_roll_stage", 0))
    tail_roll_reward_scale = float(metadata.get("tail_roll_reward_scale", 3.0))
    tail_roll_potential_gamma = float(metadata.get("tail_roll_potential_gamma", 1.0))
    tail_roll_contact_margin = float(metadata.get("tail_roll_contact_margin", 0.05))
    tail_roll_curl_reference = np.deg2rad(
        float(metadata.get("tail_roll_curl_reference_degrees", 60.0))
    )
    # Opt-in fast-forward-roll-v2 settings.  Every legacy checkpoint falls
    # through to the environment defaults, preserving its observation shape
    # and reward behaviour.
    fast_forward_observation = bool(metadata.get("fast_forward_observation", False))
    fast_forward_reward_scale = float(metadata.get("fast_forward_reward_scale", 1.0))
    fast_forward_event_degrees = float(metadata.get("fast_forward_event_degrees", 60.0))
    fast_forward_event_forward_fraction = float(
        metadata.get("fast_forward_event_forward_fraction", 0.08)
    )
    fast_forward_event_contact_nodes = float(
        metadata.get("fast_forward_event_contact_nodes", 1.5)
    )
    fast_forward_direction_fraction = float(
        metadata.get("fast_forward_direction_fraction", 0.65)
    )
    fast_forward_event_target_steps = int(
        metadata.get("fast_forward_event_target_steps", 250)
    )
    fast_forward_launch_lift = float(metadata.get("fast_forward_launch_lift", 0.20))
    fast_forward_launch_forward = float(
        metadata.get("fast_forward_launch_forward", 0.10)
    )
    fast_forward_launch_curl = float(metadata.get("fast_forward_launch_curl", 0.12))
    fast_forward_launch_head_contact = float(
        metadata.get("fast_forward_launch_head_contact", 0.50)
    )
    fast_forward_launch_hold_steps = int(
        metadata.get("fast_forward_launch_hold_steps", 8)
    )
    fast_forward_stall_steps = int(metadata.get("fast_forward_stall_steps", 150))
    fast_forward_rotation_step_ref_degrees = float(
        metadata.get("fast_forward_rotation_step_ref_degrees", 2.0)
    )
    fast_forward_translation_step_ref = float(
        metadata.get("fast_forward_translation_step_ref", 0.002)
    )
    scratch_wr_v2 = bool(metadata.get("scratch_wr_v2", False))
    scratch_wr_v2_sync_dense_weight = float(
        metadata.get("scratch_wr_v2_sync_dense_weight", 0.02)
    )
    scratch_wr_v2_penalty_start_scale = float(
        metadata.get("scratch_wr_v2_penalty_start_scale", 0.20)
    )
    scratch_wr_v2_penalty_anneal_batches = int(
        metadata.get("scratch_wr_v2_penalty_anneal_batches", 200)
    )
    scratch_wr_v2_wave_ema_beta = float(
        metadata.get("scratch_wr_v2_wave_ema_beta", 0.90)
    )
    init_pos_randomness = float(metadata.get("init_pos_randomness", 0.01))
    init_angle_range_degrees = float(metadata.get("init_angle_range_degrees", 0.0))
    init_height_jitter = float(metadata.get("init_height_jitter", 0.0))
    particle_mass = float(metadata.get("particle_mass", 0.2))
    ground_stiffness = float(metadata.get("ground_stiffness", 1e3))
    ground_damping = float(metadata.get("ground_damping", 5.0))

    env = metamaterial.env(
        num_envs=num_envs,
        material_shape=material_shape,
        num_particles=num_particles,
        max_steps=max_steps,
        observation_func=observation_func,
        terrain_type=terrain_type,
        terrain_settings=terrain_settings,
        terrain_contact_mode=terrain_contact_mode,
        control_mode=control_mode,
        feedback_gain=feedback_gain,
        max_control_gain=max_control_gain,
        k1_min=k1_min,
        k1_max=k1_max,
        k2_min=k2_min,
        k2_max=k2_max,
        k_action_scale=k_action_scale,
        fix_k1=fix_k1,
        fixed_k1=fixed_k1,
        fix_k2=fix_k2,
        fixed_k2=fixed_k2,
        min_k2_magnitude=min_k2_magnitude,
        passive_kappa=passive_kappa,
        scratch_wr_alpha=scratch_wr_alpha,
        scratch_wr_v2=scratch_wr_v2,
        scratch_wr_v2_sync_dense_weight=scratch_wr_v2_sync_dense_weight,
        scratch_wr_v2_penalty_start_scale=scratch_wr_v2_penalty_start_scale,
        scratch_wr_v2_penalty_anneal_batches=scratch_wr_v2_penalty_anneal_batches,
        scratch_wr_v2_wave_ema_beta=scratch_wr_v2_wave_ema_beta,
        reward_func=reward_func,
        rolling_observation=rolling_observation,
        rolling_direction=rolling_direction,
        rolling_curl_episodes=rolling_curl_episodes,
        rolling_transition_episodes=rolling_transition_episodes,
        rolling_speed_ref_x100=rolling_speed_ref_x100,
        rolling_omega_ref=rolling_omega_ref,
        rolling_reward_scale=rolling_reward_scale,
        action_smoothness_weight=action_smoothness_weight,
        tail_roll_observation=tail_roll_observation,
        tail_side=tail_side,
        tail_curl_sign=tail_curl_sign,
        tail_roll_stage=tail_roll_stage,
        tail_roll_reward_scale=tail_roll_reward_scale,
        tail_roll_potential_gamma=tail_roll_potential_gamma,
        tail_roll_contact_margin=tail_roll_contact_margin,
        tail_roll_curl_reference=tail_roll_curl_reference,
        fast_forward_observation=fast_forward_observation,
        fast_forward_reward_scale=fast_forward_reward_scale,
        fast_forward_event_degrees=fast_forward_event_degrees,
        fast_forward_event_forward_fraction=fast_forward_event_forward_fraction,
        fast_forward_event_contact_nodes=fast_forward_event_contact_nodes,
        fast_forward_direction_fraction=fast_forward_direction_fraction,
        fast_forward_event_target_steps=fast_forward_event_target_steps,
        fast_forward_launch_lift=fast_forward_launch_lift,
        fast_forward_launch_forward=fast_forward_launch_forward,
        fast_forward_launch_curl=fast_forward_launch_curl,
        fast_forward_launch_head_contact=fast_forward_launch_head_contact,
        fast_forward_launch_hold_steps=fast_forward_launch_hold_steps,
        fast_forward_stall_steps=fast_forward_stall_steps,
        fast_forward_rotation_step_ref_degrees=fast_forward_rotation_step_ref_degrees,
        fast_forward_translation_step_ref=fast_forward_translation_step_ref,
        init_pos_randomness=init_pos_randomness,
        init_angle_range_degrees=init_angle_range_degrees,
        init_height_jitter=init_height_jitter,
        particle_mass=particle_mass,
        ground_stiffness=ground_stiffness,
        ground_damping=ground_damping,
        render_mode=render_mode,
        window_width=window_width,
        window_height=window_height,
        render_text_lines=[],
    )
    if hasattr(env, "set_curriculum_episode"):
        scratch_wr_current_batch = metadata.get("scratch_wr_current_batch")
        env.set_curriculum_episode(0 if scratch_wr_current_batch is None else int(scratch_wr_current_batch))
    return env, resolved_name, terrain_type, terrain_settings


def load_policy_for_env(checkpoint_path: Path, env, metadata: dict[str, Any]):
    ckpt = load_checkpoint(checkpoint_path)
    meta = dict(metadata)
    meta["algorithm"] = meta.get("algorithm", "ppo")
    policy = build_policy(env, meta)
    policy.load_state_dict(ckpt["policy"], strict=True)
    policy.eval()
    return policy


def simulate_policy(
    checkpoint_path: Path,
    metadata: dict[str, Any],
    terrain: str,
    terrain_args: TerrainArgs,
    *,
    steps: int = 300,
    policy_mode: str = "deterministic",
    collect_frames: int = 0,
    follow_camera: bool = True,
    num_envs: int = 1,
    window_width: int = 1000,
    window_height: int = 420,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    env, terrain_name, terrain_type, terrain_settings = build_demo_env(
        metadata,
        terrain,
        terrain_args,
        max_steps=steps,
        render_mode="rgb_array",
        num_envs=num_envs,
        window_width=window_width,
        window_height=window_height,
    )
    if follow_camera:
        enable_follow_camera(env)
    policy = load_policy_for_env(checkpoint_path, env, metadata)

    frame_indices: set[int] = set()
    if collect_frames > 0:
        frame_indices = set(np.linspace(0, max(0, steps - 1), collect_frames, dtype=int).tolist())

    td = env.reset()
    speeds: list[float] = []
    frames: list[np.ndarray] = []
    for step in range(steps):
        action_td = choose_action(policy, td, policy_mode)
        td = env.step(action_td)["next"]
        speed = float(td["log_info", "speed"].mean().item())
        speeds.append(speed)
        if step in frame_indices:
            env.render_text_lines = [
                f"{metadata.get('scenario', 'robot')} | {metadata.get('algorithm', 'policy')} | trained: {training_terrain_name(metadata)} | demo: {terrain_name}",
                f"step {step + 1}/{steps} | speed x100: {speed * 100.0:.3f}",
            ]
            frame = env.render()
            if frame is not None:
                frames.append(np.asarray(frame).copy())

    if hasattr(env, "close"):
        try:
            env.close()
        except Exception:
            pass

    raw_mean = float(np.mean(speeds)) if speeds else 0.0
    return {
        "checkpoint": str(checkpoint_path),
        "trained_terrain": training_terrain_name(metadata),
        "demo_terrain": terrain_name,
        "terrain_type": terrain_type,
        "terrain_settings": to_plain(terrain_settings),
        "mean_speed_raw": raw_mean,
        "mean_speed_x100": raw_mean * 100.0,
        "steps": steps,
        "num_envs": num_envs,
    }, frames


def evaluate_checkpoint_on_terrains(
    checkpoint_path: Path,
    metadata: dict[str, Any],
    terrains: list[str],
    terrain_args: TerrainArgs,
    *,
    episodes: int = 5,
    steps: int = 300,
    policy_mode: str = "deterministic",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for terrain in terrains:
        episode_speeds: list[float] = []
        last_info: dict[str, Any] = {}
        for _ in range(max(1, episodes)):
            info, _ = simulate_policy(
                checkpoint_path,
                metadata,
                terrain,
                terrain_args,
                steps=steps,
                policy_mode=policy_mode,
                collect_frames=0,
                follow_camera=False,
                num_envs=1,
            )
            last_info = info
            episode_speeds.append(float(info["mean_speed_x100"]))
        rows.append(
            {
                **last_info,
                "eval_episodes": max(1, episodes),
                "mean_speed_x100": float(np.mean(episode_speeds)),
                "std_speed_x100": float(np.std(episode_speeds)),
            }
        )
    return rows


def evaluate_nonreciprocity_baseline(
    metadata: dict[str, Any],
    terrain: str,
    terrain_args: TerrainArgs,
    *,
    episodes: int = 3,
    steps: int = 300,
    kappa_alpha: float = -50.0,
) -> dict[str, float]:
    """Evaluate a simple non-reciprocity torque baseline on a matching robot/terrain."""
    terrain_type, terrain_settings, terrain_name = terrain_spec_for_demo(terrain, metadata, terrain_args)
    terrain_contact_mode = terrain_contact_mode_from_metadata(metadata)
    speeds: list[float] = []
    for _ in range(max(1, episodes)):
        env = metamaterial.env(
            num_envs=1,
            material_shape=metadata.get("scenario", metadata.get("robot", "crawler")),
            num_particles=int(metadata.get("n_particles", metadata.get("num_particles", 13))),
            max_steps=steps,
            observation_func="dth_tot",
            terrain_type=terrain_type,
            terrain_settings=terrain_settings,
            terrain_contact_mode=terrain_contact_mode,
            control_mode="direct",
            passive_kappa=float(metadata.get("passive_kappa", 4.0)),
            render_mode="rgb_array",
            window_width=1000,
            window_height=420,
            render_text_lines=[],
        )
        td = env.reset()
        ep_speeds: list[float] = []
        for _step in range(steps):
            obs = td["agents", "observation"]
            action = torch.clamp(float(kappa_alpha) * obs[..., 0:1], -9.0, 9.0)
            td.set(("agents", "action"), action)
            td = env.step(td)["next"]
            ep_speeds.append(float(td["log_info", "speed"].mean().item()) * 100.0)
        speeds.append(float(np.mean(ep_speeds)) if ep_speeds else 0.0)
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass
    return {
        "terrain": terrain_name,
        "mean_speed_x100": float(np.mean(speeds)) if speeds else 0.0,
        "std_speed_x100": float(np.std(speeds)) if speeds else 0.0,
        "episodes": max(1, episodes),
        "steps": steps,
        "kappa_alpha": float(kappa_alpha),
    }


def save_training_curve(run_dir: Path, output_dir: Path, *, baseline_speed_x100: float | None = None, dpi: int = 180) -> Path | None:
    log_path = run_dir / "training_log.csv"
    if not log_path.exists():
        return None

    episodes: list[int] = []
    speeds_x100: list[float] = []
    rewards: list[float] = []
    with log_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                episodes.append(int(float(row.get("episode", len(episodes) + 1))))
                if row.get("speed_x100") not in {None, ""}:
                    speeds_x100.append(float(row["speed_x100"]))
                else:
                    speeds_x100.append(float(row.get("speed_mean", 0.0)) * 100.0)
                rewards.append(float(row.get("reward_mean", "nan")))
            except Exception:
                continue

    if not episodes:
        return None

    arr = np.asarray(speeds_x100, dtype=float)
    window = max(1, min(25, len(arr) // 10 if len(arr) >= 10 else 1))
    if window > 1:
        kernel = np.ones(window, dtype=float) / float(window)
        smooth = np.convolve(arr, kernel, mode="same")
    else:
        smooth = arr

    fig, ax = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
    ax.plot(episodes, arr, alpha=0.35, label="episode speed")
    ax.plot(episodes, smooth, linewidth=2.0, label=f"moving mean ({window})" if window > 1 else "speed")
    if baseline_speed_x100 is not None:
        ax.axhline(float(baseline_speed_x100), linestyle="--", linewidth=1.2, label="simple non-reciprocity")
    ax.set_title("Horizontal speed during training")
    ax.set_xlabel("training episodes")
    ax.set_ylabel("mean speed per 100 timesteps")
    ax.grid(True, alpha=0.35)
    ax.legend()
    output_path = output_dir / "training_speed_curve.png"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def is_k1_k2_run(metadata: dict[str, Any]) -> bool:
    control_mode = str(metadata.get("control_mode", metadata.get("control_channel", "")))
    channel = str(metadata.get("channel", metadata.get("control_channel_label", ""))).lower()
    return control_mode in {"formula", "fixed_k1_k2_positive", "fixed_k1_k2_negative"} or channel in {
        "action",
        "formula",
        "k2_positive",
        "k2_negative",
    }


def is_equivalent_k_run(metadata: dict[str, Any]) -> bool:
    control_mode = str(metadata.get("control_mode", metadata.get("control_channel", "")))
    channel = str(metadata.get("channel", metadata.get("control_channel_label", ""))).lower()
    observation_func = str(metadata.get("observation_func", "")).lower()
    return control_mode == "direct" and (channel == "obs" or observation_func == "dth_tot")


def checkpoints_for_evolution(run_dir: Path, fallback_checkpoint: Path) -> list[Path]:
    candidates = sorted(run_dir.glob("checkpoint_*.pt"), key=lambda p: (checkpoint_episode(p), p.stat().st_mtime))
    if not candidates:
        candidates = [fallback_checkpoint]

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            deduped.append(resolved)
            seen.add(resolved)
    return deduped


def formula_coefficients_from_action(env, raw_action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    control_mode = str(getattr(env, "control_mode", "direct"))

    if control_mode in {"fixed_k1_k2_positive", "fixed_k1_k2_negative"}:
        k1 = np.zeros(raw_action.shape[:-1], dtype=np.float32) + float(getattr(env, "fixed_k1", -5.0))
        k2 = np.clip(
            raw_action[..., 0],
            float(getattr(env, "k2_min", -getattr(env, "max_control_gain", 9.0))),
            float(getattr(env, "k2_max", getattr(env, "max_control_gain", 9.0))),
        )
        return k1, k2

    if control_mode != "formula":
        raise ValueError(f"K1/K2 evolution is only available for formula/action control, got {control_mode!r}.")

    formula_fix_k1 = bool(getattr(env, "formula_fix_k1", getattr(env, "fix_k1", False)))
    formula_fix_k2 = bool(getattr(env, "formula_fix_k2", getattr(env, "fix_k2", False)))
    k_action_scale = float(getattr(env, "k_action_scale", getattr(env, "formula_action_scale", 1.0)))
    action_index = 0

    if formula_fix_k1:
        k1 = np.zeros(raw_action.shape[:-1], dtype=np.float32) + float(getattr(env, "fixed_k1", -5.0))
    else:
        k1 = np.clip(
            raw_action[..., action_index] * k_action_scale,
            float(getattr(env, "k1_min", -getattr(env, "max_control_gain", 9.0))),
            float(getattr(env, "k1_max", getattr(env, "max_control_gain", 9.0))),
        )
        action_index += 1

    if formula_fix_k2:
        k2 = np.zeros(raw_action.shape[:-1], dtype=np.float32) + float(getattr(env, "fixed_k2", 0.0))
    else:
        k2 = np.clip(
            raw_action[..., action_index] * k_action_scale,
            float(getattr(env, "k2_min", -getattr(env, "max_control_gain", 9.0))),
            float(getattr(env, "k2_max", getattr(env, "max_control_gain", 9.0))),
        )

    return k1, k2


def evaluate_k1_k2_with_policy(
    env,
    policy,
    metadata: dict[str, Any],
    dtheta_tot_values: np.ndarray,
    *,
    policy_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    obs_dim = env.observation_spec["agents", "observation"].shape[-1]
    observation_func = str(metadata.get("observation_func", "dth_tot"))

    x_next = dtheta_tot_values.astype(np.float32)
    y_prev = np.zeros_like(x_next, dtype=np.float32)
    obs_np = observation_from_grid(observation_func, x_next, y_prev, theta_dot=0.0, obs_dim=obs_dim)
    obs = torch.as_tensor(obs_np, dtype=torch.float32)
    obs_all = obs[:, None, :].repeat(1, env.num_agents, 1)
    td = TensorDict(
        {
            "agents": TensorDict(
                {"observation": obs_all},
                batch_size=[len(dtheta_tot_values), env.num_agents],
            )
        },
        batch_size=[len(dtheta_tot_values)],
    )
    out = choose_action(policy, td, policy_mode)
    raw_action = out[env.action_key].detach().cpu().numpy()
    k1, k2 = formula_coefficients_from_action(env, raw_action)
    return k1.astype(np.float32), k2.astype(np.float32)


def evaluate_direct_torque_with_policy(
    env,
    policy,
    metadata: dict[str, Any],
    dtheta_tot_values: np.ndarray,
    *,
    policy_mode: str,
) -> np.ndarray:
    obs_dim = env.observation_spec["agents", "observation"].shape[-1]
    observation_func = str(metadata.get("observation_func", "dth_tot"))

    x_next = dtheta_tot_values.astype(np.float32)
    y_prev = np.zeros_like(x_next, dtype=np.float32)
    obs_np = observation_from_grid(observation_func, x_next, y_prev, theta_dot=0.0, obs_dim=obs_dim)
    obs = torch.as_tensor(obs_np, dtype=torch.float32)
    obs_all = obs[:, None, :].repeat(1, env.num_agents, 1)
    td = TensorDict(
        {
            "agents": TensorDict(
                {"observation": obs_all},
                batch_size=[len(dtheta_tot_values), env.num_agents],
            )
        },
        batch_size=[len(dtheta_tot_values)],
    )
    out = choose_action(policy, td, policy_mode)
    return out[env.action_key].detach().cpu().numpy()[..., 0].astype(np.float32)


def equivalent_k_from_torque(torque: np.ndarray, dtheta_tot_values: np.ndarray, *, zero_epsilon: float = 1e-3) -> np.ndarray:
    denom = dtheta_tot_values.astype(np.float32)[:, None]
    k = np.full_like(torque, np.nan, dtype=np.float32)
    valid = np.abs(denom) >= float(zero_epsilon)
    np.divide(torque, denom, out=k, where=valid)
    return k


def nanmean_axis(values: np.ndarray, axis=None) -> np.ndarray:
    finite = np.isfinite(values)
    count = np.sum(finite, axis=axis)
    total = np.sum(np.where(finite, values, 0.0), axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)


def nanstd_all(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.std(finite)) if finite.size else float("nan")


def k1_k2_env_info(env, terrain_name: str, terrain_type: Any, terrain_settings: Any) -> dict[str, Any]:
    return {
        "terrain_name": terrain_name,
        "terrain_type": terrain_type,
        "terrain_settings": to_plain(terrain_settings),
        "num_agents": int(env.num_agents),
        "control_mode": str(getattr(env, "control_mode", "")),
        "fix_k1": bool(getattr(env, "formula_fix_k1", getattr(env, "fix_k1", False))),
        "fixed_k1": float(getattr(env, "fixed_k1", -5.0)),
        "fix_k2": bool(getattr(env, "formula_fix_k2", getattr(env, "fix_k2", False))),
        "fixed_k2": float(getattr(env, "fixed_k2", 0.0)),
        "k1_min": float(getattr(env, "k1_min", -getattr(env, "max_control_gain", 9.0))),
        "k1_max": float(getattr(env, "k1_max", getattr(env, "max_control_gain", 9.0))),
        "k2_min": float(getattr(env, "k2_min", -getattr(env, "max_control_gain", 9.0))),
        "k2_max": float(getattr(env, "k2_max", getattr(env, "max_control_gain", 9.0))),
        "k_action_scale": float(getattr(env, "k_action_scale", getattr(env, "formula_action_scale", 1.0))),
    }


def evaluate_k1_k2_on_observation_line(
    checkpoint_path: Path,
    metadata: dict[str, Any],
    terrain_args: TerrainArgs,
    dtheta_tot_values: np.ndarray,
    *,
    policy_mode: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    env, terrain_name, terrain_type, terrain_settings = build_demo_env(
        metadata,
        "training",
        terrain_args,
        max_steps=1,
        render_mode="rgb_array",
        num_envs=1,
    )
    policy = load_policy_for_env(checkpoint_path, env, metadata)
    k1, k2 = evaluate_k1_k2_with_policy(env, policy, metadata, dtheta_tot_values, policy_mode=policy_mode)
    info = k1_k2_env_info(env, terrain_name, terrain_type, terrain_settings)

    if hasattr(env, "close"):
        try:
            env.close()
        except Exception:
            pass

    return k1.astype(np.float32), k2.astype(np.float32), info


def symmetric_color_limits(values: np.ndarray, fallback: float = 1.0) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -fallback, fallback
    bound = float(np.max(np.abs(finite)))
    bound = max(bound, fallback, 1e-6)
    return -bound, bound


def save_k1_k2_evolution(
    run_dir: Path,
    checkpoint_path: Path,
    metadata: dict[str, Any],
    terrain_args: TerrainArgs,
    output_dir: Path,
    *,
    grid_size: int = 81,
    policy_mode: str = "deterministic",
    dpi: int = 180,
) -> list[Path]:
    if not is_k1_k2_run(metadata):
        return []

    checkpoints = checkpoints_for_evolution(run_dir, checkpoint_path)
    dtheta_tot = np.linspace(-2.0 * math.pi, 2.0 * math.pi, int(grid_size), dtype=np.float32)
    episodes: list[int] = []
    k1_surface: list[np.ndarray] = []
    k2_surface: list[np.ndarray] = []
    k1_per_joint_surface: list[np.ndarray] = []
    k2_per_joint_surface: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    per_joint_rows: list[dict[str, Any]] = []
    env, terrain_name, terrain_type, terrain_settings = build_demo_env(
        metadata,
        "training",
        terrain_args,
        max_steps=1,
        render_mode="rgb_array",
        num_envs=1,
    )
    policy = build_policy(env, metadata)
    last_info: dict[str, Any] = k1_k2_env_info(env, terrain_name, terrain_type, terrain_settings)

    try:
        for path in checkpoints:
            ckpt = load_checkpoint(path)
            ckpt_metadata = to_plain(ckpt.get("metadata", metadata))
            if not is_k1_k2_run(ckpt_metadata):
                continue
            episode = checkpoint_episode(path)
            policy.load_state_dict(ckpt["policy"], strict=True)
            policy.eval()
            k1, k2 = evaluate_k1_k2_with_policy(env, policy, ckpt_metadata, dtheta_tot, policy_mode=policy_mode)
            episodes.append(episode)
            k1_surface.append(np.mean(k1, axis=1))
            k2_surface.append(np.mean(k2, axis=1))
            k1_per_joint_surface.append(k1.copy())
            k2_per_joint_surface.append(k2.copy())
            rows.append(
                {
                    "episode": episode,
                    "checkpoint": str(path),
                    "k1_mean": float(np.mean(k1)),
                    "k1_std": float(np.std(k1)),
                    "k1_min": float(np.min(k1)),
                    "k1_max": float(np.max(k1)),
                    "k2_mean": float(np.mean(k2)),
                    "k2_std": float(np.std(k2)),
                    "k2_min": float(np.min(k2)),
                    "k2_max": float(np.max(k2)),
                    "fix_k1": bool(last_info.get("fix_k1", False)),
                    "fixed_k1": float(last_info.get("fixed_k1", 0.0)),
                    "fix_k2": bool(last_info.get("fix_k2", False)),
                    "fixed_k2": float(last_info.get("fixed_k2", 0.0)),
                }
            )
            for joint_index in range(k1.shape[1]):
                joint_k1 = k1[:, joint_index]
                joint_k2 = k2[:, joint_index]
                per_joint_rows.append(
                    {
                        "episode": episode,
                        "checkpoint": str(path),
                        "joint_index": joint_index,
                        "k1_mean": float(np.mean(joint_k1)),
                        "k1_std": float(np.std(joint_k1)),
                        "k1_min": float(np.min(joint_k1)),
                        "k1_max": float(np.max(joint_k1)),
                        "k2_mean": float(np.mean(joint_k2)),
                        "k2_std": float(np.std(joint_k2)),
                        "k2_min": float(np.min(joint_k2)),
                        "k2_max": float(np.max(joint_k2)),
                    }
                )
    finally:
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass

    if not rows:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    episode_arr = np.asarray(episodes, dtype=float)
    k1_arr = np.stack(k1_surface, axis=1)
    k2_arr = np.stack(k2_surface, axis=1)
    # [checkpoint, dtheta grid, joint]. Keep the joint axis so independent
    # policies can be inspected instead of being hidden by the legacy mean.
    k1_per_joint_arr = np.stack(k1_per_joint_surface, axis=0)
    k2_per_joint_arr = np.stack(k2_per_joint_surface, axis=0)
    joint_indices = np.arange(k1_per_joint_arr.shape[2], dtype=int)
    k1_mean = np.asarray([row["k1_mean"] for row in rows], dtype=float)
    k1_std = np.asarray([row["k1_std"] for row in rows], dtype=float)
    k2_mean = np.asarray([row["k2_mean"] for row in rows], dtype=float)
    k2_std = np.asarray([row["k2_std"] for row in rows], dtype=float)

    csv_path = save_evaluation_csv(rows, output_dir / "k1_k2_evolution_summary.csv")
    per_joint_csv_path = save_evaluation_csv(per_joint_rows, output_dir / "k1_k2_per_joint_summary.csv")
    npz_path = output_dir / "k1_k2_evolution_grid.npz"
    np.savez_compressed(
        npz_path,
        episodes=episode_arr,
        dtheta_tot=dtheta_tot,
        k1_mean_by_dtheta=k1_arr,
        k2_mean_by_dtheta=k2_arr,
        k1_mean=k1_mean,
        k1_std=k1_std,
        k2_mean=k2_mean,
        k2_std=k2_std,
        joint_indices=joint_indices,
        k1_by_checkpoint_dtheta_joint=k1_per_joint_arr,
        k2_by_checkpoint_dtheta_joint=k2_per_joint_arr,
    )

    if episode_arr.size == 1:
        x_min = float(episode_arr[0] - 0.5)
        x_max = float(episode_arr[0] + 0.5)
    else:
        x_min = float(np.min(episode_arr))
        x_max = float(np.max(episode_arr))
    extent = [x_min, x_max, float(dtheta_tot[0]), float(dtheta_tot[-1])]
    k1_vmin, k1_vmax = symmetric_color_limits(k1_arr)
    k2_vmin, k2_vmax = symmetric_color_limits(k2_arr)

    fig = plt.figure(figsize=(10.5, 8.0), constrained_layout=True)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.25, 1.25, 0.08, 1.0])
    ax_k1 = fig.add_subplot(gs[0, :])
    ax_k2 = fig.add_subplot(gs[1, :], sharex=ax_k1)
    ax_k1_summary = fig.add_subplot(gs[3, 0])
    ax_k2_summary = fig.add_subplot(gs[3, 1], sharex=ax_k1_summary)

    image_k1 = ax_k1.imshow(k1_arr, origin="lower", aspect="auto", extent=extent, cmap="coolwarm", vmin=k1_vmin, vmax=k1_vmax)
    cbar_k1 = fig.colorbar(image_k1, ax=ax_k1, pad=0.02)
    cbar_k1.set_label("K1")
    ax_k1.set_title("Evolution of K1 over training checkpoints")
    ax_k1.set_ylabel("dtheta_tot = dtheta_(i+1) - dtheta_(i-1)")

    image_k2 = ax_k2.imshow(k2_arr, origin="lower", aspect="auto", extent=extent, cmap="coolwarm", vmin=k2_vmin, vmax=k2_vmax)
    cbar_k2 = fig.colorbar(image_k2, ax=ax_k2, pad=0.02)
    cbar_k2.set_label("K2")
    ax_k2.set_title("Evolution of K2 over training checkpoints")
    ax_k2.set_xlabel("Training episode / checkpoint")
    ax_k2.set_ylabel("dtheta_tot = dtheta_(i+1) - dtheta_(i-1)")

    ax_k1_summary.plot(episode_arr, k1_mean, marker="o", label="mean K1")
    ax_k1_summary.plot(episode_arr, k1_std, marker="s", label="std K1")
    ax_k1_summary.set_title("K1 summary over observation grid")
    ax_k1_summary.set_xlabel("Training episode / checkpoint")
    ax_k1_summary.set_ylabel("value")
    ax_k1_summary.grid(True, alpha=0.3)
    ax_k1_summary.legend(fontsize=8)

    ax_k2_summary.plot(episode_arr, k2_mean, marker="o", label="mean K2")
    ax_k2_summary.plot(episode_arr, k2_std, marker="s", label="std K2")
    ax_k2_summary.set_title("K2 summary over observation grid")
    ax_k2_summary.set_xlabel("Training episode / checkpoint")
    ax_k2_summary.set_ylabel("value")
    ax_k2_summary.grid(True, alpha=0.3)
    ax_k2_summary.legend(fontsize=8)

    title = f"K1/K2 evolution: {run_dir.name}"
    subtitle = (
        f"robot={metadata.get('scenario', metadata.get('robot', 'robot'))}, "
        f"terrain={training_terrain_name(metadata)}, "
        f"channel={metadata.get('channel', 'action')}, "
        f"algorithm={metadata.get('algorithm', 'policy')}, "
        f"feedback_gain={metadata.get('feedback_gain', 1.0)}"
    )
    fig.suptitle(f"{title}\n{subtitle}", fontsize=11)

    png_path = output_dir / "k1_k2_evolution.png"
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)

    k1_joint_mean = np.mean(k1_per_joint_arr, axis=1)
    k2_joint_mean = np.mean(k2_per_joint_arr, axis=1)
    final_k1 = k1_per_joint_arr[-1].T
    final_k2 = k2_per_joint_arr[-1].T
    joint_extent = [x_min, x_max, -0.5, float(len(joint_indices)) - 0.5]
    final_extent = [float(dtheta_tot[0]), float(dtheta_tot[-1]), -0.5, float(len(joint_indices)) - 0.5]

    joint_fig, joint_axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    joint_k1_vmin, joint_k1_vmax = symmetric_color_limits(k1_per_joint_arr)
    joint_k2_vmin, joint_k2_vmax = symmetric_color_limits(k2_per_joint_arr)

    image = joint_axes[0, 0].imshow(
        k1_joint_mean.T,
        origin="lower",
        aspect="auto",
        extent=joint_extent,
        cmap="coolwarm",
        vmin=joint_k1_vmin,
        vmax=joint_k1_vmax,
    )
    joint_fig.colorbar(image, ax=joint_axes[0, 0], pad=0.02).set_label("mean K1")
    joint_axes[0, 0].set_title("K1 by joint over checkpoints\n(mean across observation grid)")
    joint_axes[0, 0].set_xlabel("training episode / checkpoint")
    joint_axes[0, 0].set_ylabel("joint index")

    image = joint_axes[0, 1].imshow(
        k2_joint_mean.T,
        origin="lower",
        aspect="auto",
        extent=joint_extent,
        cmap="coolwarm",
        vmin=joint_k2_vmin,
        vmax=joint_k2_vmax,
    )
    joint_fig.colorbar(image, ax=joint_axes[0, 1], pad=0.02).set_label("mean K2")
    joint_axes[0, 1].set_title("K2 by joint over checkpoints\n(mean across observation grid)")
    joint_axes[0, 1].set_xlabel("training episode / checkpoint")
    joint_axes[0, 1].set_ylabel("joint index")

    image = joint_axes[1, 0].imshow(
        final_k1,
        origin="lower",
        aspect="auto",
        extent=final_extent,
        cmap="coolwarm",
        vmin=joint_k1_vmin,
        vmax=joint_k1_vmax,
    )
    joint_fig.colorbar(image, ax=joint_axes[1, 0], pad=0.02).set_label("K1")
    joint_axes[1, 0].set_title(f"Final checkpoint K1 by joint (episode {int(episode_arr[-1])})")
    joint_axes[1, 0].set_xlabel("dtheta_tot")
    joint_axes[1, 0].set_ylabel("joint index")

    image = joint_axes[1, 1].imshow(
        final_k2,
        origin="lower",
        aspect="auto",
        extent=final_extent,
        cmap="coolwarm",
        vmin=joint_k2_vmin,
        vmax=joint_k2_vmax,
    )
    joint_fig.colorbar(image, ax=joint_axes[1, 1], pad=0.02).set_label("K2")
    joint_axes[1, 1].set_title(f"Final checkpoint K2 by joint (episode {int(episode_arr[-1])})")
    joint_axes[1, 1].set_xlabel("dtheta_tot")
    joint_axes[1, 1].set_ylabel("joint index")

    policy_sharing = metadata.get(
        "policy_parameter_sharing",
        "shared" if metadata.get("share_parameters_policy", metadata.get("share_policy", True)) else "independent_per_joint",
    )
    joint_fig.suptitle(f"Per-joint K1/K2 analysis: {run_dir.name}\npolicy sharing={policy_sharing}", fontsize=11)
    per_joint_png_path = output_dir / "k1_k2_per_joint_evolution.png"
    joint_fig.savefig(per_joint_png_path, dpi=dpi)
    plt.close(joint_fig)

    meta_path = safe_json_dump(
        {
            "run_dir": str(run_dir),
            "checkpoints": [str(p) for p in checkpoints],
            "episodes": episodes,
            "dtheta_tot_min": float(dtheta_tot[0]),
            "dtheta_tot_max": float(dtheta_tot[-1]),
            "grid_size": int(grid_size),
            "metadata": metadata,
            "env_info": last_info,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        output_dir / "k1_k2_evolution_metadata.json",
    )
    return [png_path, per_joint_png_path, csv_path, per_joint_csv_path, npz_path, meta_path]


def save_equivalent_k_evolution(
    run_dir: Path,
    checkpoint_path: Path,
    metadata: dict[str, Any],
    terrain_args: TerrainArgs,
    output_dir: Path,
    *,
    grid_size: int = 81,
    policy_mode: str = "deterministic",
    dpi: int = 180,
    zero_epsilon: float = 1e-3,
) -> list[Path]:
    if not is_equivalent_k_run(metadata):
        return []

    checkpoints = checkpoints_for_evolution(run_dir, checkpoint_path)
    dtheta_tot = np.linspace(-2.0 * math.pi, 2.0 * math.pi, int(grid_size), dtype=np.float32)
    episodes: list[int] = []
    k_surface: list[np.ndarray] = []
    tau_surface: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []

    env, terrain_name, terrain_type, terrain_settings = build_demo_env(
        metadata,
        "training",
        terrain_args,
        max_steps=1,
        render_mode="rgb_array",
        num_envs=1,
    )
    policy = build_policy(env, metadata)

    try:
        for path in checkpoints:
            ckpt = load_checkpoint(path)
            ckpt_metadata = to_plain(ckpt.get("metadata", metadata))
            if not is_equivalent_k_run(ckpt_metadata):
                continue
            episode = checkpoint_episode(path)
            policy.load_state_dict(ckpt["policy"], strict=True)
            policy.eval()
            tau = evaluate_direct_torque_with_policy(env, policy, ckpt_metadata, dtheta_tot, policy_mode=policy_mode)
            k = equivalent_k_from_torque(tau, dtheta_tot, zero_epsilon=zero_epsilon)
            finite_k = k[np.isfinite(k)]

            episodes.append(episode)
            k_surface.append(nanmean_axis(k, axis=1))
            tau_surface.append(np.mean(tau, axis=1))
            rows.append(
                {
                    "episode": episode,
                    "checkpoint": str(path),
                    "k_mean": float(np.mean(finite_k)) if finite_k.size else float("nan"),
                    "k_std": float(np.std(finite_k)) if finite_k.size else float("nan"),
                    "k_min": float(np.min(finite_k)) if finite_k.size else float("nan"),
                    "k_max": float(np.max(finite_k)) if finite_k.size else float("nan"),
                    "tau_mean": float(np.mean(tau)),
                    "tau_std": float(np.std(tau)),
                    "tau_min": float(np.min(tau)),
                    "tau_max": float(np.max(tau)),
                    "zero_epsilon": float(zero_epsilon),
                }
            )
    finally:
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass

    if not rows:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    episode_arr = np.asarray(episodes, dtype=float)
    k_arr = np.stack(k_surface, axis=1)
    tau_arr = np.stack(tau_surface, axis=1)
    k_mean = np.asarray([row["k_mean"] for row in rows], dtype=float)
    k_std = np.asarray([row["k_std"] for row in rows], dtype=float)
    tau_mean = np.asarray([row["tau_mean"] for row in rows], dtype=float)
    tau_std = np.asarray([row["tau_std"] for row in rows], dtype=float)

    csv_path = save_evaluation_csv(rows, output_dir / "equivalent_k_evolution_summary.csv")
    npz_path = output_dir / "equivalent_k_evolution_grid.npz"
    np.savez_compressed(
        npz_path,
        episodes=episode_arr,
        dtheta_tot=dtheta_tot,
        equivalent_k_mean_by_dtheta=k_arr,
        torque_mean_by_dtheta=tau_arr,
        k_mean=k_mean,
        k_std=k_std,
        tau_mean=tau_mean,
        tau_std=tau_std,
        zero_epsilon=float(zero_epsilon),
    )

    if episode_arr.size == 1:
        x_min = float(episode_arr[0] - 0.5)
        x_max = float(episode_arr[0] + 0.5)
    else:
        x_min = float(np.min(episode_arr))
        x_max = float(np.max(episode_arr))
    extent = [x_min, x_max, float(dtheta_tot[0]), float(dtheta_tot[-1])]
    k_vmin, k_vmax = symmetric_color_limits(k_arr)

    fig = plt.figure(figsize=(10.5, 7.0), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.55, 0.08, 1.0])
    ax_k = fig.add_subplot(gs[0, :])
    ax_summary = fig.add_subplot(gs[2, 0])
    ax_tau = fig.add_subplot(gs[2, 1], sharex=ax_summary)

    image_k = ax_k.imshow(k_arr, origin="lower", aspect="auto", extent=extent, cmap="coolwarm", vmin=k_vmin, vmax=k_vmax)
    cbar_k = fig.colorbar(image_k, ax=ax_k, pad=0.02)
    cbar_k.set_label("Equivalent K = tau / dtheta_tot")
    ax_k.set_title("Evolution of equivalent K over training checkpoints")
    ax_k.set_xlabel("Training episode / checkpoint")
    ax_k.set_ylabel("dtheta_tot = dtheta_(i+1) - dtheta_(i-1)")

    ax_summary.plot(episode_arr, k_mean, marker="o", label="mean K")
    ax_summary.plot(episode_arr, k_std, marker="s", label="std K")
    ax_summary.set_title("Equivalent K summary")
    ax_summary.set_xlabel("Training episode / checkpoint")
    ax_summary.set_ylabel("value")
    ax_summary.grid(True, alpha=0.3)
    ax_summary.legend(fontsize=8)

    ax_tau.plot(episode_arr, tau_mean, marker="o", label="mean tau")
    ax_tau.plot(episode_arr, tau_std, marker="s", label="std tau")
    ax_tau.set_title("Torque summary over observation grid")
    ax_tau.set_xlabel("Training episode / checkpoint")
    ax_tau.set_ylabel("torque")
    ax_tau.grid(True, alpha=0.3)
    ax_tau.legend(fontsize=8)

    subtitle = (
        f"robot={metadata.get('scenario', metadata.get('robot', 'robot'))}, "
        f"terrain={training_terrain_name(metadata)}, "
        f"channel={metadata.get('channel', 'obs')}, "
        f"algorithm={metadata.get('algorithm', 'policy')}, "
        f"K=tau/dtheta_tot, |dtheta_tot|>={zero_epsilon:g}"
    )
    fig.suptitle(f"Equivalent K evolution: {run_dir.name}\n{subtitle}", fontsize=11)

    png_path = output_dir / "equivalent_k_evolution.png"
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)

    meta_path = safe_json_dump(
        {
            "run_dir": str(run_dir),
            "checkpoints": [str(p) for p in checkpoints],
            "episodes": episodes,
            "dtheta_tot_min": float(dtheta_tot[0]),
            "dtheta_tot_max": float(dtheta_tot[-1]),
            "grid_size": int(grid_size),
            "zero_epsilon": float(zero_epsilon),
            "definition": "equivalent_K = direct_policy_torque / dtheta_tot",
            "metadata": metadata,
            "env_info": {
                "terrain_name": terrain_name,
                "terrain_type": terrain_type,
                "terrain_settings": to_plain(terrain_settings),
                "num_agents": int(getattr(env, "num_agents", 0)),
            },
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        output_dir / "equivalent_k_evolution_metadata.json",
    )
    return [png_path, csv_path, npz_path, meta_path]


def save_evaluation_csv(rows: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_path


def save_evaluation_plot(rows: list[dict[str, Any]], output_path: Path, *, baseline_rows: list[dict[str, Any]] | None = None, dpi: int = 180) -> Path | None:
    if not rows:
        return None
    labels = [str(row["demo_terrain"]) for row in rows]
    values = np.asarray([float(row["mean_speed_x100"]) for row in rows], dtype=float)
    stds = np.asarray([float(row.get("std_speed_x100", 0.0)) for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    x = np.arange(len(labels))
    ax.bar(x, values, yerr=stds if np.any(stds > 0) else None, alpha=0.8, label="trained policy")
    if baseline_rows:
        base_map = {str(row.get("terrain")): float(row.get("mean_speed_x100", 0.0)) for row in baseline_rows}
        base_values = [base_map.get(label, np.nan) for label in labels]
        ax.plot(x, base_values, marker="o", linestyle="--", label="simple non-reciprocity")
    ax.set_xticks(x, labels)
    ax.set_ylabel("mean speed per 100 timesteps")
    ax.set_title("Cross-terrain evaluation")
    ax.grid(axis="y", alpha=0.35)
    ax.legend()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def save_motion_contact_sheet(
    checkpoint_path: Path,
    metadata: dict[str, Any],
    terrains: list[str],
    terrain_args: TerrainArgs,
    output_path: Path,
    *,
    steps: int = 300,
    frames_per_terrain: int = 8,
    policy_mode: str = "deterministic",
    follow_camera: bool = True,
    dpi: int = 180,
) -> Path | None:
    rows_frames: list[tuple[str, list[np.ndarray], dict[str, Any]]] = []
    for terrain in terrains:
        info, frames = simulate_policy(
            checkpoint_path,
            metadata,
            terrain,
            terrain_args,
            steps=steps,
            policy_mode=policy_mode,
            collect_frames=frames_per_terrain,
            follow_camera=follow_camera,
            num_envs=1,
            window_width=1000,
            window_height=360,
        )
        if frames:
            rows_frames.append((str(info["demo_terrain"]), frames, info))

    if not rows_frames:
        return None

    max_cols = max(len(frames) for _, frames, _ in rows_frames)
    fig, axes = plt.subplots(
        len(rows_frames),
        max_cols,
        figsize=(max_cols * 2.2, max(1, len(rows_frames)) * 1.25 + 0.5),
        squeeze=False,
    )
    for r, (terrain_name, frames, info) in enumerate(rows_frames):
        for c in range(max_cols):
            ax = axes[r][c]
            ax.axis("off")
            if c < len(frames):
                ax.imshow(frames[c])
                if r == 0:
                    ax.set_title(f"frame {c + 1}", fontsize=8)
            if c == 0:
                ax.text(
                    -0.02,
                    0.5,
                    f"demo: {terrain_name}\nspeed×100: {float(info['mean_speed_x100']):.2f}",
                    transform=ax.transAxes,
                    va="center",
                    ha="right",
                    fontsize=8,
                )
    fig.suptitle(
        f"Motion frames | trained: {training_terrain_name(metadata)} | {metadata.get('scenario', 'robot')} / {metadata.get('algorithm', 'policy')}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def safe_json_dump(data: dict[str, Any], path: Path) -> Path:
    def convert(obj):
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return str(obj)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=convert)
    return path


def analyze_run(
    *,
    run_dir: str | Path | None = None,
    checkpoint: str | Path | None = None,
    output_dir: str | Path | None = None,
    terrains: Iterable[str] = ("training",),
    terrain_args: TerrainArgs | None = None,
    policy_mode: str = "deterministic",
    grid_size: int = 81,
    theta_dot_slices: int = 9,
    heatmap: bool = True,
    k1_k2_evolution: bool = True,
    equivalent_k_evolution: bool = True,
    motion: bool = True,
    training_curve: bool = True,
    evaluate: bool = True,
    baseline: bool = True,
    eval_episodes: int = 3,
    eval_steps: int = 300,
    motion_steps: int = 300,
    motion_frames: int = 8,
    dpi: int = 180,
    kappa_alpha: float = -50.0,
) -> list[Path]:
    checkpoint_path = resolve_checkpoint(checkpoint, run_dir)
    run_path = Path(run_dir).resolve() if run_dir is not None else checkpoint_path.parent
    metadata = metadata_from_checkpoint(checkpoint_path)
    terrain_args = terrain_args or TerrainArgs()
    terrain_names = expand_terrains(terrains, metadata)

    if output_dir is None:
        output_path = run_path / "analysis" / checkpoint_path.stem
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    baseline_rows: list[dict[str, Any]] = []
    if baseline:
        for terrain in terrain_names:
            try:
                baseline_rows.append(
                    evaluate_nonreciprocity_baseline(
                        metadata,
                        terrain,
                        terrain_args,
                        episodes=max(1, min(eval_episodes, 3)),
                        steps=eval_steps,
                        kappa_alpha=kappa_alpha,
                    )
                )
            except Exception as exc:
                print(f"WARNING: baseline evaluation failed for terrain={terrain}: {exc}")

    if training_curve:
        baseline_for_training = None
        trained_name = training_terrain_name(metadata)
        for row in baseline_rows:
            if str(row.get("terrain")) == trained_name:
                baseline_for_training = float(row.get("mean_speed_x100", 0.0))
        p = save_training_curve(run_path, output_path, baseline_speed_x100=baseline_for_training, dpi=dpi)
        if p is not None:
            saved.append(p)

    if heatmap:
        try:
            heatmap_dir = output_path / "policy_heatmaps"
            created = analyze_checkpoint(
                checkpoint=checkpoint_path,
                output_dir=heatmap_dir,
                grid_size=grid_size,
                theta_dot_slices=theta_dot_slices,
                dpi=dpi,
                make_baseline=True,
            )
            saved.extend(Path(p) for p in created)
        except Exception as exc:
            print(f"WARNING: policy heatmap analysis failed: {exc}")

    if k1_k2_evolution:
        try:
            saved.extend(
                save_k1_k2_evolution(
                    run_path,
                    checkpoint_path,
                    metadata,
                    terrain_args,
                    output_path,
                    grid_size=grid_size,
                    policy_mode=policy_mode,
                    dpi=dpi,
                )
            )
        except Exception as exc:
            print(f"WARNING: K1/K2 evolution analysis failed: {exc}")

    if equivalent_k_evolution:
        try:
            saved.extend(
                save_equivalent_k_evolution(
                    run_path,
                    checkpoint_path,
                    metadata,
                    terrain_args,
                    output_path,
                    grid_size=grid_size,
                    policy_mode=policy_mode,
                    dpi=dpi,
                )
            )
        except Exception as exc:
            print(f"WARNING: equivalent K evolution analysis failed: {exc}")

    eval_rows: list[dict[str, Any]] = []
    if evaluate:
        try:
            eval_rows = evaluate_checkpoint_on_terrains(
                checkpoint_path,
                metadata,
                terrain_names,
                terrain_args,
                episodes=eval_episodes,
                steps=eval_steps,
                policy_mode=policy_mode,
            )
            saved.append(save_evaluation_csv(eval_rows, output_path / "cross_terrain_evaluation.csv"))
            p = save_evaluation_plot(eval_rows, output_path / "cross_terrain_evaluation.png", baseline_rows=baseline_rows, dpi=dpi)
            if p is not None:
                saved.append(p)
        except Exception as exc:
            print(f"WARNING: cross-terrain evaluation failed: {exc}")

    if motion:
        try:
            p = save_motion_contact_sheet(
                checkpoint_path,
                metadata,
                terrain_names,
                terrain_args,
                output_path / "motion_frames.png",
                steps=motion_steps,
                frames_per_terrain=motion_frames,
                policy_mode=policy_mode,
                follow_camera=True,
                dpi=dpi,
            )
            if p is not None:
                saved.append(p)
        except Exception as exc:
            print(f"WARNING: motion-frame figure failed: {exc}")

    summary = {
        "checkpoint": str(checkpoint_path),
        "run_dir": str(run_path),
        "output_dir": str(output_path),
        "metadata": metadata,
        "terrains": terrain_names,
        "policy_mode": policy_mode,
        "grid_size": grid_size,
        "theta_dot_slices": theta_dot_slices,
        "k1_k2_evolution": k1_k2_evolution,
        "equivalent_k_evolution": equivalent_k_evolution,
        "eval_episodes": eval_episodes,
        "eval_steps": eval_steps,
        "motion_steps": motion_steps,
        "motion_frames": motion_frames,
        "baseline_rows": baseline_rows,
        "evaluation_rows": eval_rows,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "saved_files": [str(p) for p in saved],
    }
    saved.append(safe_json_dump(summary, output_path / "analysis_summary.json"))
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate thesis-style analysis files for a training run/checkpoint.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run folder containing checkpoint_*.pt and training_log.csv.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Specific checkpoint. If omitted, latest checkpoint in --run-dir is used.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Analysis output folder. Default: run_dir/analysis/checkpoint_N")
    parser.add_argument("--terrains", nargs="+", default=["training"], help="training/checkpoint, flat, stairs, tunnel, or all.")
    parser.add_argument("--policy-mode", choices=["sample", "deterministic"], default="deterministic")

    parser.add_argument("--grid-size", type=int, default=81)
    parser.add_argument("--theta-dot-slices", type=int, default=9)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--eval-steps", type=int, default=300)
    parser.add_argument("--motion-steps", type=int, default=300)
    parser.add_argument("--motion-frames", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--kappa-alpha", type=float, default=-50.0)

    parser.add_argument("--no-heatmap", action="store_true")
    parser.add_argument("--no-k1-k2-evolution", action="store_true")
    parser.add_argument("--no-equivalent-k-evolution", action="store_true")
    parser.add_argument("--no-motion", action="store_true")
    parser.add_argument("--no-training-curve", action="store_true")
    parser.add_argument("--no-evaluate", action="store_true")
    parser.add_argument("--no-baseline", action="store_true")

    parser.add_argument("--start-stairs", type=float, default=5)
    parser.add_argument("--step-width", type=float, default=5)
    parser.add_argument("--step-height", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--tunnel-start", type=float, default=10)
    parser.add_argument("--tunnel-slope", type=float, default=5)
    parser.add_argument("--tunnel-slope-height", type=float, default=1)
    parser.add_argument("--tunnel-length", type=float, default=10)
    parser.add_argument("--tunnel-height", type=float, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved = analyze_run(
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        terrains=args.terrains,
        terrain_args=TerrainArgs.from_namespace(args),
        policy_mode=args.policy_mode,
        grid_size=args.grid_size,
        theta_dot_slices=args.theta_dot_slices,
        heatmap=not args.no_heatmap,
        k1_k2_evolution=not args.no_k1_k2_evolution,
        equivalent_k_evolution=not args.no_equivalent_k_evolution,
        motion=not args.no_motion,
        training_curve=not args.no_training_curve,
        evaluate=not args.no_evaluate,
        baseline=not args.no_baseline,
        eval_episodes=args.eval_episodes,
        eval_steps=args.eval_steps,
        motion_steps=args.motion_steps,
        motion_frames=args.motion_frames,
        dpi=args.dpi,
        kappa_alpha=args.kappa_alpha,
    )
    print("Analysis files saved:")
    for path in saved:
        print(" ", path)


if __name__ == "__main__":
    main()
