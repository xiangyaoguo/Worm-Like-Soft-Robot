from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

DEFAULT_REPO = Path(__file__).resolve().parents[1]


def to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def _project_helpers(repo: Path):
    training_dir = repo / "training"
    if not training_dir.is_dir():
        raise FileNotFoundError(f"Project training directory not found: {training_dir}")
    sys.path.insert(0, str(training_dir))
    from analyze_training_results import (  # type: ignore
        TerrainArgs,
        build_demo_env,
        load_policy_for_env,
        metadata_from_checkpoint,
    )
    from demo_metamaterial import choose_action  # type: ignore

    return TerrainArgs, build_demo_env, load_policy_for_env, metadata_from_checkpoint, choose_action


def _positions(env: Any) -> np.ndarray:
    value = env.pos
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)[0].astype(np.complex128, copy=True)


def _finite_scalar(value: Any) -> float | None:
    """Convert a one-environment log value to a finite Python float."""

    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.size == 0:
        return None
    scalar = float(array.reshape(-1)[0])
    return scalar if math.isfinite(scalar) else None


def _log_info_scalar(td: Any, name: str) -> float | None:
    """Read one exported environment diagnostic without depending on TensorDict internals."""

    for key in (("log_info", name), name):
        try:
            value = td.get(key, None)
        except (AttributeError, KeyError, TypeError):
            try:
                value = td[key]
            except (KeyError, TypeError, IndexError):
                continue
        scalar = _finite_scalar(value)
        if scalar is not None:
            return scalar
    return None


def _best_fit_rotation(previous: np.ndarray, current: np.ndarray) -> float:
    previous_centered = previous - np.mean(previous)
    current_centered = current - np.mean(current)
    cross = np.sum(np.conj(previous_centered) * current_centered)
    if abs(cross) <= 1e-12:
        return 0.0
    return float(np.angle(cross))


def _body_length(pos: np.ndarray) -> float:
    return float(np.sum(np.abs(np.diff(pos))))


def _contact_material_index(pos: np.ndarray, body_length: float) -> tuple[float, int, int]:
    """Estimate which material index supports the body on flat ground.

    The weighted index is stable when several particles touch simultaneously;
    the hard index and active-contact count are retained as diagnostics.
    """

    y = np.imag(pos)
    n = len(y)
    segment_length = body_length / max(n - 1, 1)
    tolerance = max(0.18 * segment_length, 1e-6)
    height = y - float(np.min(y))
    weights = np.exp(-np.square(height / tolerance))
    weights[height > 3.0 * tolerance] = 0.0
    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-12:
        weighted_index = float(np.argmin(y))
    else:
        weighted_index = float(np.dot(weights, np.arange(n, dtype=np.float64)) / weight_sum)
    return weighted_index, int(np.argmin(y)), int(np.count_nonzero(height <= tolerance))


def _tail_geometry(
    pos: np.ndarray,
    body_length: float,
    tail_side: str,
    direction_sign: float,
    initial_tail_relative_x: float,
) -> tuple[float, float, float]:
    n = len(pos)
    tail_index = 0 if tail_side == "left" else n - 1
    tail_x = float(np.real(pos[tail_index]))
    tail_y = float(np.imag(pos[tail_index]))
    com_x = float(np.real(np.mean(pos)))
    min_y = float(np.min(np.imag(pos)))
    tail_lift_fraction = (tail_y - min_y) / max(body_length, 1e-12)
    tail_forward_fraction = (
        direction_sign * ((tail_x - com_x) - initial_tail_relative_x) / max(body_length, 1e-12)
    )

    segments = np.diff(pos)
    if len(segments) < 2:
        prefix_curvature_degrees = 0.0
    else:
        turns = np.angle(segments[1:] * np.conj(segments[:-1]))
        prefix_count = min(3, len(turns))
        prefix = turns[:prefix_count] if tail_side == "left" else turns[-prefix_count:]
        prefix_curvature_degrees = float(np.degrees(np.sum(np.abs(prefix))))
    return tail_lift_fraction, tail_forward_fraction, prefix_curvature_degrees


def _rising_edge_steps(mask: np.ndarray, minimum_separation: int = 30) -> list[int]:
    candidates = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    selected: list[int] = []
    for candidate in candidates.tolist():
        if not selected or candidate - selected[-1] >= minimum_separation:
            selected.append(int(candidate))
    return selected


def detect_roll_pulses(
    desired_rotation_cumulative: np.ndarray,
    rotation_increments: np.ndarray,
    com_x: np.ndarray,
    contact_index: np.ndarray,
    body_length: float,
    particle_count: int,
    direction_sign: float,
    rotation_threshold_radians: float,
    forward_body_fraction: float,
    contact_index_fraction: float,
    active_increment_radians: float,
    reset_drawdown_radians: float,
    reset_backward_body_fraction: float,
    contact_valid: np.ndarray | None = None,
) -> list[dict[str, float | int]]:
    """Greedily segment desired-direction rotation into independently valid pulses."""

    pulses: list[dict[str, float | int]] = []
    total_steps = len(rotation_increments)
    start = 0
    previous_end: int | None = None
    material_scale = float(max(particle_count - 1, 1))

    for end in range(1, total_steps + 1):
        desired_angle = float(desired_rotation_cumulative[end] - desired_rotation_cumulative[start])
        forward = float(direction_sign * (com_x[end] - com_x[start]))

        if (
            desired_angle <= -reset_drawdown_radians
            or forward <= -reset_backward_body_fraction * body_length
        ):
            start = end
            continue

        if desired_angle < rotation_threshold_radians:
            continue
        if forward < forward_body_fraction * body_length:
            continue

        contact_slice = contact_index[start : end + 1]
        contact_span = float(np.max(contact_slice) - np.min(contact_slice))
        normalized_contact_span = contact_span / material_scale
        if normalized_contact_span < contact_index_fraction:
            continue
        if contact_valid is not None and (
            end >= len(contact_valid) or not bool(contact_valid[end])
        ):
            # A completed roll pulse must end on a real terrain-support frame.
            # This prevents an airborne shape change from being certified as
            # contact migration.
            continue

        increments = rotation_increments[start:end]
        active = np.abs(increments) >= active_increment_radians
        desired_active_fraction = (
            float(np.mean(increments[active] > 0.0)) if np.any(active) else 0.0
        )
        interval = 0 if previous_end is None else end - previous_end
        idle_steps = 0 if previous_end is None else start - previous_end
        pulses.append(
            {
                "start_step": int(start),
                "end_step": int(end),
                "duration_steps": int(end - start),
                "interval_from_previous_end_steps": int(interval),
                "idle_steps_since_previous_pulse_end": int(idle_steps),
                "desired_net_rotation_degrees": float(np.degrees(desired_angle)),
                "forward_displacement": forward,
                "forward_body_fraction": forward / max(body_length, 1e-12),
                "contact_index_start": float(contact_index[start]),
                "contact_index_end": float(contact_index[end]),
                "contact_index_span": contact_span,
                "contact_index_span_fraction": normalized_contact_span,
                "desired_active_rotation_fraction": desired_active_fraction,
            }
        )
        previous_end = end
        start = end

    return pulses


def _episode_metrics(
    trajectory: list[np.ndarray],
    direction: str,
    tail_side: str,
    args: argparse.Namespace,
    exported_support_index: list[float | None] | None = None,
    exported_contact_strength: list[float | None] | None = None,
) -> dict[str, Any]:
    positions = np.asarray(trajectory, dtype=np.complex128)
    steps = len(positions) - 1
    particle_count = positions.shape[1]
    direction_sign = 1.0 if direction == "right" else -1.0
    initial_length = _body_length(positions[0])
    com_x = np.real(np.mean(positions, axis=1)).astype(np.float64)

    raw_rotation = np.asarray(
        [_best_fit_rotation(positions[i], positions[i + 1]) for i in range(steps)],
        dtype=np.float64,
    )
    # Clockwise rotation is desired for rightward motion; counter-clockwise for leftward motion.
    desired_rotation = -direction_sign * raw_rotation
    desired_cumulative = np.r_[0.0, np.cumsum(desired_rotation)]

    contact_index: list[float] = []
    hard_contact_index: list[int] = []
    active_contact_count: list[int] = []
    tail_lift: list[float] = []
    tail_forward: list[float] = []
    tail_prefix_curvature: list[float] = []
    tail_index = 0 if tail_side == "left" else particle_count - 1
    initial_tail_relative_x = float(
        np.real(positions[0, tail_index]) - np.real(np.mean(positions[0]))
    )

    for pos in positions:
        weighted, hard, count = _contact_material_index(pos, initial_length)
        contact_index.append(weighted)
        hard_contact_index.append(hard)
        active_contact_count.append(count)
        lift, forward, curvature = _tail_geometry(
            pos,
            initial_length,
            tail_side,
            direction_sign,
            initial_tail_relative_x,
        )
        tail_lift.append(lift)
        tail_forward.append(forward)
        tail_prefix_curvature.append(curvature)

    legacy_contact_a = np.asarray(contact_index, dtype=np.float64)
    contact_a = legacy_contact_a
    contact_metric_source = "legacy_position_height_proxy"
    ground_contact_valid_fraction: float | None = None
    mean_ground_contact_strength: float | None = None
    pulse_contact_valid: np.ndarray | None = None
    if (
        exported_support_index is not None
        and exported_contact_strength is not None
        and len(exported_support_index) == len(positions)
        and len(exported_contact_strength) == len(positions)
    ):
        exported_support_a = np.asarray(
            [np.nan if value is None else value for value in exported_support_index],
            dtype=np.float64,
        )
        exported_strength_a = np.asarray(
            [np.nan if value is None else value for value in exported_contact_strength],
            dtype=np.float64,
        )
        finite_strength = np.isfinite(exported_strength_a)
        exported_available = np.isfinite(exported_support_a) & finite_strength
        if np.any(finite_strength):
            mean_ground_contact_strength = float(np.mean(exported_strength_a[finite_strength]))
        valid_ground_contact = (
            exported_available
            & (exported_strength_a >= 0.50)
        )
        pulse_contact_valid = valid_ground_contact
        ground_contact_valid_fraction = float(np.mean(valid_ground_contact))
        if np.any(exported_available):
            # Causal sample-and-hold: update material support only on a real
            # contact frame.  Missing/airborne frames retain the latest known
            # support instead of borrowing a future landing via interpolation.
            initial_support = (
                float(exported_support_a[0])
                if exported_available[0]
                else float(legacy_contact_a[0])
            )
            contact_a = np.empty(len(positions), dtype=np.float64)
            held_support = initial_support
            for frame in range(len(positions)):
                if valid_ground_contact[frame]:
                    held_support = float(exported_support_a[frame])
                contact_a[frame] = held_support
            contact_metric_source = "env_fast_forward_log_info"

    tail_lift_a = np.asarray(tail_lift, dtype=np.float64)
    tail_forward_a = np.asarray(tail_forward, dtype=np.float64)
    tail_curvature_a = np.asarray(tail_prefix_curvature, dtype=np.float64)
    launch_mask = (
        (tail_lift_a >= args.tail_launch_lift_fraction)
        & (tail_forward_a >= args.tail_launch_forward_fraction)
        & (tail_curvature_a >= args.tail_launch_curvature_degrees)
    )
    launch_steps = _rising_edge_steps(launch_mask, args.tail_launch_min_separation)

    pulses = detect_roll_pulses(
        desired_rotation_cumulative=desired_cumulative,
        rotation_increments=desired_rotation,
        com_x=com_x,
        contact_index=contact_a,
        body_length=initial_length,
        particle_count=particle_count,
        direction_sign=direction_sign,
        rotation_threshold_radians=math.radians(args.pulse_rotation_degrees),
        forward_body_fraction=args.pulse_forward_body_fraction,
        contact_index_fraction=args.pulse_contact_index_fraction,
        active_increment_radians=math.radians(args.active_rotation_degrees),
        reset_drawdown_radians=math.radians(args.pulse_reset_drawdown_degrees),
        reset_backward_body_fraction=args.pulse_reset_backward_body_fraction,
        contact_valid=pulse_contact_valid,
    )

    previous_pulse_end = 0
    for pulse in pulses:
        pulse_end = int(pulse["end_step"])
        eligible_launches = [step for step in launch_steps if previous_pulse_end <= step <= pulse_end]
        pulse["tail_launch_detected"] = int(bool(eligible_launches))
        pulse["tail_launch_step"] = int(eligible_launches[0]) if eligible_launches else -1
        previous_pulse_end = pulse_end

    matched_launch_steps = sorted(
        {
            int(pulse["tail_launch_step"])
            for pulse in pulses
            if int(pulse["tail_launch_step"]) >= 0
        }
    )

    active = np.abs(desired_rotation) >= math.radians(args.active_rotation_degrees)
    desired_fraction = float(np.mean(desired_rotation[active] > 0.0)) if np.any(active) else 0.0
    pulse_intervals = [
        int(pulse["interval_from_previous_end_steps"])
        for pulse in pulses[1:]
    ]
    displacement = float(direction_sign * (com_x[-1] - com_x[0]))

    if len(pulses) >= 2:
        classification = "repeated_fast_forward_roll"
    elif len(pulses) == 1:
        classification = "single_fast_forward_roll"
    elif displacement >= args.pulse_forward_body_fraction * initial_length:
        classification = "forward_motion_without_roll_pulse"
    else:
        classification = "no_meaningful_forward_roll"

    return {
        "steps": steps,
        "classification": classification,
        "particle_count": particle_count,
        "initial_body_length": initial_length,
        "forward_displacement": displacement,
        "forward_body_lengths": displacement / max(initial_length, 1e-12),
        "net_best_fit_rotation_degrees": float(np.degrees(np.sum(raw_rotation))),
        "desired_net_rotation_degrees": float(np.degrees(np.sum(desired_rotation))),
        "desired_positive_rotation_degrees": float(
            np.degrees(np.sum(np.clip(desired_rotation, 0.0, None)))
        ),
        "reverse_rotation_degrees": float(
            np.degrees(np.sum(np.clip(-desired_rotation, 0.0, None)))
        ),
        "desired_active_rotation_fraction": desired_fraction,
        "contact_metric_source": contact_metric_source,
        "environment_support_index_available": contact_metric_source == "env_fast_forward_log_info",
        "ground_contact_valid_fraction": ground_contact_valid_fraction,
        "mean_ground_contact_strength": mean_ground_contact_strength,
        "contact_material_index_span": float(np.max(contact_a) - np.min(contact_a)),
        "contact_material_index_span_fraction": float(
            (np.max(contact_a) - np.min(contact_a)) / max(particle_count - 1, 1)
        ),
        "hard_support_unique_material_indices": int(len(set(hard_contact_index))),
        "mean_active_contact_particles": float(np.mean(active_contact_count)),
        # A formal tail launch must initiate a valid roll pulse. Raw geometric
        # candidates are kept separately because ordinary crawling also lifts
        # and advances the tail periodically.
        "tail_launch_detected": bool(matched_launch_steps),
        "tail_launch_count": len(matched_launch_steps),
        "tail_launch_steps": matched_launch_steps,
        "tail_launch_candidate_count": len(launch_steps),
        "tail_launch_candidate_steps": launch_steps,
        "peak_tail_lift_body_fraction": float(np.max(tail_lift_a)),
        "peak_tail_forward_body_fraction": float(np.max(tail_forward_a)),
        "peak_tail_prefix_curvature_degrees": float(np.max(tail_curvature_a)),
        "roll_pulse_count": len(pulses),
        "roll_pulse_intervals_steps": pulse_intervals,
        "mean_roll_pulse_interval_steps": (
            float(np.mean(pulse_intervals)) if pulse_intervals else None
        ),
        "roll_pulses": pulses,
    }


def _metadata_value(metadata: dict[str, Any], name: str, default: Any = None) -> Any:
    if name in metadata:
        return metadata[name]
    training_args = metadata.get("training_args")
    if isinstance(training_args, dict) and name in training_args:
        return training_args[name]
    return default


def evaluate_checkpoint(
    checkpoint: Path,
    checkpoint_index: int,
    args: argparse.Namespace,
    helpers: tuple[Any, ...],
) -> dict[str, Any]:
    TerrainArgs, build_demo_env, load_policy_for_env, metadata_from_checkpoint, choose_action = helpers
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    metadata = metadata_from_checkpoint(checkpoint)
    terrain_contact_mode = str(
        _metadata_value(metadata, "terrain_contact_mode", "legacy_flat")
    )
    if terrain_contact_mode not in {"legacy_flat", "mesh_v1"}:
        raise ValueError(
            f"Unsupported checkpoint terrain_contact_mode: {terrain_contact_mode!r}"
        )
    direction = args.direction
    if direction == "auto":
        direction = str(_metadata_value(metadata, "rolling_direction", "right")).lower()
    if direction not in {"right", "left"}:
        raise ValueError(f"Unsupported direction: {direction}")
    tail_side = args.tail_side
    if tail_side == "auto":
        tail_side = str(_metadata_value(metadata, "tail_side", "left")).lower()
    if tail_side not in {"left", "right"}:
        raise ValueError(f"Unsupported tail side: {tail_side}")

    env, resolved_terrain_name, resolved_terrain_type, resolved_terrain_settings = build_demo_env(
        metadata,
        args.terrain,
        TerrainArgs(),
        max_steps=args.steps,
        render_mode="rgb_array",
        num_envs=1,
    )
    actual_terrain_contact_mode = str(
        getattr(env, "terrain_contact_mode", "missing")
    )
    if actual_terrain_contact_mode != terrain_contact_mode:
        close = getattr(env, "close", None)
        if callable(close):
            close()
        raise RuntimeError(
            "Evaluator environment contact mode does not match checkpoint metadata: "
            f"environment={actual_terrain_contact_mode!r}, "
            f"checkpoint={terrain_contact_mode!r}"
        )
    expected_terrain_type = str(_metadata_value(metadata, "terrain_type", "flat"))
    expected_terrain_settings = _metadata_value(metadata, "terrain_settings", None)
    terrain_metadata_validated = args.terrain not in {"checkpoint", "training"}
    if args.terrain in {"checkpoint", "training"}:
        terrain_metadata_validated = (
            str(resolved_terrain_type) == expected_terrain_type
            and to_jsonable(resolved_terrain_settings)
            == to_jsonable(expected_terrain_settings)
        )
        if not terrain_metadata_validated:
            raise ValueError(
                "Checkpoint terrain reconstruction mismatch: "
                f"expected type/settings={expected_terrain_type!r}/"
                f"{expected_terrain_settings!r}, resolved="
                f"{resolved_terrain_type!r}/{resolved_terrain_settings!r}"
            )
    policy = load_policy_for_env(checkpoint, env, metadata)

    episodes: list[dict[str, Any]] = []
    try:
        for episode_index in range(args.episodes):
            # Reuse the same initial-state seeds for every checkpoint so model
            # comparisons are paired rather than confounded by different resets.
            seed = args.seed + episode_index
            torch.manual_seed(seed)
            np.random.seed(seed)
            td = env.reset()
            trajectory = [_positions(env)]
            exported_support_index = [
                _log_info_scalar(td, "fast_forward_support_index")
            ]
            exported_contact_strength = [
                _log_info_scalar(td, "fast_forward_ground_contact_strength")
            ]
            exported_mesh_floor_contact = [
                _log_info_scalar(td, "terrain_floor_contact_strength")
            ]
            for _ in range(args.steps):
                action_td = choose_action(policy, td, "deterministic")
                td = env.step(action_td)["next"]
                trajectory.append(_positions(env))
                exported_support_index.append(
                    _log_info_scalar(td, "fast_forward_support_index")
                )
                exported_contact_strength.append(
                    _log_info_scalar(td, "fast_forward_ground_contact_strength")
                )
                exported_mesh_floor_contact.append(
                    _log_info_scalar(td, "terrain_floor_contact_strength")
                )
            mesh_contact_diagnostics_available = all(
                value is not None for value in exported_mesh_floor_contact
            )
            if (
                terrain_contact_mode == "mesh_v1"
                and not mesh_contact_diagnostics_available
            ):
                raise RuntimeError(
                    "mesh_v1 checkpoint evaluation is missing the mesh-only "
                    "terrain_floor_contact_strength diagnostic."
                )
            metrics = _episode_metrics(
                trajectory,
                direction,
                tail_side,
                args,
                exported_support_index,
                exported_contact_strength,
            )
            metrics["terrain_contact_mode"] = terrain_contact_mode
            metrics["environment_terrain_contact_mode"] = (
                actual_terrain_contact_mode
            )
            metrics["mesh_contact_diagnostics_available"] = (
                mesh_contact_diagnostics_available
            )
            metrics["terrain_aware_contact_used"] = bool(
                terrain_contact_mode == "mesh_v1"
                and actual_terrain_contact_mode == "mesh_v1"
                and mesh_contact_diagnostics_available
                and metrics["contact_metric_source"] == "env_fast_forward_log_info"
            )
            metrics["episode"] = episode_index + 1
            metrics["seed"] = seed
            episodes.append(metrics)
            print(
                f"[{checkpoint.name}] episode {episode_index + 1}/{args.episodes}: "
                f"dx={metrics['forward_displacement']:.3f}, "
                f"rotation={metrics['net_best_fit_rotation_degrees']:.1f} deg, "
                f"pulses={metrics['roll_pulse_count']}",
                file=sys.stderr,
                flush=True,
            )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()

    interval_values = [
        interval
        for episode in episodes
        for interval in episode["roll_pulse_intervals_steps"]
    ]
    aggregate = {
        "evaluation_episodes": len(episodes),
        "mean_forward_displacement": float(
            np.mean([episode["forward_displacement"] for episode in episodes])
        ),
        "median_forward_displacement": float(
            np.median([episode["forward_displacement"] for episode in episodes])
        ),
        "mean_forward_body_lengths": float(
            np.mean([episode["forward_body_lengths"] for episode in episodes])
        ),
        "median_forward_body_lengths": float(
            np.median([episode["forward_body_lengths"] for episode in episodes])
        ),
        "mean_net_best_fit_rotation_degrees": float(
            np.mean([episode["net_best_fit_rotation_degrees"] for episode in episodes])
        ),
        "median_net_best_fit_rotation_degrees": float(
            np.median([episode["net_best_fit_rotation_degrees"] for episode in episodes])
        ),
        "mean_desired_net_rotation_degrees": float(
            np.mean([episode["desired_net_rotation_degrees"] for episode in episodes])
        ),
        "median_desired_net_rotation_degrees": float(
            np.median([episode["desired_net_rotation_degrees"] for episode in episodes])
        ),
        "mean_desired_active_rotation_fraction": float(
            np.mean([episode["desired_active_rotation_fraction"] for episode in episodes])
        ),
        "median_desired_active_rotation_fraction": float(
            np.median([episode["desired_active_rotation_fraction"] for episode in episodes])
        ),
        "mean_contact_material_index_span_fraction": float(
            np.mean([episode["contact_material_index_span_fraction"] for episode in episodes])
        ),
        "contact_metric_sources": sorted(
            {str(episode["contact_metric_source"]) for episode in episodes}
        ),
        "episodes_using_environment_support": int(
            sum(
                episode["contact_metric_source"] == "env_fast_forward_log_info"
                for episode in episodes
            )
        ),
        "terrain_aware_contact_episode_rate": float(
            np.mean([bool(episode["terrain_aware_contact_used"]) for episode in episodes])
        ),
        "total_roll_pulses": int(sum(episode["roll_pulse_count"] for episode in episodes)),
        "mean_roll_pulses_per_episode": float(
            np.mean([episode["roll_pulse_count"] for episode in episodes])
        ),
        "episodes_with_roll_pulse": int(
            sum(episode["roll_pulse_count"] > 0 for episode in episodes)
        ),
        "roll_pulse_episode_rate": float(
            np.mean([episode["roll_pulse_count"] > 0 for episode in episodes])
        ),
        "episodes_with_tail_launch": int(
            sum(bool(episode["tail_launch_detected"]) for episode in episodes)
        ),
        "tail_launch_episode_rate": float(
            np.mean([bool(episode["tail_launch_detected"]) for episode in episodes])
        ),
        "mean_inter_pulse_interval_steps": (
            float(np.mean(interval_values)) if interval_values else None
        ),
    }

    return {
        "checkpoint": str(checkpoint),
        "checkpoint_name": checkpoint.name,
        "run_name": _metadata_value(metadata, "run_name", checkpoint.parent.name),
        "control_mode": _metadata_value(metadata, "control_mode"),
        "channel": _metadata_value(metadata, "channel"),
        "reward_func": _metadata_value(metadata, "reward_func"),
        "terrain_label": _metadata_value(metadata, "terrain_label", resolved_terrain_name),
        "terrain_type": resolved_terrain_type,
        "terrain_settings": to_jsonable(resolved_terrain_settings),
        "terrain_contact_mode": terrain_contact_mode,
        "environment_terrain_contact_mode": actual_terrain_contact_mode,
        "requested_terrain": args.terrain,
        "resolved_terrain": resolved_terrain_name,
        "terrain_metadata_validated": terrain_metadata_validated,
        "direction": direction,
        "tail_side": tail_side,
        "episodes": episodes,
        "aggregate": aggregate,
    }


def _self_test() -> None:
    steps = 200
    angle = np.r_[np.zeros(20), np.linspace(0.0, math.radians(75.0), steps - 19)]
    rotation = np.diff(angle)
    com = np.linspace(0.0, 1.0, steps + 1)
    contact = np.linspace(4.5, 1.5, steps + 1)
    pulses = detect_roll_pulses(
        angle,
        rotation,
        com,
        contact,
        body_length=9.0,
        particle_count=10,
        direction_sign=1.0,
        rotation_threshold_radians=math.radians(60.0),
        forward_body_fraction=0.08,
        contact_index_fraction=0.20,
        active_increment_radians=math.radians(0.05),
        reset_drawdown_radians=math.radians(20.0),
        reset_backward_body_fraction=0.03,
    )
    assert len(pulses) == 1, pulses

    no_rotation = np.zeros(steps + 1)
    assert not detect_roll_pulses(
        no_rotation,
        np.diff(no_rotation),
        np.linspace(0.0, 20.0, steps + 1),
        contact,
        body_length=9.0,
        particle_count=10,
        direction_sign=1.0,
        rotation_threshold_radians=math.radians(60.0),
        forward_body_fraction=0.08,
        contact_index_fraction=0.20,
        active_increment_radians=math.radians(0.05),
        reset_drawdown_radians=math.radians(20.0),
        reset_backward_body_fraction=0.03,
    )

    no_migration = np.full(steps + 1, 4.5)
    assert not detect_roll_pulses(
        angle,
        rotation,
        com,
        no_migration,
        body_length=9.0,
        particle_count=10,
        direction_sign=1.0,
        rotation_threshold_radians=math.radians(60.0),
        forward_body_fraction=0.08,
        contact_index_fraction=0.20,
        active_increment_radians=math.radians(0.05),
        reset_drawdown_radians=math.radians(20.0),
        reset_backward_body_fraction=0.03,
    )

    metric_args = _parser().parse_args([])
    static_body = np.arange(10, dtype=np.float64).astype(np.complex128)
    static_trajectory = [static_body.copy() for _ in range(4)]
    exported_metrics = _episode_metrics(
        static_trajectory,
        "right",
        "left",
        metric_args,
        [None, 4.5, 4.0, 3.5],
        [None, 1.0, 1.0, 1.0],
    )
    assert exported_metrics["contact_metric_source"] == "env_fast_forward_log_info"
    fallback_metrics = _episode_metrics(
        static_trajectory,
        "right",
        "left",
        metric_args,
        [None, None, None, None],
        [None, None, None, None],
    )
    assert fallback_metrics["contact_metric_source"] == "legacy_position_height_proxy"

    # A rotating, translating airborne body may change which material point is
    # geometrically lowest, but exported zero contact strength must veto every
    # pulse.  This is the formal-gate adversary, not merely an env reward test.
    ring_theta = np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False)
    ring = np.cos(ring_theta) + 1j * np.sin(ring_theta)
    airborne_trajectory = []
    for frame in range(steps + 1):
        rigid_rotation = np.exp(-1j * math.radians(0.75 * frame))
        airborne_trajectory.append(
            np.asarray(
                ring * rigid_rotation + 0.06 * frame + 5.0j,
                dtype=np.complex128,
            )
        )
    airborne_metrics = _episode_metrics(
        airborne_trajectory,
        "right",
        "left",
        metric_args,
        np.linspace(4.5, 1.5, steps + 1).tolist(),
        np.zeros(steps + 1, dtype=np.float64).tolist(),
    )
    assert airborne_metrics["contact_metric_source"] == "env_fast_forward_log_info"
    assert airborne_metrics["ground_contact_valid_fraction"] == 0.0
    assert airborne_metrics["roll_pulse_count"] == 0, airborne_metrics
    print(
        "Self-test passed: true pulse detected; crawl, fixed-contact, and airborne "
        "cases rejected; environment support preferred with legacy fallback."
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministic fast-forward-roll evaluator for R31 per-joint and tail_wave checkpoints."
        )
    )
    parser.add_argument("--checkpoint", action="append", type=Path, default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--terrain",
        choices=("checkpoint", "training", "flat", "stairs", "tunnel"),
        default="flat",
    )
    parser.add_argument("--direction", choices=("auto", "right", "left"), default="auto")
    parser.add_argument("--tail-side", choices=("auto", "left", "right"), default="auto")
    parser.add_argument("--pulse-rotation-degrees", type=float, default=60.0)
    parser.add_argument("--pulse-forward-body-fraction", type=float, default=0.08)
    parser.add_argument("--pulse-contact-index-fraction", type=float, default=0.20)
    parser.add_argument("--active-rotation-degrees", type=float, default=0.05)
    parser.add_argument("--pulse-reset-drawdown-degrees", type=float, default=20.0)
    parser.add_argument("--pulse-reset-backward-body-fraction", type=float, default=0.03)
    parser.add_argument("--tail-launch-lift-fraction", type=float, default=0.025)
    parser.add_argument("--tail-launch-forward-fraction", type=float, default=0.015)
    parser.add_argument("--tail-launch-curvature-degrees", type=float, default=15.0)
    parser.add_argument("--tail-launch-min-separation", type=int, default=30)
    parser.add_argument("--quiet", action="store_true", help="Do not print the JSON payload to stdout")
    parser.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        if not args.checkpoint:
            return
    if not args.checkpoint:
        parser.error("at least one --checkpoint is required unless --self-test is used")
    if args.output is None:
        parser.error("--output is required when evaluating checkpoints")
    if args.episodes <= 0 or args.steps <= 0:
        parser.error("--episodes and --steps must be positive")

    helpers = _project_helpers(args.repo.resolve())
    results = [
        evaluate_checkpoint(checkpoint, index, args, helpers)
        for index, checkpoint in enumerate(args.checkpoint)
    ]
    payload = {
        "method": {
            "schema": "fast_forward_eval/v1",
            "name": "fast_forward_roll_v2",
            "policy_mode": "deterministic",
            "terrain": args.terrain,
            "contact_metric_preference": (
                "environment log_info fast_forward_support_index gated by "
                "fast_forward_ground_contact_strength>=0.50; legacy position-height "
                "proxy only when exports are unavailable"
            ),
            "steps_per_episode": args.steps,
            "episodes_per_checkpoint": args.episodes,
            "definition": (
                "one pulse requires desired net best-fit rotation, forward COM displacement, "
                "and contact material-index migration in the same interval; no closure or "
                "360-degree requirement"
            ),
            "pulse_rotation_degrees": args.pulse_rotation_degrees,
            "pulse_forward_body_fraction": args.pulse_forward_body_fraction,
            "pulse_contact_index_fraction": args.pulse_contact_index_fraction,
            "active_rotation_increment_degrees": args.active_rotation_degrees,
            "tail_launch_lift_body_fraction": args.tail_launch_lift_fraction,
            "tail_launch_forward_body_fraction": args.tail_launch_forward_fraction,
            "tail_launch_prefix_curvature_degrees": args.tail_launch_curvature_degrees,
            "base_seed": args.seed,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.quiet:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved: {args.output.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
