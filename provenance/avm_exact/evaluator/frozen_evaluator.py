"""Fail-closed evaluator for the formal paired HPR O1-sham/O2 study.

The default action is a read-only contract scan.  Full rollouts require the
explicit ``--execute`` flag and always cover the complete matrix frozen in the
JSON configuration.  No CLI option can change seeds, checkpoints or endpoints.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA = "formal_hpr_o1_sham_frozen_evaluator/v1"
TASK_SCHEMA = "formal_hpr_o1_sham_frozen_evaluator/task/v1"
REQUIRED_CHECKPOINT_KEYS = {"policy", "critic", "metadata"}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite JSON float: {value}")
        return value
    if hasattr(value, "item"):
        return json_safe(value.item())
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def metadata_value(metadata: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in metadata:
        return metadata[key]
    training_args = metadata.get("training_args")
    if isinstance(training_args, dict) and key in training_args:
        return training_args[key]
    return default


def exact_number(actual: Any, expected: Any, name: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"Locked contract drift: {name}={actual!r}, expected {expected!r}")


def common_kinematic_success(metrics: dict[str, Any], criterion: dict[str, Any]) -> bool:
    if criterion.get("pulse_or_contact_gate_used") is not False:
        raise RuntimeError("Common criterion unexpectedly contains a pulse/contact gate")
    return bool(
        float(metrics["desired_net_rotation_degrees"])
        >= float(criterion["minimum_desired_net_rotation_degrees"])
        and float(metrics["desired_active_rotation_fraction"])
        >= float(criterion["minimum_desired_active_rotation_fraction"])
        and float(metrics["forward_body_lengths"])
        >= float(criterion["minimum_forward_body_lengths"])
    )


def infer_actor_observation_mode(
    metadata: dict[str, Any], arm_id: str, arm_config: dict[str, Any], seed: int
) -> tuple[str, str]:
    expected_mode = str(arm_config["expected_actor_observation_mode"])
    raw_mode = metadata_value(metadata, "actor_observation_mode", None)
    expected_run = str(arm_config["expected_run_name_template"]).format(seed=seed)
    actual_run = str(metadata_value(metadata, "run_name", ""))
    if actual_run != expected_run:
        raise RuntimeError(f"Run-name drift for {arm_id}: {actual_run!r} != {expected_run!r}")
    if raw_mode is None:
        if not bool(arm_config.get("permit_locked_legacy_missing_mode", False)):
            raise RuntimeError(f"{arm_id} checkpoint omits actor_observation_mode")
        if arm_id != "O2" or expected_mode != "full_o2":
            raise RuntimeError("Legacy observation-mode inference is permitted only for O2")
        return "full_o2", "locked_archived_o2_contract"
    mode = str(raw_mode)
    if mode != expected_mode:
        raise RuntimeError(f"Actor observation mode drift for {arm_id}: {mode!r} != {expected_mode!r}")
    return mode, "checkpoint_metadata"


def validate_metadata(
    metadata: dict[str, Any], config: dict[str, Any], arm_id: str, seed: int
) -> dict[str, Any]:
    locked = config["locked_controller"]
    arm_config = config["arms"][arm_id]
    mode, mode_source = infer_actor_observation_mode(metadata, arm_id, arm_config, seed)
    exact_fields = (
        "scenario", "num_particles", "observation_func", "control_mode",
        "reward_func", "algorithm", "per_joint_k1_k2", "share_policy",
        "share_critic", "centralised_critic", "terrain_type",
        "terrain_contact_mode", "init_pos_randomness",
        "init_angle_range_degrees", "init_height_jitter",
    )
    aliases = {"num_particles": ("n_particles", "num_particles")}
    for key in exact_fields:
        candidates = aliases.get(key, (key,))
        actual = None
        for candidate in candidates:
            actual = metadata_value(metadata, candidate, None)
            if actual is not None:
                break
        if actual != locked[key]:
            raise RuntimeError(f"Locked metadata drift: {key}={actual!r}, expected {locked[key]!r}")
    for key in ("feedback_gain", "k_action_scale"):
        exact_number(metadata_value(metadata, key), locked[key], key)
    if tuple(metadata_value(metadata, "formula_action_names", ())) != tuple(
        locked["formula_action_names"]
    ):
        raise RuntimeError("Formula action order is not [k1, k2]")
    if bool(metadata_value(metadata, "fix_k1", True)) or bool(
        metadata_value(metadata, "fix_k2", True)
    ):
        raise RuntimeError("K1/K2 output channel is fixed")
    if metadata_value(metadata, "pretrained_model_path", None) is not None:
        raise RuntimeError("Checkpoint is not a from-scratch formal run")
    if arm_id == "O1_sham":
        critic_mode = metadata_value(metadata, "critic_observation_mode", None)
        if critic_mode != locked["critic_observation_mode"]:
            raise RuntimeError(
                f"O1-sham critic mode drift: {critic_mode!r} != {locked['critic_observation_mode']!r}"
            )
    return {
        "actor_observation_mode": mode,
        "actor_observation_mode_source": mode_source,
        "critic_observation_mode": (
            str(metadata_value(metadata, "critic_observation_mode"))
            if arm_id == "O1_sham"
            else "full_o2_by_archived_centralised_critic_contract"
        ),
        "metadata_contract_valid": True,
    }


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != "formal_hpr_o1_sham_frozen_evaluator_config/v1":
        raise RuntimeError("Evaluator config schema drift")
    if list(config["training_seeds"]) != list(range(9201, 9206)):
        raise RuntimeError("Training-seed matrix drift")
    if config["seed_to_paper_run"] != {str(seed): seed - 9201 for seed in range(9201, 9206)}:
        raise RuntimeError("Paper run mapping drift")
    if list(config["checkpoint_batches"]) != list(range(100, 1501, 100)):
        raise RuntimeError("Checkpoint schedule drift")
    resets = list(config["evaluation"]["reset_seeds"])
    if resets != list(range(20264101, 20264121)):
        raise RuntimeError("Reset-seed panel drift")
    if int(config["evaluation"]["steps"]) != 1000:
        raise RuntimeError("Rollout duration drift")
    criterion = config["evaluation"]["common_kinematic_criterion"]
    expected = {
        "minimum_desired_net_rotation_degrees": 360.0,
        "minimum_desired_active_rotation_fraction": 0.7,
        "minimum_forward_body_lengths": 1.0,
        "pulse_or_contact_gate_used": False,
    }
    if criterion != expected:
        raise RuntimeError("Common kinematic criterion drift")
    if set(config["arms"]) != {"O2", "O1_sham"}:
        raise RuntimeError("Arm matrix drift")
    if config["evaluation"]["trace_export"] != "all_episodes_all_checkpoints":
        raise RuntimeError("Trace-export matrix drift")


@dataclass(frozen=True)
class Task:
    arm_id: str
    seed: int
    checkpoint_batch: int

    @property
    def paper_run(self) -> int:
        return self.seed - 9201

    @property
    def task_id(self) -> str:
        return f"{self.arm_id}__run{self.paper_run}__checkpoint{self.checkpoint_batch:04d}"


def checkpoint_path(config: dict[str, Any], task: Task) -> Path:
    root = Path(config["parent_formal_root"])
    template = config["arms"][task.arm_id]["run_relative_template"]
    return (root / template.format(seed=task.seed) / f"checkpoint_{task.checkpoint_batch}.pt").resolve()


def all_tasks(config: dict[str, Any]) -> list[Task]:
    return [
        Task(arm_id, int(seed), int(batch))
        for arm_id in ("O2", "O1_sham")
        for seed in config["training_seeds"]
        for batch in config["checkpoint_batches"]
    ]


class Dependencies:
    def __init__(self, config: dict[str, Any]) -> None:
        site_packages = str(Path(config["runtime"]["site_packages"]).resolve())
        formal_root = Path(config["parent_formal_root"]).resolve()
        training = formal_root / config["immutable_runtime"]["training_relative"]
        for path in (site_packages, str(training)):
            if path not in sys.path:
                sys.path.insert(0, path)
        import numpy as np  # type: ignore
        import torch  # type: ignore
        from analyze_training_results import (  # type: ignore
            TerrainArgs,
            build_demo_env,
            load_policy_for_env,
            metadata_from_checkpoint,
        )
        from demo_metamaterial import choose_action  # type: ignore

        torch.set_num_threads(int(config["runtime"]["torch_num_threads_per_worker"]))
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        self.np = np
        self.torch = torch
        self.TerrainArgs = TerrainArgs
        self.build_demo_env = build_demo_env
        self.load_policy_for_env = load_policy_for_env
        self.metadata_from_checkpoint = metadata_from_checkpoint
        self.choose_action = choose_action
        metric_path = formal_root / config["immutable_runtime"]["metric_helper_relative"]
        self.metric_path = metric_path
        self.metric = load_module(metric_path, f"formal_frozen_metric_{os.getpid()}")
        self.metric_args = self.metric._parser().parse_args([])


def policy_sha256(policy: Any) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(policy.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def validate_environment(env: Any, config: dict[str, Any]) -> None:
    locked = config["locked_controller"]
    if int(getattr(env, "num_particles", -1)) != int(locked["num_particles"]):
        raise RuntimeError("Environment particle count drift")
    if str(getattr(env, "observation_func", "")) != locked["observation_func"]:
        raise RuntimeError("Environment observation function drift")
    if str(getattr(env, "control_mode", "")) != locked["control_mode"]:
        raise RuntimeError("Environment control mode drift")
    if tuple(getattr(env, "formula_action_names", ())) != tuple(locked["formula_action_names"]):
        raise RuntimeError("Environment action order drift")
    for name in ("feedback_gain", "k_action_scale", "max_torque"):
        exact_number(getattr(env, name), locked[name], f"environment.{name}")
    obs_shape = tuple(env.observation_spec[("agents", "observation")].shape)
    action_shape = tuple(env.action_spec[env.action_key].shape)
    if obs_shape != (1, 8, 2) or action_shape != (1, 8, 2):
        raise RuntimeError(f"Environment shape drift: observation={obs_shape}, action={action_shape}")


def actor_input(td: Any, mode: str, torch: Any) -> tuple[Any, Any, Any]:
    raw = td["agents", "observation"].detach().clone()
    if tuple(raw.shape) != (1, 8, 2) or not bool(torch.isfinite(raw).all().item()):
        raise RuntimeError(f"Raw observation contract failed: {tuple(raw.shape)}")
    actor = raw.clone()
    if mode == "spatial_only_sham":
        actor[..., 1] = 0.0
    elif mode != "full_o2":
        raise RuntimeError(f"Unsupported actor observation mode: {mode}")
    actor_td = td.clone(recurse=True)
    actor_td["agents", "observation"] = actor
    return raw, actor, actor_td


def run_episode(
    deps: Dependencies,
    env: Any,
    policy: Any,
    config: dict[str, Any],
    mode: str,
    reset_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    np, torch = deps.np, deps.torch
    steps = int(config["evaluation"]["steps"])
    torch.manual_seed(int(reset_seed))
    np.random.seed(int(reset_seed))
    td = env.reset()
    positions = [deps.metric._positions(env)]
    support = [deps.metric._log_info_scalar(td, "fast_forward_support_index")]
    contact = [deps.metric._log_info_scalar(td, "fast_forward_ground_contact_strength")]
    raw_obs = np.empty((steps, 8, 2), dtype=np.float32)
    actor_obs = np.empty_like(raw_obs)
    action = np.empty_like(raw_obs)
    gain = np.empty_like(raw_obs)
    tau1 = np.empty((steps, 8), dtype=np.float32)
    tau2 = np.empty_like(tau1)
    tau_unclipped = np.empty_like(tau1)
    tau_clipped = np.empty_like(tau1)
    saturated = np.empty((steps, 8), dtype=np.uint8)
    scale = float(env.k_action_scale)
    max_torque = float(env.max_torque)
    for step in range(steps):
        raw_t, actor_t, actor_td = actor_input(td, mode, torch)
        chosen = deps.choose_action(policy, actor_td, "deterministic")
        action_t = chosen["agents", "action"].detach().clone()
        if tuple(action_t.shape) != (1, 8, 2) or not bool(torch.isfinite(action_t).all().item()):
            raise RuntimeError(f"Actor action contract failed at step {step}")
        raw_np = raw_t[0].cpu().numpy().astype(np.float32, copy=True)
        actor_np = actor_t[0].cpu().numpy().astype(np.float32, copy=True)
        action_np = action_t[0].cpu().numpy().astype(np.float32, copy=True)
        gain_np = scale * action_np
        tau1_np = gain_np[:, 0] * raw_np[:, 0]
        # raw observation channel 1 is exactly feedback_gain * theta_dot in the
        # locked environment, so multiplying by K2 reconstructs the active term.
        tau2_np = gain_np[:, 1] * raw_np[:, 1]
        total_np = tau1_np + tau2_np
        raw_obs[step], actor_obs[step], action[step], gain[step] = (
            raw_np, actor_np, action_np, gain_np
        )
        tau1[step], tau2[step], tau_unclipped[step] = tau1_np, tau2_np, total_np
        tau_clipped[step] = np.clip(total_np, -max_torque, max_torque)
        saturated[step] = (np.abs(total_np) >= max_torque).astype(np.uint8)
        env_td = td.clone(recurse=True)
        env_td["agents", "action"] = action_t
        td = env.step(env_td)["next"]
        position = deps.metric._positions(env)
        if not np.isfinite(np.real(position)).all() or not np.isfinite(np.imag(position)).all():
            raise RuntimeError(f"Non-finite position at step {step + 1}")
        positions.append(position)
        support.append(deps.metric._log_info_scalar(td, "fast_forward_support_index"))
        contact.append(deps.metric._log_info_scalar(td, "fast_forward_ground_contact_strength"))
    if mode == "spatial_only_sham" and not np.array_equal(actor_obs[..., 1], np.zeros_like(actor_obs[..., 1])):
        raise RuntimeError("O1-sham actor second channel was not exactly zero")
    if mode == "full_o2" and not np.array_equal(actor_obs, raw_obs):
        raise RuntimeError("O2 actor input differs from raw environment observation")
    if mode == "spatial_only_sham" and float(np.max(np.abs(raw_obs[..., 1]))) <= 1e-8:
        raise RuntimeError("O1-sham manipulation check failed: raw theta-dot channel was always zero")
    metrics = deps.metric._episode_metrics(
        positions,
        config["evaluation"]["desired_direction"],
        config["evaluation"]["tail_side"],
        deps.metric_args,
        support,
        contact,
    )
    metrics["reset_seed"] = int(reset_seed)
    metrics["success_common_kinematic"] = common_kinematic_success(
        metrics, config["evaluation"]["common_kinematic_criterion"]
    )
    metrics["mean_abs_k1"] = float(np.mean(np.abs(gain[..., 0])))
    metrics["mean_abs_k2"] = float(np.mean(np.abs(gain[..., 1])))
    metrics["torque_saturation_fraction"] = float(np.mean(saturated))
    metrics["raw_theta_dot_abs_mean"] = float(np.mean(np.abs(raw_obs[..., 1])))
    metrics["actor_theta_dot_abs_mean"] = float(np.mean(np.abs(actor_obs[..., 1])))
    position_a = np.asarray(positions, dtype=np.complex128)
    arrays = {
        "positions_xy": np.stack((np.real(position_a), np.imag(position_a)), axis=-1).astype(np.float32),
        "raw_environment_observation": raw_obs,
        "actor_input_observation": actor_obs,
        "normalised_action": action,
        "physical_gain": gain,
        "tau_k1_unclipped": tau1,
        "tau_k2_unclipped": tau2,
        "tau_active_unclipped": tau_unclipped,
        "tau_active_clipped": tau_clipped,
        "torque_saturated": saturated,
        "support_index": np.asarray([np.nan if x is None else x for x in support], dtype=np.float32),
        "ground_contact_strength": np.asarray([np.nan if x is None else x for x in contact], dtype=np.float32),
    }
    return metrics, arrays


def metric_projection(row: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(row[key])
        for key in (
            "initial_body_length", "forward_displacement", "forward_body_lengths",
            "net_best_fit_rotation_degrees", "desired_net_rotation_degrees",
            "desired_active_rotation_fraction",
        )
    }


def o2_endpoint_identity_gate(
    config: dict[str, Any], task: Task, episodes: list[dict[str, Any]]
) -> dict[str, Any] | None:
    if task.arm_id != "O2" or task.checkpoint_batch != int(config["evaluation"]["endpoint_batch"]):
        return None
    template = config["arms"]["O2"]["official_endpoint_relative_template"]
    official_path = Path(config["parent_formal_root"]) / template.format(seed=task.seed)
    payload = read_json(official_path)
    official = payload["results"][0]["episodes"]
    expected = {int(row["seed"]): row for row in official}
    tolerance = float(config["evaluation"]["official_identity_absolute_tolerance"])
    maximum = {key: 0.0 for key in metric_projection(episodes[0])}
    if len(episodes) != 20 or len(expected) != 20:
        raise RuntimeError("O2 endpoint identity episode-count mismatch")
    for row in episodes:
        reset_seed = int(row["reset_seed"])
        if reset_seed not in expected:
            raise RuntimeError(f"O2 official endpoint lacks reset seed {reset_seed}")
        actual_fields = metric_projection(row)
        expected_fields = metric_projection(expected[reset_seed])
        for key, actual in actual_fields.items():
            error = abs(actual - expected_fields[key])
            maximum[key] = max(maximum[key], error)
            if error > tolerance:
                raise RuntimeError(
                    f"O2 endpoint identity mismatch run {task.paper_run}, reset {reset_seed}, "
                    f"{key}: {error:.3g} > {tolerance:.3g}"
                )
    return {
        "passed": True,
        "official_path": str(official_path.resolve()),
        "official_sha256": sha256_file(official_path),
        "absolute_tolerance": tolerance,
        "maximum_absolute_error": maximum,
    }


def task_paths(output_root: Path, task: Task) -> tuple[Path, Path]:
    directory = output_root / "tasks" / task.arm_id / f"run{task.paper_run}"
    base = directory / f"checkpoint_{task.checkpoint_batch:04d}"
    return base.with_suffix(".json"), base.with_suffix(".npz")


def atomic_npz(path: Path, np: Any, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def evaluate_task(config_path: str, output_root_string: str, task_dict: dict[str, Any]) -> dict[str, Any]:
    config_path_p = Path(config_path).resolve()
    config = read_json(config_path_p)
    validate_config(config)
    task = Task(**task_dict)
    output_root = Path(output_root_string).resolve()
    manifest_path, trace_path = task_paths(output_root, task)
    if manifest_path.exists() or trace_path.exists():
        raise FileExistsError(f"Refusing to overwrite task output: {manifest_path} / {trace_path}")
    checkpoint = checkpoint_path(config, task)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha_before = sha256_file(checkpoint)
    deps = Dependencies(config)
    raw_checkpoint = deps.torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (set(raw_checkpoint) & REQUIRED_CHECKPOINT_KEYS) != REQUIRED_CHECKPOINT_KEYS:
        raise RuntimeError(f"Checkpoint payload keys drift: {set(raw_checkpoint)}")
    metadata = dict(deps.metadata_from_checkpoint(checkpoint))
    contract = validate_metadata(metadata, config, task.arm_id, task.seed)
    env, resolved_name, resolved_type, resolved_settings = deps.build_demo_env(
        metadata,
        config["evaluation"]["terrain"],
        deps.TerrainArgs(),
        max_steps=int(config["evaluation"]["steps"]),
        render_mode="rgb_array",
        num_envs=1,
    )
    validate_environment(env, config)
    policy = deps.load_policy_for_env(checkpoint, env, metadata)
    policy_hash_before = policy_sha256(policy)
    mode = contract["actor_observation_mode"]
    episodes: list[dict[str, Any]] = []
    episode_arrays: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for reset_seed in config["evaluation"]["reset_seeds"]:
            metrics, arrays = run_episode(
                deps, env, policy, config, mode, int(reset_seed)
            )
            episodes.append(metrics)
            episode_arrays.append(arrays)
        policy_hash_after = policy_sha256(policy)
    finally:
        close_env(env)
    checkpoint_sha_after = sha256_file(checkpoint)
    if checkpoint_sha_after != checkpoint_sha_before:
        raise RuntimeError("Checkpoint changed during evaluation")
    if policy_hash_after != policy_hash_before:
        raise RuntimeError("Policy state changed during evaluation")
    identity = o2_endpoint_identity_gate(config, task, episodes)
    np = deps.np
    trace_arrays = {
        "reset_seeds": np.asarray(config["evaluation"]["reset_seeds"], dtype=np.int64)
    }
    for key in config["required_trace_arrays"]:
        if key == "reset_seeds":
            continue
        trace_arrays[key] = np.stack([episode[key] for episode in episode_arrays], axis=0)
    atomic_npz(trace_path, np, trace_arrays)
    success_count = int(sum(bool(row["success_common_kinematic"]) for row in episodes))
    task_signature = {
        "config_sha256": sha256_file(config_path_p),
        "task": task_dict,
        "checkpoint_sha256": checkpoint_sha_before,
        "steps": config["evaluation"]["steps"],
        "reset_seeds": config["evaluation"]["reset_seeds"],
        "actor_observation_mode": mode,
    }
    payload = {
        "schema": TASK_SCHEMA,
        "study_id": config["study_id"],
        "status": "complete",
        "task": {
            **task_dict,
            "paper_run": task.paper_run,
            "task_id": task.task_id,
        },
        "task_signature": task_signature,
        "task_signature_sha256": sha256_json(task_signature),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256_before_after": checkpoint_sha_before,
            "unchanged": True,
        },
        "runtime_contract": contract,
        "environment": {
            "terrain_name": resolved_name,
            "terrain_type": resolved_type,
            "terrain_settings": resolved_settings,
            "raw_observation": "[s_i, theta_dot_i]",
            "actor_observation": "[s_i, 0]" if mode == "spatial_only_sham" else "[s_i, theta_dot_i]",
            "physical_k2_theta_dot_enabled": True,
        },
        "evaluation": {
            "steps": config["evaluation"]["steps"],
            "reset_seeds": config["evaluation"]["reset_seeds"],
            "common_kinematic_criterion": config["evaluation"]["common_kinematic_criterion"],
            "success_count": success_count,
            "success_rate": success_count / len(episodes),
            "training_run_discovered_rolling": success_count
            >= int(config["evaluation"]["training_run_discovery_threshold_successes"]),
            "wall_seconds": time.perf_counter() - started,
        },
        "policy": {
            "state_sha256_before_after": policy_hash_before,
            "unchanged": True,
        },
        "o2_endpoint_identity_gate": identity,
        "trace_archive": {
            "path": str(trace_path),
            "sha256": sha256_file(trace_path),
            "array_shapes": {key: list(value.shape) for key, value in trace_arrays.items()},
        },
        "episodes": episodes,
    }
    atomic_json(manifest_path, payload)
    return {
        "task_id": task.task_id,
        "manifest": str(manifest_path),
        "success_count": success_count,
        "wall_seconds": payload["evaluation"]["wall_seconds"],
    }


def contract_scan(config_path: Path, require_all_checkpoints: bool) -> dict[str, Any]:
    config = read_json(config_path)
    validate_config(config)
    formal_root = Path(config["parent_formal_root"]).resolve()
    source_hashes: dict[str, str] = {}
    for key, relative in config["immutable_runtime"].items():
        if not key.endswith("_relative"):
            continue
        path = formal_root / relative
        if not path.is_file() and key != "training_relative":
            raise FileNotFoundError(path)
        if path.is_file():
            source_hashes[key] = sha256_file(path)
    missing: list[str] = []
    present = 0
    for task in all_tasks(config):
        path = checkpoint_path(config, task)
        if path.is_file():
            present += 1
        else:
            missing.append(str(path))
    if require_all_checkpoints and missing:
        raise FileNotFoundError(
            f"Formal matrix is incomplete: {len(missing)} checkpoints missing; first={missing[0]}"
        )
    return {
        "schema": SCHEMA,
        "mode": "contract_scan",
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "task_count": len(all_tasks(config)),
        "checkpoint_count_present": present,
        "checkpoint_count_missing": len(missing),
        "missing_checkpoints": missing,
        "source_sha256": source_hashes,
        "ready_for_complete_evaluation": not missing,
    }


def episode_csv_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    task = payload["task"]
    fields = (
        "reset_seed", "success_common_kinematic", "desired_net_rotation_degrees",
        "desired_active_rotation_fraction", "forward_body_lengths", "mean_abs_k1",
        "mean_abs_k2", "torque_saturation_fraction", "raw_theta_dot_abs_mean",
        "actor_theta_dot_abs_mean",
    )
    return [
        {
            "arm_id": task["arm_id"],
            "paper_run": task["paper_run"],
            "internal_training_seed": task["seed"],
            "checkpoint_batch": task["checkpoint_batch"],
            **{field: row[field] for field in fields},
        }
        for row in payload["episodes"]
    ]


def aggregate(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    records: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    for task in all_tasks(config):
        manifest, trace = task_paths(output_root, task)
        if not manifest.is_file() or not trace.is_file():
            raise RuntimeError(f"Incomplete evaluation task: {task.task_id}")
        payload = read_json(manifest)
        if payload.get("schema") != TASK_SCHEMA or payload.get("status") != "complete":
            raise RuntimeError(f"Invalid task manifest: {manifest}")
        if sha256_file(trace) != payload["trace_archive"]["sha256"]:
            raise RuntimeError(f"Trace hash drift: {trace}")
        records.append(
            {
                **payload["task"],
                "success_count": payload["evaluation"]["success_count"],
                "success_rate": payload["evaluation"]["success_rate"],
                "training_run_discovered_rolling": payload["evaluation"]["training_run_discovered_rolling"],
                "manifest": str(manifest),
                "trace": str(trace),
            }
        )
        episode_rows.extend(episode_csv_rows(payload))
    csv_path = output_root / "episode_results.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episode_rows[0]))
        writer.writeheader()
        writer.writerows(episode_rows)
    endpoint = int(config["evaluation"]["endpoint_batch"])
    endpoint_records = [row for row in records if int(row["checkpoint_batch"]) == endpoint]
    result = {
        "schema": SCHEMA,
        "mode": "complete_evaluation",
        "study_id": config["study_id"],
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "task_count": len(records),
        "episode_count": len(episode_rows),
        "complete_locked_matrix": len(records) == 150 and len(episode_rows) == 3000,
        "endpoint_batch": endpoint,
        "common_kinematic_criterion": config["evaluation"]["common_kinematic_criterion"],
        "endpoint_records": endpoint_records,
        "task_records": records,
        "episode_results_csv": str(csv_path),
        "episode_results_csv_sha256": sha256_file(csv_path),
    }
    if not result["complete_locked_matrix"]:
        raise RuntimeError("Aggregate is not the complete 2 x 5 x 15 x 20 matrix")
    atomic_json(output_root / "STUDY_MANIFEST.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=Path(__file__).with_name("evaluator_config.json"))
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--contract-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    result.add_argument("--workers", type=int, default=1)
    return result


def main() -> None:
    args = parser().parse_args()
    config_path = args.config.resolve()
    config = read_json(config_path)
    validate_config(config)
    if not args.execute:
        result = contract_scan(config_path, require_all_checkpoints=False)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    gate = contract_scan(config_path, require_all_checkpoints=True)
    if not gate["ready_for_complete_evaluation"]:
        raise RuntimeError("Formal matrix is not complete")
    workers = int(args.workers)
    if workers < 1 or workers > int(config["runtime"]["maximum_workers"]):
        raise ValueError(f"workers must be 1..{config['runtime']['maximum_workers']}")
    output_root = Path(config["output_root"]).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    start_marker = output_root / "EVALUATION_STARTED.json"
    if start_marker.exists():
        raise FileExistsError(f"Refusing a second full evaluation: {start_marker}")
    atomic_json(
        start_marker,
        {
            "schema": SCHEMA,
            "status": "started",
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "task_count": 150,
            "episode_count": 3000,
        },
    )
    tasks = all_tasks(config)
    started = time.perf_counter()
    if workers == 1:
        for index, task in enumerate(tasks, start=1):
            result = evaluate_task(str(config_path), str(output_root), task.__dict__)
            print(f"[{index:03d}/150] {result['task_id']}: {result['success_count']}/20", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(evaluate_task, str(config_path), str(output_root), task.__dict__): task
                for task in tasks
            }
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                print(f"[{completed:03d}/150] {result['task_id']}: {result['success_count']}/20", flush=True)
    study = aggregate(config_path, output_root)
    atomic_json(
        output_root / "EVALUATION_COMPLETE.json",
        {
            "schema": SCHEMA,
            "status": "complete",
            "wall_seconds": time.perf_counter() - started,
            "study_manifest": str(output_root / "STUDY_MANIFEST.json"),
            "study_manifest_sha256": sha256_file(output_root / "STUDY_MANIFEST.json"),
            "complete_locked_matrix": study["complete_locked_matrix"],
        },
    )


if __name__ == "__main__":
    main()
