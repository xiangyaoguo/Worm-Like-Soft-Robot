from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from condition_matrix import Condition, build_conditions


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "study_config.json"
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_config_path(config: dict[str, Any], key: str, fallback: Path) -> Path:
    """Resolve portable paths relative to the config that selected them."""

    raw = config.get(key)
    if raw is None:
        return fallback.resolve()
    path = Path(str(raw))
    if not path.is_absolute():
        path = CONFIG_PATH.resolve().parent / path
    return path.resolve()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def tensor_max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if tuple(left.shape) != tuple(right.shape):
        return math.inf
    if torch.equal(left, right):
        return 0.0
    return float(torch.max(torch.abs(left - right)).detach().cpu().item())


def load_frozen_evaluator(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("frozen_fast_forward_evaluator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import frozen evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metadata_value(metadata: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in metadata:
        return metadata[key]
    training_args = metadata.get("training_args")
    if isinstance(training_args, dict) and key in training_args:
        return training_args[key]
    return default


def episode_success(metrics: dict[str, Any], criteria: dict[str, Any]) -> bool:
    interval = metrics.get("mean_roll_pulse_interval_steps")
    return bool(
        int(metrics["roll_pulse_count"]) >= int(criteria["minimum_roll_pulses"])
        and float(metrics["desired_net_rotation_degrees"])
        >= float(criteria["minimum_desired_net_rotation_degrees"])
        and float(metrics["desired_active_rotation_fraction"])
        >= float(criteria["minimum_direction_fraction"])
        and float(metrics["forward_body_lengths"])
        >= float(criteria["minimum_forward_body_lengths"])
        and interval is not None
        and float(interval) <= float(criteria["maximum_mean_inter_pulse_interval_steps"])
    )


def metric_projection(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "steps",
        "initial_body_length",
        "forward_displacement",
        "forward_body_lengths",
        "net_best_fit_rotation_degrees",
        "desired_net_rotation_degrees",
        "desired_active_rotation_fraction",
        "contact_material_index_span_fraction",
        "contact_metric_source",
        "roll_pulse_count",
        "roll_pulse_intervals_steps",
        "mean_roll_pulse_interval_steps",
        "tail_launch_detected",
        "tail_launch_count",
    )
    return {key: metrics.get(key) for key in keys}


def compare_metric_value(actual: Any, expected: Any, tolerance: float, path: str) -> None:
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if actual != expected:
            raise RuntimeError(f"Identity mismatch at {path}: {actual!r} != {expected!r}")
        return
    if isinstance(expected, (int, float)):
        if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance):
            raise RuntimeError(f"Identity mismatch at {path}: {actual!r} != {expected!r}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise RuntimeError(f"Identity sequence mismatch at {path}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            compare_metric_value(left, right, tolerance, f"{path}[{index}]")
        return
    if actual != expected:
        raise RuntimeError(f"Identity mismatch at {path}: {actual!r} != {expected!r}")


class SeedRuntime:
    COMPATIBILITY_KEYS = (
        "scenario",
        "n_particles",
        "observation_func",
        "control_mode",
        "k_action_scale",
        "fix_k1",
        "fix_k2",
        "per_joint_k1_k2",
        "share_policy",
        "share_critic",
        "centralised_critic",
        "max_control_gain",
        "feedback_gain",
        "max_torque",
        "terrain_type",
        "terrain_settings",
        "terrain_contact_mode",
        "init_pos_randomness",
        "init_angle_range",
        "init_height_jitter",
    )

    def __init__(self, seed: int, stage: str, environment_arm: str = "Rroll") -> None:
        self.config = load_json(CONFIG_PATH)
        self.seed = int(seed)
        self.stage = stage
        if environment_arm not in {"R0", "Rroll"}:
            raise ValueError(f"Unsupported environment arm: {environment_arm}")
        self.environment_arm = environment_arm
        self.formal_root = resolve_config_path(
            self.config, "formal_root", CONFIG_PATH.resolve().parent
        )
        self.snapshot = resolve_config_path(
            self.config,
            "source_snapshot_root",
            self.formal_root / "_control" / "code_snapshot",
        )
        self.training_dir = self.snapshot / "training"
        sys.path.insert(0, str(self.training_dir))
        from analyze_training_results import (  # type: ignore
            TerrainArgs,
            build_demo_env,
            load_policy_for_env,
            metadata_from_checkpoint,
        )
        from demo_metamaterial import choose_action  # type: ignore

        self.TerrainArgs = TerrainArgs
        self.build_demo_env = build_demo_env
        self.load_policy_for_env = load_policy_for_env
        self.metadata_from_checkpoint = metadata_from_checkpoint
        self.choose_action = choose_action
        self.frozen_eval_path = self.training_dir / "evaluate_fast_forward_roll.py"
        self.frozen_eval = load_frozen_evaluator(self.frozen_eval_path)
        self.metric_args = self.frozen_eval._parser().parse_args([])
        self._validate_metric_contract()

        run_root = resolve_config_path(
            self.config,
            "formal_runs_root",
            self.formal_root / "formal" / "runs",
        )
        self.r0_checkpoint = (
            run_root / f"formal__seed{seed}__R0" / "checkpoint_1500.pt"
        ).resolve()
        self.roll_checkpoint = (
            run_root / f"formal__seed{seed}__Rroll" / "checkpoint_1500.pt"
        ).resolve()
        for path in (self.r0_checkpoint, self.roll_checkpoint):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.checkpoint_hashes_before = {
            "R0": sha256_file(self.r0_checkpoint),
            "Rroll": sha256_file(self.roll_checkpoint),
        }
        self.r0_metadata = dict(self.metadata_from_checkpoint(self.r0_checkpoint))
        self.roll_metadata = dict(self.metadata_from_checkpoint(self.roll_checkpoint))
        self._validate_metadata_pair()

        environment_metadata = (
            self.r0_metadata if environment_arm == "R0" else self.roll_metadata
        )
        self.env, *_ = self.build_demo_env(
            environment_metadata,
            "flat",
            self.TerrainArgs(),
            max_steps=self.steps,
            render_mode="rgb_array",
            num_envs=1,
        )
        self._validate_environment()
        self.r0_policy = self.load_policy_for_env(
            self.r0_checkpoint, self.env, self.r0_metadata
        )
        self.roll_policy = self.load_policy_for_env(
            self.roll_checkpoint, self.env, self.roll_metadata
        )
        self.policy_hashes_before = {
            "R0": policy_sha256(self.r0_policy),
            "Rroll": policy_sha256(self.roll_policy),
        }
        self.calibration = self._load_calibration() if stage == "main" else None
        self.permutation = np.random.default_rng(
            int(self.config["main_evaluation"]["fixed_time_permutation_seed"])
        ).permutation(self.steps)

    @property
    def steps(self) -> int:
        section = "identity_gate" if self.stage == "identity" else "main_evaluation"
        return int(self.config[section]["steps"])

    def _validate_metric_contract(self) -> None:
        expected = self.config["metric_parameters"]
        mapping = {
            "pulse_rotation_degrees": "pulse_rotation_degrees",
            "pulse_forward_body_fraction": "pulse_forward_body_fraction",
            "pulse_contact_index_fraction": "pulse_contact_index_fraction",
            "active_rotation_degrees": "active_rotation_degrees",
            "pulse_reset_drawdown_degrees": "pulse_reset_drawdown_degrees",
            "pulse_reset_backward_body_fraction": "pulse_reset_backward_body_fraction",
            "tail_launch_lift_fraction": "tail_launch_lift_fraction",
            "tail_launch_forward_fraction": "tail_launch_forward_fraction",
            "tail_launch_curvature_degrees": "tail_launch_curvature_degrees",
            "tail_launch_min_separation": "tail_launch_min_separation",
        }
        for config_key, arg_key in mapping.items():
            actual = getattr(self.metric_args, arg_key)
            if float(actual) != float(expected[config_key]):
                raise RuntimeError(
                    f"Frozen evaluator parameter drift: {arg_key}={actual}, "
                    f"expected {expected[config_key]}"
                )

    def _validate_metadata_pair(self) -> None:
        mismatches: list[str] = []
        for key in self.COMPATIBILITY_KEYS:
            left = metadata_value(self.r0_metadata, key)
            right = metadata_value(self.roll_metadata, key)
            if left != right:
                mismatches.append(f"{key}: R0={left!r}, Rroll={right!r}")
        if mismatches:
            raise RuntimeError("R0/Rroll physical-controller mismatch:\n" + "\n".join(mismatches))
        locked = self.config["locked_contract"]
        required = {
            "control_mode": locked["control_mode"],
            "observation_func": "dth_tot_plus_friction_thdot",
            "per_joint_k1_k2": True,
            "k_action_scale": locked["k_action_scale"],
            "max_torque": locked["max_torque"],
            "terrain_contact_mode": locked["terrain_contact_mode"],
        }
        for key, expected in required.items():
            actual = metadata_value(self.roll_metadata, key)
            if key == "max_torque" and actual is None:
                # This formal checkpoint does not duplicate max_torque in its
                # metadata.  The immutable constructed environment is checked
                # for exactly 9.0 in _validate_environment instead.
                continue
            if actual != expected:
                raise RuntimeError(f"Locked metadata drift: {key}={actual!r}, expected={expected!r}")

    def _validate_environment(self) -> None:
        if int(getattr(self.env, "num_particles", -1)) != 10:
            raise RuntimeError("Environment particle count is not 10")
        if float(getattr(self.env, "k_action_scale", math.nan)) != 100.0:
            raise RuntimeError("Environment K scale is not 100")
        if float(getattr(self.env, "max_torque", math.nan)) != 9.0:
            raise RuntimeError("Environment max torque is not 9")
        names = tuple(getattr(self.env, "formula_action_names", ()))
        if names != ("k1", "k2"):
            raise RuntimeError(f"Action order drift: {names!r}")

    def _load_calibration(self) -> dict[str, np.ndarray]:
        path = ROOT / "calibration" / f"seed{self.seed}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing identity-gate calibration: {path}")
        with np.load(path, allow_pickle=False) as data:
            template = np.asarray(data["k2_time_template"], dtype=np.float32)
            static = np.asarray(data["k2_static_mean"], dtype=np.float32)
        if template.shape != (self.steps, 8) or static.shape != (8,):
            raise RuntimeError(
                f"Calibration shape mismatch: template={template.shape}, static={static.shape}"
            )
        if not np.isfinite(template).all() or not np.isfinite(static).all():
            raise RuntimeError("Calibration contains NaN/Inf")
        return {"template": template, "static": static}

    def actor_actions(self, td: Any) -> tuple[torch.Tensor, torch.Tensor, float]:
        observation_a = td["agents", "observation"].detach().clone()
        observation_b = observation_a.detach().clone()
        r0_td = self.choose_action(
            self.r0_policy, td.clone(recurse=True), "deterministic"
        )
        roll_td = self.choose_action(
            self.roll_policy, td.clone(recurse=True), "deterministic"
        )
        r0 = r0_td["agents", "action"].detach().clone()
        roll = roll_td["agents", "action"].detach().clone()
        expected_shape = (1, 8, 2)
        for label, value in (("R0", r0), ("Rroll", roll)):
            if tuple(value.shape) != expected_shape:
                raise RuntimeError(f"{label} action shape {tuple(value.shape)} != {expected_shape}")
            if not bool(torch.isfinite(value).all().item()):
                raise RuntimeError(f"{label} action contains NaN/Inf")
        return r0, roll, tensor_max_abs(observation_a, observation_b)

    def apply_condition(
        self,
        condition: Condition,
        r0: torch.Tensor,
        roll: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        spec = condition.spec
        op = str(spec["op"])
        result = r0.clone()
        if op == "source_mix":
            for joint in spec.get("k1_roll_joints", []):
                result[..., int(joint), 0] = roll[..., int(joint), 0]
            for joint in spec.get("k2_roll_joints", []):
                result[..., int(joint), 1] = roll[..., int(joint), 1]
        elif op == "k1_zero":
            result = roll.clone()
            result[..., :, 0] = 0.0
        elif op == "k1_sign":
            signs = torch.as_tensor(spec["signs"], dtype=roll.dtype, device=roll.device)
            if tuple(signs.shape) != (8,):
                raise RuntimeError(f"Invalid K1 sign vector: {tuple(signs.shape)}")
            result = roll.clone()
            result[..., :, 0] = signs * torch.abs(roll[..., :, 0])
        elif op == "k1_reverse":
            result = roll.clone()
            result[..., :, 0] = torch.flip(roll[..., :, 0], dims=(-1,))
        elif op == "k2_scale":
            result = roll.clone()
            result[..., :, 1] = float(spec["alpha"]) * roll[..., :, 1]
        elif op == "k2_sign_force":
            result = roll.clone()
            sign = float(spec["sign"])
            if sign not in (-1.0, 1.0):
                raise RuntimeError(f"Invalid K2 sign: {sign}")
            result[..., :, 1] = sign * torch.abs(roll[..., :, 1])
        elif op == "k2_region":
            result = roll.clone()
            keep = {int(value) for value in spec["keep_joints"]}
            for joint in range(8):
                if joint not in keep:
                    result[..., joint, 1] = 0.0
        elif op == "k2_calibration_static_mean":
            if self.calibration is None:
                raise RuntimeError("Static calibration requested outside main stage")
            result = roll.clone()
            value = torch.as_tensor(
                self.calibration["static"], dtype=roll.dtype, device=roll.device
            )
            result[..., :, 1] = value
        elif op in {
            "k2_calibration_time_template",
            "k2_calibration_permuted_template",
        }:
            if self.calibration is None:
                raise RuntimeError("Time calibration requested outside main stage")
            result = roll.clone()
            index = (
                int(self.permutation[step])
                if op == "k2_calibration_permuted_template"
                else step
            )
            value = torch.as_tensor(
                self.calibration["template"][index],
                dtype=roll.dtype,
                device=roll.device,
            )
            result[..., :, 1] = value
        else:
            raise ValueError(f"Unsupported intervention op: {op}")
        if tuple(result.shape) != (1, 8, 2) or not bool(torch.isfinite(result).all().item()):
            raise RuntimeError(f"Applied action contract failed for {condition.id}")
        return result

    def run_episode(
        self,
        condition: Condition,
        episode_seed: int,
        capture_calibration: bool = False,
    ) -> tuple[dict[str, Any], np.ndarray | None]:
        torch.manual_seed(int(episode_seed))
        np.random.seed(int(episode_seed))
        td = self.env.reset()
        trajectory = [self.frozen_eval._positions(self.env)]
        support = [self.frozen_eval._log_info_scalar(td, "fast_forward_support_index")]
        contact = [
            self.frozen_eval._log_info_scalar(td, "fast_forward_ground_contact_strength")
        ]
        k2_trace = np.zeros((self.steps, 8), dtype=np.float32) if capture_calibration else None
        sums = {
            "k": np.zeros((8, 2), dtype=np.float64),
            "abs_k": np.zeros((8, 2), dtype=np.float64),
            "positive": np.zeros((8, 2), dtype=np.float64),
            "source_delta_abs": np.zeros((8, 2), dtype=np.float64),
            "tau1_sq": np.zeros(8, dtype=np.float64),
            "tau2_sq": np.zeros(8, dtype=np.float64),
            "tau_sq": np.zeros(8, dtype=np.float64),
            "power_abs": np.zeros(8, dtype=np.float64),
            "saturated": np.zeros(8, dtype=np.float64),
        }
        same_observation_error = 0.0
        action_scale = float(self.env.k_action_scale)
        feedback_gain = float(getattr(self.env, "feedback_gain", 1.0))
        max_torque = float(self.env.max_torque)
        for step in range(self.steps):
            observation = td["agents", "observation"][0].detach().cpu().numpy().copy()
            if observation.shape != (8, 2) or not np.isfinite(observation).all():
                raise RuntimeError(f"Observation contract failed at step {step}")
            r0, roll, obs_error = self.actor_actions(td)
            same_observation_error = max(same_observation_error, obs_error)
            applied = self.apply_condition(condition, r0, roll, step)
            action_td = td.clone(recurse=True)
            action_td["agents", "action"] = applied

            applied_np = applied[0].detach().cpu().numpy().astype(np.float64)
            roll_np = roll[0].detach().cpu().numpy().astype(np.float64)
            physical_k = action_scale * applied_np
            tau1 = physical_k[:, 0] * observation[:, 0]
            tau2 = physical_k[:, 1] * feedback_gain * observation[:, 1]
            tau = tau1 + tau2
            sums["k"] += physical_k
            sums["abs_k"] += np.abs(physical_k)
            sums["positive"] += physical_k > 0.0
            sums["source_delta_abs"] += np.abs(applied_np - roll_np) * action_scale
            sums["tau1_sq"] += np.square(tau1)
            sums["tau2_sq"] += np.square(tau2)
            sums["tau_sq"] += np.square(tau)
            sums["power_abs"] += np.abs(np.clip(tau, -max_torque, max_torque) * observation[:, 1])
            sums["saturated"] += np.abs(tau) >= max_torque
            if k2_trace is not None:
                k2_trace[step] = roll_np[:, 1]

            td = self.env.step(action_td)["next"]
            trajectory.append(self.frozen_eval._positions(self.env))
            support.append(
                self.frozen_eval._log_info_scalar(td, "fast_forward_support_index")
            )
            contact.append(
                self.frozen_eval._log_info_scalar(
                    td, "fast_forward_ground_contact_strength"
                )
            )

        metrics = self.frozen_eval._episode_metrics(
            trajectory,
            "right",
            "left",
            self.metric_args,
            support,
            contact,
        )
        if (
            self.environment_arm == "Rroll"
            and metrics.get("contact_metric_source") != "env_fast_forward_log_info"
        ):
            raise RuntimeError(
                f"Condition {condition.id} used noncanonical contact metric: "
                f"{metrics.get('contact_metric_source')!r}"
            )
        metrics["seed"] = int(episode_seed)
        metrics["success"] = episode_success(metrics, self.config["episode_success"])
        metrics["condition_id"] = condition.id
        metrics["training_seed"] = self.seed
        metrics["same_observation_error"] = same_observation_error
        metrics["joint_summary"] = {
            "K_mean": (sums["k"] / self.steps).tolist(),
            "K_abs_mean": (sums["abs_k"] / self.steps).tolist(),
            "K_positive_fraction": (sums["positive"] / self.steps).tolist(),
            "abs_delta_from_Rroll_K_mean": (
                sums["source_delta_abs"] / self.steps
            ).tolist(),
            "tau1_boundary_rms": np.sqrt(sums["tau1_sq"] / self.steps).tolist(),
            "tau2_boundary_rms": np.sqrt(sums["tau2_sq"] / self.steps).tolist(),
            "tau_boundary_rms": np.sqrt(sums["tau_sq"] / self.steps).tolist(),
            "power_boundary_abs_mean": (sums["power_abs"] / self.steps).tolist(),
            "torque_boundary_saturation_fraction": (
                sums["saturated"] / self.steps
            ).tolist(),
        }
        return metrics, k2_trace

    def official_evaluation(self, arm: str) -> dict[str, Any]:
        evaluation_root = resolve_config_path(
            self.config,
            "official_evaluations_root",
            self.formal_root / "formal" / "evaluations",
        )
        path = evaluation_root / f"formal__seed{self.seed}__{arm}__eval_attempt1.json"
        payload = load_json(path)
        results = payload.get("results")
        if not isinstance(results, list) or len(results) != 1:
            raise RuntimeError(f"Invalid official evaluation payload: {path}")
        return results[0]

    def verify_unchanged(self) -> dict[str, Any]:
        checkpoint_after = {
            "R0": sha256_file(self.r0_checkpoint),
            "Rroll": sha256_file(self.roll_checkpoint),
        }
        policy_after = {
            "R0": policy_sha256(self.r0_policy),
            "Rroll": policy_sha256(self.roll_policy),
        }
        if checkpoint_after != self.checkpoint_hashes_before:
            raise RuntimeError("Formal checkpoint changed during analysis")
        if policy_after != self.policy_hashes_before:
            raise RuntimeError("Actor parameters changed during analysis")
        return {
            "checkpoints_before_after_equal": True,
            "checkpoint_sha256": checkpoint_after,
            "policies_before_after_equal": True,
            "policy_state_sha256": policy_after,
        }

    def close(self) -> None:
        close_env(self.env)


def identity_conditions() -> dict[str, Condition]:
    conditions = {condition.id: condition for condition in build_conditions()}
    return {"R0": conditions["C00"], "Rroll": conditions["C11"]}


def run_identity_seed(seed: int) -> dict[str, Any]:
    runtimes = {
        "R0": SeedRuntime(seed, "identity", environment_arm="R0"),
        "Rroll": SeedRuntime(seed, "identity", environment_arm="Rroll"),
    }
    config = runtimes["Rroll"].config
    base_seed = int(config["identity_gate"]["base_seed"])
    episodes = int(config["identity_gate"]["episodes"])
    tolerance = float(config["identity_gate"]["metric_absolute_tolerance"])
    result: dict[str, Any] = {
        "schema": "obs2_v2_1_k_identity_seed/v1",
        "training_seed": seed,
        "base_seed": base_seed,
        "episodes": episodes,
        "arms": {},
    }
    calibration_traces: list[np.ndarray] = []
    try:
        for arm, condition in identity_conditions().items():
            runtime = runtimes[arm]
            official = runtime.official_evaluation(arm)
            official_episodes = official["episodes"]
            if len(official_episodes) != episodes:
                raise RuntimeError(f"Official {arm} episode count drift")
            observed: list[dict[str, Any]] = []
            successes: list[bool] = []
            for index in range(episodes):
                episode_seed = base_seed + index
                metrics, k2_trace = runtime.run_episode(
                    condition,
                    episode_seed,
                    capture_calibration=(arm == "Rroll"),
                )
                actual_projection = metric_projection(metrics)
                expected_projection = metric_projection(official_episodes[index])
                for key, expected in expected_projection.items():
                    compare_metric_value(
                        actual_projection[key],
                        expected,
                        tolerance,
                        f"{arm}.episode{index + 1}.{key}",
                    )
                expected_success = episode_success(
                    official_episodes[index], config["episode_success"]
                )
                if bool(metrics["success"]) != expected_success:
                    raise RuntimeError(f"Identity success mismatch: {arm} episode {index + 1}")
                successes.append(bool(metrics["success"]))
                observed.append(
                    {
                        "episode": index + 1,
                        "seed": episode_seed,
                        "success": bool(metrics["success"]),
                        **actual_projection,
                    }
                )
                if k2_trace is not None:
                    calibration_traces.append(k2_trace)
            result["arms"][arm] = {
                "success_episodes": int(sum(successes)),
                "official_success_episodes": int(
                    sum(
                        episode_success(item, config["episode_success"])
                        for item in official_episodes
                    )
                ),
                "exact_metric_match": True,
                "episode_results": observed,
            }
        required = int(config["identity_gate"]["required_formal_success_counts"][str(seed)])
        if int(result["arms"]["Rroll"]["success_episodes"]) != required:
            raise RuntimeError(f"Rroll identity count mismatch for seed {seed}")
        if int(result["arms"]["R0"]["success_episodes"]) != 0:
            raise RuntimeError(f"R0 identity count mismatch for seed {seed}")
        calibration_array = np.stack(calibration_traces, axis=0)
        calibration_dir = ROOT / "calibration"
        calibration_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            calibration_dir / f"seed{seed}.npz",
            k2_time_template=np.mean(calibration_array, axis=0).astype(np.float32),
            k2_static_mean=np.mean(calibration_array, axis=(0, 1)).astype(np.float32),
            calibration_episode_seeds=np.arange(base_seed, base_seed + episodes),
        )
        result["calibration"] = {
            "source": "Rroll proposed K2 on identity-gate episodes",
            "trace_shape": list(calibration_array.shape),
            "time_template_shape": [runtimes["Rroll"].steps, 8],
            "static_mean_shape": [8],
            "uses_main_evaluation_states": False,
        }
        result["immutability"] = {
            arm: runtime.verify_unchanged() for arm, runtime in runtimes.items()
        }
        result["passed"] = True
        return result
    finally:
        for runtime in runtimes.values():
            runtime.close()


def run_main_seed(seed: int, condition_ids: Iterable[str] | None = None) -> dict[str, Any]:
    gate = ROOT / "IDENTITY_GATE_PASS.json"
    if not gate.is_file() or load_json(gate).get("passed") is not True:
        raise RuntimeError("Main interventions are blocked until identity gate passes")
    runtime = SeedRuntime(seed, "main")
    config = runtime.config
    base_seed = int(config["main_evaluation"]["base_seed"])
    episodes = int(config["main_evaluation"]["episodes"])
    selected = list(build_conditions())
    if condition_ids is not None:
        wanted = set(condition_ids)
        selected = [condition for condition in selected if condition.id in wanted]
        missing = wanted - {condition.id for condition in selected}
        if missing:
            raise ValueError(f"Unknown condition IDs: {sorted(missing)}")
    output_dir = ROOT / "results" / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    try:
        for condition in selected:
            output = output_dir / f"{condition.id}.json"
            if output.is_file():
                existing = load_json(output)
                if (
                    existing.get("training_seed") == seed
                    and existing.get("condition") == condition.to_dict()
                    and len(existing.get("episodes", [])) == episodes
                ):
                    completed.append(condition.id)
                    continue
                raise RuntimeError(f"Refusing incompatible existing result: {output}")
            records: list[dict[str, Any]] = []
            for index in range(episodes):
                metrics, _ = runtime.run_episode(condition, base_seed + index)
                records.append(metrics)
            payload = {
                "schema": "obs2_v2_1_k_condition_seed/v1",
                "study_id": config["study_id"],
                "training_seed": seed,
                "condition": condition.to_dict(),
                "evaluation_base_seed": base_seed,
                "evaluation_episodes": episodes,
                "evaluation_steps": runtime.steps,
                "success_episodes": int(sum(bool(item["success"]) for item in records)),
                "episodes": records,
                "checkpoint_sha256": runtime.checkpoint_hashes_before,
                "frozen_evaluator_sha256": sha256_file(runtime.frozen_eval_path),
            }
            atomic_json(output, payload)
            completed.append(condition.id)
            atomic_json(
                ROOT / "progress" / f"seed{seed}.json",
                {
                    "training_seed": seed,
                    "status": "running",
                    "completed_conditions": completed,
                    "completed_count": len(completed),
                    "target_count": len(selected),
                    "latest_condition": condition.id,
                },
            )
        immutability = runtime.verify_unchanged()
        summary = {
            "schema": "obs2_v2_1_k_seed_complete/v1",
            "training_seed": seed,
            "status": "complete",
            "completed_conditions": completed,
            "completed_count": len(completed),
            "target_count": len(selected),
            "immutability": immutability,
        }
        atomic_json(ROOT / "progress" / f"seed{seed}.json", summary)
        return summary
    finally:
        runtime.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen K1/K2 intervention rollout")
    parser.add_argument("--stage", required=True, choices=("identity", "main"))
    parser.add_argument("--training-seed", required=True, type=int)
    parser.add_argument("--condition", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(CONFIG_PATH)
    allowed = {int(value) for value in config["training_seeds"]}
    if args.training_seed not in allowed:
        raise ValueError(f"Training seed is outside the frozen contract: {args.training_seed}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.stage == "identity":
        payload = run_identity_seed(args.training_seed)
        atomic_json(ROOT / "identity" / f"seed{args.training_seed}.json", payload)
    else:
        payload = run_main_seed(
            args.training_seed,
            condition_ids=args.condition if args.condition else None,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
