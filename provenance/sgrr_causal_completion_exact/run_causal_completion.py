"""Frozen-checkpoint K1/K2 causal-completion rollouts.

This program never trains, resumes, or writes a checkpoint.  It imports the
already validated ``SeedRuntime`` from the earlier mechanism study, points that
runtime at this study's frozen contract, and writes only below this directory.

Condition matrix (113 total):
  * C11 identity baseline;
  * 16 individual K channels x six interventions in the complete C11
    background (zero, 0.5x, 1.5x, sign flip, calibrated static mean, fixed
    time-permuted calibrated template);
  * eight joint-pair necessity interventions that zero one joint's K1+K2 in
    the complete C11 controller;
  * eight joint-pair sufficiency interventions that transplant one joint's
    Rroll K1+K2 into the complete C00/R0 controller.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "study_config.json"
CONTRACT_PATH = ROOT / "study_contract.json"
MANIFEST_PATH = ROOT / "SOURCE_MANIFEST.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
CONTRACT = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
LEGACY_ROOT = Path(CONFIG["legacy_mechanism_root"]).resolve()
sys.path.insert(0, str(LEGACY_ROOT))
import mechanism_rollout as legacy  # type: ignore  # noqa: E402


# SeedRuntime resolves its config through a module global.  Every worker is a
# separate process, so redirecting it here cannot affect the sealed legacy run.
legacy.CONFIG_PATH = CONFIG_PATH
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")


TRANSFORMS: tuple[tuple[str, str, dict[str, Any]], ...] = (
    ("ZERO", "zero", {"transform": "zero"}),
    ("SCALE_0P5", "scale_0p5", {"transform": "scale", "alpha": 0.5}),
    ("SCALE_1P5", "scale_1p5", {"transform": "scale", "alpha": 1.5}),
    ("SIGN_FLIP", "sign_flip", {"transform": "sign_flip"}),
    ("STATIC_MEAN", "static_mean", {"transform": "static_mean"}),
    ("TIME_PERMUTED", "time_permuted", {"transform": "time_permuted"}),
)
METRIC_COLUMNS = (
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
    "mean_roll_pulse_interval_steps",
    "tail_launch_detected",
    "tail_launch_count",
)
SUMMARY_METRICS = (
    "forward_body_lengths",
    "desired_net_rotation_degrees",
    "desired_active_rotation_fraction",
    "roll_pulse_count",
    "mean_roll_pulse_interval_steps",
    "tail_launch_count",
)


@dataclass(frozen=True)
class Condition:
    id: str
    family: str
    description: str
    spec: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "description": self.description,
            "spec": self.spec,
        }


def build_conditions() -> tuple[Condition, ...]:
    conditions: list[Condition] = [
        Condition(
            "C11",
            "identity_baseline",
            "Complete Rroll K1 and K2 on the unchanged deterministic policy.",
            {"op": "baseline_c11"},
        )
    ]
    for joint in range(8):
        label = f"J{joint + 1:02d}"
        for channel, channel_label in enumerate(("K1", "K2")):
            for suffix, family, transform in TRANSFORMS:
                conditions.append(
                    Condition(
                        f"C11_{label}_{channel_label}_{suffix}",
                        f"full_c11_channel_{family}",
                        f"In complete C11, apply {family} only to {label} {channel_label}.",
                        {
                            "op": "c11_channel_intervention",
                            "joint": joint,
                            "channel": channel,
                            "channel_label": channel_label,
                            **transform,
                        },
                    )
                )
    for joint in range(8):
        label = f"J{joint + 1:02d}"
        conditions.append(
            Condition(
                f"C11_PAIR_NEC_{label}",
                "c11_joint_pair_necessity",
                f"In complete C11/Rroll, simultaneously set {label} K1+K2 to zero.",
                {"op": "c11_joint_pair_necessity", "joint": joint},
            )
        )
    for joint in range(8):
        label = f"J{joint + 1:02d}"
        conditions.append(
            Condition(
                f"C00_PAIR_SUFF_{label}",
                "c00_joint_pair_sufficiency",
                f"In complete C00/R0, transplant simultaneous Rroll K1+K2 only at {label}.",
                {"op": "c00_joint_pair_sufficiency", "joint": joint},
            )
        )
    if len(conditions) != 113 or len({item.id for item in conditions}) != 113:
        raise RuntimeError("Frozen condition generator did not produce 113 unique conditions")
    return tuple(conditions)


CONDITIONS = build_conditions()
CONDITION_BY_ID = {condition.id: condition for condition in CONDITIONS}


def canonical_condition_sha256() -> str:
    payload = [condition.to_dict() for condition in CONDITIONS]
    raw = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_source_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Missing frozen source manifest: {MANIFEST_PATH}")
    manifest = load_json(MANIFEST_PATH)
    failures: list[str] = []
    verified: list[dict[str, str]] = []
    for record in manifest.get("files", []):
        path = Path(record["path"]).resolve()
        expected = str(record["sha256"]).lower()
        if not path.is_file():
            failures.append(f"missing: {path}")
            continue
        actual = sha256_file(path)
        if actual != expected:
            failures.append(f"hash drift: {path}: {actual} != {expected}")
        verified.append({"path": str(path), "sha256": actual})
        if record.get("verify_nested_files"):
            nested = load_json(path).get("files", {})
            if not isinstance(nested, dict):
                failures.append(f"invalid nested source manifest: {path}")
                continue
            for nested_record in nested.values():
                nested_path = Path(nested_record["path"]).resolve()
                nested_expected = str(nested_record["sha256"]).lower()
                if not nested_path.is_file():
                    failures.append(f"nested missing: {nested_path}")
                    continue
                nested_actual = sha256_file(nested_path)
                if nested_actual != nested_expected:
                    failures.append(
                        f"nested hash drift: {nested_path}: "
                        f"{nested_actual} != {nested_expected}"
                    )
                verified.append(
                    {"path": str(nested_path), "sha256": nested_actual}
                )
    if failures:
        raise RuntimeError("Frozen source/checkpoint audit failed:\n" + "\n".join(failures))
    if CONTRACT.get("study_id") != CONFIG.get("study_id"):
        raise RuntimeError("Config/contract study_id mismatch")
    if len(CONDITIONS) != int(CONFIG["condition_matrix"]["total_count"]):
        raise RuntimeError("Condition count drift from config")
    expected_condition_hash = str(CONFIG["condition_matrix"]["canonical_sha256"])
    if canonical_condition_sha256() != expected_condition_hash:
        raise RuntimeError("Canonical condition hash drift from frozen config")
    if str(CONTRACT["conditions"]["canonical_sha256"]) != expected_condition_hash:
        raise RuntimeError("Canonical condition hash drift between config and contract")
    return {
        "passed": True,
        "verified_file_count": len(verified),
        "canonical_condition_sha256": canonical_condition_sha256(),
        "files": verified,
    }


class CompletionRuntime(legacy.SeedRuntime):
    """Legacy validated runtime with this study's two-channel calibration."""

    def _load_calibration(self) -> dict[str, np.ndarray]:
        path = ROOT / "calibration" / f"seed{self.seed}.npz"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing two-channel calibration for seed {self.seed}: {path}"
            )
        with np.load(path, allow_pickle=False) as data:
            template = np.asarray(data["roll_action_time_template"], dtype=np.float32)
            static = np.asarray(data["roll_action_static_mean"], dtype=np.float32)
            seeds = np.asarray(data["calibration_episode_seeds"], dtype=np.int64)
        expected_seeds = np.arange(
            int(CONFIG["calibration"]["base_seed"]),
            int(CONFIG["calibration"]["base_seed"])
            + int(CONFIG["calibration"]["episodes"]),
            dtype=np.int64,
        )
        if template.shape != (self.steps, 8, 2) or static.shape != (8, 2):
            raise RuntimeError(
                f"Calibration shape drift: template={template.shape}, static={static.shape}"
            )
        if not np.array_equal(seeds, expected_seeds):
            raise RuntimeError("Calibration episode seeds drifted from frozen contract")
        if not np.isfinite(template).all() or not np.isfinite(static).all():
            raise RuntimeError("Calibration contains NaN/Inf")
        return {"template": template, "static": static}

    def apply_completion_condition(
        self,
        condition: Condition,
        r0: torch.Tensor,
        roll: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        op = str(condition.spec["op"])
        if op == "baseline_c11":
            result = roll.clone()
        elif op == "c11_joint_pair_necessity":
            result = roll.clone()
            joint = int(condition.spec["joint"])
            result[..., joint, :] = 0.0
        elif op == "c00_joint_pair_sufficiency":
            result = r0.clone()
            joint = int(condition.spec["joint"])
            result[..., joint, :] = roll[..., joint, :]
        elif op == "c11_channel_intervention":
            result = roll.clone()
            joint = int(condition.spec["joint"])
            channel = int(condition.spec["channel"])
            transform = str(condition.spec["transform"])
            if transform == "zero":
                result[..., joint, channel] = 0.0
            elif transform == "scale":
                result[..., joint, channel] *= float(condition.spec["alpha"])
            elif transform == "sign_flip":
                result[..., joint, channel] *= -1.0
            elif transform == "static_mean":
                if self.calibration is None:
                    raise RuntimeError("Static intervention requires frozen calibration")
                result[..., joint, channel] = float(self.calibration["static"][joint, channel])
            elif transform == "time_permuted":
                if self.calibration is None:
                    raise RuntimeError("Time intervention requires frozen calibration")
                source_step = int(self.permutation[step])
                result[..., joint, channel] = float(
                    self.calibration["template"][source_step, joint, channel]
                )
            else:
                raise ValueError(f"Unsupported channel transform: {transform}")
        else:
            raise ValueError(f"Unsupported completion operation: {op}")
        if tuple(result.shape) != (1, 8, 2):
            raise RuntimeError(f"Applied action shape failed for {condition.id}")
        if not bool(torch.isfinite(result).all().item()):
            raise RuntimeError(f"Applied action contains NaN/Inf for {condition.id}")
        return result


def _nan_scalar(value: Any) -> float:
    return math.nan if value is None else float(value)


def _clip(values: np.ndarray, limit: float) -> np.ndarray:
    return np.clip(values, -limit, limit)


def _rotation_from_positions(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = positions - np.mean(positions, axis=1, keepdims=True)
    cross = np.sum(np.conjugate(centered[:-1]) * centered[1:], axis=1)
    desired_increment = -np.angle(cross)
    return desired_increment, np.cumsum(desired_increment)


def run_rollout(
    runtime: CompletionRuntime,
    condition: Condition,
    episode_seed: int,
    capture_trace: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray] | None]:
    torch.manual_seed(int(episode_seed))
    np.random.seed(int(episode_seed))
    td = runtime.env.reset()
    steps = runtime.steps
    action_scale = float(runtime.env.k_action_scale)
    feedback_gain = float(getattr(runtime.env, "feedback_gain", 1.0))
    max_torque = float(runtime.env.max_torque)
    trajectory = [np.asarray(runtime.frozen_eval._positions(runtime.env)).copy()]
    support = [
        runtime.frozen_eval._log_info_scalar(td, "fast_forward_support_index")
    ]
    contact = [
        runtime.frozen_eval._log_info_scalar(
            td, "fast_forward_ground_contact_strength"
        )
    ]
    trace: dict[str, np.ndarray] | None = None
    if capture_trace:
        trace = {
            "observation": np.empty((steps, 8, 2), dtype=np.float32),
            "r0_action": np.empty((steps, 8, 2), dtype=np.float32),
            "rroll_action": np.empty((steps, 8, 2), dtype=np.float32),
            "applied_action": np.empty((steps, 8, 2), dtype=np.float32),
            "physical_k": np.empty((steps, 8, 2), dtype=np.float32),
            "tau1_unclipped": np.empty((steps, 8), dtype=np.float32),
            "tau2_unclipped": np.empty((steps, 8), dtype=np.float32),
            "tau_unclipped": np.empty((steps, 8), dtype=np.float32),
            "tau_clipped": np.empty((steps, 8), dtype=np.float32),
            "boundary_power": np.empty((steps, 8), dtype=np.float32),
            "shapley_tau1": np.empty((steps, 8), dtype=np.float32),
            "shapley_tau2": np.empty((steps, 8), dtype=np.float32),
            "shapley_power1": np.empty((steps, 8), dtype=np.float32),
            "shapley_power2": np.empty((steps, 8), dtype=np.float32),
        }
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
    for step in range(steps):
        observation = td["agents", "observation"][0].detach().cpu().numpy().copy()
        if observation.shape != (8, 2) or not np.isfinite(observation).all():
            raise RuntimeError(f"Observation contract failed at step {step}")
        r0, roll, observation_error = runtime.actor_actions(td)
        same_observation_error = max(same_observation_error, observation_error)
        applied = runtime.apply_completion_condition(condition, r0, roll, step)
        action_td = td.clone(recurse=True)
        action_td["agents", "action"] = applied

        r0_np = r0[0].detach().cpu().numpy().astype(np.float64)
        roll_np = roll[0].detach().cpu().numpy().astype(np.float64)
        applied_np = applied[0].detach().cpu().numpy().astype(np.float64)
        physical_k = action_scale * applied_np
        tau1 = physical_k[:, 0] * observation[:, 0]
        tau2 = physical_k[:, 1] * feedback_gain * observation[:, 1]
        tau = tau1 + tau2
        clipped = _clip(tau, max_torque)
        # Exact two-player Shapley attribution through the nonlinear clip.
        phi1 = 0.5 * (
            _clip(tau1, max_torque)
            + clipped
            - _clip(tau2, max_torque)
        )
        phi2 = clipped - phi1

        sums["k"] += physical_k
        sums["abs_k"] += np.abs(physical_k)
        sums["positive"] += physical_k > 0.0
        sums["source_delta_abs"] += np.abs(applied_np - roll_np) * action_scale
        sums["tau1_sq"] += np.square(tau1)
        sums["tau2_sq"] += np.square(tau2)
        sums["tau_sq"] += np.square(tau)
        sums["power_abs"] += np.abs(clipped * observation[:, 1])
        sums["saturated"] += np.abs(tau) >= max_torque
        if trace is not None:
            trace["observation"][step] = observation
            trace["r0_action"][step] = r0_np
            trace["rroll_action"][step] = roll_np
            trace["applied_action"][step] = applied_np
            trace["physical_k"][step] = physical_k
            trace["tau1_unclipped"][step] = tau1
            trace["tau2_unclipped"][step] = tau2
            trace["tau_unclipped"][step] = tau
            trace["tau_clipped"][step] = clipped
            trace["boundary_power"][step] = clipped * observation[:, 1]
            trace["shapley_tau1"][step] = phi1
            trace["shapley_tau2"][step] = phi2
            trace["shapley_power1"][step] = phi1 * observation[:, 1]
            trace["shapley_power2"][step] = phi2 * observation[:, 1]

        td = runtime.env.step(action_td)["next"]
        trajectory.append(np.asarray(runtime.frozen_eval._positions(runtime.env)).copy())
        support.append(
            runtime.frozen_eval._log_info_scalar(td, "fast_forward_support_index")
        )
        contact.append(
            runtime.frozen_eval._log_info_scalar(
                td, "fast_forward_ground_contact_strength"
            )
        )

    metrics = runtime.frozen_eval._episode_metrics(
        trajectory,
        CONFIG["locked_contract"]["direction"],
        CONFIG["locked_contract"]["tail_side"],
        runtime.metric_args,
        support,
        contact,
    )
    if metrics.get("contact_metric_source") != "env_fast_forward_log_info":
        raise RuntimeError(
            f"Condition {condition.id} used noncanonical contact metric "
            f"{metrics.get('contact_metric_source')!r}"
        )
    metrics["seed"] = int(episode_seed)
    metrics["success"] = legacy.episode_success(metrics, CONFIG["episode_success"])
    metrics["condition_id"] = condition.id
    metrics["training_seed"] = runtime.seed
    metrics["same_observation_error"] = same_observation_error
    metrics["joint_summary"] = {
        "K_mean": (sums["k"] / steps).tolist(),
        "K_abs_mean": (sums["abs_k"] / steps).tolist(),
        "K_positive_fraction": (sums["positive"] / steps).tolist(),
        "abs_delta_from_Rroll_K_mean": (sums["source_delta_abs"] / steps).tolist(),
        "tau1_boundary_rms": np.sqrt(sums["tau1_sq"] / steps).tolist(),
        "tau2_boundary_rms": np.sqrt(sums["tau2_sq"] / steps).tolist(),
        "tau_boundary_rms": np.sqrt(sums["tau_sq"] / steps).tolist(),
        "power_boundary_abs_mean": (sums["power_abs"] / steps).tolist(),
        "torque_boundary_saturation_fraction": (sums["saturated"] / steps).tolist(),
    }
    if trace is not None:
        positions = np.asarray(trajectory)
        desired_increment, desired_cumulative = _rotation_from_positions(positions)
        trace.update(
            {
                "step": np.arange(steps, dtype=np.int32),
                "positions": positions,
                "support_index": np.asarray(
                    [_nan_scalar(value) for value in support], dtype=np.float32
                ),
                "ground_contact_strength": np.asarray(
                    [_nan_scalar(value) for value in contact], dtype=np.float32
                ),
                "desired_rotation_increment_rad": desired_increment.astype(np.float32),
                "desired_cumulative_rotation_rad": desired_cumulative.astype(np.float32),
                "training_seed": np.asarray(runtime.seed, dtype=np.int64),
                "evaluation_seed": np.asarray(episode_seed, dtype=np.int64),
                "condition_id": np.asarray(condition.id),
                "condition_sha256": np.asarray(canonical_condition_sha256()),
            }
        )
    return metrics, trace


def _expected_checkpoint_hashes(seed: int) -> dict[str, str]:
    values = CONTRACT["checkpoint_sha256"][str(seed)]
    return {arm: str(value).lower() for arm, value in values.items()}


def _validate_runtime_hashes(runtime: CompletionRuntime) -> None:
    expected = _expected_checkpoint_hashes(runtime.seed)
    if runtime.checkpoint_hashes_before != expected:
        raise RuntimeError(
            f"Checkpoint hash drift for seed {runtime.seed}: "
            f"{runtime.checkpoint_hashes_before} != {expected}"
        )


def _set_torch_threads() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # A caller in the same process may already have fixed interop threads.
        pass


def run_calibration_seed(seed: int) -> dict[str, Any]:
    verify_source_manifest()
    _set_torch_threads()
    output = ROOT / "calibration" / f"seed{seed}.npz"
    receipt_path = ROOT / "calibration" / f"seed{seed}.json"
    expected_episode_seeds = np.arange(
        int(CONFIG["calibration"]["base_seed"]),
        int(CONFIG["calibration"]["base_seed"])
        + int(CONFIG["calibration"]["episodes"]),
        dtype=np.int64,
    )
    if output.is_file() and receipt_path.is_file():
        receipt = load_json(receipt_path)
        with np.load(output, allow_pickle=False) as existing:
            template = np.asarray(existing["roll_action_time_template"])
            static = np.asarray(existing["roll_action_static_mean"])
            seeds = np.asarray(existing["calibration_episode_seeds"])
        if (
            template.shape == (1000, 8, 2)
            and static.shape == (8, 2)
            and np.array_equal(seeds, expected_episode_seeds)
            and receipt.get("study_id") == CONFIG["study_id"]
            and receipt.get("training_seed") == seed
            and receipt.get("calibration_npz_sha256") == sha256_file(output)
        ):
            return receipt
        raise RuntimeError(f"Refusing incompatible calibration artifact: {output}")

    runtime = CompletionRuntime(seed, "identity")
    _validate_runtime_hashes(runtime)
    condition = CONDITION_BY_ID["C11"]
    actions: list[np.ndarray] = []
    try:
        for episode_seed in expected_episode_seeds:
            _, trace = run_rollout(
                runtime, condition, int(episode_seed), capture_trace=True
            )
            assert trace is not None
            actions.append(trace["rroll_action"])
        action_array = np.stack(actions, axis=0).astype(np.float32)
        time_template = np.mean(action_array, axis=0).astype(np.float32)
        static_mean = np.mean(action_array, axis=(0, 1)).astype(np.float32)
        atomic_npz(
            output,
            roll_action_time_template=time_template,
            roll_action_static_mean=static_mean,
            physical_k_time_template=(
                float(CONFIG["locked_contract"]["k_action_scale"]) * time_template
            ).astype(np.float32),
            physical_k_static_mean=(
                float(CONFIG["locked_contract"]["k_action_scale"]) * static_mean
            ).astype(np.float32),
            calibration_episode_seeds=expected_episode_seeds,
            training_seed=np.asarray(seed, dtype=np.int64),
            source_condition=np.asarray("C11"),
        )
        immutability = runtime.verify_unchanged()
        receipt = {
            "schema": "obs2_v2_1_k_causal_completion/calibration/v1",
            "study_id": CONFIG["study_id"],
            "training_seed": seed,
            "source_condition": "C11",
            "base_seed": int(expected_episode_seeds[0]),
            "episodes": len(expected_episode_seeds),
            "steps": runtime.steps,
            "raw_action_stack_shape": list(action_array.shape),
            "time_template_shape": list(time_template.shape),
            "static_mean_shape": list(static_mean.shape),
            "main_outcome_states_used": False,
            "calibration_npz_sha256": sha256_file(output),
            "immutability": immutability,
            "status": "complete",
        }
        atomic_json(receipt_path, receipt)
        return receipt
    finally:
        runtime.close()


def _validate_existing_condition_result(
    path: Path, seed: int, condition: Condition, episodes: int
) -> bool:
    if not path.is_file():
        return False
    payload = load_json(path)
    valid = (
        payload.get("study_id") == CONFIG["study_id"]
        and payload.get("training_seed") == seed
        and payload.get("condition") == condition.to_dict()
        and payload.get("canonical_condition_sha256") == canonical_condition_sha256()
        and len(payload.get("episodes", [])) == episodes
    )
    if valid:
        expected_trace_count = episodes if condition.id == "C11" else 1
        trace_files = payload.get("trace_files", [])
        if len(trace_files) != expected_trace_count:
            raise RuntimeError(f"Trace receipt count drift: {path}")
        for record in trace_files:
            trace_path = ROOT / record["path"]
            if not trace_path.is_file() or sha256_file(trace_path) != record["sha256"]:
                raise RuntimeError(f"Trace missing or hash-drifted: {trace_path}")
        return True
    raise RuntimeError(f"Refusing incompatible existing result: {path}")


def _trace_path(seed: int, condition: Condition, episode_seed: int) -> Path:
    return (
        ROOT
        / "traces"
        / f"seed{seed}"
        / f"{condition.id}__evalseed{episode_seed}.npz"
    )


def run_main_seed(
    seed: int, condition_ids: Iterable[str] | None = None
) -> dict[str, Any]:
    audit = verify_source_manifest()
    _set_torch_threads()
    selected = list(CONDITIONS)
    if condition_ids is not None:
        wanted = list(dict.fromkeys(condition_ids))
        missing = sorted(set(wanted) - set(CONDITION_BY_ID))
        if missing:
            raise ValueError(f"Unknown condition IDs: {missing}")
        selected = [CONDITION_BY_ID[value] for value in wanted]
    runtime = CompletionRuntime(seed, "main")
    _validate_runtime_hashes(runtime)
    base_seed = int(CONFIG["main_evaluation"]["base_seed"])
    episodes = int(CONFIG["main_evaluation"]["episodes"])
    output_dir = ROOT / "results" / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    try:
        for condition in selected:
            output = output_dir / f"{condition.id}.json"
            if _validate_existing_condition_result(
                output, seed, condition, episodes
            ):
                completed.append(condition.id)
                continue
            records: list[dict[str, Any]] = []
            trace_paths: list[dict[str, str]] = []
            for index in range(episodes):
                episode_seed = base_seed + index
                capture_trace = condition.id == "C11" or index == 0
                metrics, trace = run_rollout(
                    runtime, condition, episode_seed, capture_trace=capture_trace
                )
                records.append(metrics)
                if trace is not None:
                    path = _trace_path(seed, condition, episode_seed)
                    atomic_npz(path, **trace)
                    trace_paths.append(
                        {
                            "path": str(path.relative_to(ROOT)),
                            "sha256": sha256_file(path),
                        }
                    )
            payload = {
                "schema": "obs2_v2_1_k_causal_completion/condition_seed/v1",
                "study_id": CONFIG["study_id"],
                "training_seed": seed,
                "condition": condition.to_dict(),
                "canonical_condition_sha256": canonical_condition_sha256(),
                "evaluation_base_seed": base_seed,
                "evaluation_episodes": episodes,
                "evaluation_steps": runtime.steps,
                "success_episodes": int(sum(bool(item["success"]) for item in records)),
                "episodes": records,
                "trace_files": trace_paths,
                "checkpoint_sha256": runtime.checkpoint_hashes_before,
                "policy_state_sha256": runtime.policy_hashes_before,
                "frozen_evaluator_sha256": sha256_file(runtime.frozen_eval_path),
                "source_audit": {
                    "passed": audit["passed"],
                    "verified_file_count": audit["verified_file_count"],
                },
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
            "schema": "obs2_v2_1_k_causal_completion/seed_complete/v1",
            "study_id": CONFIG["study_id"],
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


def run_smoke(seed: int, condition_id: str) -> dict[str, Any]:
    verify_source_manifest()
    _set_torch_threads()
    condition = CONDITION_BY_ID[condition_id]
    needs_calibration = condition.spec.get("transform") in {
        "static_mean",
        "time_permuted",
    }
    runtime = CompletionRuntime(seed, "main" if needs_calibration else "identity")
    _validate_runtime_hashes(runtime)
    episode_seed = int(CONFIG["smoke"]["base_seed"])
    try:
        metrics, trace = run_rollout(
            runtime, condition, episode_seed, capture_trace=True
        )
        assert trace is not None
        trace_path = (
            ROOT
            / "smoke"
            / f"seed{seed}"
            / f"{condition.id}__evalseed{episode_seed}.npz"
        )
        atomic_npz(trace_path, **trace)
        payload = {
            "schema": "obs2_v2_1_k_causal_completion/smoke/v1",
            "study_id": CONFIG["study_id"],
            "training_seed": seed,
            "condition": condition.to_dict(),
            "evaluation_seed": episode_seed,
            "metrics": metrics,
            "trace_file": str(trace_path.relative_to(ROOT)),
            "trace_sha256": sha256_file(trace_path),
            "immutability": runtime.verify_unchanged(),
            "scientific_outcome": False,
            "purpose": "technical contract smoke test only",
            "status": "complete",
        }
        atomic_json(
            ROOT / "smoke" / f"seed{seed}" / f"{condition.id}.json", payload
        )
        return payload
    finally:
        runtime.close()


def _failure_flags(metrics: dict[str, Any]) -> dict[str, int]:
    criteria = CONFIG["episode_success"]
    interval = metrics.get("mean_roll_pulse_interval_steps")
    return {
        "pulse_fail": int(
            int(metrics["roll_pulse_count"]) < int(criteria["minimum_roll_pulses"])
        ),
        "rotation_fail": int(
            float(metrics["desired_net_rotation_degrees"])
            < float(criteria["minimum_desired_net_rotation_degrees"])
        ),
        "direction_fail": int(
            float(metrics["desired_active_rotation_fraction"])
            < float(criteria["minimum_direction_fraction"])
        ),
        "forward_fail": int(
            float(metrics["forward_body_lengths"])
            < float(criteria["minimum_forward_body_lengths"])
        ),
        "interval_fail": int(
            interval is None
            or float(interval)
            > float(criteria["maximum_mean_inter_pulse_interval_steps"])
        ),
    }


def _numeric(values: Iterable[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            result.append(number)
    return result


def summarize_results() -> dict[str, Any]:
    episode_rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}
    condition_meta: dict[str, dict[str, Any]] = {}
    for seed in (int(value) for value in CONFIG["training_seeds"]):
        directory = ROOT / "results" / f"seed{seed}"
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            payload = load_json(path)
            condition = payload["condition"]
            condition_id = condition["id"]
            condition_meta[condition_id] = condition
            for episode_number, metrics in enumerate(payload["episodes"], start=1):
                row = {
                    "condition_id": condition_id,
                    "family": condition["family"],
                    "training_seed": seed,
                    "episode": episode_number,
                    "evaluation_seed": metrics["seed"],
                    "success": int(bool(metrics["success"])),
                    **{key: metrics.get(key) for key in METRIC_COLUMNS},
                    **_failure_flags(metrics),
                }
                episode_rows.append(row)
                grouped.setdefault(condition_id, []).append(row)
    episode_fields = (
        "condition_id",
        "family",
        "training_seed",
        "episode",
        "evaluation_seed",
        "success",
        *METRIC_COLUMNS,
        "pulse_fail",
        "rotation_fail",
        "direction_fail",
        "forward_fail",
        "interval_fail",
    )
    atomic_csv(ROOT / "analysis" / "episode_results.csv", episode_rows, episode_fields)

    summary_rows: list[dict[str, Any]] = []
    for condition_id in [item.id for item in CONDITIONS]:
        rows = grouped.get(condition_id)
        if not rows:
            continue
        seed_success = {
            str(seed): int(
                sum(
                    row["success"]
                    for row in rows
                    if int(row["training_seed"]) == seed
                )
            )
            for seed in (int(value) for value in CONFIG["training_seeds"])
            if any(int(row["training_seed"]) == seed for row in rows)
        }
        record: dict[str, Any] = {
            "condition_id": condition_id,
            "family": condition_meta[condition_id]["family"],
            "episode_count": len(rows),
            "training_seed_count": len(seed_success),
            "success_episodes": int(sum(row["success"] for row in rows)),
            "success_rate": float(np.mean([row["success"] for row in rows])),
            "successful_training_seeds_ge_10of20": int(
                sum(value >= 10 for value in seed_success.values())
            ),
            "success_episodes_by_training_seed": json.dumps(
                seed_success, sort_keys=True, separators=(",", ":")
            ),
        }
        for name in SUMMARY_METRICS:
            values = _numeric(row.get(name) for row in rows)
            record[f"{name}_mean"] = float(np.mean(values)) if values else None
            record[f"{name}_std"] = float(np.std(values)) if values else None
        for name in (
            "pulse_fail",
            "rotation_fail",
            "direction_fail",
            "forward_fail",
            "interval_fail",
        ):
            record[name] = int(sum(int(row[name]) for row in rows))
        summary_rows.append(record)
    summary_fields = tuple(
        dict.fromkeys(key for row in summary_rows for key in row).keys()
    )
    atomic_csv(
        ROOT / "analysis" / "condition_summary.csv", summary_rows, summary_fields
    )
    complete = (
        len(summary_rows) == len(CONDITIONS)
        and len(episode_rows)
        == int(CONFIG["main_evaluation"]["total_episodes"])
    )
    payload = {
        "schema": "obs2_v2_1_k_causal_completion/summary/v1",
        "study_id": CONFIG["study_id"],
        "canonical_condition_sha256": canonical_condition_sha256(),
        "condition_rows": len(summary_rows),
        "episode_rows": len(episode_rows),
        "expected_condition_rows": len(CONDITIONS),
        "expected_episode_rows": int(CONFIG["main_evaluation"]["total_episodes"]),
        "complete": complete,
        "condition_summary": summary_rows,
    }
    atomic_json(ROOT / "analysis" / "condition_summary.json", payload)
    return payload


def _worker(stage: str, seed: int) -> dict[str, Any]:
    if stage == "calibration":
        return run_calibration_seed(seed)
    if stage == "main":
        return run_main_seed(seed)
    raise ValueError(stage)


def _parallel(stage: str, seeds: Sequence[int], workers: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_worker, stage, seed): seed for seed in seeds}
        try:
            for future in as_completed(futures):
                results.append(future.result())
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("verify", "smoke", "calibration", "main", "all", "summarize"),
    )
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--condition", action="append", default=[])
    parser.add_argument("--workers", type=int, default=5)
    return parser.parse_args()


def _validate_seed(seed: int | None) -> int:
    if seed is None:
        raise ValueError("--training-seed is required for this stage")
    allowed = {int(value) for value in CONFIG["training_seeds"]}
    if seed not in allowed:
        raise ValueError(f"Training seed is outside frozen contract: {seed}")
    return seed


def main() -> None:
    args = parse_args()
    allowed_seeds = [int(value) for value in CONFIG["training_seeds"]]
    if args.workers < 1 or args.workers > int(
        CONFIG["main_evaluation"]["max_parallel_seed_workers"]
    ):
        raise ValueError("--workers must be between 1 and 5")
    if args.stage == "verify":
        payload: Any = {
            **verify_source_manifest(),
            "condition_count": len(CONDITIONS),
            "conditions": [item.to_dict() for item in CONDITIONS],
        }
    elif args.stage == "smoke":
        seed = _validate_seed(args.training_seed)
        condition_id = (
            args.condition[0]
            if args.condition
            else str(CONFIG["smoke"]["default_condition"])
        )
        if condition_id not in CONDITION_BY_ID:
            raise ValueError(f"Unknown smoke condition: {condition_id}")
        payload = run_smoke(seed, condition_id)
    elif args.stage == "calibration":
        seed = _validate_seed(args.training_seed)
        payload = run_calibration_seed(seed)
    elif args.stage == "main":
        seed = _validate_seed(args.training_seed)
        payload = run_main_seed(seed, args.condition or None)
        summarize_results()
    elif args.stage == "all":
        verify_source_manifest()
        calibrations = _parallel("calibration", allowed_seeds, args.workers)
        seeds = _parallel("main", allowed_seeds, args.workers)
        payload = {
            "calibration": calibrations,
            "main": seeds,
            "summary": summarize_results(),
        }
    else:
        payload = summarize_results()
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
