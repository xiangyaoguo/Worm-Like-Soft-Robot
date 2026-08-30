from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from condition_matrix import build_conditions, canonical_sha256
from mechanism_rollout import (
    ROOT,
    SeedRuntime,
    atomic_json,
    compare_metric_value,
    episode_success,
    load_json,
    metric_projection,
    sha256_file,
    tensor_max_abs,
)
from smoke_test import independent_expected


TRACE_ROOT = ROOT / "independent_trace_audit"
PASS_PATH = ROOT / "INDEPENDENT_TRACE_AUDIT_PASS.json"
FIVE_METRIC_KEYS = (
    "roll_pulse_count",
    "desired_net_rotation_degrees",
    "desired_active_rotation_fraction",
    "forward_body_lengths",
    "mean_roll_pulse_interval_steps",
)


def tensor_bit_exact(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
        return False
    left_bytes = left.detach().cpu().contiguous().numpy().tobytes()
    right_bytes = right.detach().cpu().contiguous().numpy().tobytes()
    return left_bytes == right_bytes


def td_snapshot(td: Any) -> dict[Any, torch.Tensor]:
    result: dict[Any, torch.Tensor] = {}
    for key in td.keys(include_nested=True, leaves_only=True):
        value = td.get(key)
        if not torch.is_tensor(value):
            raise RuntimeError(f"Cannot bit-audit non-tensor TensorDict leaf: {key!r}")
        if key in result:
            raise RuntimeError(f"Duplicate TensorDict leaf key: {key!r}")
        result[key] = value.detach().clone()
    return result


def snapshot_error(before: Mapping[Any, torch.Tensor], td: Any) -> float:
    after = td_snapshot(td)
    if set(before) != set(after):
        return float("inf")
    return max(
        (tensor_max_abs(before[key], after[key]) for key in before), default=0.0
    )


def snapshot_bit_exact(before: Mapping[Any, torch.Tensor], td: Any) -> bool:
    after = td_snapshot(td)
    return set(before) == set(after) and all(
        tensor_bit_exact(before[key], after[key]) for key in before
    )


def independent_five_metrics(
    positions: np.ndarray,
    support: np.ndarray,
    contact: np.ndarray,
    metric_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute the five frozen success metrics without calling the evaluator."""

    if positions.ndim != 2 or positions.shape[1] != 10 or positions.shape[0] < 2:
        raise RuntimeError(f"Independent metric position shape invalid: {positions.shape}")
    if support.shape != (positions.shape[0],) or contact.shape != support.shape:
        raise RuntimeError("Independent metric support/contact shape mismatch")
    if (
        not np.isfinite(positions.real).all()
        or not np.isfinite(positions.imag).all()
        or not np.isfinite(support).all()
        or not np.isfinite(contact).all()
    ):
        raise RuntimeError("Independent metric trace contains NaN/Inf")

    initial_length = float(np.sum(np.abs(np.diff(positions[0]))))
    if not math.isfinite(initial_length) or initial_length <= 1e-12:
        raise RuntimeError(f"Independent metric initial body length invalid: {initial_length}")
    com_x = np.real(np.mean(positions, axis=1)).astype(np.float64)
    raw_rotation = np.empty(positions.shape[0] - 1, dtype=np.float64)
    for step in range(len(raw_rotation)):
        previous = positions[step] - np.mean(positions[step])
        current = positions[step + 1] - np.mean(positions[step + 1])
        cross = np.sum(np.conj(previous) * current)
        raw_rotation[step] = 0.0 if abs(cross) <= 1e-12 else float(np.angle(cross))

    # The frozen contract is rightward, for which clockwise rotation is desired.
    desired_rotation = -raw_rotation
    desired_cumulative = np.r_[0.0, np.cumsum(desired_rotation)]
    valid_ground_contact = (contact >= 0.50)
    held_support = float(support[0])
    contact_index = np.empty_like(support, dtype=np.float64)
    for frame in range(len(support)):
        if valid_ground_contact[frame]:
            held_support = float(support[frame])
        contact_index[frame] = held_support

    rotation_threshold = math.radians(
        float(metric_parameters["pulse_rotation_degrees"])
    )
    active_threshold = math.radians(
        float(metric_parameters["active_rotation_degrees"])
    )
    reset_drawdown = math.radians(
        float(metric_parameters["pulse_reset_drawdown_degrees"])
    )
    forward_fraction = float(metric_parameters["pulse_forward_body_fraction"])
    contact_fraction = float(metric_parameters["pulse_contact_index_fraction"])
    reset_backward_fraction = float(
        metric_parameters["pulse_reset_backward_body_fraction"]
    )
    pulse_ends: list[int] = []
    start = 0
    total_steps = len(desired_rotation)
    for end in range(1, total_steps + 1):
        desired_angle = float(desired_cumulative[end] - desired_cumulative[start])
        forward = float(com_x[end] - com_x[start])
        if (
            desired_angle <= -reset_drawdown
            or forward <= -reset_backward_fraction * initial_length
        ):
            start = end
            continue
        if desired_angle < rotation_threshold:
            continue
        if forward < forward_fraction * initial_length:
            continue
        contact_slice = contact_index[start : end + 1]
        contact_span_fraction = float(
            (np.max(contact_slice) - np.min(contact_slice)) / 9.0
        )
        if contact_span_fraction < contact_fraction:
            continue
        if not bool(valid_ground_contact[end]):
            continue
        pulse_ends.append(end)
        start = end

    intervals = [
        int(pulse_ends[index] - pulse_ends[index - 1])
        for index in range(1, len(pulse_ends))
    ]
    active = np.abs(desired_rotation) >= active_threshold
    desired_fraction = (
        float(np.mean(desired_rotation[active] > 0.0)) if np.any(active) else 0.0
    )
    displacement = float(com_x[-1] - com_x[0])
    return {
        "roll_pulse_count": len(pulse_ends),
        "desired_net_rotation_degrees": float(np.degrees(np.sum(desired_rotation))),
        "desired_active_rotation_fraction": desired_fraction,
        "forward_body_lengths": displacement / max(initial_length, 1e-12),
        "mean_roll_pulse_interval_steps": (
            float(np.mean(intervals)) if intervals else None
        ),
    }


def independent_success(metrics: Mapping[str, Any], criteria: Mapping[str, Any]) -> bool:
    interval = metrics["mean_roll_pulse_interval_steps"]
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


def main_result_episode(
    training_seed: int,
    condition: Any,
    config: Mapping[str, Any],
    checkpoint_hashes: Mapping[str, str],
    evaluator_hash: str,
) -> tuple[dict[str, Any], Path, str]:
    condition_id = str(condition.id)
    path = ROOT / "results" / f"seed{training_seed}" / f"{condition_id}.json"
    payload = load_json(path)
    base_seed = int(config["main_evaluation"]["base_seed"])
    episode_count = int(config["main_evaluation"]["episodes"])
    expected_seeds = list(range(base_seed, base_seed + episode_count))
    if payload.get("schema") != "obs2_v2_1_k_condition_seed/v1":
        raise RuntimeError(f"Stored result schema mismatch: {path}")
    if payload.get("study_id") != config["study_id"]:
        raise RuntimeError(f"Stored result study mismatch: {path}")
    if int(payload.get("training_seed", -1)) != training_seed:
        raise RuntimeError(f"Stored result training seed mismatch: {path}")
    if payload.get("condition") != condition.to_dict():
        raise RuntimeError(f"Stored result condition mismatch: {path}")
    if int(payload.get("evaluation_base_seed", -1)) != base_seed:
        raise RuntimeError(f"Stored result evaluation base seed mismatch: {path}")
    if int(payload.get("evaluation_episodes", -1)) != episode_count:
        raise RuntimeError(f"Stored result episode count mismatch: {path}")
    if int(payload.get("evaluation_steps", -1)) != int(config["main_evaluation"]["steps"]):
        raise RuntimeError(f"Stored result step count mismatch: {path}")
    if payload.get("checkpoint_sha256") != dict(checkpoint_hashes):
        raise RuntimeError(f"Stored result checkpoint hash mismatch: {path}")
    if payload.get("frozen_evaluator_sha256") != evaluator_hash:
        raise RuntimeError(f"Stored result evaluator hash mismatch: {path}")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != episode_count:
        raise RuntimeError(f"Stored result episode inventory invalid: {path}")
    actual_seeds = [int(item.get("seed", -1)) for item in episodes]
    if actual_seeds != expected_seeds:
        raise RuntimeError(f"Stored result evaluation seed inventory mismatch: {path}")
    if int(payload.get("success_episodes", -1)) != sum(
        bool(item.get("success")) for item in episodes
    ):
        raise RuntimeError(f"Stored result success count mismatch: {path}")
    for item in episodes:
        if (
            int(item.get("training_seed", -1)) != training_seed
            or item.get("condition_id") != condition_id
        ):
            raise RuntimeError(f"Stored episode pairing mismatch: {path}")
    records = [item for item in episodes if int(item["seed"]) == base_seed]
    if len(records) != 1:
        raise RuntimeError(
            f"Expected one stored main episode for seed {training_seed}/{condition_id}"
        )
    return records[0], path.resolve(), sha256_file(path)


def latest_contract_path() -> Path:
    revision = len(list(ROOT.glob("TECHNICAL_AMENDMENT_*.json")))
    return (
        ROOT / "FROZEN_CONTRACT.json"
        if revision == 0
        else ROOT / f"FROZEN_CONTRACT_R{revision}.json"
    )


def validate_frozen_hashes(
    config: Mapping[str, Any], conditions: list[Any]
) -> dict[str, Any]:
    contract_path = latest_contract_path()
    if not contract_path.is_file():
        raise FileNotFoundError(f"Missing latest frozen contract: {contract_path}")
    contract = load_json(contract_path)
    if (
        contract.get("schema") != "obs2_v2_1_k_frozen_contract/v1"
        or contract.get("study_id") != config["study_id"]
    ):
        raise RuntimeError("Latest frozen contract identity mismatch")
    if int(contract.get("condition_count", -1)) != len(conditions):
        raise RuntimeError("Frozen contract condition count mismatch")
    condition_hash = canonical_sha256(conditions)
    if contract.get("conditions_sha256") != condition_hash:
        raise RuntimeError("Frozen condition matrix hash mismatch")
    if contract.get("conditions") != [condition.to_dict() for condition in conditions]:
        raise RuntimeError("Frozen condition inventory mismatch")

    expected_source_names = {
        "study_config.json",
        "condition_matrix.py",
        "mechanism_rollout.py",
        "smoke_test.py",
        "run_study.py",
    }
    source_entries = contract.get("source_files")
    if not isinstance(source_entries, dict) or set(source_entries) != expected_source_names:
        raise RuntimeError("Frozen source inventory mismatch")
    source_hashes: dict[str, str] = {}
    for name in sorted(expected_source_names):
        path = ROOT / name
        evidence = source_entries[name]
        actual_hash = sha256_file(path)
        if (
            not isinstance(evidence, dict)
            or evidence.get("sha256") != actual_hash
            or int(evidence.get("size", -1)) != path.stat().st_size
        ):
            raise RuntimeError(f"Frozen source hash/size drift: {name}")
        source_hashes[name] = actual_hash

    formal_root = Path(str(config["formal_root"])).resolve()
    expected_protected: dict[str, Path] = {
        "formal_config": formal_root / "_control" / "experiment_config.json",
        "formal_result": formal_root / "FORMAL_RESULT.json",
        "formal_source_manifest": formal_root / "_control" / "source_manifest.json",
        "frozen_evaluator": formal_root
        / "_control"
        / "code_snapshot"
        / "training"
        / "evaluate_fast_forward_roll.py",
    }
    for seed_value in config["training_seeds"]:
        seed = int(seed_value)
        for arm in ("R0", "Rroll"):
            expected_protected[f"checkpoint_seed{seed}_{arm}"] = (
                formal_root
                / "formal"
                / "runs"
                / f"formal__seed{seed}__{arm}"
                / "checkpoint_1500.pt"
            )
    protected_entries = contract.get("protected_formal_files")
    if not isinstance(protected_entries, dict) or set(protected_entries) != set(
        expected_protected
    ):
        raise RuntimeError("Frozen protected-file inventory mismatch")
    protected_hashes: dict[str, str] = {}
    for name, path in expected_protected.items():
        evidence = protected_entries[name]
        if not path.is_file():
            raise FileNotFoundError(f"Missing protected frozen file: {path}")
        actual_hash = sha256_file(path)
        if (
            not isinstance(evidence, dict)
            or evidence.get("sha256") != actual_hash
            or int(evidence.get("size", -1)) != path.stat().st_size
        ):
            raise RuntimeError(f"Protected frozen file drift: {name}")
        protected_hashes[name] = actual_hash
    amendment_paths = sorted(ROOT.glob("TECHNICAL_AMENDMENT_*.json"))
    amendment_entries = contract.get("technical_amendments")
    if not isinstance(amendment_entries, list) or len(amendment_entries) != len(
        amendment_paths
    ):
        raise RuntimeError("Frozen technical-amendment inventory mismatch")
    amendment_hashes: dict[str, str] = {}
    for index, (path, evidence) in enumerate(
        zip(amendment_paths, amendment_entries), start=1
    ):
        actual_hash = sha256_file(path)
        if not isinstance(evidence, dict) or evidence.get("sha256") != actual_hash:
            raise RuntimeError(f"Technical amendment hash drift: {path.name}")
        amendment = load_json(path)
        previous_path = (
            ROOT / "FROZEN_CONTRACT.json"
            if index == 1
            else ROOT / f"FROZEN_CONTRACT_R{index - 1}.json"
        )
        expected_previous = amendment.get(
            "previous_contract_sha256", amendment.get("original_contract_sha256")
        )
        if expected_previous != sha256_file(previous_path):
            raise RuntimeError(f"Technical amendment chain mismatch: {path.name}")
        amendment_hashes[path.name] = actual_hash
    return {
        "contract_path": str(contract_path.resolve()),
        "contract_sha256": sha256_file(contract_path),
        "condition_matrix_sha256": condition_hash,
        "source_sha256": source_hashes,
        "protected_sha256": protected_hashes,
        "technical_amendment_sha256": amendment_hashes,
    }


def validate_execution_complete(
    config: Mapping[str, Any], conditions: list[Any]
) -> dict[str, Any]:
    completion_path = ROOT / "MAIN_EXECUTION_COMPLETE.json"
    completion = load_json(completion_path)
    seeds = [int(value) for value in config["training_seeds"]]
    condition_ids = sorted(condition.id for condition in conditions)
    expected_inventory = {
        str(seed): condition_ids for seed in seeds
    }
    if (
        completion.get("schema") != "obs2_v2_1_k_execution_complete/v1"
        or completion.get("study_id") != config["study_id"]
        or completion.get("status") != "complete"
        or completion.get("protected_formal_files_unchanged") is not True
    ):
        raise RuntimeError("Trace audit requires a valid complete main execution")
    if (
        int(completion.get("training_seed_count", -1)) != len(seeds)
        or int(completion.get("condition_count", -1)) != len(conditions)
        or int(completion.get("policy_condition_cases", -1))
        != len(seeds) * len(conditions)
        or int(completion.get("episodes", -1))
        != int(config["main_evaluation"]["total_episodes"])
        or completion.get("inventory") != expected_inventory
    ):
        raise RuntimeError("Complete-main inventory/count mismatch")
    return completion


def validate_calibration_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    gate = load_json(ROOT / "IDENTITY_GATE_PASS.json")
    seeds = [int(value) for value in config["training_seeds"]]
    evidence = gate.get("calibration_sha256")
    if (
        gate.get("schema") != "obs2_v2_1_k_identity_gate/v1"
        or gate.get("study_id") != config["study_id"]
        or gate.get("passed") is not True
        or not isinstance(evidence, dict)
        or set(evidence) != {str(seed) for seed in seeds}
    ):
        raise RuntimeError("Identity-gate calibration evidence mismatch")
    actual: dict[str, str] = {}
    expected_steps = int(config["main_evaluation"]["steps"])
    for seed in seeds:
        path = ROOT / "calibration" / f"seed{seed}.npz"
        digest = sha256_file(path)
        if evidence[str(seed)] != digest:
            raise RuntimeError(f"Calibration hash drift: seed {seed}")
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {
                "k2_time_template",
                "k2_static_mean",
                "calibration_episode_seeds",
            }:
                raise RuntimeError(f"Calibration key inventory mismatch: seed {seed}")
            template = data["k2_time_template"]
            static = data["k2_static_mean"]
            calibration_seeds = data["calibration_episode_seeds"]
            identity_base = int(config["identity_gate"]["base_seed"])
            identity_count = int(config["identity_gate"]["episodes"])
            if (
                template.shape != (expected_steps, 8)
                or template.dtype != np.float32
                or static.shape != (8,)
                or static.dtype != np.float32
                or calibration_seeds.dtype != np.int64
                or not np.array_equal(
                    calibration_seeds,
                    np.arange(
                        identity_base, identity_base + identity_count, dtype=np.int64
                    ),
                )
                or not np.isfinite(template).all()
                or not np.isfinite(static).all()
            ):
                raise RuntimeError(f"Calibration array contract mismatch: seed {seed}")
        actual[str(seed)] = digest
    return actual


def run_seed(training_seed: int) -> dict[str, Any]:
    runtime = SeedRuntime(training_seed, "main", environment_arm="Rroll")
    config = runtime.config
    allowed_seeds = [int(value) for value in config["training_seeds"]]
    if training_seed not in allowed_seeds or len(allowed_seeds) != len(set(allowed_seeds)):
        raise RuntimeError(f"Invalid trace-audit training seed: {training_seed}")
    episode_seed = int(config["main_evaluation"]["base_seed"])
    metric_tolerance = float(config["identity_gate"]["metric_absolute_tolerance"])
    conditions = list(build_conditions())
    if (
        len(conditions) != int(config["main_evaluation"]["condition_count"])
        or runtime.steps != int(config["main_evaluation"]["steps"])
    ):
        raise RuntimeError("Trace-audit condition/step contract mismatch")
    source_paths = {
        "study_config.json": ROOT / "study_config.json",
        "condition_matrix.py": ROOT / "condition_matrix.py",
        "mechanism_rollout.py": ROOT / "mechanism_rollout.py",
        "smoke_test.py": ROOT / "smoke_test.py",
        "frozen_evaluator": runtime.frozen_eval_path,
    }
    source_hashes_before = {
        name: sha256_file(path) for name, path in source_paths.items()
    }
    evaluator_hash = source_hashes_before["frozen_evaluator"]
    seed_trace_dir = TRACE_ROOT / f"seed{training_seed}"
    seed_trace_dir.mkdir(parents=True, exist_ok=True)
    case_rows: list[dict[str, Any]] = []
    maximum_transform_error = 0.0
    maximum_actor_input_error = 0.0
    maximum_original_td_mutation_error = 0.0
    maximum_input_observation_mutation_error = 0.0
    maximum_source_action_mutation_error = 0.0
    try:
        for condition in conditions:
            torch.manual_seed(episode_seed)
            np.random.seed(episode_seed)
            td = runtime.env.reset()
            r0_observations = np.empty((runtime.steps, 8, 2), dtype=np.float32)
            roll_observations = np.empty((runtime.steps, 8, 2), dtype=np.float32)
            r0_actions = np.empty((runtime.steps, 8, 2), dtype=np.float32)
            roll_actions = np.empty((runtime.steps, 8, 2), dtype=np.float32)
            applied_actions = np.empty((runtime.steps, 8, 2), dtype=np.float32)
            independent_actions = np.empty((runtime.steps, 8, 2), dtype=np.float32)
            positions = np.empty((runtime.steps + 1, 10), dtype=np.complex128)
            support = np.empty(runtime.steps + 1, dtype=np.float64)
            contact = np.empty(runtime.steps + 1, dtype=np.float64)
            positions[0] = runtime.frozen_eval._positions(runtime.env)
            initial_support = runtime.frozen_eval._log_info_scalar(
                td, "fast_forward_support_index"
            )
            initial_contact = runtime.frozen_eval._log_info_scalar(
                td, "fast_forward_ground_contact_strength"
            )
            if initial_support is None or initial_contact is None:
                raise RuntimeError("Canonical environment lacks support/contact at reset")
            support[0] = initial_support
            contact[0] = initial_contact

            for step in range(runtime.steps):
                original_snapshot = td_snapshot(td)
                r0_input = td.clone(recurse=True)
                roll_input = td.clone(recurse=True)
                r0_observation_before = (
                    r0_input["agents", "observation"].detach().clone()
                )
                roll_observation_before = (
                    roll_input["agents", "observation"].detach().clone()
                )
                input_error = tensor_max_abs(
                    r0_observation_before, roll_observation_before
                )
                maximum_actor_input_error = max(maximum_actor_input_error, input_error)
                input_bit_exact = tensor_bit_exact(
                    r0_observation_before, roll_observation_before
                )
                r0_output = runtime.choose_action(
                    runtime.r0_policy, r0_input, "deterministic"
                )
                roll_output = runtime.choose_action(
                    runtime.roll_policy, roll_input, "deterministic"
                )
                input_mutation_error = max(
                    tensor_max_abs(
                        r0_observation_before,
                        r0_output["agents", "observation"].detach(),
                    ),
                    tensor_max_abs(
                        roll_observation_before,
                        roll_output["agents", "observation"].detach(),
                    ),
                )
                maximum_input_observation_mutation_error = max(
                    maximum_input_observation_mutation_error, input_mutation_error
                )
                original_mutation_error = snapshot_error(original_snapshot, td)
                maximum_original_td_mutation_error = max(
                    maximum_original_td_mutation_error, original_mutation_error
                )
                input_observations_unchanged = (
                    tensor_bit_exact(
                        r0_observation_before,
                        r0_output["agents", "observation"].detach(),
                    )
                    and tensor_bit_exact(
                        roll_observation_before,
                        roll_output["agents", "observation"].detach(),
                    )
                )
                original_td_unchanged = snapshot_bit_exact(original_snapshot, td)
                r0 = r0_output["agents", "action"].detach().clone()
                roll = roll_output["agents", "action"].detach().clone()
                r0_before_apply = r0.detach().clone()
                roll_before_apply = roll.detach().clone()
                actual = runtime.apply_condition(condition, r0, roll, step)
                expected_np = independent_expected(
                    condition,
                    r0[0].detach().cpu().numpy(),
                    roll[0].detach().cpu().numpy(),
                    runtime.calibration,
                    runtime.permutation,
                    step,
                ).astype(np.float32)
                expected = torch.from_numpy(expected_np).to(
                    device=actual.device, dtype=actual.dtype
                ).unsqueeze(0)
                transform_error = tensor_max_abs(actual, expected)
                maximum_transform_error = max(
                    maximum_transform_error, transform_error
                )
                transform_bit_exact = tensor_bit_exact(actual, expected)
                source_action_mutation_error = max(
                    tensor_max_abs(r0_before_apply, r0),
                    tensor_max_abs(roll_before_apply, roll),
                )
                maximum_source_action_mutation_error = max(
                    maximum_source_action_mutation_error, source_action_mutation_error
                )
                source_actions_unchanged = tensor_bit_exact(
                    r0_before_apply, r0
                ) and tensor_bit_exact(roll_before_apply, roll)
                if (
                    not input_bit_exact
                    or not input_observations_unchanged
                    or not original_td_unchanged
                    or not transform_bit_exact
                    or not source_actions_unchanged
                ):
                    raise RuntimeError(
                        f"Step audit failed seed={training_seed} condition={condition.id} "
                        f"step={step}: input={input_error}, input_mut={input_mutation_error}, "
                        f"original_mut={original_mutation_error}, transform={transform_error}, "
                        f"source_action_mut={source_action_mutation_error}"
                    )

                r0_observations[step] = r0_observation_before[0].cpu().numpy()
                roll_observations[step] = roll_observation_before[0].cpu().numpy()
                r0_actions[step] = r0[0].cpu().numpy()
                roll_actions[step] = roll[0].cpu().numpy()
                applied_actions[step] = actual[0].cpu().numpy()
                independent_actions[step] = expected_np
                action_td = td.clone(recurse=True)
                action_td["agents", "action"] = actual
                if not snapshot_bit_exact(original_snapshot, td):
                    raise RuntimeError(
                        f"Original TensorDict changed before env step: "
                        f"{training_seed}/{condition.id}/{step}"
                    )
                td = runtime.env.step(action_td)["next"]
                positions[step + 1] = runtime.frozen_eval._positions(runtime.env)
                support_value = runtime.frozen_eval._log_info_scalar(
                    td, "fast_forward_support_index"
                )
                contact_value = runtime.frozen_eval._log_info_scalar(
                    td, "fast_forward_ground_contact_strength"
                )
                if support_value is None or contact_value is None:
                    raise RuntimeError("Canonical environment lost support/contact exports")
                support[step + 1] = support_value
                contact[step + 1] = contact_value

            metrics = runtime.frozen_eval._episode_metrics(
                list(positions),
                "right",
                "left",
                runtime.metric_args,
                support.tolist(),
                contact.tolist(),
            )
            metrics["success"] = episode_success(
                metrics, config["episode_success"]
            )
            if metrics.get("contact_metric_source") != "env_fast_forward_log_info":
                raise RuntimeError(
                    f"Noncanonical contact metric for {training_seed}/{condition.id}"
                )
            independent_metrics = independent_five_metrics(
                positions, support, contact, config["metric_parameters"]
            )
            independent_metric_success = independent_success(
                independent_metrics, config["episode_success"]
            )
            frozen_five = {key: metrics.get(key) for key in FIVE_METRIC_KEYS}
            for key in FIVE_METRIC_KEYS:
                compare_metric_value(
                    independent_metrics[key],
                    frozen_five[key],
                    metric_tolerance,
                    f"independent_vs_frozen.{training_seed}.{condition.id}.{key}",
                )
            if independent_metric_success != bool(metrics["success"]):
                raise RuntimeError(
                    f"Independent success mismatch for {training_seed}/{condition.id}"
                )
            stored, result_path, result_hash = main_result_episode(
                training_seed,
                condition,
                config,
                runtime.checkpoint_hashes_before,
                evaluator_hash,
            )
            actual_projection = metric_projection(metrics)
            stored_projection = metric_projection(stored)
            for key, expected_value in stored_projection.items():
                compare_metric_value(
                    actual_projection[key],
                    expected_value,
                    metric_tolerance,
                    f"{training_seed}.{condition.id}.{key}",
                )
            if bool(metrics["success"]) != bool(stored["success"]):
                raise RuntimeError(
                    f"Stored success mismatch for {training_seed}/{condition.id}"
                )
            for key in FIVE_METRIC_KEYS:
                compare_metric_value(
                    independent_metrics[key],
                    stored.get(key),
                    metric_tolerance,
                    f"independent_vs_stored.{training_seed}.{condition.id}.{key}",
                )
            if independent_metric_success != bool(stored["success"]):
                raise RuntimeError(
                    f"Independent/stored success mismatch for {training_seed}/{condition.id}"
                )
            trace_path = seed_trace_dir / f"{condition.id}.npz"
            np.savez_compressed(
                trace_path,
                step=np.arange(runtime.steps, dtype=np.int64),
                observation=r0_observations,
                r0_observation_input=r0_observations,
                rroll_observation_input=roll_observations,
                r0_action=r0_actions,
                roll_action=roll_actions,
                applied_action=applied_actions,
                independent_action=independent_actions,
                positions=positions,
                support_index=support,
                ground_contact_strength=contact,
                training_seed=np.asarray(training_seed, dtype=np.int64),
                evaluation_seed=np.asarray(episode_seed, dtype=np.int64),
                condition_id=np.asarray(condition.id),
            )
            case_rows.append(
                {
                    "condition_id": condition.id,
                    "trace": str(trace_path),
                    "trace_sha256": sha256_file(trace_path),
                    "result": str(result_path),
                    "result_sha256": result_hash,
                    "metrics_match_main_result": True,
                    "independent_five_metrics": independent_metrics,
                    "independent_success": independent_metric_success,
                    "success": bool(metrics["success"]),
                }
            )

        source_hashes_after = {
            name: sha256_file(path) for name, path in source_paths.items()
        }
        if source_hashes_after != source_hashes_before:
            raise RuntimeError(f"Audit source drift during seed {training_seed}")
        payload = {
            "schema": "obs2_v2_1_k_independent_trace_seed/v1",
            "training_seed": training_seed,
            "evaluation_seed": episode_seed,
            "condition_count": len(conditions),
            "maximum_actor_input_observation_error": maximum_actor_input_error,
            "maximum_actor_input_observation_mutation_error": maximum_input_observation_mutation_error,
            "maximum_original_td_mutation_error": maximum_original_td_mutation_error,
            "maximum_independent_transform_error": maximum_transform_error,
            "maximum_source_action_mutation_error": maximum_source_action_mutation_error,
            "cases": case_rows,
            "immutability": runtime.verify_unchanged(),
            "source_sha256": source_hashes_after,
        }
        atomic_json(TRACE_ROOT / f"seed{training_seed}_summary.json", payload)
        return payload
    finally:
        runtime.close()


def run_child(seed: int) -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    payload = run_seed(seed)
    print(
        json.dumps(
            {
                "training_seed": seed,
                "condition_count": payload["condition_count"],
                "max_transform_error": payload[
                    "maximum_independent_transform_error"
                ],
            }
        )
    )


def audit_environment(config: Mapping[str, Any]) -> dict[str, str]:
    formal_config = load_json(
        Path(str(config["formal_root"])) / "_control" / "experiment_config.json"
    )
    site_packages = str(formal_config["runtime"]["site_packages"])
    environment = dict(os.environ)
    old_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = site_packages + (
        os.pathsep + old_pythonpath if old_pythonpath else ""
    )
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "PYGAME_HIDE_SUPPORT_PROMPT": "1",
        }
    )
    return environment


def relative_artifact_key(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise RuntimeError(f"Audit artifact escaped study root: {resolved}") from error


def validate_trace_archive(
    path: Path, training_seed: int, evaluation_seed: int, condition_id: str, steps: int
) -> None:
    expected_keys = {
        "step",
        "observation",
        "r0_observation_input",
        "rroll_observation_input",
        "r0_action",
        "roll_action",
        "applied_action",
        "independent_action",
        "positions",
        "support_index",
        "ground_contact_strength",
        "training_seed",
        "evaluation_seed",
        "condition_id",
    }
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != expected_keys:
            raise RuntimeError(f"Persisted trace key inventory mismatch: {path}")
        if not np.array_equal(data["step"], np.arange(steps, dtype=np.int64)):
            raise RuntimeError(f"Persisted trace step index mismatch: {path}")
        for key in (
            "observation",
            "r0_observation_input",
            "rroll_observation_input",
            "r0_action",
            "roll_action",
            "applied_action",
            "independent_action",
        ):
            value = data[key]
            if value.shape != (steps, 8, 2) or value.dtype != np.float32:
                raise RuntimeError(f"Persisted trace array contract mismatch: {path}:{key}")
            if not np.isfinite(value).all():
                raise RuntimeError(f"Persisted trace contains NaN/Inf: {path}:{key}")
        if data["positions"].shape != (steps + 1, 10) or data["positions"].dtype != np.complex128:
            raise RuntimeError(f"Persisted position trace contract mismatch: {path}")
        if not np.isfinite(data["positions"].real).all() or not np.isfinite(
            data["positions"].imag
        ).all():
            raise RuntimeError(f"Persisted position trace contains NaN/Inf: {path}")
        for key in ("support_index", "ground_contact_strength"):
            if data[key].shape != (steps + 1,) or data[key].dtype != np.float64:
                raise RuntimeError(f"Persisted contact trace contract mismatch: {path}:{key}")
            if not np.isfinite(data[key]).all():
                raise RuntimeError(f"Persisted contact trace contains NaN/Inf: {path}:{key}")
        if not np.array_equal(data["observation"], data["r0_observation_input"]):
            raise RuntimeError(f"Persisted observation alias mismatch: {path}")
        if data["r0_observation_input"].tobytes() != data[
            "rroll_observation_input"
        ].tobytes():
            raise RuntimeError(f"Persisted actor input is not bit-exact: {path}")
        if data["applied_action"].tobytes() != data["independent_action"].tobytes():
            raise RuntimeError(f"Persisted transform is not bit-exact: {path}")
        if int(data["training_seed"].item()) != training_seed:
            raise RuntimeError(f"Persisted training seed mismatch: {path}")
        if int(data["evaluation_seed"].item()) != evaluation_seed:
            raise RuntimeError(f"Persisted evaluation seed mismatch: {path}")
        if str(data["condition_id"].item()) != condition_id:
            raise RuntimeError(f"Persisted condition ID mismatch: {path}")


def orchestrate() -> None:
    config = load_json(ROOT / "study_config.json")
    audit_hash_before = sha256_file(Path(__file__).resolve())
    running_payload = {
        "schema": "obs2_v2_1_k_independent_trace_audit/v1",
        "study_id": config["study_id"],
        "passed": False,
        "status": "running",
        "audit_source_sha256": audit_hash_before,
        "transform_source_sha256": sha256_file(ROOT / "smoke_test.py"),
        "result_file_sha256": {},
        "checkpoint_sha256": {},
        "trace_file_sha256": {},
        "audited_episodes": 0,
    }
    # Replacing any old true marker before validation/work makes a rerun fail closed.
    PASS_PATH.unlink(missing_ok=True)
    atomic_json(PASS_PATH, running_payload)
    conditions = list(build_conditions())
    seeds = [int(value) for value in config["training_seeds"]]
    worker_count = int(config["main_evaluation"]["max_parallel_seed_workers"])
    expected_cases = len(seeds) * len(conditions)
    if (
        len(seeds) != 5
        or len(set(seeds)) != 5
        or len(conditions) != 59
        or worker_count != 5
        or int(config["main_evaluation"]["training_seed_count"]) != 5
        or int(config["main_evaluation"]["condition_count"]) != 59
        or config["locked_contract"].get("direction") != "right"
        or config["locked_contract"].get("tail_side") != "left"
        or int(config["locked_contract"].get("particle_count", -1)) != 10
        or int(config["main_evaluation"]["total_episodes"])
        != 5 * 59 * int(config["main_evaluation"]["episodes"])
    ):
        raise RuntimeError("Trace audit is locked to five workers over 5x59 cases")
    validate_execution_complete(config, conditions)
    frozen_before = validate_frozen_hashes(config, conditions)
    calibration_before = validate_calibration_hashes(config)
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    child_environment = audit_environment(config)
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures: list[tuple[int, Path, Any]] = []
        for seed in seeds:
            stdout = logs / f"trace_audit_seed{seed}.stdout.log"
            stderr = logs / f"trace_audit_seed{seed}.stderr.log"
            command = [sys.executable, str(Path(__file__).resolve()), "--seed", str(seed)]

            def invoke(
                command: list[str] = command,
                stdout: Path = stdout,
                stderr: Path = stderr,
            ) -> int:
                with stdout.open("w", encoding="utf-8") as out, stderr.open(
                    "w", encoding="utf-8"
                ) as err:
                    return subprocess.run(
                        command,
                        cwd=str(ROOT),
                        env=child_environment,
                        stdout=out,
                        stderr=err,
                        check=False,
                    ).returncode

            futures.append((seed, stderr, executor.submit(invoke)))
        for seed, stderr, future in futures:
            if future.result() != 0:
                raise RuntimeError(
                    f"Independent trace audit seed {seed} failed:\n"
                    + stderr.read_text(encoding="utf-8", errors="replace")[-6000:]
                )

    frozen_after = validate_frozen_hashes(config, conditions)
    if frozen_after != frozen_before:
        raise RuntimeError("Frozen source/checkpoint hashes changed during trace audit")
    if sha256_file(Path(__file__).resolve()) != audit_hash_before:
        raise RuntimeError("Independent audit source changed during trace audit")
    calibration_after = validate_calibration_hashes(config)
    if calibration_after != calibration_before:
        raise RuntimeError("Calibration hashes changed during trace audit")
    summaries = [load_json(TRACE_ROOT / f"seed{seed}_summary.json") for seed in seeds]
    trace_hashes: dict[str, str] = {}
    result_hashes: dict[str, str] = {}
    checkpoint_hashes: dict[str, dict[str, str]] = {}
    condition_ids = [condition.id for condition in conditions]
    evaluation_seed = int(config["main_evaluation"]["base_seed"])
    steps = int(config["main_evaluation"]["steps"])
    for expected_seed, summary in zip(seeds, summaries):
        if (
            summary.get("schema") != "obs2_v2_1_k_independent_trace_seed/v1"
            or int(summary.get("training_seed", -1)) != expected_seed
            or int(summary.get("evaluation_seed", -1)) != evaluation_seed
            or int(summary.get("condition_count", -1)) != len(conditions)
        ):
            raise RuntimeError("Trace audit condition inventory mismatch")
        for key in (
            "maximum_actor_input_observation_error",
            "maximum_actor_input_observation_mutation_error",
            "maximum_original_td_mutation_error",
            "maximum_independent_transform_error",
            "maximum_source_action_mutation_error",
        ):
            if float(summary[key]) != 0.0:
                raise RuntimeError(f"Trace audit nonzero {key}")
        cases = summary.get("cases")
        if not isinstance(cases, list) or [case.get("condition_id") for case in cases] != condition_ids:
            raise RuntimeError("Trace audit case ID inventory mismatch")
        immutability = summary.get("immutability")
        if (
            not isinstance(immutability, dict)
            or immutability.get("checkpoints_before_after_equal") is not True
            or immutability.get("policies_before_after_equal") is not True
        ):
            raise RuntimeError(f"Runtime immutability evidence invalid: seed {expected_seed}")
        expected_checkpoint_pair = {
            arm: frozen_before["protected_sha256"][
                f"checkpoint_seed{expected_seed}_{arm}"
            ]
            for arm in ("R0", "Rroll")
        }
        if immutability.get("checkpoint_sha256") != expected_checkpoint_pair:
            raise RuntimeError(f"Checkpoint evidence mismatch: seed {expected_seed}")
        checkpoint_hashes[str(expected_seed)] = expected_checkpoint_pair
        expected_seed_sources = {
            **frozen_before["source_sha256"],
            "frozen_evaluator": frozen_before["protected_sha256"]["frozen_evaluator"],
        }
        expected_seed_sources.pop("run_study.py")
        if summary.get("source_sha256") != expected_seed_sources:
            raise RuntimeError(f"Seed source evidence mismatch: seed {expected_seed}")
        for case in cases:
            condition_id = str(case["condition_id"])
            if (
                case.get("metrics_match_main_result") is not True
                or case.get("independent_success") != case.get("success")
                or set(case.get("independent_five_metrics", {})) != set(FIVE_METRIC_KEYS)
            ):
                raise RuntimeError(
                    f"Independent metric evidence invalid: {expected_seed}/{condition_id}"
                )
            trace_path = Path(str(case["trace"])).resolve()
            result_path = Path(str(case["result"])).resolve()
            expected_trace_path = (
                TRACE_ROOT / f"seed{expected_seed}" / f"{condition_id}.npz"
            ).resolve()
            expected_result_path = (
                ROOT / "results" / f"seed{expected_seed}" / f"{condition_id}.json"
            ).resolve()
            if trace_path != expected_trace_path or result_path != expected_result_path:
                raise RuntimeError(f"Audit artifact path mismatch: {expected_seed}/{condition_id}")
            validate_trace_archive(
                trace_path, expected_seed, evaluation_seed, condition_id, steps
            )
            trace_actual = sha256_file(trace_path)
            result_actual = sha256_file(result_path)
            if trace_actual != case["trace_sha256"]:
                raise RuntimeError(f"Trace hash drift: {trace_path}")
            if result_actual != case["result_sha256"]:
                raise RuntimeError(f"Result hash drift: {result_path}")
            trace_key = relative_artifact_key(trace_path)
            result_key = relative_artifact_key(result_path)
            if trace_key in trace_hashes or result_key in result_hashes:
                raise RuntimeError("Duplicate trace/result artifact in audit inventory")
            trace_hashes[trace_key] = trace_actual
            result_hashes[result_key] = result_actual
    if len(trace_hashes) != expected_cases or len(result_hashes) != expected_cases:
        raise RuntimeError("Trace/result hash manifest is incomplete")
    payload = {
        "schema": "obs2_v2_1_k_independent_trace_audit/v1",
        "study_id": config["study_id"],
        "passed": True,
        "status": "complete",
        "training_seed_count": len(seeds),
        "condition_count": len(conditions),
        "parallel_process_workers": worker_count,
        "audited_policy_condition_cases": expected_cases,
        "audited_episode_count": expected_cases,
        "audited_episodes": expected_cases,
        "evaluation_seed": evaluation_seed,
        "step_count_per_case": steps,
        "actor_inputs_bit_exact": True,
        "actor_input_observations_unchanged": True,
        "original_tensordict_unmodified_by_policy_calls": True,
        "source_actions_unmodified_by_transform": True,
        "independent_transform_bit_exact": True,
        "independent_five_metrics_match_frozen_evaluator": True,
        "rerun_metrics_match_stored_main_results": True,
        "checkpoint_and_policy_hashes_unchanged": True,
        "condition_matrix_sha256": frozen_before["condition_matrix_sha256"],
        "frozen_contract_sha256": frozen_before["contract_sha256"],
        "frozen_source_sha256": frozen_before["source_sha256"],
        "frozen_evaluator_sha256": frozen_before["protected_sha256"][
            "frozen_evaluator"
        ],
        "calibration_sha256": calibration_before,
        "checkpoint_sha256": checkpoint_hashes,
        "result_file_sha256": result_hashes,
        "trace_file_sha256": trace_hashes,
        "audit_source_sha256": audit_hash_before,
        "independent_transform_source": str((ROOT / "smoke_test.py").resolve()),
        "transform_source_sha256": sha256_file(ROOT / "smoke_test.py"),
    }
    if sha256_file(Path(__file__).resolve()) != audit_hash_before:
        raise RuntimeError("Independent audit source changed before PASS publication")
    atomic_json(PASS_PATH, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.seed is None:
        try:
            orchestrate()
        except Exception as error:
            config = load_json(ROOT / "study_config.json")
            failure = {
                "schema": "obs2_v2_1_k_independent_trace_audit/v1",
                "study_id": config.get("study_id"),
                "passed": False,
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
                "audit_source_sha256": sha256_file(Path(__file__).resolve()),
                "transform_source_sha256": sha256_file(ROOT / "smoke_test.py"),
                "result_file_sha256": {},
                "checkpoint_sha256": {},
                "trace_file_sha256": {},
                "audited_episodes": 0,
            }
            PASS_PATH.unlink(missing_ok=True)
            atomic_json(PASS_PATH, failure)
            raise
    else:
        run_child(args.seed)


if __name__ == "__main__":
    main()
