from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "study_config.json"
ACTIVE_CONTRACT_PATH = ROOT / "FROZEN_CONTRACT_R2.json"
PASS_PATH = ROOT / "VALIDATION_PASS.json"
FAIL_PATH = ROOT / "VALIDATION_FAIL.json"
TRACE_AUDIT_PATH = ROOT / "INDEPENDENT_TRACE_AUDIT_PASS.json"
CROSS_AUDIT_PATH = ROOT / "CROSS_ENVIRONMENT_IDENTITY_AUDIT_PASS.json"

EXPECTED_TRAINING_SEEDS = [9201, 9202, 9203, 9204, 9205]
EXPECTED_TRAINING_SEED_COUNT = 5
EXPECTED_CONDITION_COUNT = 59
EXPECTED_EPISODES_PER_CASE = 20
EXPECTED_CASE_COUNT = EXPECTED_TRAINING_SEED_COUNT * EXPECTED_CONDITION_COUNT
EXPECTED_TOTAL_EPISODES = EXPECTED_CASE_COUNT * EXPECTED_EPISODES_PER_CASE
EXPECTED_ARMS = ("R0", "Rroll")
SHA256_HEX_LENGTH = 64

RESULT_KEYS = {
    "schema",
    "study_id",
    "training_seed",
    "condition",
    "evaluation_base_seed",
    "evaluation_episodes",
    "evaluation_steps",
    "success_episodes",
    "episodes",
    "checkpoint_sha256",
    "frozen_evaluator_sha256",
}
EPISODE_KEYS = {
    "steps",
    "classification",
    "particle_count",
    "initial_body_length",
    "forward_displacement",
    "forward_body_lengths",
    "net_best_fit_rotation_degrees",
    "desired_net_rotation_degrees",
    "desired_positive_rotation_degrees",
    "reverse_rotation_degrees",
    "desired_active_rotation_fraction",
    "contact_metric_source",
    "environment_support_index_available",
    "ground_contact_valid_fraction",
    "mean_ground_contact_strength",
    "contact_material_index_span",
    "contact_material_index_span_fraction",
    "hard_support_unique_material_indices",
    "mean_active_contact_particles",
    "tail_launch_detected",
    "tail_launch_count",
    "tail_launch_steps",
    "tail_launch_candidate_count",
    "tail_launch_candidate_steps",
    "peak_tail_lift_body_fraction",
    "peak_tail_forward_body_fraction",
    "peak_tail_prefix_curvature_degrees",
    "roll_pulse_count",
    "roll_pulse_intervals_steps",
    "mean_roll_pulse_interval_steps",
    "roll_pulses",
    "seed",
    "success",
    "condition_id",
    "training_seed",
    "same_observation_error",
    "joint_summary",
}
PULSE_KEYS = {
    "start_step",
    "end_step",
    "duration_steps",
    "interval_from_previous_end_steps",
    "idle_steps_since_previous_pulse_end",
    "desired_net_rotation_degrees",
    "forward_displacement",
    "forward_body_fraction",
    "contact_index_start",
    "contact_index_end",
    "contact_index_span",
    "contact_index_span_fraction",
    "desired_active_rotation_fraction",
    "tail_launch_detected",
    "tail_launch_step",
}
JOINT_KEYS = {
    "K_mean",
    "K_abs_mean",
    "K_positive_fraction",
    "abs_delta_from_Rroll_K_mean",
    "tau1_boundary_rms",
    "tau2_boundary_rms",
    "tau_boundary_rms",
    "power_boundary_abs_mean",
    "torque_boundary_saturation_fraction",
}
CLASSIFICATIONS = {
    "repeated_fast_forward_roll",
    "single_fast_forward_roll",
    "forward_motion_without_roll_pulse",
    "no_meaningful_forward_roll",
}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"Duplicate JSON object key: {key!r}")
        payload[key] = value
    return payload


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys
    )
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


CONFIG: dict[str, Any] = {}


def build_conditions() -> list[Any]:
    # Delay project-source import until main() has invalidated any old pass marker.
    from condition_matrix import build_conditions as frozen_build_conditions

    return frozen_build_conditions()


def canonical_sha256(conditions: list[Any]) -> str:
    from condition_matrix import canonical_sha256 as frozen_canonical_sha256

    return frozen_canonical_sha256(conditions)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = canonical_json_text(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_json_text(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def strict_json_equal(actual: Any, expected: Any) -> bool:
    return canonical_json_text(actual) == canonical_json_text(expected)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def remove_file_if_present(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys mismatch; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
    return value


def strict_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be a real bool")
    return value


def strict_string(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    return value


def strict_number(
    value: Any,
    label: str,
    *,
    allow_none: bool = False,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number, got {type(value).__name__}")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be finite, got {converted!r}")
    if minimum is not None and converted < minimum:
        raise ValueError(f"{label} must be >= {minimum}, got {converted}")
    if maximum is not None and converted > maximum:
        raise ValueError(f"{label} must be <= {maximum}, got {converted}")
    return converted


def strict_int(
    value: Any,
    label: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be a non-bool integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} must be <= {maximum}, got {value}")
    return value


def strict_hash(value: Any, label: str) -> str:
    digest = strict_string(value, label)
    if (
        len(digest) != SHA256_HEX_LENGTH
        or digest.lower() != digest
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256 hex digest")
    return digest


def require_expected_value(actual: Any, expected: Any, label: str) -> None:
    if type(expected) is bool:
        if strict_bool(actual, label) is not expected:
            raise RuntimeError(f"{label} mismatch")
    elif actual != expected:
        raise RuntimeError(f"{label} mismatch")


def strict_int_list(
    value: Any,
    label: str,
    *,
    minimum: int,
    maximum: int,
    strictly_increasing: bool = False,
    unique: bool = True,
) -> list[int]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    result = [
        strict_int(item, f"{label}[{index}]", minimum=minimum, maximum=maximum)
        for index, item in enumerate(value)
    ]
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{label} contains duplicate values")
    if strictly_increasing and result != sorted(result):
        raise ValueError(f"{label} must be strictly increasing")
    return result


def close_enough(actual: float, expected: float, label: str) -> None:
    tolerance = 1e-9 * max(1.0, abs(actual), abs(expected))
    if abs(actual - expected) > tolerance:
        raise ValueError(f"{label} mismatch: {actual!r} != {expected!r}")


def ensure_regular_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise RuntimeError(f"{label} is a symlink: {path}")
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    return resolved


def ensure_regular_file_inside(path: Path, root: Path, label: str) -> Path:
    resolved = ensure_regular_file(path, label)
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(f"{label} escapes its root: {resolved}") from error
    return resolved


def expected_checkpoint_hashes(contract: dict[str, Any]) -> dict[str, dict[str, str]]:
    formal = contract["protected_formal_files"]
    return {
        str(seed): {
            arm: strict_hash(
                formal[f"checkpoint_seed{seed}_{arm}"]["sha256"],
                f"contract checkpoint seed{seed}/{arm}",
            )
            for arm in EXPECTED_ARMS
        }
        for seed in EXPECTED_TRAINING_SEEDS
    }


def verify_checkpoint_map(
    value: Any, expected: dict[str, dict[str, str]], label: str
) -> None:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{label} seed inventory mismatch")
    for seed, expected_arms in expected.items():
        arms = value[seed]
        if not isinstance(arms, dict) or set(arms) != set(EXPECTED_ARMS):
            raise ValueError(f"{label}.{seed} arm inventory mismatch")
        for arm, expected_hash in expected_arms.items():
            actual = strict_hash(arms[arm], f"{label}.{seed}.{arm}")
            if actual != expected_hash:
                raise RuntimeError(f"{label}.{seed}.{arm} hash mismatch")


def verify_config_contract() -> None:
    if CONFIG.get("schema") != "obs2_v2_1_k_mechanism/v1":
        raise RuntimeError("Study config schema mismatch")
    if CONFIG.get("study_id") != "obs2_v2_1_k_mechanism_20260804":
        raise RuntimeError("Study config ID mismatch")
    if CONFIG.get("analysis_type") != "frozen_checkpoint_evaluation_only":
        raise RuntimeError("Study analysis type mismatch")
    seeds = CONFIG.get("training_seeds")
    if seeds != EXPECTED_TRAINING_SEEDS or len(set(seeds)) != len(seeds):
        raise RuntimeError("Study training-seed order/uniqueness mismatch")
    main = CONFIG.get("main_evaluation")
    if not isinstance(main, dict):
        raise TypeError("main_evaluation must be an object")
    expected = {
        "episodes": EXPECTED_EPISODES_PER_CASE,
        "condition_count": EXPECTED_CONDITION_COUNT,
        "training_seed_count": EXPECTED_TRAINING_SEED_COUNT,
        "total_episodes": EXPECTED_TOTAL_EPISODES,
    }
    for key, value in expected.items():
        if main.get(key) != value:
            raise RuntimeError(f"Frozen main_evaluation.{key} mismatch")
    strict_int(main.get("base_seed"), "main_evaluation.base_seed", minimum=0)
    strict_int(main.get("steps"), "main_evaluation.steps", minimum=1)


def verify_frozen_bindings(conditions_list: list[Any]) -> dict[str, Any]:
    verify_config_contract()
    contract_path = ensure_regular_file(ACTIVE_CONTRACT_PATH, "active frozen contract")
    contract = load_json(contract_path)
    expected_headers = {
        "schema": "obs2_v2_1_k_frozen_contract/v1",
        "study_id": CONFIG["study_id"],
        "analysis_type": CONFIG["analysis_type"],
        "condition_count": EXPECTED_CONDITION_COUNT,
        "training_seeds": EXPECTED_TRAINING_SEEDS,
        "identity_gate": CONFIG["identity_gate"],
        "main_evaluation": CONFIG["main_evaluation"],
        "episode_success": CONFIG["episode_success"],
        "locked_contract": CONFIG["locked_contract"],
    }
    for key, expected in expected_headers.items():
        if not strict_json_equal(contract.get(key), expected):
            raise RuntimeError(f"Active contract mismatch for {key}")
    condition_dicts = [condition.to_dict() for condition in conditions_list]
    condition_ids = [condition["id"] for condition in condition_dicts]
    if len(condition_dicts) != EXPECTED_CONDITION_COUNT:
        raise RuntimeError("Canonical condition count is not exactly 59")
    if len(set(condition_ids)) != len(condition_ids):
        raise RuntimeError("Canonical condition IDs are not unique")
    matrix_hash = canonical_sha256(conditions_list)
    if not strict_json_equal(contract.get("conditions"), condition_dicts):
        raise RuntimeError("Frozen condition list/order/content mismatch")
    if contract.get("conditions_sha256") != matrix_hash:
        raise RuntimeError("Canonical condition matrix hash mismatch")

    expected_sources = {
        "study_config.json",
        "condition_matrix.py",
        "mechanism_rollout.py",
        "smoke_test.py",
        "run_study.py",
    }
    source_files = contract.get("source_files")
    if not isinstance(source_files, dict) or set(source_files) != expected_sources:
        raise RuntimeError("Frozen execution-source inventory mismatch")
    for name, evidence in source_files.items():
        if not isinstance(evidence, dict):
            raise TypeError(f"Contract source evidence is not an object: {name}")
        path = ensure_regular_file(Path(evidence["path"]), f"frozen source {name}")
        if path.name != name:
            raise RuntimeError(f"Frozen source path/name mismatch: {name}")
        if path.stat().st_size != strict_int(evidence["size"], f"{name}.size", minimum=0):
            raise RuntimeError(f"Frozen execution-source size changed: {name}")
        if sha256_file(path) != strict_hash(evidence["sha256"], f"{name}.sha256"):
            raise RuntimeError(f"Frozen execution source changed: {name}")

    expected_formal = {"formal_config", "formal_result", "formal_source_manifest", "frozen_evaluator"}
    expected_formal.update(
        f"checkpoint_seed{seed}_{arm}"
        for seed in EXPECTED_TRAINING_SEEDS
        for arm in EXPECTED_ARMS
    )
    formal_files = contract.get("protected_formal_files")
    if not isinstance(formal_files, dict) or set(formal_files) != expected_formal:
        raise RuntimeError("Protected formal-file inventory mismatch")
    for name, evidence in formal_files.items():
        if not isinstance(evidence, dict):
            raise TypeError(f"Formal evidence is not an object: {name}")
        path = ensure_regular_file(Path(evidence["path"]), f"protected formal file {name}")
        if path.stat().st_size != strict_int(evidence["size"], f"{name}.size", minimum=0):
            raise RuntimeError(f"Protected formal-file size changed: {name}")
        if sha256_file(path) != strict_hash(evidence["sha256"], f"{name}.sha256"):
            raise RuntimeError(f"Protected formal source changed: {name}")

    amendments = contract.get("technical_amendments")
    if not isinstance(amendments, list) or len(amendments) != 2:
        raise RuntimeError("Technical-amendment inventory mismatch")
    for index, evidence in enumerate(amendments):
        if not isinstance(evidence, dict) or set(evidence) != {"path", "sha256"}:
            raise RuntimeError(f"Technical amendment {index} evidence mismatch")
        path = ensure_regular_file(Path(evidence["path"]), f"technical amendment {index}")
        if sha256_file(path) != strict_hash(evidence["sha256"], f"amendment[{index}].sha256"):
            raise RuntimeError(f"Technical amendment changed: {path}")
    return contract


def verify_smoke_and_identity(
    contract: dict[str, Any], checkpoint_hashes: dict[str, dict[str, str]]
) -> dict[str, str]:
    matrix_hash = contract["conditions_sha256"]
    smoke_path = ensure_regular_file(ROOT / "SMOKE_TEST_PASS.json", "smoke pass")
    smoke = load_json(smoke_path)
    expected_smoke = {
        "schema": "obs2_v2_1_k_smoke/v1",
        "passed": True,
        "condition_count": EXPECTED_CONDITION_COUNT,
        "condition_matrix_sha256": matrix_hash,
        "frozen_evaluator_self_test": "passed",
        "C00_endpoint_bit_exact": True,
        "C11_endpoint_bit_exact": True,
        "invalid_sign_vector_failed_closed": True,
    }
    for key, expected in expected_smoke.items():
        require_expected_value(smoke.get(key), expected, f"smoke.{key}")
    if strict_number(smoke.get("maximum_transform_error"), "smoke.maximum_transform_error") != 0.0:
        raise RuntimeError("Smoke transform error is nonzero")
    if strict_number(smoke.get("same_observation_error"), "smoke.same_observation_error") != 0.0:
        raise RuntimeError("Smoke same-observation error is nonzero")
    immutability = smoke.get("immutability")
    if not isinstance(immutability, dict):
        raise TypeError("Smoke immutability evidence must be an object")
    if strict_bool(immutability.get("checkpoints_before_after_equal"), "smoke checkpoint equality") is not True:
        raise RuntimeError("Smoke checkpoint immutability failed")
    if strict_bool(immutability.get("policies_before_after_equal"), "smoke policy equality") is not True:
        raise RuntimeError("Smoke policy immutability failed")
    smoke_checkpoints = immutability.get("checkpoint_sha256")
    if smoke_checkpoints != checkpoint_hashes[str(EXPECTED_TRAINING_SEEDS[0])]:
        raise RuntimeError("Smoke checkpoint hashes are not bound to seed 9201")
    policy_hashes = immutability.get("policy_state_sha256")
    if not isinstance(policy_hashes, dict) or set(policy_hashes) != set(EXPECTED_ARMS):
        raise RuntimeError("Smoke policy-state hash inventory mismatch")
    for arm in EXPECTED_ARMS:
        strict_hash(policy_hashes[arm], f"smoke policy hash {arm}")

    identity_path = ensure_regular_file(ROOT / "IDENTITY_GATE_PASS.json", "identity-gate pass")
    identity = load_json(identity_path)
    if identity.get("schema") != "obs2_v2_1_k_identity_gate/v1":
        raise RuntimeError("Identity-gate schema mismatch")
    if identity.get("study_id") != CONFIG["study_id"]:
        raise RuntimeError("Identity-gate study mismatch")
    if strict_bool(identity.get("passed"), "identity.passed") is not True:
        raise RuntimeError("Identity gate did not pass")
    seed_keys = {str(seed) for seed in EXPECTED_TRAINING_SEEDS}
    expected_success = CONFIG["identity_gate"]["required_formal_success_counts"]
    r0_counts = identity.get("R0_success_episodes_by_seed")
    roll_counts = identity.get("Rroll_success_episodes_by_seed")
    if not isinstance(r0_counts, dict) or set(r0_counts) != seed_keys:
        raise RuntimeError("Identity R0 seed inventory mismatch")
    if not isinstance(roll_counts, dict) or set(roll_counts) != seed_keys:
        raise RuntimeError("Identity Rroll seed inventory mismatch")
    for seed in EXPECTED_TRAINING_SEEDS:
        key = str(seed)
        strict_int(r0_counts[key], f"identity.R0.{key}", minimum=0, maximum=20)
        count = strict_int(roll_counts[key], f"identity.Rroll.{key}", minimum=0, maximum=20)
        if count != expected_success[key]:
            raise RuntimeError(f"Identity Rroll success count mismatch for seed {seed}")

    calibration_hashes = identity.get("calibration_sha256")
    if not isinstance(calibration_hashes, dict) or set(calibration_hashes) != seed_keys:
        raise RuntimeError("Identity calibration inventory mismatch")
    verified_calibration: dict[str, str] = {}
    for seed in EXPECTED_TRAINING_SEEDS:
        key = str(seed)
        expected = strict_hash(calibration_hashes[key], f"identity calibration {key}")
        calibration = ensure_regular_file_inside(
            ROOT / "calibration" / f"seed{seed}.npz", ROOT / "calibration", "calibration template"
        )
        actual = sha256_file(calibration)
        if actual != expected:
            raise RuntimeError(f"Calibration template drift for seed {seed}")
        verified_calibration[key] = actual

    receipts = identity.get("receipts")
    if not isinstance(receipts, list) or len(receipts) != EXPECTED_TRAINING_SEED_COUNT:
        raise RuntimeError("Identity receipt inventory mismatch")
    if [receipt.get("seed") for receipt in receipts if isinstance(receipt, dict)] != EXPECTED_TRAINING_SEEDS:
        raise RuntimeError("Identity receipt seed order/uniqueness mismatch")
    for index, receipt in enumerate(receipts):
        if not isinstance(receipt, dict):
            raise TypeError(f"Identity receipt {index} is not an object")
        if receipt.get("stage") != "identity" or receipt.get("exit_code") != 0:
            raise RuntimeError(f"Identity receipt {index} is not successful")

    return {
        "smoke_sha256": sha256_file(smoke_path),
        "identity_gate_sha256": sha256_file(identity_path),
        "calibration_manifest_sha256": canonical_json_sha256(verified_calibration),
    }


def verify_completion(condition_ids: list[str]) -> dict[str, Any]:
    completion_path = ensure_regular_file(
        ROOT / "MAIN_EXECUTION_COMPLETE.json", "main completion marker"
    )
    completion = load_json(completion_path)
    expected = {
        "schema": "obs2_v2_1_k_execution_complete/v1",
        "study_id": CONFIG["study_id"],
        "status": "complete",
        "training_seed_count": EXPECTED_TRAINING_SEED_COUNT,
        "condition_count": EXPECTED_CONDITION_COUNT,
        "policy_condition_cases": EXPECTED_CASE_COUNT,
        "episodes": EXPECTED_TOTAL_EPISODES,
        "protected_formal_files_unchanged": True,
    }
    for key, value in expected.items():
        require_expected_value(completion.get(key), value, f"completion.{key}")
    inventory = completion.get("inventory")
    seed_keys = [str(seed) for seed in EXPECTED_TRAINING_SEEDS]
    if not isinstance(inventory, dict) or list(inventory) != seed_keys:
        raise RuntimeError("Completion seed inventory/order mismatch")
    expected_ids = sorted(condition_ids)
    for seed in seed_keys:
        ids = inventory[seed]
        if not isinstance(ids, list) or ids != expected_ids or len(set(ids)) != len(ids):
            raise RuntimeError(f"Completion condition inventory/order mismatch for seed {seed}")
    receipts = completion.get("receipts")
    if not isinstance(receipts, list) or len(receipts) != EXPECTED_TRAINING_SEED_COUNT:
        raise RuntimeError("Main completion receipt inventory mismatch")
    if [receipt.get("seed") for receipt in receipts if isinstance(receipt, dict)] != EXPECTED_TRAINING_SEEDS:
        raise RuntimeError("Main receipt seed order/uniqueness mismatch")
    for index, receipt in enumerate(receipts):
        if not isinstance(receipt, dict):
            raise TypeError(f"Main receipt {index} is not an object")
        if receipt.get("stage") != "main" or receipt.get("exit_code") != 0:
            raise RuntimeError(f"Main receipt {index} is not successful")
    return completion


def verify_joint_summary(joint: Any, label: str) -> None:
    joint = require_exact_keys(joint, JOINT_KEYS, label)
    matrix_keys = (
        "K_mean",
        "K_abs_mean",
        "K_positive_fraction",
        "abs_delta_from_Rroll_K_mean",
    )
    vector_keys = (
        "tau1_boundary_rms",
        "tau2_boundary_rms",
        "tau_boundary_rms",
        "power_boundary_abs_mean",
        "torque_boundary_saturation_fraction",
    )
    for key in matrix_keys:
        value = joint[key]
        if not isinstance(value, list) or len(value) != 8:
            raise ValueError(f"{label}.{key} is not [8,2]")
        for row_index, row in enumerate(value):
            if not isinstance(row, list) or len(row) != 2:
                raise ValueError(f"{label}.{key}[{row_index}] is not [2]")
            for column_index, item in enumerate(row):
                strict_number(item, f"{label}.{key}[{row_index}][{column_index}]")
    for key in vector_keys:
        value = joint[key]
        if not isinstance(value, list) or len(value) != 8:
            raise ValueError(f"{label}.{key} is not [8]")
        for index, item in enumerate(value):
            strict_number(item, f"{label}.{key}[{index}]")

    for joint_index in range(8):
        for channel in range(2):
            mean = float(joint["K_mean"][joint_index][channel])
            absolute = float(joint["K_abs_mean"][joint_index][channel])
            positive = float(joint["K_positive_fraction"][joint_index][channel])
            delta = float(joint["abs_delta_from_Rroll_K_mean"][joint_index][channel])
            if absolute < 0.0 or absolute + 1e-8 < abs(mean):
                raise ValueError(f"{label}: invalid K_abs_mean")
            if not 0.0 <= positive <= 1.0:
                raise ValueError(f"{label}: K positive fraction outside [0,1]")
            if delta < 0.0:
                raise ValueError(f"{label}: negative absolute K delta")
        for key in (
            "tau1_boundary_rms",
            "tau2_boundary_rms",
            "tau_boundary_rms",
            "power_boundary_abs_mean",
        ):
            if float(joint[key][joint_index]) < 0.0:
                raise ValueError(f"{label}: negative {key}")
        saturation = float(joint["torque_boundary_saturation_fraction"][joint_index])
        if not 0.0 <= saturation <= 1.0:
            raise ValueError(f"{label}: saturation outside [0,1]")


def verify_pulses(episode: dict[str, Any], label: str, steps: int, particles: int) -> None:
    pulse_count = strict_int(
        episode["roll_pulse_count"], f"{label}.roll_pulse_count", minimum=0, maximum=steps
    )
    pulses = episode["roll_pulses"]
    if not isinstance(pulses, list) or len(pulses) != pulse_count:
        raise ValueError(f"{label}.roll_pulses count mismatch")
    intervals = strict_int_list(
        episode["roll_pulse_intervals_steps"],
        f"{label}.roll_pulse_intervals_steps",
        minimum=1,
        maximum=steps,
        unique=False,
    )
    if len(intervals) != max(0, pulse_count - 1):
        raise ValueError(f"{label}.roll_pulse_intervals_steps length mismatch")
    mean_interval = strict_number(
        episode["mean_roll_pulse_interval_steps"],
        f"{label}.mean_roll_pulse_interval_steps",
        allow_none=True,
        minimum=0.0,
        maximum=float(steps),
    )
    if intervals:
        if mean_interval is None:
            raise ValueError(f"{label}: pulse interval mean is absent")
        close_enough(mean_interval, sum(intervals) / len(intervals), f"{label}.pulse interval mean")
    elif mean_interval is not None:
        raise ValueError(f"{label}: pulse interval mean must be null")

    previous_end: int | None = None
    pulse_launch_steps: list[int] = []
    for index, raw_pulse in enumerate(pulses):
        pulse_label = f"{label}.roll_pulses[{index}]"
        pulse = require_exact_keys(raw_pulse, PULSE_KEYS, pulse_label)
        start = strict_int(pulse["start_step"], f"{pulse_label}.start_step", minimum=0, maximum=steps - 1)
        end = strict_int(pulse["end_step"], f"{pulse_label}.end_step", minimum=1, maximum=steps)
        if start >= end:
            raise ValueError(f"{pulse_label}: start must precede end")
        duration = strict_int(pulse["duration_steps"], f"{pulse_label}.duration_steps", minimum=1, maximum=steps)
        if duration != end - start:
            raise ValueError(f"{pulse_label}: duration mismatch")
        interval = strict_int(
            pulse["interval_from_previous_end_steps"],
            f"{pulse_label}.interval_from_previous_end_steps",
            minimum=0,
            maximum=steps,
        )
        idle = strict_int(
            pulse["idle_steps_since_previous_pulse_end"],
            f"{pulse_label}.idle_steps_since_previous_pulse_end",
            minimum=0,
            maximum=steps,
        )
        if previous_end is None:
            if interval != 0 or idle != 0:
                raise ValueError(f"{pulse_label}: first-pulse interval/idle must be zero")
        else:
            if start < previous_end or interval != end - previous_end or idle != start - previous_end:
                raise ValueError(f"{pulse_label}: temporal linkage mismatch")
            if interval != intervals[index - 1]:
                raise ValueError(f"{pulse_label}: interval list mismatch")
        strict_number(pulse["desired_net_rotation_degrees"], f"{pulse_label}.desired_net_rotation_degrees", minimum=0.0)
        strict_number(pulse["forward_displacement"], f"{pulse_label}.forward_displacement", minimum=0.0)
        strict_number(pulse["forward_body_fraction"], f"{pulse_label}.forward_body_fraction", minimum=0.0)
        strict_number(pulse["contact_index_start"], f"{pulse_label}.contact_index_start", minimum=0.0, maximum=float(particles - 1))
        strict_number(pulse["contact_index_end"], f"{pulse_label}.contact_index_end", minimum=0.0, maximum=float(particles - 1))
        span = strict_number(pulse["contact_index_span"], f"{pulse_label}.contact_index_span", minimum=0.0, maximum=float(particles - 1))
        span_fraction = strict_number(pulse["contact_index_span_fraction"], f"{pulse_label}.contact_index_span_fraction", minimum=0.0, maximum=1.0)
        assert span is not None and span_fraction is not None
        close_enough(span_fraction, span / max(particles - 1, 1), f"{pulse_label}.contact span fraction")
        strict_number(pulse["desired_active_rotation_fraction"], f"{pulse_label}.desired_active_rotation_fraction", minimum=0.0, maximum=1.0)
        launch_flag = strict_int(pulse["tail_launch_detected"], f"{pulse_label}.tail_launch_detected", minimum=0, maximum=1)
        launch_step = strict_int(pulse["tail_launch_step"], f"{pulse_label}.tail_launch_step", minimum=-1, maximum=steps)
        if (launch_flag == 0) != (launch_step == -1):
            raise ValueError(f"{pulse_label}: tail-launch flag/step mismatch")
        if launch_step >= 0:
            lower_bound = 0 if previous_end is None else previous_end
            if not lower_bound <= launch_step <= end:
                raise ValueError(f"{pulse_label}: tail launch is outside its matching interval")
            pulse_launch_steps.append(launch_step)
        previous_end = end
    # The frozen evaluator matches launch candidates to pulse intervals with
    # inclusive end points.  A launch exactly on a shared pulse boundary can
    # therefore be attached to both adjacent pulses.  The episode-level field
    # intentionally stores the sorted unique set (see
    # evaluate_fast_forward_roll.py:382-388), so validate that same estimand
    # instead of requiring the per-pulse list to be duplicate-free.
    if sorted(set(pulse_launch_steps)) != episode["tail_launch_steps"]:
        raise ValueError(f"{label}: pulse/episode tail-launch steps mismatch")


def recompute_episode_success(episode: dict[str, Any], label: str) -> bool:
    criteria = CONFIG["episode_success"]
    pulses = strict_int(episode["roll_pulse_count"], f"{label}.roll_pulse_count", minimum=0)
    rotation = strict_number(episode["desired_net_rotation_degrees"], f"{label}.desired_net_rotation_degrees")
    direction = strict_number(episode["desired_active_rotation_fraction"], f"{label}.desired_active_rotation_fraction", minimum=0.0, maximum=1.0)
    forward = strict_number(episode["forward_body_lengths"], f"{label}.forward_body_lengths")
    interval = strict_number(
        episode["mean_roll_pulse_interval_steps"],
        f"{label}.mean_roll_pulse_interval_steps",
        allow_none=True,
        minimum=0.0,
    )
    assert rotation is not None and direction is not None and forward is not None
    # These are the five frozen conjunctive success criteria; each is recomputed.
    return bool(
        pulses >= int(criteria["minimum_roll_pulses"])
        and rotation >= float(criteria["minimum_desired_net_rotation_degrees"])
        and direction >= float(criteria["minimum_direction_fraction"])
        and forward >= float(criteria["minimum_forward_body_lengths"])
        and interval is not None
        and interval <= float(criteria["maximum_mean_inter_pulse_interval_steps"])
    )


def verify_episode(
    raw_episode: Any,
    label: str,
    training_seed: int,
    condition_id: str,
    evaluation_seed: int,
) -> bool:
    episode = require_exact_keys(raw_episode, EPISODE_KEYS, label)
    expected_steps = int(CONFIG["main_evaluation"]["steps"])
    steps = strict_int(episode["steps"], f"{label}.steps", minimum=1)
    if steps != expected_steps:
        raise RuntimeError(f"{label}: step count mismatch")
    classification = strict_string(episode["classification"], f"{label}.classification")
    if classification not in CLASSIFICATIONS:
        raise ValueError(f"{label}: unknown classification")
    particles = strict_int(episode["particle_count"], f"{label}.particle_count", minimum=1)
    if particles != int(CONFIG["locked_contract"]["particle_count"]):
        raise RuntimeError(f"{label}: particle count mismatch")
    initial_length = strict_number(episode["initial_body_length"], f"{label}.initial_body_length", minimum=1e-12)
    displacement = strict_number(episode["forward_displacement"], f"{label}.forward_displacement")
    body_lengths = strict_number(episode["forward_body_lengths"], f"{label}.forward_body_lengths")
    assert initial_length is not None and displacement is not None and body_lengths is not None
    close_enough(body_lengths, displacement / initial_length, f"{label}.forward_body_lengths")
    net_rotation = strict_number(episode["net_best_fit_rotation_degrees"], f"{label}.net_best_fit_rotation_degrees")
    desired_rotation = strict_number(episode["desired_net_rotation_degrees"], f"{label}.desired_net_rotation_degrees")
    positive_rotation = strict_number(episode["desired_positive_rotation_degrees"], f"{label}.desired_positive_rotation_degrees", minimum=0.0)
    reverse_rotation = strict_number(episode["reverse_rotation_degrees"], f"{label}.reverse_rotation_degrees", minimum=0.0)
    assert net_rotation is not None and desired_rotation is not None
    assert positive_rotation is not None and reverse_rotation is not None
    close_enough(desired_rotation, -net_rotation, f"{label}.desired rotation sign")
    close_enough(desired_rotation, positive_rotation - reverse_rotation, f"{label}.rotation decomposition")
    strict_number(episode["desired_active_rotation_fraction"], f"{label}.desired_active_rotation_fraction", minimum=0.0, maximum=1.0)
    if episode["contact_metric_source"] != "env_fast_forward_log_info":
        raise RuntimeError(f"{label}: contact metric source drift")
    if strict_bool(episode["environment_support_index_available"], f"{label}.environment_support_index_available") is not True:
        raise RuntimeError(f"{label}: environment support index unavailable")
    strict_number(episode["ground_contact_valid_fraction"], f"{label}.ground_contact_valid_fraction", minimum=0.0, maximum=1.0)
    strict_number(episode["mean_ground_contact_strength"], f"{label}.mean_ground_contact_strength", minimum=0.0, maximum=1.0)
    contact_span = strict_number(episode["contact_material_index_span"], f"{label}.contact_material_index_span", minimum=0.0, maximum=float(particles - 1))
    contact_fraction = strict_number(episode["contact_material_index_span_fraction"], f"{label}.contact_material_index_span_fraction", minimum=0.0, maximum=1.0)
    assert contact_span is not None and contact_fraction is not None
    close_enough(contact_fraction, contact_span / max(particles - 1, 1), f"{label}.contact span fraction")
    strict_int(episode["hard_support_unique_material_indices"], f"{label}.hard_support_unique_material_indices", minimum=0, maximum=particles)
    strict_number(episode["mean_active_contact_particles"], f"{label}.mean_active_contact_particles", minimum=0.0, maximum=float(particles))
    tail_detected = strict_bool(episode["tail_launch_detected"], f"{label}.tail_launch_detected")
    tail_steps = strict_int_list(episode["tail_launch_steps"], f"{label}.tail_launch_steps", minimum=0, maximum=steps, strictly_increasing=True)
    tail_count = strict_int(episode["tail_launch_count"], f"{label}.tail_launch_count", minimum=0, maximum=steps + 1)
    if tail_count != len(tail_steps) or tail_detected != (tail_count > 0):
        raise ValueError(f"{label}: tail-launch count/detection mismatch")
    candidate_steps = strict_int_list(episode["tail_launch_candidate_steps"], f"{label}.tail_launch_candidate_steps", minimum=0, maximum=steps, strictly_increasing=True)
    candidate_count = strict_int(episode["tail_launch_candidate_count"], f"{label}.tail_launch_candidate_count", minimum=0, maximum=steps + 1)
    if candidate_count != len(candidate_steps) or not set(tail_steps).issubset(candidate_steps):
        raise ValueError(f"{label}: tail-launch candidate mismatch")
    strict_number(episode["peak_tail_lift_body_fraction"], f"{label}.peak_tail_lift_body_fraction")
    strict_number(episode["peak_tail_forward_body_fraction"], f"{label}.peak_tail_forward_body_fraction")
    strict_number(episode["peak_tail_prefix_curvature_degrees"], f"{label}.peak_tail_prefix_curvature_degrees", minimum=0.0)
    verify_pulses(episode, label, steps, particles)

    pulse_count = int(episode["roll_pulse_count"])
    forward_threshold = float(CONFIG["metric_parameters"]["pulse_forward_body_fraction"]) * initial_length
    expected_classification = (
        "repeated_fast_forward_roll"
        if pulse_count >= 2
        else "single_fast_forward_roll"
        if pulse_count == 1
        else "forward_motion_without_roll_pulse"
        if displacement >= forward_threshold
        else "no_meaningful_forward_roll"
    )
    if classification != expected_classification:
        raise ValueError(f"{label}: classification recomputation mismatch")
    if strict_int(episode["seed"], f"{label}.seed", minimum=0) != evaluation_seed:
        raise RuntimeError(f"{label}: evaluation seed mismatch")
    if episode["condition_id"] != condition_id:
        raise RuntimeError(f"{label}: condition mismatch")
    if strict_int(episode["training_seed"], f"{label}.training_seed", minimum=0) != training_seed:
        raise RuntimeError(f"{label}: training seed mismatch")
    if strict_number(episode["same_observation_error"], f"{label}.same_observation_error") != 0.0:
        raise RuntimeError(f"{label}: nonzero same-observation diagnostic")
    stored_success = strict_bool(episode["success"], f"{label}.success")
    expected_success = recompute_episode_success(episode, label)
    if stored_success != expected_success:
        raise RuntimeError(f"{label}: five-criterion success recomputation mismatch")
    verify_joint_summary(episode["joint_summary"], f"{label}.joint_summary")
    return expected_success


def collect_and_verify_results(
    conditions: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[dict[str, str], dict[str, dict[str, int]], int]:
    result_root = ROOT / "results"
    if result_root.is_symlink() or not result_root.is_dir():
        raise RuntimeError("Result root is missing, not a directory, or a symlink")
    expected_seed_dirs = [f"seed{seed}" for seed in EXPECTED_TRAINING_SEEDS]
    root_entries = list(result_root.iterdir())
    if any(path.is_symlink() or not path.is_dir() for path in root_entries):
        raise RuntimeError("Result root contains an unexpected non-directory or symlink")
    if sorted(path.name for path in root_entries) != sorted(expected_seed_dirs):
        raise RuntimeError("Result seed-directory inventory is not exactly five frozen seeds")

    expected_eval_seeds = list(
        range(
            int(CONFIG["main_evaluation"]["base_seed"]),
            int(CONFIG["main_evaluation"]["base_seed"]) + EXPECTED_EPISODES_PER_CASE,
        )
    )
    identity_eval_seeds = set(
        range(
            int(CONFIG["identity_gate"]["base_seed"]),
            int(CONFIG["identity_gate"]["base_seed"])
            + int(CONFIG["identity_gate"]["episodes"]),
        )
    )
    if identity_eval_seeds.intersection(expected_eval_seeds):
        raise RuntimeError("Identity/calibration and main evaluation seeds overlap")

    formal_hashes = contract["protected_formal_files"]
    frozen_evaluator_hash = formal_hashes["frozen_evaluator"]["sha256"]
    result_file_hashes: dict[str, str] = {}
    success_counts: dict[str, dict[str, int]] = {}
    total_episodes = 0
    expected_ids = set(conditions)
    for training_seed in EXPECTED_TRAINING_SEEDS:
        result_dir = result_root / f"seed{training_seed}"
        entries = list(result_dir.iterdir())
        if any(path.is_symlink() or not path.is_file() or path.suffix != ".json" for path in entries):
            raise RuntimeError(f"Unexpected result entry for seed {training_seed}")
        if len(entries) != EXPECTED_CONDITION_COUNT:
            raise RuntimeError(f"Result file count is not 59 for seed {training_seed}")
        actual_ids = [path.stem for path in entries]
        if set(actual_ids) != expected_ids or len(set(actual_ids)) != len(actual_ids):
            raise RuntimeError(f"Condition inventory/uniqueness mismatch for seed {training_seed}")
        success_counts[str(training_seed)] = {}
        expected_case_checkpoints = {
            arm: formal_hashes[f"checkpoint_seed{training_seed}_{arm}"]["sha256"]
            for arm in EXPECTED_ARMS
        }
        for condition_id, condition in conditions.items():
            path = ensure_regular_file_inside(
                result_dir / f"{condition_id}.json", result_dir, "result file"
            )
            relative = path.relative_to(ROOT).as_posix()
            payload = require_exact_keys(load_json(path), RESULT_KEYS, str(path))
            result_file_hashes[relative] = sha256_file(path)
            expected_header = {
                "schema": "obs2_v2_1_k_condition_seed/v1",
                "study_id": CONFIG["study_id"],
                "training_seed": training_seed,
                "condition": condition.to_dict(),
                "evaluation_base_seed": int(CONFIG["main_evaluation"]["base_seed"]),
                "evaluation_episodes": EXPECTED_EPISODES_PER_CASE,
                "evaluation_steps": int(CONFIG["main_evaluation"]["steps"]),
                "checkpoint_sha256": expected_case_checkpoints,
                "frozen_evaluator_sha256": frozen_evaluator_hash,
            }
            for key, expected in expected_header.items():
                if not strict_json_equal(payload[key], expected):
                    raise RuntimeError(f"{path}: header mismatch for {key}")
            records = payload["episodes"]
            if not isinstance(records, list) or len(records) != EXPECTED_EPISODES_PER_CASE:
                raise RuntimeError(f"Episode count is not exactly 20: {path}")
            actual_eval_seeds = [
                item.get("seed") if isinstance(item, dict) else None for item in records
            ]
            if actual_eval_seeds != expected_eval_seeds or len(set(actual_eval_seeds)) != len(actual_eval_seeds):
                raise RuntimeError(f"Evaluation seed order/uniqueness mismatch: {path}")
            recomputed_success = 0
            for episode_index, (item, evaluation_seed) in enumerate(zip(records, expected_eval_seeds)):
                label = f"{training_seed}.{condition_id}.episode{episode_index + 1}"
                recomputed_success += int(
                    verify_episode(item, label, training_seed, condition_id, evaluation_seed)
                )
            stored_count = strict_int(
                payload["success_episodes"], f"{path}.success_episodes", minimum=0, maximum=20
            )
            if stored_count != recomputed_success:
                raise RuntimeError(f"Success count mismatch: {path}")
            success_counts[str(training_seed)][condition_id] = recomputed_success
            total_episodes += len(records)

    if len(result_file_hashes) != EXPECTED_CASE_COUNT:
        raise RuntimeError("Full results manifest does not contain exactly 295 files")
    if total_episodes != EXPECTED_TOTAL_EPISODES:
        raise RuntimeError(f"Total episode inventory is not exactly 5900: {total_episodes}")
    return dict(sorted(result_file_hashes.items())), success_counts, total_episodes


def verify_result_hash_map(
    value: Any,
    expected: dict[str, str],
    label: str,
    *,
    require_all: bool,
    required_paths: set[str] | None = None,
) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise TypeError(f"{label} must be a nonempty object")
    verified: dict[str, str] = {}
    for key, digest in value.items():
        if not isinstance(key, str) or key not in expected:
            raise RuntimeError(f"{label} contains an unknown result path: {key!r}")
        actual = strict_hash(digest, f"{label}.{key}")
        if actual != expected[key]:
            raise RuntimeError(f"{label} stale/mismatched result hash: {key}")
        path = ensure_regular_file_inside(ROOT / Path(key), ROOT / "results", f"{label} result")
        if sha256_file(path) != actual:
            raise RuntimeError(f"{label} result changed after validator inventory: {key}")
        verified[key] = actual
    if require_all and verified != expected:
        raise RuntimeError(f"{label} is not the full 295-file results manifest")
    if required_paths is not None and not required_paths.issubset(verified):
        raise RuntimeError(f"{label} lacks required endpoint result coverage")
    return dict(sorted(verified.items()))


def verify_trace_audit(
    contract: dict[str, Any],
    results: dict[str, str],
    checkpoints: dict[str, dict[str, str]],
) -> dict[str, Any]:
    path = ensure_regular_file(TRACE_AUDIT_PATH, "independent trace-audit pass")
    audit = load_json(path)
    expected_scalars = {
        "schema": "obs2_v2_1_k_independent_trace_audit/v1",
        "study_id": CONFIG["study_id"],
        "passed": True,
        "status": "complete",
        "training_seed_count": EXPECTED_TRAINING_SEED_COUNT,
        "condition_count": EXPECTED_CONDITION_COUNT,
        "parallel_process_workers": EXPECTED_TRAINING_SEED_COUNT,
        "audited_policy_condition_cases": EXPECTED_CASE_COUNT,
        "audited_episode_count": EXPECTED_CASE_COUNT,
        "audited_episodes": EXPECTED_CASE_COUNT,
        "evaluation_seed": int(CONFIG["main_evaluation"]["base_seed"]),
        "step_count_per_case": int(CONFIG["main_evaluation"]["steps"]),
        "actor_inputs_bit_exact": True,
        "actor_input_observations_unchanged": True,
        "original_tensordict_unmodified_by_policy_calls": True,
        "source_actions_unmodified_by_transform": True,
        "independent_transform_bit_exact": True,
        "independent_five_metrics_match_frozen_evaluator": True,
        "rerun_metrics_match_stored_main_results": True,
        "checkpoint_and_policy_hashes_unchanged": True,
        "condition_matrix_sha256": contract["conditions_sha256"],
    }
    for key, expected in expected_scalars.items():
        require_expected_value(audit.get(key), expected, f"trace.{key}")
    source_hash = strict_hash(audit.get("audit_source_sha256"), "trace audit source hash")
    if source_hash != sha256_file(ROOT / "independent_trace_audit.py"):
        raise RuntimeError("Independent trace audit source hash is stale")
    if audit.get("frozen_contract_sha256") != sha256_file(ACTIVE_CONTRACT_PATH):
        raise RuntimeError("Independent trace audit frozen-contract hash is stale")
    expected_frozen_sources = {
        name: evidence["sha256"] for name, evidence in contract["source_files"].items()
    }
    if audit.get("frozen_source_sha256") != expected_frozen_sources:
        raise RuntimeError("Independent trace audit source manifest is stale")
    if audit.get("frozen_evaluator_sha256") != contract["protected_formal_files"]["frozen_evaluator"]["sha256"]:
        raise RuntimeError("Independent trace audit evaluator hash is stale")
    transform_hash = audit.get("transform_source_sha256")
    if transform_hash is None:
        transform_hash = audit.get("independent_transform_source_sha256")
    transform_hash = strict_hash(transform_hash, "trace transform source hash")
    if transform_hash != sha256_file(ROOT / "smoke_test.py"):
        raise RuntimeError("Independent transform source hash is stale")
    verify_result_hash_map(
        audit.get("result_file_sha256"), results, "trace.result_file_sha256", require_all=True
    )
    verify_checkpoint_map(audit.get("checkpoint_sha256"), checkpoints, "trace.checkpoint_sha256")
    trace_hashes = audit.get("trace_file_sha256")
    if not isinstance(trace_hashes, dict) or len(trace_hashes) != EXPECTED_CASE_COUNT:
        raise RuntimeError("Trace-file inventory is not exactly 295")
    for trace_name, digest in trace_hashes.items():
        trace_path = ensure_regular_file_inside(ROOT / Path(trace_name), ROOT / "independent_trace_audit", "trace file")
        if sha256_file(trace_path) != strict_hash(digest, f"trace hash {trace_name}"):
            raise RuntimeError(f"Independent trace file hash drift: {trace_path}")
    return {
        "independent_trace_audit_sha256": sha256_file(path),
        "independent_trace_audit_source_sha256": source_hash,
    }


def verify_cross_environment_audit(
    results: dict[str, str], checkpoints: dict[str, dict[str, str]]
) -> dict[str, Any]:
    path = ensure_regular_file(CROSS_AUDIT_PATH, "cross-environment identity-audit pass")
    audit = load_json(path)
    if audit.get("schema") != "obs2_v2_1_k_cross_environment_identity_audit/v2":
        raise RuntimeError("Cross-environment audit schema mismatch")
    if audit.get("study_id") != CONFIG["study_id"]:
        raise RuntimeError("Cross-environment audit study mismatch")
    if strict_bool(audit.get("passed"), "cross audit passed") is not True:
        raise RuntimeError("Cross-environment identity audit did not pass")
    if audit.get("training_seeds") != EXPECTED_TRAINING_SEEDS:
        raise RuntimeError("Cross-environment training-seed order mismatch")
    failures = audit.get("failure_reasons")
    if not isinstance(failures, list) or failures:
        raise RuntimeError("Cross-environment audit contains failure reasons")
    binding = audit.get("stored_main_result_binding")
    if not isinstance(binding, dict):
        raise RuntimeError("Cross-environment stored-main binding is absent")
    if strict_bool(binding.get("passed"), "cross stored-main binding passed") is not True:
        raise RuntimeError("Cross-environment stored-main result binding failed")
    if strict_int(binding.get("checked_episodes"), "cross binding checked episodes", minimum=0) != 200:
        raise RuntimeError("Cross-environment stored-main binding did not check 200 episodes")
    if strict_int(binding.get("expected_episodes"), "cross binding expected episodes", minimum=0) != 200:
        raise RuntimeError("Cross-environment stored-main expected inventory mismatch")
    if binding.get("mismatches") != []:
        raise RuntimeError("Cross-environment stored-main binding contains mismatches")
    results_by_seed = audit.get("results_by_seed")
    if not isinstance(results_by_seed, dict) or list(results_by_seed) != [str(seed) for seed in EXPECTED_TRAINING_SEEDS]:
        raise RuntimeError("Cross-environment per-seed inventory/order mismatch")
    for seed, seed_payload in results_by_seed.items():
        if not isinstance(seed_payload, dict) or strict_bool(seed_payload.get("passed"), f"cross seed {seed} passed") is not True:
            raise RuntimeError(f"Cross-environment seed {seed} did not pass")
        arms = seed_payload.get("arms")
        if not isinstance(arms, dict) or set(arms) != set(EXPECTED_ARMS):
            raise RuntimeError(f"Cross-environment arm inventory mismatch for seed {seed}")
        for arm in EXPECTED_ARMS:
            if not isinstance(arms[arm], dict) or strict_bool(arms[arm].get("passed"), f"cross {seed}/{arm} passed") is not True:
                raise RuntimeError(f"Cross-environment arm {seed}/{arm} did not pass")
    maxima = audit.get("global_max_errors")
    if not isinstance(maxima, dict) or set(maxima) != {"position", "velocity", "observation", "action"}:
        raise RuntimeError("Cross-environment global-error schema mismatch")
    for key, value in maxima.items():
        converted = strict_number(value, f"cross.global_max_errors.{key}", allow_none=(key == "velocity"), minimum=0.0)
        if converted not in (None, 0.0):
            raise RuntimeError(f"Cross-environment nonzero {key} error")
    source_value = audit.get("audit_source_sha256", audit.get("source_sha256"))
    source_hash = strict_hash(source_value, "cross audit source hash")
    if source_hash != sha256_file(ROOT / "cross_environment_identity_audit.py"):
        raise RuntimeError("Cross-environment audit source hash is stale")
    required_endpoints = {
        f"results/seed{seed}/{condition}.json"
        for seed in EXPECTED_TRAINING_SEEDS
        for condition in ("C00", "C11")
    }
    verified_cross_results = verify_result_hash_map(
        audit.get("result_file_sha256"),
        results,
        "cross.result_file_sha256",
        require_all=False,
        required_paths=required_endpoints,
    )
    if binding.get("result_file_sha256") != verified_cross_results:
        raise RuntimeError("Cross-environment top-level/binding result manifests differ")
    checkpoint_evidence = audit.get("checkpoint_sha256")
    if not isinstance(checkpoint_evidence, dict) or set(checkpoint_evidence) != {"before", "after"}:
        raise RuntimeError("Cross-environment checkpoint before/after evidence mismatch")
    verify_checkpoint_map(checkpoint_evidence["before"], checkpoints, "cross.checkpoint_sha256.before")
    verify_checkpoint_map(checkpoint_evidence["after"], checkpoints, "cross.checkpoint_sha256.after")
    return {
        "cross_environment_identity_audit_sha256": sha256_file(path),
        "cross_environment_identity_audit_source_sha256": source_hash,
    }


def main() -> None:
    global CONFIG
    # A failed or interrupted fresh validation must never leave a reusable old pass.
    remove_file_if_present(PASS_PATH)
    CONFIG = load_json(CONFIG_PATH)
    conditions_list = build_conditions()
    conditions = {condition.id: condition for condition in conditions_list}
    if len(conditions) != len(conditions_list):
        raise RuntimeError("Condition IDs are not unique")
    contract = verify_frozen_bindings(conditions_list)
    checkpoints = expected_checkpoint_hashes(contract)
    binding_hashes = verify_smoke_and_identity(contract, checkpoints)
    completion = verify_completion(list(conditions))
    result_hashes, success_counts, total_episodes = collect_and_verify_results(
        conditions, contract
    )
    audit_hashes = verify_trace_audit(contract, result_hashes, checkpoints)
    audit_hashes.update(
        verify_cross_environment_audit(result_hashes, checkpoints)
    )
    results_manifest_sha256 = canonical_json_sha256(result_hashes)
    payload = {
        "schema": "obs2_v2_1_k_validation/v3",
        "study_id": CONFIG["study_id"],
        "passed": True,
        "validation_scope": "Strict 5x59x20 inventory, schema/type/finite/range/relational validation, five-criterion success recomputation, immutable hash bindings, and independent trace/cross-environment audit sealing.",
        "analysis_timing_status": "Post-result transparent validator implementation, not preregistered code.",
        "training_seed_count": EXPECTED_TRAINING_SEED_COUNT,
        "condition_count": EXPECTED_CONDITION_COUNT,
        "policy_condition_case_count": EXPECTED_CASE_COUNT,
        "episode_count": total_episodes,
        "identity_and_main_eval_seeds_disjoint": True,
        "all_contact_metrics_canonical": True,
        "stored_same_observation_diagnostics_zero": True,
        "same_observation_diagnostic_limitation": "The runner field is structurally zero by construction; independent evidence is bound through the trace audit.",
        "all_success_values_strictly_recomputed_from_five_criteria": True,
        "all_required_fields_types_finite_ranges_and_relations_valid": True,
        "all_joint_summaries_finite_and_physical_invariants_valid": True,
        "frozen_execution_sources_unchanged": True,
        "protected_formal_files_unchanged": True,
        "calibration_templates_match_identity_gate": True,
        "independent_trace_audit_passed_and_hash_bound": True,
        "cross_environment_identity_audit_passed_and_hash_bound": True,
        "success_counts": success_counts,
        "result_file_sha256": result_hashes,
        "results_manifest_sha256": results_manifest_sha256,
        "checkpoint_sha256": checkpoints,
        "completion_marker_sha256": sha256_file(ROOT / "MAIN_EXECUTION_COMPLETE.json"),
        "active_contract_sha256": sha256_file(ACTIVE_CONTRACT_PATH),
        "condition_matrix_sha256": contract["conditions_sha256"],
        "formal_source_manifest_sha256": contract["protected_formal_files"]["formal_source_manifest"]["sha256"],
        "frozen_evaluator_sha256": contract["protected_formal_files"]["frozen_evaluator"]["sha256"],
        **binding_hashes,
        **audit_hashes,
        "validator_source_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(PASS_PATH, payload)
    remove_file_if_present(FAIL_PATH)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        remove_file_if_present(PASS_PATH)
        atomic_json(
            FAIL_PATH,
            {
                "schema": "obs2_v2_1_k_validation_failure/v2",
                "study_id": CONFIG.get("study_id"),
                "passed": False,
                "error": f"{type(error).__name__}: {error}",
                "validator_source_sha256": sha256_file(Path(__file__).resolve()),
            },
        )
        raise
