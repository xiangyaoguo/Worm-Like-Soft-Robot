from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from condition_matrix import build_conditions


ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "study_config.json").read_text(encoding="utf-8"))
OUTPUT = ROOT / "analysis"
SEEDS = [int(value) for value in CONFIG["training_seeds"]]
CONDITIONS = {condition.id: condition for condition in build_conditions()}
RESULT_SCHEMA = "obs2_v2_1_k_condition_seed/v1"
IMPLEMENTATION_STATUS = "post_result_transparent_implementation"
EVALUATION_SEED_COUNT = 20
AUDIT_FILES = (
    "VALIDATION_PASS.json",
    "INDEPENDENT_TRACE_AUDIT_PASS.json",
    "CROSS_ENVIRONMENT_IDENTITY_AUDIT_PASS.json",
)
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_prerequisite_audits() -> dict[str, dict[str, Any]]:
    """Fail closed before reading any intervention result file."""
    audits: dict[str, dict[str, Any]] = {}
    for filename in AUDIT_FILES:
        path = ROOT / filename
        require(path.is_file(), f"Required prerequisite audit is missing: {path}")
        payload = load_json(path)
        require(
            payload.get("passed") is True,
            f"Prerequisite audit does not have literal passed=true: {path}",
        )
        audits[filename] = payload
    return audits


def _declared_result_hashes(value: Any, trail: tuple[str, ...] = ()) -> set[str]:
    """Collect only hashes whose JSON key path explicitly identifies results/files."""
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, child in value.items():
            found.update(_declared_result_hashes(child, trail + (str(key).lower(),)))
    elif isinstance(value, list):
        for child in value:
            found.update(_declared_result_hashes(child, trail))
    elif isinstance(value, str) and SHA256_PATTERN.fullmatch(value):
        leaf = trail[-1] if trail else ""
        parents = "/".join(trail[:-1])
        explicitly_named = (
            any(token in leaf for token in ("result", "file", "inventory"))
            and any(token in leaf for token in ("sha256", "hash", "digest"))
        )
        nested_content_hash = (
            leaf in {"sha256", "hash", "content_sha256", "content_hash"}
            and any(token in parents for token in ("result", "file", "inventory"))
        )
        path_keyed_hash = any(part.endswith(".json") for part in trail)
        if explicitly_named or nested_content_hash or path_keyed_hash:
            found.add(value.lower())
    return found


def _condition_id_from_header(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        candidate = value.get("id", value.get("condition_id"))
        return str(candidate) if candidate is not None else None
    return None


def _expected_condition_header(condition_id: str) -> dict[str, Any]:
    condition = CONDITIONS[condition_id]
    if hasattr(condition, "to_dict"):
        raw = condition.to_dict()
    elif is_dataclass(condition):
        raw = asdict(condition)
    else:
        raw = vars(condition)
    # Normalize tuples and other JSON-compatible containers exactly as the
    # frozen evaluator writes them.
    normalized = json.loads(json.dumps(raw, ensure_ascii=False))
    require(isinstance(normalized, dict), f"Invalid condition definition: {condition_id}")
    return normalized


def _joint_summary(episode: Mapping[str, Any], context: str) -> Mapping[str, Any]:
    value = episode.get("joint_summary")
    if isinstance(value, list):
        require(len(value) == 1, f"joint_summary must contain exactly one item: {context}")
        value = value[0]
    require(isinstance(value, Mapping), f"Invalid joint_summary object: {context}")
    return value


def _joint_vector(joint: Mapping[str, Any], key: str, context: str) -> list[float]:
    value = joint.get(key)
    require(isinstance(value, list) and len(value) == 8, f"Invalid {key}[8]: {context}")
    vector = [float(item) for item in value]
    require(all(math.isfinite(item) for item in vector), f"Non-finite {key}: {context}")
    return vector


def _joint_channel_matrix(
    joint: Mapping[str, Any], key: str, context: str
) -> list[list[float]]:
    value = joint.get(key)
    require(isinstance(value, list) and len(value) == 8, f"Invalid {key}[8][2]: {context}")
    matrix: list[list[float]] = []
    for item in value:
        require(isinstance(item, list) and len(item) == 2, f"Invalid {key}[8][2]: {context}")
        pair = [float(item[0]), float(item[1])]
        require(all(math.isfinite(number) for number in pair), f"Non-finite {key}: {context}")
        matrix.append(pair)
    return matrix


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def finite_mean(values: Iterable[Any]) -> float | None:
    converted = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(converted)) if converted else None


def percentile_ci(values: np.ndarray, seed: int, draws: int = 20000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(draws, values.size))
    means = np.mean(values[indices], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def exact_sign_flip(values: np.ndarray, alternative: str = "two-sided") -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    observed = float(np.sum(values))
    distribution = np.array([0.0], dtype=np.float64)
    for value in values:
        distribution = np.concatenate((distribution + value, distribution - value))
    tolerance = 1e-12
    if alternative == "greater":
        return float(np.mean(distribution >= observed - tolerance))
    if alternative == "less":
        return float(np.mean(distribution <= observed + tolerance))
    return float(np.mean(np.abs(distribution) >= abs(observed) - tolerance))


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    finite = [(name, value) for name, value in p_values.items() if math.isfinite(value)]
    ordered = sorted(finite, key=lambda item: item[1])
    count = len(ordered)
    adjusted: dict[str, float] = {name: math.nan for name in p_values}
    running = 0.0
    for rank, (name, value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * value)
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def episode_row(condition_id: str, training_seed: int, episode: dict[str, Any]) -> dict[str, Any]:
    condition = CONDITIONS[condition_id]
    context = f"seed={training_seed}, condition={condition_id}, evaluation_seed={episode.get('seed')}"
    joint = _joint_summary(episode, context)
    k_mean = _joint_channel_matrix(joint, "K_mean", context)
    k_abs_mean = _joint_channel_matrix(joint, "K_abs_mean", context)
    k_positive_fraction = _joint_channel_matrix(
        joint, "K_positive_fraction", context
    )
    tau1_rms = _joint_vector(joint, "tau1_boundary_rms", context)
    tau2_rms = _joint_vector(joint, "tau2_boundary_rms", context)
    saturation = _joint_vector(
        joint, "torque_boundary_saturation_fraction", context
    )
    power_abs_mean = _joint_vector(joint, "power_boundary_abs_mean", context)
    interval_value = episode.get("mean_roll_pulse_interval_steps")
    interval_defined = (
        interval_value is not None and math.isfinite(float(interval_value))
    )
    row: dict[str, Any] = {
        "implementation_status": IMPLEMENTATION_STATUS,
        "condition_id": condition_id,
        "module": condition.module,
        "family": condition.family,
        "training_seed": training_seed,
        "evaluation_seed": int(episode["seed"]),
        "success": int(bool(episode["success"])),
        "roll_pulses": int(episode["roll_pulse_count"]),
        "desired_net_rotation_degrees": float(episode["desired_net_rotation_degrees"]),
        "net_best_fit_rotation_degrees": float(episode["net_best_fit_rotation_degrees"]),
        "direction_fraction": float(episode["desired_active_rotation_fraction"]),
        "forward_body_lengths": float(episode["forward_body_lengths"]),
        "forward_displacement": float(episode["forward_displacement"]),
        "mean_pulse_interval": interval_value,
        "mean_pulse_interval_defined": int(interval_defined),
        "tail_launch": int(bool(episode.get("tail_launch_detected"))),
        "contact_metric_source": episode.get("contact_metric_source"),
        # These names deliberately reflect the frozen result schema.  It has no
        # separate K1/K2 saturation or signed mean-torque fields.
        "mean_abs_power_boundary": float(np.mean(power_abs_mean)),
        "mean_joint_K1_abs_mean": float(np.mean([item[0] for item in k_abs_mean])),
        "mean_joint_K2_abs_mean": float(np.mean([item[1] for item in k_abs_mean])),
        "mean_joint_K1_positive_fraction": float(
            np.mean([item[0] for item in k_positive_fraction])
        ),
        "mean_joint_K2_positive_fraction": float(
            np.mean([item[1] for item in k_positive_fraction])
        ),
        "mean_joint_tau1_boundary_rms": float(np.mean(tau1_rms)),
        "mean_joint_tau2_boundary_rms": float(np.mean(tau2_rms)),
        "mean_torque_boundary_saturation_fraction": float(np.mean(saturation)),
    }
    for index in range(8):
        label = f"J{index + 1:02d}"
        row[f"{label}_K1_mean"] = k_mean[index][0]
        row[f"{label}_K2_mean"] = k_mean[index][1]
        row[f"{label}_K1_abs_mean"] = k_abs_mean[index][0]
        row[f"{label}_K2_abs_mean"] = k_abs_mean[index][1]
        row[f"{label}_K1_positive_fraction"] = k_positive_fraction[index][0]
        row[f"{label}_K2_positive_fraction"] = k_positive_fraction[index][1]
        row[f"{label}_tau1_rms"] = tau1_rms[index]
        row[f"{label}_tau2_rms"] = tau2_rms[index]
        row[f"{label}_saturation"] = saturation[index]
        row[f"{label}_power_abs_mean"] = power_abs_mean[index]
    return row


def load_all_rows() -> tuple[
    list[dict[str, Any]],
    dict[tuple[int, str], dict[int, dict[str, Any]]],
    list[int],
    dict[str, Any],
]:
    require(len(SEEDS) == 5 and len(set(SEEDS)) == 5, "Exactly five unique training seeds are required")
    require(len(CONDITIONS) == 59, "Exactly 59 unique frozen conditions are required")
    require(
        int(CONFIG["main_evaluation"]["condition_count"]) == 59
        and int(CONFIG["main_evaluation"]["training_seed_count"]) == 5
        and int(CONFIG["main_evaluation"]["episodes"]) == EVALUATION_SEED_COUNT,
        "Configured 5x59x20 dimensions have drifted",
    )
    audits = validate_prerequisite_audits()
    completion = ROOT / "MAIN_EXECUTION_COMPLETE.json"
    completion_payload = load_json(completion) if completion.is_file() else {}
    if completion_payload.get("status") != "complete":
        raise RuntimeError("Main intervention execution is not complete")
    rows: list[dict[str, Any]] = []
    episodes: dict[tuple[int, str], dict[int, dict[str, Any]]] = {}
    file_hashes: dict[str, str] = {}
    canonical_evaluation_seeds: list[int] | None = None
    evaluator_hash: str | None = None
    for seed in SEEDS:
        for condition_id in CONDITIONS:
            path = ROOT / "results" / f"seed{seed}" / f"{condition_id}.json"
            require(path.is_file(), f"Missing result file: {path}")
            payload = load_json(path)
            relative = path.relative_to(ROOT).as_posix()
            file_hashes[relative] = sha256_file(path)
            require(payload.get("schema") == RESULT_SCHEMA, f"Schema mismatch: {path}")
            require(
                payload.get("study_id") == CONFIG["study_id"],
                f"Study identity mismatch: {path}",
            )
            require(
                type(payload.get("training_seed")) is int
                and payload["training_seed"] == seed,
                f"Training-seed header mismatch: {path}",
            )
            header_condition = _condition_id_from_header(payload.get("condition"))
            require(header_condition == condition_id, f"Condition header mismatch: {path}")
            require(
                payload["condition"] == _expected_condition_header(condition_id),
                f"Full frozen condition header mismatch: {path}",
            )
            current_evaluator_hash = payload.get("frozen_evaluator_sha256")
            require(
                isinstance(current_evaluator_hash, str)
                and SHA256_PATTERN.fullmatch(current_evaluator_hash) is not None,
                f"Invalid frozen_evaluator_sha256 header: {path}",
            )
            if evaluator_hash is None:
                evaluator_hash = current_evaluator_hash.lower()
            require(
                current_evaluator_hash.lower() == evaluator_hash,
                f"Frozen-evaluator hash changed across result files: {path}",
            )
            records = payload.get("episodes")
            require(isinstance(records, list), f"Invalid episodes array: {path}")
            require(
                len(records) == EVALUATION_SEED_COUNT,
                f"Expected exactly {EVALUATION_SEED_COUNT} episodes: {path}",
            )
            indexed: dict[int, dict[str, Any]] = {}
            for item in records:
                require(isinstance(item, dict), f"Invalid episode record: {path}")
                evaluation_seed = item.get("seed")
                require(
                    type(evaluation_seed) is int,
                    f"Non-integer evaluation seed: {path}",
                )
                require(
                    evaluation_seed not in indexed,
                    f"Duplicate evaluation seed {evaluation_seed}: {path}",
                )
                require(
                    item.get("condition_id") == condition_id,
                    f"Episode condition mismatch at evaluation seed {evaluation_seed}: {path}",
                )
                if "training_seed" in item:
                    require(
                        item["training_seed"] == seed,
                        f"Episode training-seed mismatch at evaluation seed {evaluation_seed}: {path}",
                    )
                indexed[evaluation_seed] = item
            current_seeds = sorted(indexed)
            if canonical_evaluation_seeds is None:
                canonical_evaluation_seeds = current_seeds
            require(
                current_seeds == canonical_evaluation_seeds,
                f"Fixed evaluation-seed inventory mismatch: {path}",
            )
            episodes[(seed, condition_id)] = indexed
            rows.extend(
                episode_row(condition_id, seed, indexed[evaluation_seed])
                for evaluation_seed in canonical_evaluation_seeds
            )
    require(
        canonical_evaluation_seeds is not None
        and len(canonical_evaluation_seeds) == EVALUATION_SEED_COUNT,
        "No valid fixed evaluation-seed inventory was loaded",
    )
    configured_base_seed = int(CONFIG["main_evaluation"]["base_seed"])
    configured_episode_count = int(CONFIG["main_evaluation"]["episodes"])
    configured_evaluation_seeds = list(
        range(configured_base_seed, configured_base_seed + configured_episode_count)
    )
    require(
        canonical_evaluation_seeds == configured_evaluation_seeds,
        "Result evaluation seeds do not match the frozen contiguous base_seed/episodes set",
    )
    expected = len(SEEDS) * len(CONDITIONS) * EVALUATION_SEED_COUNT
    configured_expected = int(CONFIG["main_evaluation"]["total_episodes"])
    require(
        configured_expected == expected,
        f"Configured total episodes {configured_expected} != inventory-derived {expected}",
    )
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} episode rows, got {len(rows)}")
    inventory_text = "".join(
        f"{name}\t{file_hashes[name]}\n" for name in sorted(file_hashes)
    )
    inventory_sha256 = hashlib.sha256(inventory_text.encode("utf-8")).hexdigest()
    validator_manifest_sha256 = hashlib.sha256(
        json.dumps(
            file_hashes,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    validation = audits["VALIDATION_PASS.json"]
    validation_files = validation.get("result_file_sha256")
    require(
        isinstance(validation_files, Mapping)
        and dict(validation_files) == file_hashes,
        "VALIDATION_PASS result_file_sha256 does not exactly bind all 295 loaded results",
    )
    require(
        validation.get("results_manifest_sha256") == validator_manifest_sha256,
        "VALIDATION_PASS results_manifest_sha256 does not bind the loaded result map",
    )

    trace = audits["INDEPENDENT_TRACE_AUDIT_PASS.json"]
    trace_files = trace.get("result_file_sha256")
    require(
        isinstance(trace_files, Mapping) and dict(trace_files) == file_hashes,
        "INDEPENDENT_TRACE_AUDIT_PASS does not exactly bind all 295 loaded results",
    )

    cross = audits["CROSS_ENVIRONMENT_IDENTITY_AUDIT_PASS.json"]
    cross_files = cross.get("result_file_sha256")
    expected_cross_paths = {
        f"results/seed{seed}/{condition}.json"
        for seed in SEEDS
        for condition in ("C00", "C11")
    }
    require(
        isinstance(cross_files, Mapping)
        and set(cross_files) == expected_cross_paths
        and all(file_hashes[path] == digest for path, digest in cross_files.items()),
        "CROSS_ENVIRONMENT_IDENTITY_AUDIT_PASS does not exactly bind the 10 C00/C11 endpoint files",
    )
    declared_hashes: set[str] = set()
    for payload in (*audits.values(), completion_payload):
        declared_hashes.update(_declared_result_hashes(payload))
    matched_hashes = declared_hashes.intersection(
        set(file_hashes.values()) | {inventory_sha256}
    )
    if declared_hashes:
        require(
            bool(matched_hashes),
            "Prerequisite audits declare result/file hashes, but none bind to the loaded result inventory",
        )
    integrity = {
        "implementation_status": IMPLEMENTATION_STATUS,
        "result_schema": RESULT_SCHEMA,
        "result_file_count": len(file_hashes),
        "result_file_sha256": file_hashes,
        "result_inventory_sha256": inventory_sha256,
        "validator_results_manifest_sha256": validator_manifest_sha256,
        "result_inventory_hash_algorithm": (
            "sha256 of UTF-8 lines '<root-relative-posix-path>\\t<file-sha256>\\n' "
            "in lexical path order"
        ),
        "frozen_evaluator_sha256": evaluator_hash,
        "audit_sha256": {
            filename: sha256_file(ROOT / filename) for filename in AUDIT_FILES
        },
        "main_execution_complete_sha256": sha256_file(completion),
        "audit_declared_result_hash_count": len(declared_hashes),
        "audit_matched_result_hash_count": len(matched_hashes),
        "audit_result_hash_binding": "exact_validation_trace_full_and_cross_endpoint_maps",
    }
    return rows, episodes, canonical_evaluation_seeds, integrity


def condition_seed_rows(
    episodes: dict[tuple[int, str], dict[int, dict[str, Any]]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for seed in SEEDS:
        for condition_id, condition in CONDITIONS.items():
            records = list(episodes[(seed, condition_id)].values())
            success_count = int(sum(bool(item["success"]) for item in records))
            interval_count = sum(
                item.get("mean_roll_pulse_interval_steps") is not None
                and math.isfinite(float(item["mean_roll_pulse_interval_steps"]))
                for item in records
            )
            result.append(
                {
                    "implementation_status": IMPLEMENTATION_STATUS,
                    "condition_id": condition_id,
                    "module": condition.module,
                    "family": condition.family,
                    "training_seed": seed,
                    "success_episodes": success_count,
                    "success_rate": success_count / len(records),
                    "mean_forward_body_lengths": finite_mean(
                        item["forward_body_lengths"] for item in records
                    ),
                    "mean_desired_rotation_degrees": finite_mean(
                        item["desired_net_rotation_degrees"] for item in records
                    ),
                    "mean_net_rotation_degrees": finite_mean(
                        item["net_best_fit_rotation_degrees"] for item in records
                    ),
                    "mean_direction_fraction": finite_mean(
                        item["desired_active_rotation_fraction"] for item in records
                    ),
                    "mean_roll_pulses": finite_mean(
                        item["roll_pulse_count"] for item in records
                    ),
                    "mean_pulse_interval": finite_mean(
                        item.get("mean_roll_pulse_interval_steps") for item in records
                    ),
                    "mean_pulse_interval_defined_episodes": interval_count,
                    "mean_pulse_interval_coverage": interval_count / len(records),
                    "tail_launch_rate": finite_mean(
                        bool(item.get("tail_launch_detected")) for item in records
                    ),
                }
            )
    return result


def condition_summary_rows(seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[row["condition_id"]].append(row)
    result: list[dict[str, Any]] = []
    threshold = int(CONFIG["condition_seed_success_min_episodes"])
    for condition_id, records in grouped.items():
        condition = CONDITIONS[condition_id]
        rates = np.asarray([row["success_rate"] for row in records], dtype=np.float64)
        successful_seeds = [
            int(row["training_seed"])
            for row in records
            if int(row["success_episodes"]) >= threshold
        ]
        ci = percentile_ci(rates, seed=910000 + sum(ord(c) for c in condition_id))
        result.append(
            {
                "implementation_status": IMPLEMENTATION_STATUS,
                "condition_id": condition_id,
                "module": condition.module,
                "family": condition.family,
                "description": condition.description,
                "mean_success_rate": float(np.mean(rates)),
                "min_seed_success_rate": float(np.min(rates)),
                "max_seed_success_rate": float(np.max(rates)),
                "training_seed_bootstrap_ci_low": ci[0],
                "training_seed_bootstrap_ci_high": ci[1],
                "successful_training_seeds": ",".join(map(str, successful_seeds)),
                "successful_training_seed_count": len(successful_seeds),
                "reproducible_3_of_5": int(
                    len(successful_seeds) >= int(CONFIG["reproducible_min_training_seeds"])
                ),
                "robust_4_of_5": int(
                    len(successful_seeds) >= int(CONFIG["robust_min_training_seeds"])
                ),
                "mean_forward_body_lengths": finite_mean(
                    row["mean_forward_body_lengths"] for row in records
                ),
                "mean_desired_rotation_degrees": finite_mean(
                    row["mean_desired_rotation_degrees"] for row in records
                ),
                "mean_net_rotation_degrees": finite_mean(
                    row["mean_net_rotation_degrees"] for row in records
                ),
                "mean_direction_fraction": finite_mean(
                    row["mean_direction_fraction"] for row in records
                ),
                "mean_roll_pulses": finite_mean(row["mean_roll_pulses"] for row in records),
                "mean_pulse_interval": finite_mean(
                    row["mean_pulse_interval"] for row in records
                ),
                "mean_pulse_interval_defined_episodes": int(
                    sum(row["mean_pulse_interval_defined_episodes"] for row in records)
                ),
                "mean_pulse_interval_possible_episodes": int(
                    sum(EVALUATION_SEED_COUNT for _ in records)
                ),
                "mean_pulse_interval_coverage": float(
                    sum(row["mean_pulse_interval_defined_episodes"] for row in records)
                    / sum(EVALUATION_SEED_COUNT for _ in records)
                ),
            }
        )
    return sorted(result, key=lambda row: (row["module"], row["condition_id"]))


def contrast_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {"id": "A_K1_AT_R0_K2", "target": "C10", "base": "C00", "family": "A_5", "orientation": 1},
        {"id": "A_K1_AT_RROLL_K2", "target": "C11", "base": "C01", "family": "A_5", "orientation": 1},
        {"id": "A_K2_AT_R0_K1", "target": "C01", "base": "C00", "family": "A_5", "orientation": 1},
        {"id": "A_K2_AT_RROLL_K1", "target": "C11", "base": "C10", "family": "A_5", "orientation": 1},
    ]
    for joint in range(1, 9):
        label = f"J{joint:02d}"
        specs.extend(
            (
                {
                    "id": f"B_K1_SUFF_{label}",
                    "target": f"K1_SUFF_{label}",
                    "base": "C00",
                    "family": "B_32",
                    "orientation": 1,
                },
                {
                    "id": f"B_K1_NEC_{label}",
                    "target": f"K1_NEC_{label}",
                    "base": "C10",
                    "family": "B_32",
                    "orientation": -1,
                },
                {
                    "id": f"B_K2_SUFF_{label}",
                    "target": f"K2_SUFF_{label}",
                    "base": "C00",
                    "family": "B_32",
                    "orientation": 1,
                },
                {
                    "id": f"B_K2_NEC_{label}",
                    "target": f"K2_NEC_{label}",
                    "base": "C11",
                    "family": "B_32",
                    "orientation": -1,
                },
            )
        )
    for condition in CONDITIONS.values():
        if condition.module == "C":
            base = "C00" if condition.family == "k1_subset_sufficiency" else "C11"
            family = "C_13"
            specs.append(
                {
                    "id": f"{condition.id}_VS_{base}",
                    "target": condition.id,
                    "base": base,
                    "family": family,
                    "orientation": 1,
                }
            )
        elif condition.module == "D":
            specs.append(
                {
                    "id": f"{condition.id}_VS_C11",
                    "target": condition.id,
                    "base": "C11",
                    "family": "D_10",
                    "orientation": 1,
                }
            )
    counts: dict[str, int] = defaultdict(int)
    for spec in specs:
        counts[spec["family"]] += 1
    # A's fifth member (factorial interaction) is constructed below.
    expected = {"A_5": 4, "B_32": 32, "C_13": 13, "D_10": 10}
    require(counts == expected, f"Holm-family inventory mismatch: {dict(counts)} != {expected}")
    return specs


def contrast_rows(
    episodes: dict[tuple[int, str], dict[int, dict[str, Any]]],
    evaluation_seeds: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed_p: dict[str, dict[str, float]] = defaultdict(dict)
    eval_p: dict[str, dict[str, float]] = defaultdict(dict)
    for spec in contrast_specs():
        target = spec["target"]
        base = spec["base"]
        orientation = float(spec["orientation"])
        seed_effects: list[float] = []
        eval_effects: list[float] = []
        for seed in SEEDS:
            left = episodes[(seed, target)]
            right = episodes[(seed, base)]
            seed_effects.append(
                orientation
                * float(
                    np.mean([bool(item["success"]) for item in left.values()])
                    - np.mean([bool(item["success"]) for item in right.values()])
                )
            )
        for evaluation_seed in evaluation_seeds:
            eval_effects.append(
                orientation
                * float(
                    np.mean(
                        [
                            bool(episodes[(seed, target)][evaluation_seed]["success"])
                            - bool(episodes[(seed, base)][evaluation_seed]["success"])
                            for seed in SEEDS
                        ]
                    )
                )
            )
        seed_array = np.asarray(seed_effects)
        eval_array = np.asarray(eval_effects)
        seed_ci = percentile_ci(seed_array, 100000 + sum(ord(c) for c in spec["id"]))
        eval_ci = percentile_ci(eval_array, 200000 + sum(ord(c) for c in spec["id"]))
        seed_p_two = exact_sign_flip(seed_array, "two-sided")
        seed_p_one = exact_sign_flip(seed_array, "greater")
        eval_p_two = exact_sign_flip(eval_array, "two-sided")
        row = {
            "implementation_status": IMPLEMENTATION_STATUS,
            "contrast_id": spec["id"],
            "family": spec["family"],
            "target": target,
            "base": base,
            "orientation": "target_minus_base" if orientation > 0 else "base_minus_target",
            "mean_success_rate_effect": float(np.mean(seed_array)),
            "median_training_seed_effect": float(np.median(seed_array)),
            "training_seed_effects": ";".join(f"{value:.6f}" for value in seed_array),
            "training_seed_direction_consistency": float(np.mean(seed_array > 0.0)),
            "training_seed_bootstrap_ci_low": seed_ci[0],
            "training_seed_bootstrap_ci_high": seed_ci[1],
            "training_seed_exact_two_sided_p": seed_p_two,
            "posthoc_unadjusted_training_seed_exact_one_sided_p": seed_p_one,
            "one_sided_p_interpretation": "posthoc descriptive only; not confirmatory and not multiplicity adjusted",
            "fixed_policy_initial_state_effect": float(np.mean(eval_array)),
            "initial_state_bootstrap_ci_low": eval_ci[0],
            "initial_state_bootstrap_ci_high": eval_ci[1],
            "fixed_policy_initial_state_exact_two_sided_p": eval_p_two,
            "large_effect_RD_ge_0P25": int(float(np.mean(seed_array)) >= 0.25),
        }
        rows.append(row)
        seed_p[spec["family"]][spec["id"]] = seed_p_two
        eval_p[spec["family"]][spec["id"]] = eval_p_two
    interaction_seed: list[float] = []
    interaction_eval: list[float] = []
    for seed in SEEDS:
        values = {
            condition_id: np.mean(
                [bool(item["success"]) for item in episodes[(seed, condition_id)].values()]
            )
            for condition_id in ("C00", "C10", "C01", "C11")
        }
        interaction_seed.append(
            float(values["C11"] - values["C10"] - values["C01"] + values["C00"])
        )
    for evaluation_seed in evaluation_seeds:
        values = {
            condition_id: np.mean(
                [
                    bool(episodes[(seed, condition_id)][evaluation_seed]["success"])
                    for seed in SEEDS
                ]
            )
            for condition_id in ("C00", "C10", "C01", "C11")
        }
        interaction_eval.append(
            float(values["C11"] - values["C10"] - values["C01"] + values["C00"])
        )
    interaction_seed_array = np.asarray(interaction_seed)
    interaction_eval_array = np.asarray(interaction_eval)
    interaction_seed_ci = percentile_ci(interaction_seed_array, 454501)
    interaction_eval_ci = percentile_ci(interaction_eval_array, 454502)
    interaction_row = {
        "implementation_status": IMPLEMENTATION_STATUS,
        "contrast_id": "A_K1_K2_INTERACTION",
        "family": "A_5",
        "target": "C11-C10-C01+C00",
        "base": "factorial_additive_null",
        "orientation": "positive_nonadditive_interaction",
        "mean_success_rate_effect": float(np.mean(interaction_seed_array)),
        "median_training_seed_effect": float(np.median(interaction_seed_array)),
        "training_seed_effects": ";".join(
            f"{value:.6f}" for value in interaction_seed_array
        ),
        "training_seed_direction_consistency": float(
            np.mean(interaction_seed_array > 0.0)
        ),
        "training_seed_bootstrap_ci_low": interaction_seed_ci[0],
        "training_seed_bootstrap_ci_high": interaction_seed_ci[1],
        "training_seed_exact_two_sided_p": exact_sign_flip(
            interaction_seed_array, "two-sided"
        ),
        "posthoc_unadjusted_training_seed_exact_one_sided_p": exact_sign_flip(
            interaction_seed_array, "greater"
        ),
        "one_sided_p_interpretation": "posthoc descriptive only; not confirmatory and not multiplicity adjusted",
        "fixed_policy_initial_state_effect": float(np.mean(interaction_eval_array)),
        "initial_state_bootstrap_ci_low": interaction_eval_ci[0],
        "initial_state_bootstrap_ci_high": interaction_eval_ci[1],
        "fixed_policy_initial_state_exact_two_sided_p": exact_sign_flip(
            interaction_eval_array, "two-sided"
        ),
        "large_effect_RD_ge_0P25": int(float(np.mean(interaction_seed_array)) >= 0.25),
    }
    rows.append(interaction_row)
    seed_p["A_5"]["A_K1_K2_INTERACTION"] = interaction_row[
        "training_seed_exact_two_sided_p"
    ]
    eval_p["A_5"]["A_K1_K2_INTERACTION"] = interaction_row[
        "fixed_policy_initial_state_exact_two_sided_p"
    ]

    by_id = {row["contrast_id"]: row for row in rows}
    require(
        {family: len(values) for family, values in seed_p.items()}
        == {"A_5": 5, "B_32": 32, "C_13": 13, "D_10": 10},
        "Final Holm-family sizes are not exactly A=5, B=32, C=13, D=10",
    )
    for family, values in seed_p.items():
        adjusted = holm_adjust(values)
        adjusted_eval = holm_adjust(eval_p[family])
        for contrast_id in values:
            by_id[contrast_id]["holm_family_size"] = len(values)
            by_id[contrast_id]["training_seed_holm_two_sided_p"] = adjusted[contrast_id]
            by_id[contrast_id]["fixed_policy_initial_state_holm_two_sided_p"] = adjusted_eval[
                contrast_id
            ]
    return rows


def subset_synergy_rows(
    episodes: dict[tuple[int, str], dict[int, dict[str, Any]]]
) -> list[dict[str, Any]]:
    definitions = {
        "J02_J03": ("K1_SUFF_J02_J03", ["K1_SUFF_J02", "K1_SUFF_J03"]),
        "J02_J05": ("K1_SUFF_J02_J05", ["K1_SUFF_J02", "K1_SUFF_J05"]),
        "J03_J05": ("K1_SUFF_J03_J05", ["K1_SUFF_J03", "K1_SUFF_J05"]),
    }
    rows: list[dict[str, Any]] = []
    for name, (pair, singles) in definitions.items():
        values: list[float] = []
        for seed in SEEDS:
            y00 = np.mean(
                [bool(item["success"]) for item in episodes[(seed, "C00")].values()]
            )
            ypair = np.mean(
                [bool(item["success"]) for item in episodes[(seed, pair)].values()]
            )
            ysingle = [
                np.mean(
                    [
                        bool(item["success"])
                        for item in episodes[(seed, single)].values()
                    ]
                )
                for single in singles
            ]
            values.append(float(ypair - ysingle[0] - ysingle[1] + y00))
        array = np.asarray(values)
        ci = percentile_ci(array, 300000 + sum(ord(c) for c in name))
        rows.append(
            {
                "implementation_status": IMPLEMENTATION_STATUS,
                "holm_family": "THREE_JOINT_SYNERGY_3",
                "combination": name,
                "interaction_definition": "Y_pair-Y_single_a-Y_single_b+Y_C00",
                "mean_synergy_RD": float(np.mean(array)),
                "training_seed_effects": ";".join(f"{value:.6f}" for value in array),
                "bootstrap_ci_low": ci[0],
                "bootstrap_ci_high": ci[1],
                "exact_two_sided_p": exact_sign_flip(array, "two-sided"),
            }
        )
    adjusted = holm_adjust(
        {row["combination"]: row["exact_two_sided_p"] for row in rows}
    )
    require(len(adjusted) == 3, "Three-joint synergy Holm family must contain 3 tests")
    for row in rows:
        row["holm_family_size"] = 3
        row["holm_two_sided_p"] = adjusted[row["combination"]]
    return rows


def continuous_contrast_rows(
    episode_rows: list[dict[str, Any]],
    evaluation_seeds: list[int],
) -> list[dict[str, Any]]:
    """Paired continuous effects; the evaluation seed, never row order, is the key."""
    metrics = (
        ("forward_body_lengths", "forward_body_lengths", False, "episode forward displacement divided by frozen body length"),
        ("desired_rotation_degrees", "desired_net_rotation_degrees", False, "frozen desired-direction net rotation in degrees"),
        ("roll_pulses", "roll_pulses", False, "frozen roll pulse count"),
        ("direction_fraction", "direction_fraction", False, "fraction of active rotation in desired direction"),
        ("pulse_interval_defined", "mean_pulse_interval_defined", False, "binary indicator that a mean roll-pulse interval is defined; all 100 pairs retained"),
        ("mean_pulse_interval_steps", "mean_pulse_interval", True, "complete-case mean roll-pulse interval conditional on both conditions defining an interval; selection-sensitive"),
        ("mean_abs_power_boundary", "mean_abs_power_boundary", False, "mean across 8 joints of power_boundary_abs_mean"),
        ("K1_abs_magnitude", "mean_joint_K1_abs_mean", False, "mean across 8 joints of frozen K_abs_mean channel K1"),
        ("K2_abs_magnitude", "mean_joint_K2_abs_mean", False, "mean across 8 joints of frozen K_abs_mean channel K2"),
        ("K1_positive_fraction", "mean_joint_K1_positive_fraction", False, "mean across 8 joints of frozen K_positive_fraction channel K1"),
        ("K2_positive_fraction", "mean_joint_K2_positive_fraction", False, "mean across 8 joints of frozen K_positive_fraction channel K2"),
        ("tau1_boundary_rms", "mean_joint_tau1_boundary_rms", False, "mean across 8 joints of tau1_boundary_rms; not signed mean torque"),
        ("tau2_boundary_rms", "mean_joint_tau2_boundary_rms", False, "mean across 8 joints of tau2_boundary_rms; not signed mean torque"),
        ("torque_boundary_saturation", "mean_torque_boundary_saturation_fraction", False, "mean across 8 joints of the sole frozen torque_boundary_saturation_fraction; no channel-specific saturation exists"),
    )
    indexed = {
        (int(row["training_seed"]), str(row["condition_id"]), int(row["evaluation_seed"])): row
        for row in episode_rows
    }
    result: list[dict[str, Any]] = []
    for spec in contrast_specs():
        orientation = float(spec["orientation"])
        for metric_id, field, defined_only, definition in metrics:
            pair_effects: dict[tuple[int, int], float] = {}
            for training_seed in SEEDS:
                for evaluation_seed in evaluation_seeds:
                    target = indexed[(training_seed, spec["target"], evaluation_seed)].get(field)
                    base = indexed[(training_seed, spec["base"], evaluation_seed)].get(field)
                    target_defined = target is not None and math.isfinite(float(target))
                    base_defined = base is not None and math.isfinite(float(base))
                    if not (target_defined and base_defined):
                        require(
                            defined_only,
                            f"Undefined required continuous metric {metric_id}: {spec['id']}, "
                            f"training_seed={training_seed}, evaluation_seed={evaluation_seed}",
                        )
                        continue
                    pair_effects[(training_seed, evaluation_seed)] = orientation * (
                        float(target) - float(base)
                    )
            expected_pairs = len(SEEDS) * len(evaluation_seeds)
            if not defined_only:
                require(
                    len(pair_effects) == expected_pairs,
                    f"Incomplete paired inventory for {spec['id']} / {metric_id}",
                )
            if not pair_effects:
                require(
                    defined_only,
                    f"No defined pairs for required metric {spec['id']} / {metric_id}",
                )
                zero_training_coverage = {str(seed): 0.0 for seed in SEEDS}
                zero_evaluation_coverage = {
                    str(seed): 0.0 for seed in evaluation_seeds
                }
                result.append(
                    {
                        "implementation_status": IMPLEMENTATION_STATUS,
                        "contrast_id": spec["id"],
                        "family": (
                            "A_SIMPLE_4"
                            if spec["family"] == "A_5"
                            else spec["family"]
                        ),
                        "target": spec["target"],
                        "base": spec["base"],
                        "orientation": (
                            "target_minus_base"
                            if orientation > 0
                            else "base_minus_target"
                        ),
                        "metric": metric_id,
                        "operational_definition": definition,
                        "complete_pair_weighted_mean_effect": None,
                        "training_seed_equal_weight_mean_effect": None,
                        "evaluation_seed_equal_weight_mean_effect": None,
                        "defined_pair_count": 0,
                        "possible_pair_count": expected_pairs,
                        "defined_pair_coverage": 0.0,
                        "defined_pair_coverage_by_training_seed": json.dumps(
                            zero_training_coverage,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "defined_pair_coverage_by_evaluation_seed": json.dumps(
                            zero_evaluation_coverage,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                        "training_seed_effect_count": 0,
                        "training_seed_inference_available": False,
                        "training_seed_bootstrap_ci_low": None,
                        "training_seed_bootstrap_ci_high": None,
                        "evaluation_seed_effect_count": 0,
                        "fixed_evaluation_seed_inference_available": False,
                        "fixed_evaluation_seed_bootstrap_ci_low": None,
                        "fixed_evaluation_seed_bootstrap_ci_high": None,
                        "ci_scope": "posthoc_pointwise_descriptive_unadjusted",
                        "ci_interpretation": (
                            "no complete cases; interval-defined indicator carries the full-pair result"
                        ),
                    }
                )
                continue
            training_effects = np.asarray(
                [
                    np.mean(
                        [
                            value
                            for (pair_seed, _), value in pair_effects.items()
                            if pair_seed == training_seed
                        ]
                    )
                    for training_seed in SEEDS
                    if any(pair_seed == training_seed for pair_seed, _ in pair_effects)
                ],
                dtype=np.float64,
            )
            evaluation_effects = np.asarray(
                [
                    np.mean(
                        [
                            value
                            for (_, pair_evaluation_seed), value in pair_effects.items()
                            if pair_evaluation_seed == evaluation_seed
                        ]
                    )
                    for evaluation_seed in evaluation_seeds
                    if any(
                        pair_evaluation_seed == evaluation_seed
                        for _, pair_evaluation_seed in pair_effects
                    )
                ],
                dtype=np.float64,
            )
            bootstrap_seed = sum(ord(char) for char in f"{spec['id']}:{metric_id}")
            training_inference_available = training_effects.size == len(SEEDS)
            evaluation_inference_available = evaluation_effects.size == len(evaluation_seeds)
            training_ci = (
                percentile_ci(training_effects, 510000 + bootstrap_seed)
                if training_inference_available
                else (math.nan, math.nan)
            )
            evaluation_ci = (
                percentile_ci(evaluation_effects, 610000 + bootstrap_seed)
                if evaluation_inference_available
                else (math.nan, math.nan)
            )
            coverage_by_training_seed = {
                str(training_seed): sum(
                    key[0] == training_seed for key in pair_effects
                )
                / len(evaluation_seeds)
                for training_seed in SEEDS
            }
            coverage_by_evaluation_seed = {
                str(evaluation_seed): sum(
                    key[1] == evaluation_seed for key in pair_effects
                )
                / len(SEEDS)
                for evaluation_seed in evaluation_seeds
            }
            result.append(
                {
                    "implementation_status": IMPLEMENTATION_STATUS,
                    "contrast_id": spec["id"],
                    "family": (
                        "A_SIMPLE_4" if spec["family"] == "A_5" else spec["family"]
                    ),
                    "target": spec["target"],
                    "base": spec["base"],
                    "orientation": (
                        "target_minus_base" if orientation > 0 else "base_minus_target"
                    ),
                    "metric": metric_id,
                    "operational_definition": definition,
                    "complete_pair_weighted_mean_effect": float(
                        np.mean(list(pair_effects.values()))
                    ),
                    "training_seed_equal_weight_mean_effect": float(
                        np.mean(training_effects)
                    ),
                    "evaluation_seed_equal_weight_mean_effect": float(
                        np.mean(evaluation_effects)
                    ),
                    "defined_pair_count": len(pair_effects),
                    "possible_pair_count": expected_pairs,
                    "defined_pair_coverage": len(pair_effects) / expected_pairs,
                    "defined_pair_coverage_by_training_seed": json.dumps(
                        coverage_by_training_seed, sort_keys=True, separators=(",", ":")
                    ),
                    "defined_pair_coverage_by_evaluation_seed": json.dumps(
                        coverage_by_evaluation_seed, sort_keys=True, separators=(",", ":")
                    ),
                    "training_seed_effect_count": int(training_effects.size),
                    "training_seed_inference_available": training_inference_available,
                    "training_seed_bootstrap_ci_low": training_ci[0],
                    "training_seed_bootstrap_ci_high": training_ci[1],
                    "evaluation_seed_effect_count": int(evaluation_effects.size),
                    "fixed_evaluation_seed_inference_available": evaluation_inference_available,
                    "fixed_evaluation_seed_bootstrap_ci_low": evaluation_ci[0],
                    "fixed_evaluation_seed_bootstrap_ci_high": evaluation_ci[1],
                    "ci_scope": "posthoc_pointwise_descriptive_unadjusted",
                    "ci_interpretation": (
                        "descriptive only; exclusion of zero is not a multiplicity-adjusted significance test; "
                        "the n=5 training-seed percentile bootstrap has limited coverage reliability"
                    ),
                }
            )
    return result


def condition_joint_rows(episode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for condition_id, condition in CONDITIONS.items():
        selected = [row for row in episode_rows if row["condition_id"] == condition_id]
        require(bool(selected), f"No episode rows for condition {condition_id}")
        for index in range(8):
            label = f"J{index + 1:02d}"
            result.append(
                {
                    "implementation_status": IMPLEMENTATION_STATUS,
                    "condition_id": condition_id,
                    "module": condition.module,
                    "family": condition.family,
                    "joint": label,
                    "K1_mean": finite_mean(row[f"{label}_K1_mean"] for row in selected),
                    "K2_mean": finite_mean(row[f"{label}_K2_mean"] for row in selected),
                    "K1_abs_mean": finite_mean(row[f"{label}_K1_abs_mean"] for row in selected),
                    "K2_abs_mean": finite_mean(row[f"{label}_K2_abs_mean"] for row in selected),
                    "K1_positive_fraction": finite_mean(row[f"{label}_K1_positive_fraction"] for row in selected),
                    "K2_positive_fraction": finite_mean(row[f"{label}_K2_positive_fraction"] for row in selected),
                    "tau1_boundary_rms": finite_mean(row[f"{label}_tau1_rms"] for row in selected),
                    "tau2_boundary_rms": finite_mean(row[f"{label}_tau2_rms"] for row in selected),
                    "torque_boundary_saturation_fraction": finite_mean(row[f"{label}_saturation"] for row in selected),
                    "power_boundary_abs_mean": finite_mean(row[f"{label}_power_abs_mean"] for row in selected),
                }
            )
    return result


def baseline_joint_rows(episode_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [row for row in episode_rows if row["condition_id"] == "C11"]
    result: list[dict[str, Any]] = []
    for index in range(8):
        label = f"J{index + 1:02d}"
        result.append(
            {
                "implementation_status": IMPLEMENTATION_STATUS,
                "joint": label,
                "K1_mean": finite_mean(row[f"{label}_K1_mean"] for row in selected),
                "K2_mean": finite_mean(row[f"{label}_K2_mean"] for row in selected),
                "K1_abs_mean": finite_mean(row[f"{label}_K1_abs_mean"] for row in selected),
                "K2_abs_mean": finite_mean(row[f"{label}_K2_abs_mean"] for row in selected),
                "K1_positive_fraction": finite_mean(
                    row[f"{label}_K1_positive_fraction"] for row in selected
                ),
                "K2_positive_fraction": finite_mean(
                    row[f"{label}_K2_positive_fraction"] for row in selected
                ),
                "tau1_rms": finite_mean(row[f"{label}_tau1_rms"] for row in selected),
                "tau2_rms": finite_mean(row[f"{label}_tau2_rms"] for row in selected),
                "saturation_fraction": finite_mean(
                    row[f"{label}_saturation"] for row in selected
                ),
                "power_abs_mean": finite_mean(
                    row[f"{label}_power_abs_mean"] for row in selected
                ),
            }
        )
    return result


def make_plots(
    condition_summary: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
    joints: list[dict[str, Any]],
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []
    plot_dir = OUTPUT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    by_condition = {row["condition_id"]: row for row in condition_summary}
    paths: list[str] = []

    matrix = np.asarray(
        [
            [by_condition["C00"]["mean_success_rate"], by_condition["C01"]["mean_success_rate"]],
            [by_condition["C10"]["mean_success_rate"], by_condition["C11"]["mean_success_rate"]],
        ]
    )
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks([0, 1], ["R0 K2", "Rroll K2"])
    ax.set_yticks([0, 1], ["R0 K1", "Rroll K1"])
    for row in range(2):
        for column in range(2):
            ax.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", color="white")
    ax.set_title("K1/K2 source factorial: roll success")
    fig.colorbar(image, ax=ax, label="success rate")
    fig.tight_layout()
    path = plot_dir / "A_channel_factorial.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    contrast_map = {row["contrast_id"]: row for row in contrasts}
    x = np.arange(1, 9)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, prefix, title in (
        (axes[0, 0], "B_K1_SUFF_", "K1 single-joint sufficiency"),
        (axes[0, 1], "B_K1_NEC_", "K1 leave-one-out necessity"),
        (axes[1, 0], "B_K2_SUFF_", "K2 single-joint sufficiency"),
        (axes[1, 1], "B_K2_NEC_", "K2 conditional necessity"),
    ):
        values = [contrast_map[f"{prefix}J{joint:02d}"]["mean_success_rate_effect"] for joint in x]
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.bar(x, values, color="#35618f")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xlabel("joint")
        ax.set_ylabel("oriented success-rate effect")
    fig.tight_layout()
    path = plot_dir / "B_per_joint_causal_effects.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    labels = [row["condition_id"] for row in condition_summary if row["module"] == "C"]
    values = [by_condition[label]["mean_success_rate"] for label in labels]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(np.arange(len(labels)), values, color="#8d4f78")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=65, ha="right", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("success rate")
    ax.set_title("K1 subset/sign/spatial interventions")
    fig.tight_layout()
    path = plot_dir / "C_k1_patterns.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    dose_ids = ["K2_SCALE_0", "K2_SCALE_0P5", "C11", "K2_SCALE_1P5"]
    dose_x = [0.0, 0.5, 1.0, 1.5]
    success = [by_condition[item]["mean_success_rate"] for item in dose_ids]
    speed = [by_condition[item]["mean_forward_body_lengths"] for item in dose_ids]
    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
    ax2 = ax1.twinx()
    ax1.plot(dose_x, success, "o-", color="#244f73", label="roll success")
    ax2.plot(dose_x, speed, "s--", color="#b05b3b", label="forward body lengths")
    ax1.set_xlabel("K2 scale")
    ax1.set_ylabel("success rate", color="#244f73")
    ax2.set_ylabel("mean forward body lengths", color="#b05b3b")
    ax1.set_title("K2 dose: rolling vs forward progress")
    fig.tight_layout()
    path = plot_dir / "D_k2_dose.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    labels = [row["joint"] for row in joints]
    axes[0].bar(labels, [row["K1_mean"] for row in joints], color="#287c6f")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Natural Rroll K1 mean")
    axes[0].set_ylabel("physical K")
    axes[1].bar(labels, [row["K2_mean"] for row in joints], color="#c17a2f")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_title("Natural Rroll K2 mean")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    path = plot_dir / "natural_joint_k_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(str(path))
    return paths


def make_report(
    summary: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
    synergy: list[dict[str, Any]],
    joints: list[dict[str, Any]],
) -> str:
    by_condition = {row["condition_id"]: row for row in summary}
    by_contrast = {row["contrast_id"]: row for row in contrasts}
    k1_nec = sorted(
        [by_contrast[f"B_K1_NEC_J{joint:02d}"] for joint in range(1, 9)],
        key=lambda row: row["mean_success_rate_effect"],
        reverse=True,
    )
    k1_suff = sorted(
        [by_contrast[f"B_K1_SUFF_J{joint:02d}"] for joint in range(1, 9)],
        key=lambda row: row["mean_success_rate_effect"],
        reverse=True,
    )
    k2_nec = sorted(
        [by_contrast[f"B_K2_NEC_J{joint:02d}"] for joint in range(1, 9)],
        key=lambda row: row["mean_success_rate_effect"],
        reverse=True,
    )
    k2_suff = sorted(
        [by_contrast[f"B_K2_SUFF_J{joint:02d}"] for joint in range(1, 9)],
        key=lambda row: row["mean_success_rate_effect"],
        reverse=True,
    )
    robust_c = sorted(
        [row for row in summary if row["module"] == "C"],
        key=lambda row: row["mean_success_rate"],
        reverse=True,
    )
    top_nec = ", ".join(
        f"{row['contrast_id'][-3:]}(Δ={row['mean_success_rate_effect']:.2f})"
        for row in k1_nec[:4]
    )
    top_suff = ", ".join(
        f"{row['contrast_id'][-3:]}(Δ={row['mean_success_rate_effect']:.2f})"
        for row in k1_suff[:4]
    )
    top_k2_nec = ", ".join(
        f"{row['contrast_id'][-3:]}(Δ={row['mean_success_rate_effect']:.2f})"
        for row in k2_nec[:8]
    )
    top_k2_suff = ", ".join(
        f"{row['contrast_id'][-3:]}(Δ={row['mean_success_rate_effect']:.2f})"
        for row in k2_suff[:8]
    )
    natural = "; ".join(
        f"{row['joint']} K1={row['K1_mean']:.1f}, K2={row['K2_mean']:.1f}"
        for row in joints
    )
    k2_dose = ", ".join(
        f"{label}:{by_condition[label]['mean_success_rate']:.2f}/"
        f"{by_condition[label]['mean_forward_body_lengths']:.2f}BL"
        for label in ("K2_SCALE_0", "K2_SCALE_0P5", "C11", "K2_SCALE_1P5")
    )
    lines = [
        "# Frozen K1/K2 Rolling-Mechanism Intervention Results",
        "",
        "## Evidence Boundary",
        "",
        "This study applies deterministic evaluation interventions only to the five completed pairs of formal checkpoints; it performs no training, changes no observation or action channel, and modifies neither the reward, PPO, nor the physics. Every condition shares 20 new initial states. The independent units across trained policies are the five seeds, so effect sizes and directional consistency are reported; initial-state permutation tests generalize only to new initial states for these five frozen policy pairs.",
        "With five training seeds, the smallest attainable two-sided exact sign-flip p-value is 0.0625, so 0.05 is mathematically unattainable. Holm-adjusted results only guard against overstatement; primary judgments must rely on effect size, directional consistency across seeds, and preregistered robustness thresholds. The 20 shared initial states are not 20 independent trained policies.",
        "",
        "## All-Channel Causal Decomposition",
        "",
        f"- C00={by_condition['C00']['mean_success_rate']:.3f}, C10={by_condition['C10']['mean_success_rate']:.3f}, C01={by_condition['C01']['mean_success_rate']:.3f}, and C11={by_condition['C11']['mean_success_rate']:.3f}.",
        f"- Add all Rroll K1 channels on the R0-K2 background: delta={by_contrast['A_K1_AT_R0_K2']['mean_success_rate_effect']:.3f}; add all Rroll K1 channels on the Rroll-K2 background: delta={by_contrast['A_K1_AT_RROLL_K2']['mean_success_rate_effect']:.3f}.",
        f"- Add all Rroll K2 channels on the R0-K1 background: delta={by_contrast['A_K2_AT_R0_K1']['mean_success_rate_effect']:.3f}; add Rroll K2 given Rroll K1: delta={by_contrast['A_K2_AT_RROLL_K1']['mean_success_rate_effect']:.3f}.",
        f"- Nonadditive K1 x K2 interaction C11-C10-C01+C00={by_contrast['A_K1_K2_INTERACTION']['mean_success_rate_effect']:.3f}. A positive value is consistent with positive nonadditivity between the channels, but this contrast alone cannot establish a unique mechanism.",
        "",
        "## Per-Joint K1",
        "",
        f"- Joint with the largest descriptive necessity drop: {top_nec}.",
        f"- Joint with the largest descriptive single-joint sufficiency effect: {top_suff}.",
        "- Necessity is defined on the C10 background (all other K1=Rroll, K2=R0); sufficiency is defined on the C00 background. Neither can be interpreted outside its background.",
        "",
        "## Per-Joint K2",
        "",
        f"- Conditional-necessity ranking when each joint is replaced by R0 on the full C11 background: {top_k2_nec}.",
        f"- Conditional-sufficiency ranking when only one Rroll K2 is transplanted onto C00: {top_k2_suff}.",
        "- The K2 necessity effect is evaluated on the full Rroll K1/K2 background and directly tests each joint's K2 contribution to maintaining an established rolling feedback loop. Critical joints may differ by seed; the aggregate ranking cannot be interpreted as one unique joint shared by all policies.",
        "",
        "## K1 Sign and Spatial Distribution",
        "",
        *[
            f"- {row['condition_id']}: success rate={row['mean_success_rate']:.3f}, "
            f"successful training seeds={row['successful_training_seed_count']}/5, forward displacement={row['mean_forward_body_lengths']:.3f} body lengths."
            for row in robust_c
        ],
        "",
        "## K2 Effects on Speed and Persistence",
        "",
        f"- K2 dose (success rate/forward body lengths): {k2_dose}.",
        "- K2=0, regional, sign, and temporal-template interventions are conditional or exploratory mechanism evidence. Only preservation of rolling success alongside altered forward displacement or pulse interval supports the interpretation that K2 primarily modulates speed/stability rather than direction.",
        "",
        "## Joint Distribution in the Natural Policy",
        "",
        natural,
        "",
        "## Combination Synergy",
        "",
        *[
            f"- {row['combination']}: second-order interaction RD={row['mean_synergy_RD']:.3f}, "
            f"seed-bootstrap 95% CI [{row['bootstrap_ci_low']:.3f}, {row['bootstrap_ci_high']:.3f}]，"
            f"Holm two-sided p={row['holm_two_sided_p']:.3f}; this synergy3 analysis is an independent post hoc family and does not share FWER control with C13."
            for row in synergy
        ],
        "",
        "Complete tables are in condition_summary.csv, contrasts.csv, subset_synergy.csv, and baseline_joint_profile.csv; figures are in plots/.",
        "Intervals for continuous metrics in continuous_mechanism_contrasts.csv are post hoc, per-metric descriptive intervals without multiplicity correction; exclusion of zero must not be interpreted as statistical significance. Pulse-interval values include only pairs for which the interval is defined under both conditions and therefore have a selection boundary. Changes in definability across all pairs are reported separately as pulse_interval_defined.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    rows, episodes, evaluation_seeds, integrity = load_all_rows()
    seed_summary = condition_seed_rows(episodes)
    condition_summary = condition_summary_rows(seed_summary)
    contrasts = contrast_rows(episodes, evaluation_seeds)
    synergy = subset_synergy_rows(episodes)
    continuous = continuous_contrast_rows(rows, evaluation_seeds)
    condition_joints = condition_joint_rows(rows)
    joints = baseline_joint_rows(rows)
    write_csv(OUTPUT / "episode_results.csv", rows)
    write_csv(OUTPUT / "condition_seed_summary.csv", seed_summary)
    write_csv(OUTPUT / "condition_summary.csv", condition_summary)
    write_csv(OUTPUT / "contrasts.csv", contrasts)
    write_csv(OUTPUT / "subset_synergy.csv", synergy)
    write_csv(OUTPUT / "continuous_mechanism_contrasts.csv", continuous)
    write_csv(OUTPUT / "condition_joint_profile.csv", condition_joints)
    write_csv(
        OUTPUT / "B_all_condition_results.csv",
        [row for row in condition_summary if row["module"] == "B"],
    )
    write_csv(
        OUTPUT / "D_all_condition_results.csv",
        [row for row in condition_summary if row["module"] == "D"],
    )
    write_csv(
        OUTPUT / "B_all_contrasts.csv",
        [row for row in contrasts if row["family"] == "B_32"],
    )
    write_csv(
        OUTPUT / "D_all_contrasts.csv",
        [row for row in contrasts if row["family"] == "D_10"],
    )
    write_csv(
        OUTPUT / "K2_per_joint_results.csv",
        [
            {
                key: row[key]
                for key in (
                    "implementation_status",
                    "condition_id",
                    "module",
                    "family",
                    "joint",
                    "K2_mean",
                    "K2_abs_mean",
                    "K2_positive_fraction",
                    "tau2_boundary_rms",
                    "torque_boundary_saturation_fraction",
                    "power_boundary_abs_mean",
                )
            }
            for row in condition_joints
        ],
    )
    write_csv(OUTPUT / "baseline_joint_profile.csv", joints)
    plots = make_plots(condition_summary, contrasts, joints)
    report = (
        "# K1/K2 mechanism analysis\n\n"
        f"Implementation status: `{IMPLEMENTATION_STATUS}`.\n\n"
        "The 59-condition design, sample sizes, fixed initial-state split, and five-part endpoint were frozen before outcomes; this concrete statistical analyzer was implemented after main result generation began and is not preregistered analysis code.\n\n"
        "Continuous paired effects are in `continuous_mechanism_contrasts.csv`; "
        "the pulse-interval metric reports only evaluation-seed pairs where both "
        "conditions define an interval, together with coverage. The frozen schema "
        "does not contain separate K1/K2 saturation or signed mean torque. Therefore "
        "the analysis transparently reports K1/K2 absolute magnitude and sign "
        "fraction, tau1/tau2 boundary RMS, and the sole actuator torque-boundary "
        "saturation fraction. Full B and D outputs and all-condition K2 joint "
        "profiles are provided as separate CSV files.\n\n"
        + make_report(condition_summary, contrasts, synergy, joints)
    )
    (OUTPUT / "RESULTS_REPORT.md").write_text(report, encoding="utf-8")
    manifest = {
        "schema": "obs2_v2_1_k_analysis_manifest/v1",
        "implementation_status": IMPLEMENTATION_STATUS,
        "study_id": CONFIG["study_id"],
        "prerequisite_audits_passed": list(AUDIT_FILES),
        "integrity": integrity,
        "fixed_evaluation_seeds": evaluation_seeds,
        "fixed_evaluation_seed_count": len(evaluation_seeds),
        "episode_rows": len(rows),
        "condition_seed_rows": len(seed_summary),
        "condition_rows": len(condition_summary),
        "contrast_rows": len(contrasts),
        "continuous_mechanism_contrast_rows": len(continuous),
        "condition_joint_rows": len(condition_joints),
        "holm_families": {
            "A_5": 5,
            "B_32": 32,
            "C_13": 13,
            "D_10": 10,
            "THREE_JOINT_SYNERGY_3": 3,
        },
        "continuous_metric_boundary": (
            "No separate K1/K2 saturation or signed mean torque exists in the frozen "
            "result schema; uses K abs/sign summaries, tau1/tau2 boundary RMS, and "
            "the sole torque-boundary saturation fraction without substitution. All "
            "continuous intervals are posthoc pointwise descriptive and unadjusted; "
            "pulse-interval values are conditional on both members being defined."
        ),
        "plots": plots,
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "statistical_boundary": "Five independent training seeds; 20 evaluation seeds are paired repeated initial states.",
        "analysis_timing_status": IMPLEMENTATION_STATUS,
    }
    atomic_json(OUTPUT / "analysis_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
