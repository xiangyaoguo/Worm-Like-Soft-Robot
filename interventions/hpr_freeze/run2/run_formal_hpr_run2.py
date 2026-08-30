from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REFERENCE_ROOT = Path(
    r"C:\Users\PUBLIC_USER\Documents\GraduateThesisProject"
    r"\formal_hpr_freeze_validation_20260810"
)
REFERENCE_RUNNER = REFERENCE_ROOT / "scripts" / "run_formal_hpr_freeze_study.py"
REFERENCE_VALIDATOR = REFERENCE_ROOT / "scripts" / "validate_study.py"
REFERENCE_ANALYSIS = REFERENCE_ROOT / "scripts" / "analyse_and_plot.py"
MECHANISM_ROLLOUT = Path(
    r"C:\Users\PUBLIC_USER\Documents\GraduateThesisProject"
    r"\obs2_v2_1_k_mechanism_20260804\mechanism_rollout.py"
)
FORMAL_ROOT = Path(
    r"C:\Users\PUBLIC_USER\CloudStorage\Desktop\finalproject\job\roll_learning"
    r"\obs2_roll_repro_v2_1_formal_20260803_r2"
)
FORMAL_CONFIG = FORMAL_ROOT / "_control" / "experiment_config.json"
FORMAL_CODE_SNAPSHOT = FORMAL_ROOT / "_control" / "code_snapshot"
CHECKPOINT = (
    FORMAL_ROOT
    / "formal"
    / "runs"
    / "formal__seed9203__R0"
    / "checkpoint_1500.pt"
)
OFFICIAL_EVALUATION = (
    FORMAL_ROOT
    / "formal"
    / "evaluations"
    / "formal__seed9203__R0__eval_attempt1.json"
)
REFERENCE_CONDITION = (
    REFERENCE_ROOT / "data" / "raw" / "seed9201" / "BASELINE.json"
)

PAPER_RUN_ID = 2
INTERNAL_TRAINING_SEED = 9203
RESET_SEEDS = tuple(range(20264101, 20264121))
EXPECTED_BASELINE_SUCCESSES = 7
EXPECTED_CHECKPOINT_SHA256 = (
    "0428d9a86b6622d924738c68fe09df4c6ab922e2a3225a5c24ba41e96eb1c4b8"
)
IDENTITY_TOLERANCE = 1e-6
CONTINUOUS_IDENTITY_FIELDS = (
    "initial_body_length",
    "forward_displacement",
    "forward_body_lengths",
    "net_best_fit_rotation_degrees",
    "desired_net_rotation_degrees",
    "desired_active_rotation_fraction",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_receipt(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    entries: list[dict[str, Any]] = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        file_hash = sha256_file(path)
        size = path.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\n")
        entries.append({"relative_path": relative, "bytes": size, "sha256": file_hash})
    return {
        "root": str(root),
        "file_count": len(entries),
        "aggregate_sha256": digest.hexdigest(),
        "files": entries,
    }


def output_ledger(root: Path, excluded: set[str] | None = None) -> list[dict[str, Any]]:
    excluded = excluded or set()
    rows: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative in excluded or relative.endswith(".tmp"):
            continue
        rows.append(
            {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_reference_runner() -> Any:
    if not REFERENCE_RUNNER.is_file():
        raise FileNotFoundError(REFERENCE_RUNNER)
    spec = importlib.util.spec_from_file_location("reference_hpr_freeze_runner", REFERENCE_RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import reference runner: {REFERENCE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def common_success(row: dict[str, Any]) -> bool:
    return bool(
        float(row["desired_net_rotation_degrees"]) >= 360.0
        and float(row["desired_active_rotation_fraction"]) >= 0.70
        and float(row["forward_body_lengths"]) >= 1.0
    )


def baseline_identity_gate(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    official_payload = load_json(OFFICIAL_EVALUATION)
    official_rows = official_payload["results"][0]["episodes"]
    if len(official_rows) != 20 or len(episodes) != 20:
        raise RuntimeError("Baseline identity episode count mismatch")
    official_by_reset = {int(row["seed"]): row for row in official_rows}
    if tuple(sorted(official_by_reset)) != RESET_SEEDS:
        raise RuntimeError("Official evaluation reset set drift")
    maximum_error = {field: 0.0 for field in CONTINUOUS_IDENTITY_FIELDS}
    per_reset_success_exact = True
    for row in episodes:
        reset_seed = int(row["reset_seed"])
        expected = official_by_reset.get(reset_seed)
        if expected is None:
            raise RuntimeError(f"Official baseline missing reset seed {reset_seed}")
        for field in CONTINUOUS_IDENTITY_FIELDS:
            actual_value = float(row[field])
            expected_value = float(expected[field])
            if not math.isfinite(actual_value) or not math.isfinite(expected_value):
                raise RuntimeError(f"Non-finite baseline identity value: {reset_seed}/{field}")
            error = abs(actual_value - expected_value)
            maximum_error[field] = max(maximum_error[field], error)
            if error > IDENTITY_TOLERANCE:
                raise RuntimeError(
                    f"Baseline identity mismatch for reset {reset_seed}, {field}: "
                    f"error={error:.12g} > {IDENTITY_TOLERANCE}"
                )
        observed_success = bool(row["success_kinematic"])
        expected_success = common_success(expected)
        if observed_success != expected_success:
            per_reset_success_exact = False
            raise RuntimeError(f"Baseline success mismatch for reset {reset_seed}")
    observed_count = sum(bool(row["success_kinematic"]) for row in episodes)
    official_count = sum(common_success(row) for row in official_rows)
    if observed_count != EXPECTED_BASELINE_SUCCESSES or official_count != EXPECTED_BASELINE_SUCCESSES:
        raise RuntimeError(
            f"Run 2 baseline gate failed: observed={observed_count}, "
            f"official={official_count}, expected={EXPECTED_BASELINE_SUCCESSES}"
        )
    return {
        "schema": "formal_hpr_freeze_validation/baseline_identity/v2",
        "paper_run_id": PAPER_RUN_ID,
        "internal_training_seed": INTERNAL_TRAINING_SEED,
        "official_evaluation": str(OFFICIAL_EVALUATION),
        "official_evaluation_sha256": sha256_file(OFFICIAL_EVALUATION),
        "episode_count": 20,
        "expected_common_criterion_success_count": EXPECTED_BASELINE_SUCCESSES,
        "observed_common_criterion_success_count": observed_count,
        "official_common_criterion_success_count": official_count,
        "common_criterion_success_count_exact": True,
        "per_reset_success_classification_exact": per_reset_success_exact,
        "continuous_metrics_exact_within_tolerance": True,
        "continuous_metric_fields": list(CONTINUOUS_IDENTITY_FIELDS),
        "absolute_tolerance": IDENTITY_TOLERANCE,
        "maximum_absolute_error": maximum_error,
    }


def archive_existing_data(path: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = path.with_name(f"{path.name}_superseded_{timestamp}")
    if target.exists():
        raise FileExistsError(target)
    path.replace(target)
    return target


def prepare_output(path: Path, replace: bool) -> Path | None:
    if not path.exists():
        path.mkdir(parents=True)
        return None
    has_files = any(item.is_file() for item in path.rglob("*"))
    if not has_files:
        return None
    if not replace:
        raise RuntimeError(
            f"Output directory is not empty: {path}. Use --replace to archive and rerun."
        )
    archived = archive_existing_data(path)
    path.mkdir(parents=True)
    return archived


def source_manifest() -> dict[str, Any]:
    required_files = {
        "adapter_runner": Path(__file__).resolve(),
        "adapter_validator": ROOT / "validate_run2_study.py",
        "adapter_plotter": ROOT / "plot_run2_study.py",
        "preregistered_run2_protocol": ROOT / "PREREGISTERED_RUN2_PROTOCOL.md",
        "reference_runner": REFERENCE_RUNNER,
        "reference_validator": REFERENCE_VALIDATOR,
        "reference_analysis": REFERENCE_ANALYSIS,
        "mechanism_rollout": MECHANISM_ROLLOUT,
        "formal_experiment_config": FORMAL_CONFIG,
        "formal_checkpoint": CHECKPOINT,
        "formal_official_evaluation": OFFICIAL_EVALUATION,
        "reference_condition_payload": REFERENCE_CONDITION,
    }
    for path in required_files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    files = {
        name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for name, path in required_files.items()
    }
    if files["formal_checkpoint"]["sha256"] != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("Run 2 formal checkpoint SHA-256 mismatch")
    reference = load_json(REFERENCE_CONDITION)
    if reference.get("schema") != "formal_hpr_freeze_validation/condition/v1":
        raise RuntimeError("Reference condition schema drift")
    return {
        "schema": "formal_hpr_run2_freeze/source_manifest/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_run_id": PAPER_RUN_ID,
        "internal_training_seed": INTERNAL_TRAINING_SEED,
        "files": files,
        "formal_code_snapshot_tree": tree_receipt(FORMAL_CODE_SNAPSHOT),
        "expected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "reference_condition_schema": reference["schema"],
    }


def run(mode: str, replace: bool) -> dict[str, Any]:
    pilot = mode == "pilot"
    data_root = ROOT / ("pilot_data" if pilot else "data")
    archived = prepare_output(data_root, replace)
    sources = source_manifest()
    atomic_json(data_root / "SOURCE_MANIFEST.json", sources)

    reference = load_reference_runner()
    reference.DATA_ROOT = data_root
    reference.EXPECTED_CHECKPOINT_HASHES = {
        INTERNAL_TRAINING_SEED: EXPECTED_CHECKPOINT_SHA256
    }

    def patched_compare(training_seed: int, episodes: list[dict[str, Any]]) -> dict[str, Any]:
        if int(training_seed) != INTERNAL_TRAINING_SEED:
            raise RuntimeError(f"Unexpected internal training seed: {training_seed}")
        payload = baseline_identity_gate(episodes)
        payload["checkpoint_sha256"] = EXPECTED_CHECKPOINT_SHA256
        return payload

    reference.compare_baseline_to_official = patched_compare
    conditions = reference.build_conditions("full")
    condition_ids = [condition.id for condition in conditions]
    if len(condition_ids) != 36 or len(set(condition_ids)) != 36:
        raise RuntimeError("Reference condition matrix is not the locked 36-condition design")

    started = time.perf_counter()
    seed_manifest = reference.run_training_seed(
        INTERNAL_TRAINING_SEED,
        "full",
        pilot=pilot,
    )
    study_manifest = reference.aggregate_outputs(
        [INTERNAL_TRAINING_SEED],
        "full",
        pilot=pilot,
    )
    elapsed = time.perf_counter() - started
    expected_rollouts = 20 if pilot else 720
    if int(study_manifest["total_rollouts"]) != expected_rollouts:
        raise RuntimeError(
            f"Rollout count mismatch: {study_manifest['total_rollouts']} != {expected_rollouts}"
        )
    execution = {
        "schema": "formal_hpr_run2_freeze/execution_manifest/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "paper_run_id": PAPER_RUN_ID,
        "internal_training_seed": INTERNAL_TRAINING_SEED,
        "condition_schema": "formal_hpr_freeze_validation/condition/v1",
        "condition_ids": condition_ids[:1] if pilot else condition_ids,
        "condition_count": 1 if pilot else 36,
        "paired_reset_seeds": list(RESET_SEEDS),
        "episode_count_per_condition": 20,
        "steps_per_rollout": 1000,
        "total_rollouts": expected_rollouts,
        "expected_baseline_common_success_count": EXPECTED_BASELINE_SUCCESSES,
        "identity_absolute_tolerance": IDENTITY_TOLERANCE,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "archived_previous_output": None if archived is None else str(archived),
        "wall_seconds": elapsed,
        "seed_manifest_summary": {
            "condition_count": seed_manifest["condition_count"],
            "total_rollouts": seed_manifest["total_rollouts"],
            "post_run_integrity": seed_manifest["post_run_integrity"],
        },
        "source_manifest_sha256": sha256_file(data_root / "SOURCE_MANIFEST.json"),
    }
    atomic_json(data_root / "RUN2_EXECUTION_MANIFEST.json", execution)
    execution["output_ledger"] = output_ledger(
        data_root,
        excluded={"RUN2_EXECUTION_MANIFEST.json"},
    )
    atomic_json(data_root / "RUN2_EXECUTION_MANIFEST.json", execution)
    return execution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate paper-facing HPR run 2 under the locked freeze matrix."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pilot", action="store_true", help="Run the 20-rollout baseline gate only.")
    mode.add_argument("--full", action="store_true", help="Run all 36 x 20 = 720 rollouts.")
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Archive an existing output directory by rename before rerunning.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run("pilot" if args.pilot else "full", replace=bool(args.replace))
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
