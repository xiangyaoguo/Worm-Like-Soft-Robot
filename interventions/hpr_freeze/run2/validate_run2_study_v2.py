from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from run_formal_hpr_run2 import (
    CHECKPOINT,
    EXPECTED_BASELINE_SUCCESSES,
    EXPECTED_CHECKPOINT_SHA256,
    IDENTITY_TOLERANCE,
    INTERNAL_TRAINING_SEED,
    PAPER_RUN_ID,
    RESET_SEEDS,
    ROOT,
    sha256_file,
)


EXPECTED_FIGURES = (
    "fig_run2_global_channel_outcomes.png",
    "fig_run2_joint_ablation_retention.png",
    "fig_run2_joint_channel_effects.png",
    "fig_run2_effect_agreement_runs0_2_4.png",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def common_success(row: dict[str, Any]) -> bool:
    return bool(
        float(row["desired_net_rotation_degrees"]) >= 360.0
        and float(row["desired_active_rotation_fraction"]) >= 0.70
        and float(row["forward_body_lengths"]) >= 1.0
    )


def validate_recorded_ledger(data_root: Path, execution: dict[str, Any], issues: list[str]) -> None:
    for item in execution.get("output_ledger", []):
        path = data_root / item["relative_path"]
        if not path.is_file():
            issues.append(f"recorded output missing: {item['relative_path']}")
            continue
        if path.stat().st_size != int(item["bytes"]):
            issues.append(f"recorded output size drift: {item['relative_path']}")
        if sha256_file(path) != item["sha256"]:
            issues.append(f"recorded output hash drift: {item['relative_path']}")


def validate_source_tree(source: dict[str, Any], issues: list[str]) -> None:
    """Recheck every file recorded for the immutable formal code snapshot."""
    tree = source.get("formal_code_snapshot_tree", {})
    root_value = tree.get("root")
    entries = tree.get("files", [])
    if not root_value or not isinstance(entries, list):
        issues.append("formal code-snapshot receipt is missing or malformed")
        return
    root = Path(root_value)
    if not root.is_dir():
        issues.append("formal code-snapshot root is missing")
        return
    if int(tree.get("file_count", -1)) != len(entries):
        issues.append("formal code-snapshot receipt file count is inconsistent")
    recorded_relatives = {str(item.get("relative_path")) for item in entries}
    current_relatives = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    if current_relatives != recorded_relatives:
        issues.append("formal code-snapshot file set drift")
    for item in entries:
        relative = str(item.get("relative_path"))
        path = root / relative
        if not path.is_file():
            issues.append(f"formal code-snapshot file missing: {relative}")
            continue
        if path.stat().st_size != int(item.get("bytes", -1)):
            issues.append(f"formal code-snapshot size drift: {relative}")
        if sha256_file(path) != item.get("sha256"):
            issues.append(f"formal code-snapshot hash drift: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--require-figures", action="store_true")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    issues: list[str] = []

    required = (
        "SOURCE_MANIFEST.json",
        "RUN2_EXECUTION_MANIFEST.json",
        "STUDY_MANIFEST.json",
        "episode_results.csv",
        "condition_policy_summary.csv",
        "raw/seed9203/BASELINE_IDENTITY_AUDIT.json",
        "raw/seed9203/SEED_MANIFEST.json",
    )
    for relative in required:
        if not (data_root / relative).is_file():
            issues.append(f"missing required output: {relative}")
    if issues:
        payload = {"status": "FAIL", "issues": issues}
        print(json.dumps(payload, ensure_ascii=False))
        raise SystemExit(1)

    execution = load_json(data_root / "RUN2_EXECUTION_MANIFEST.json")
    source = load_json(data_root / "SOURCE_MANIFEST.json")
    study = load_json(data_root / "STUDY_MANIFEST.json")
    episodes = read_csv(data_root / "episode_results.csv")
    summaries = read_csv(data_root / "condition_policy_summary.csv")

    validate_source_tree(source, issues)

    if execution.get("paper_run_id") != PAPER_RUN_ID:
        issues.append("paper-facing run id is not 2")
    if execution.get("internal_training_seed") != INTERNAL_TRAINING_SEED:
        issues.append("internal training seed is not 9203")
    if execution.get("mode") != "full":
        issues.append("execution manifest is not a full run")
    if int(execution.get("condition_count", -1)) != 36:
        issues.append("execution condition count is not 36")
    if int(execution.get("total_rollouts", -1)) != 720:
        issues.append("execution rollout count is not 720")
    if len(episodes) != 720:
        issues.append(f"episode_results row count {len(episodes)} != 720")
    if len(summaries) != 36:
        issues.append(f"condition summary row count {len(summaries)} != 36")
    if int(study.get("total_rollouts", -1)) != 720:
        issues.append("base study manifest rollout count is not 720")
    if study.get("training_seeds") != [INTERNAL_TRAINING_SEED]:
        issues.append("base study manifest seed mapping drift")
    criterion = study.get("common_kinematic_criterion", {})
    if criterion != {
        "minimum_desired_net_rotation_degrees": 360.0,
        "minimum_desired_active_rotation_fraction": 0.70,
        "minimum_forward_body_lengths": 1.0,
        "pulse_or_contact_gate_used": False,
    }:
        issues.append("common kinematic criterion drift")

    expected_condition_ids = tuple(execution.get("condition_ids", []))
    if len(expected_condition_ids) != 36 or len(set(expected_condition_ids)) != 36:
        issues.append("execution condition id list is not a unique 36-condition matrix")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    duplicate_counter: Counter[tuple[str, int]] = Counter()
    for row in episodes:
        if int(row["training_seed"]) != INTERNAL_TRAINING_SEED:
            issues.append("episode row contains a non-run2 internal seed")
            continue
        condition_id = row["condition_id"]
        reset_seed = int(row["reset_seed"])
        grouped[condition_id].append(row)
        duplicate_counter[(condition_id, reset_seed)] += 1
        for field in (
            "desired_net_rotation_degrees",
            "desired_active_rotation_fraction",
            "forward_body_lengths",
            "desired_revolutions",
            "mean_abs_k1",
            "mean_abs_k2",
        ):
            if not math.isfinite(float(row[field])):
                issues.append(f"non-finite metric: {condition_id}/{reset_seed}/{field}")
        if truth(row["success_kinematic"]) != common_success(row):
            issues.append(f"kinematic success recomputation mismatch: {condition_id}/{reset_seed}")
    if any(value != 1 for value in duplicate_counter.values()):
        issues.append("duplicate condition/reset row detected")
    if tuple(grouped) != expected_condition_ids:
        issues.append("condition ordering or membership drift")
    for condition_id, rows in grouped.items():
        reset_set = tuple(sorted(int(row["reset_seed"]) for row in rows))
        if reset_set != RESET_SEEDS:
            issues.append(f"paired reset set mismatch: {condition_id}")

    raw_dir = data_root / "raw" / f"seed{INTERNAL_TRAINING_SEED}"
    condition_files = sorted(
        path for path in raw_dir.glob("*.json")
        if path.name not in {"BASELINE_IDENTITY_AUDIT.json", "SEED_MANIFEST.json"}
    )
    if len(condition_files) != 36:
        issues.append(f"raw condition file count {len(condition_files)} != 36")
    for path in condition_files:
        payload = load_json(path)
        if payload.get("schema") != "formal_hpr_freeze_validation/condition/v1":
            issues.append(f"condition schema drift: {path.name}")
        if int(payload.get("training_seed", -1)) != INTERNAL_TRAINING_SEED:
            issues.append(f"condition seed mapping drift: {path.name}")
        if len(payload.get("episodes", [])) != 20:
            issues.append(f"condition episode count drift: {path.name}")

    identity = load_json(raw_dir / "BASELINE_IDENTITY_AUDIT.json")
    observed_baseline = sum(
        truth(row["success_kinematic"]) for row in grouped.get("BASELINE", [])
    )
    if observed_baseline != EXPECTED_BASELINE_SUCCESSES:
        issues.append(f"baseline common success {observed_baseline} != 7")
    for key in (
        "expected_common_criterion_success_count",
        "observed_common_criterion_success_count",
        "official_common_criterion_success_count",
    ):
        if int(identity.get(key, -1)) != EXPECTED_BASELINE_SUCCESSES:
            issues.append(f"identity gate {key} != 7")
    if not identity.get("per_reset_success_classification_exact"):
        issues.append("baseline per-reset success classification is not exact")
    if not identity.get("continuous_metrics_exact_within_tolerance"):
        issues.append("baseline continuous identity gate is false")
    errors = identity.get("maximum_absolute_error", {})
    if not errors or max(float(value) for value in errors.values()) > IDENTITY_TOLERANCE:
        issues.append("baseline continuous metric error exceeds 1e-6")

    if sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA256:
        issues.append("current formal checkpoint hash drift")
    if source.get("expected_checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256:
        issues.append("source manifest checkpoint hash drift")
    for item in source.get("files", {}).values():
        path = Path(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            issues.append(f"source file missing or changed: {path}")

    seed_manifest = load_json(raw_dir / "SEED_MANIFEST.json")
    integrity = seed_manifest.get("post_run_integrity", {})
    if not integrity.get("checkpoints_before_after_equal"):
        issues.append("checkpoint before/after immutability gate failed")
    if not integrity.get("policies_before_after_equal"):
        issues.append("policy before/after immutability gate failed")
    if int(seed_manifest.get("condition_count", -1)) != 36:
        issues.append("seed manifest condition count is not 36")
    if int(seed_manifest.get("total_rollouts", -1)) != 720:
        issues.append("seed manifest rollout count is not 720")
    validate_recorded_ledger(data_root, execution, issues)

    figure_receipts: list[dict[str, Any]] = []
    if args.require_figures:
        figure_root = ROOT / "figures"
        for filename in EXPECTED_FIGURES:
            path = figure_root / filename
            if not path.is_file() or path.stat().st_size <= 0:
                issues.append(f"missing or empty figure: {filename}")
            else:
                figure_receipts.append(
                    {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
                )
        if not (data_root / "paired_effects.csv").is_file():
            issues.append("paired_effects.csv missing after plotting")
        if not (data_root / "ANALYSIS_SUMMARY.json").is_file():
            issues.append("ANALYSIS_SUMMARY.json missing after plotting")

    payload = {
        "schema": "formal_hpr_run2_freeze/validation/v1",
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "paper_run_id": PAPER_RUN_ID,
        "internal_training_seed": INTERNAL_TRAINING_SEED,
        "episode_rows": len(episodes),
        "condition_groups": len(grouped),
        "paired_resets_per_condition": 20,
        "baseline_common_success_count": observed_baseline,
        "identity_absolute_tolerance": IDENTITY_TOLERANCE,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "figure_receipts": figure_receipts,
    }
    output = data_root / ("VALIDATION_PASS.json" if not issues else "VALIDATION_FAIL.json")
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
