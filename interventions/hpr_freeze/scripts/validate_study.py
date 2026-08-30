from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SOURCE_STUDY_ROOT = Path(__file__).resolve().parents[1]
STUDY_ROOT = Path(
    os.environ.get("THESIS_HPR_OUTPUT", str(SOURCE_STUDY_ROOT))
).resolve()
DATA_ROOT = STUDY_ROOT / "data"
FIGURE_ROOT = STUDY_ROOT / "figures"
EXPECTED_SEEDS = (9201, 9205)
EXPECTED_RESETS = tuple(range(20264101, 20264121))
EXPECTED_CHECKPOINT_HASHES = {
    9201: "90f37918b9d3c2a0d50752c0db73940b03b79626d94dfde3381df2b0d3a0ae52",
    9205: "3c10cbb25159a01416240af82dc1bd6a0cf3f608d27e7a01e02ec785994aec7e",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truth(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes"}


def main() -> None:
    issues: list[str] = []
    episodes = read_csv(DATA_ROOT / "episode_results.csv")
    summaries = read_csv(DATA_ROOT / "condition_policy_summary.csv")
    effects = read_csv(DATA_ROOT / "paired_effects.csv")
    study_manifest = load_json(DATA_ROOT / "STUDY_MANIFEST.json")

    if len(episodes) != 1440:
        issues.append(f"episode_results row count {len(episodes)} != 1440")
    if len(summaries) != 72:
        issues.append(f"condition summary row count {len(summaries)} != 72")
    if len(effects) != 72:
        issues.append(f"paired effect row count {len(effects)} != 72")

    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    duplicate_counter: Counter[tuple[int, str, int]] = Counter()
    recomputed_success = 0
    for row in episodes:
        training_seed = int(row["training_seed"])
        condition_id = row["condition_id"]
        reset_seed = int(row["reset_seed"])
        grouped[(training_seed, condition_id)].append(row)
        duplicate_counter[(training_seed, condition_id, reset_seed)] += 1
        numeric_fields = (
            "desired_net_rotation_degrees",
            "desired_active_rotation_fraction",
            "forward_body_lengths",
            "desired_revolutions",
        )
        if not all(math.isfinite(float(row[field])) for field in numeric_fields):
            issues.append(f"non-finite metric: {training_seed}/{condition_id}/{reset_seed}")
        expected_success = bool(
            float(row["desired_net_rotation_degrees"]) >= 360.0
            and float(row["desired_active_rotation_fraction"]) >= 0.70
            and float(row["forward_body_lengths"]) >= 1.0
        )
        if truth(row["success_kinematic"]) != expected_success:
            issues.append(f"success mismatch: {training_seed}/{condition_id}/{reset_seed}")
        recomputed_success += int(expected_success)

    if any(count != 1 for count in duplicate_counter.values()):
        issues.append("duplicate policy-condition-reset row detected")
    if len(grouped) != 72:
        issues.append(f"policy-condition groups {len(grouped)} != 72")
    for key, rows in grouped.items():
        resets = tuple(sorted(int(row["reset_seed"]) for row in rows))
        if resets != EXPECTED_RESETS:
            issues.append(f"paired reset set mismatch for {key}")

    for seed in EXPECTED_SEEDS:
        audit = load_json(DATA_ROOT / "raw" / f"seed{seed}" / "BASELINE_IDENTITY_AUDIT.json")
        if not audit.get("all_common_criterion_success"):
            issues.append(f"baseline identity success gate failed for seed {seed}")
        if max(float(value) for value in audit["maximum_absolute_error"].values()) > 1e-6:
            issues.append(f"baseline identity numerical error exceeded tolerance for seed {seed}")
        if audit.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_HASHES[seed]:
            issues.append(f"checkpoint hash mismatch in identity audit for seed {seed}")
        manifest = load_json(DATA_ROOT / "raw" / f"seed{seed}" / "SEED_MANIFEST.json")
        integrity = manifest["post_run_integrity"]
        if not integrity.get("checkpoints_before_after_equal") or not integrity.get("policies_before_after_equal"):
            issues.append(f"post-run immutability gate failed for seed {seed}")
        if int(manifest["condition_count"]) != 36 or int(manifest["total_rollouts"]) != 720:
            issues.append(f"seed manifest matrix count mismatch for seed {seed}")

    expected_figures = [
        "fig_01_intervention_design.png",
        "fig_02_global_channel_outcomes.png",
        "fig_03_joint_ablation_retention_heatmap.png",
        "fig_04_joint_channel_effect_heatmap.png",
        "fig_05_cross_seed_agreement.png",
        "fig_06_morphology_seed9201.png",
        "fig_07_morphology_seed9205.png",
        "fig_08_representative_time_series.png",
    ]
    figure_receipts = []
    for filename in expected_figures:
        path = FIGURE_ROOT / filename
        if not path.is_file() or path.stat().st_size <= 0:
            issues.append(f"missing or empty figure: {filename}")
        else:
            figure_receipts.append(
                {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            )

    if int(study_manifest.get("total_rollouts", -1)) != 1440:
        issues.append("study manifest total_rollouts mismatch")
    if study_manifest.get("common_kinematic_criterion", {}).get("pulse_or_contact_gate_used") is not False:
        issues.append("primary endpoint incorrectly uses a pulse/contact gate")

    payload = {
        "schema": "formal_hpr_freeze_validation/validation/v1",
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
        "episode_rows": len(episodes),
        "condition_policy_groups": len(grouped),
        "paired_resets_per_group": 20,
        "recomputed_kinematic_successes_across_all_conditions": recomputed_success,
        "training_seed_count": 2,
        "checkpoint_hashes": EXPECTED_CHECKPOINT_HASHES,
        "figure_receipts": figure_receipts,
    }
    output = DATA_ROOT / ("VALIDATION_PASS.json" if not issues else "VALIDATION_FAIL.json")
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
