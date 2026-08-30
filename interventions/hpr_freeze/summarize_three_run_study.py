"""Validate and summarize the thesis HPR runs 0, 2, and 4 intervention matrix."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


EXPECTED_SEEDS = (9201, 9203, 9205)
EXPECTED_CONDITIONS = 36
EPISODES_PER_CELL = 20


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    rows = read_csv(data_root / "episode_results.csv")
    expected_rows = len(EXPECTED_SEEDS) * EXPECTED_CONDITIONS * EPISODES_PER_CELL
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} episode rows, found {len(rows)}")

    groups: dict[tuple[int, str], list[int]] = defaultdict(list)
    for row in rows:
        seed = int(row["training_seed"])
        if seed not in EXPECTED_SEEDS:
            raise RuntimeError(f"Unexpected training seed: {seed}")
        raw_success = row.get("success_kinematic", row.get("success", ""))
        success = str(raw_success).strip().lower() in {"1", "true", "yes"}
        groups[(seed, row["condition_id"])].append(int(success))
    if len(groups) != len(EXPECTED_SEEDS) * EXPECTED_CONDITIONS:
        raise RuntimeError(f"Expected 108 policy-condition cells, found {len(groups)}")
    if any(len(values) != EPISODES_PER_CELL for values in groups.values()):
        raise RuntimeError("At least one policy-condition cell does not contain 20 paired resets")
    expected_baseline_counts = {9201: 20, 9203: 7, 9205: 20}
    for seed, expected in expected_baseline_counts.items():
        actual = sum(groups[(seed, "BASELINE")])
        if actual != expected:
            raise RuntimeError(
                f"BASELINE strict success count mismatch for seed {seed}: "
                f"{actual} != archived {expected}"
            )

    summary_rows = [
        {
            "training_seed": seed,
            "formal_run": {9201: 0, 9203: 2, 9205: 4}[seed],
            "condition_id": condition,
            "rolling_successes": sum(values),
            "episodes": len(values),
            "success_rate": sum(values) / len(values),
        }
        for (seed, condition), values in sorted(groups.items())
    ]
    csv_path = data_root / "three_run_condition_summary.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    payload = {
        "schema": "thesis_hpr_three_run_summary/v1",
        "training_seeds": list(EXPECTED_SEEDS),
        "formal_runs": [0, 2, 4],
        "condition_count_per_run": EXPECTED_CONDITIONS,
        "paired_resets_per_condition": EPISODES_PER_CELL,
        "episode_rows": len(rows),
        "policy_condition_cells": len(groups),
        "summary_csv": str(csv_path),
    }
    json_path = data_root / "THREE_RUN_VALIDATION_PASS.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
