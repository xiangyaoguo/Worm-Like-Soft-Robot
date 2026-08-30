# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


ALLOWED_LABELS = {
    "crawling",
    "rolling",
    "partial_roll_or_rocking",
    "sliding",
    "failed_or_other",
    "uncertain",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        if not rows:
            raise ValueError(f"No rows for {path}")
        fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge two independent anonymous gait reviews.")
    parser.add_argument("--rater-1", type=Path, required=True)
    parser.add_argument("--rater-2", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--episode-features", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path)
    return parser.parse_args()


def normalize_form(path: Path, reviewer_id: str) -> dict[str, dict[str, str]]:
    rows = read_csv(path)
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        anonymous_id = row.get("anonymous_id", "").strip()
        label = row.get("rater_label", "").strip().lower()
        if not anonymous_id:
            raise ValueError(f"Missing anonymous_id in {path}")
        if anonymous_id in result:
            raise ValueError(f"Duplicate anonymous_id {anonymous_id} in {path}")
        if label not in ALLOWED_LABELS:
            raise ValueError(
                f"{reviewer_id}: {anonymous_id} has invalid or blank label {label!r}; "
                f"allowed={sorted(ALLOWED_LABELS)}"
            )
        result[anonymous_id] = row
    return result


def cohen_kappa(labels_1: list[str], labels_2: list[str]) -> tuple[float, float, float, float]:
    total = len(labels_1)
    agreement = sum(left == right for left, right in zip(labels_1, labels_2)) / total
    counts_1 = Counter(labels_1)
    counts_2 = Counter(labels_2)
    expected = sum((counts_1[label] / total) * (counts_2[label] / total) for label in ALLOWED_LABELS)
    kappa = (agreement - expected) / (1.0 - expected) if expected < 1.0 else 1.0
    crawl_1 = sum(label == "crawling" for label in labels_1)
    crawl_2 = sum(label == "crawling" for label in labels_2)
    crawl_agree = sum(
        left == "crawling" and right == "crawling"
        for left, right in zip(labels_1, labels_2)
    )
    positive_agreement = (2.0 * crawl_agree / (crawl_1 + crawl_2)) if crawl_1 + crawl_2 else 1.0
    return agreement, expected, kappa, positive_agreement


def main() -> None:
    args = parse_args()
    rater_1 = normalize_form(args.rater_1, "rater_1")
    rater_2 = normalize_form(args.rater_2, "rater_2")
    if set(rater_1) != set(rater_2):
        raise ValueError("The two rating forms do not contain the same anonymous IDs")

    anonymous_ids = sorted(rater_1)
    labels_1 = [rater_1[item]["rater_label"].strip().lower() for item in anonymous_ids]
    labels_2 = [rater_2[item]["rater_label"].strip().lower() for item in anonymous_ids]
    raw_agreement, expected_agreement, kappa, positive_agreement = cohen_kappa(labels_1, labels_2)

    confusion_rows = []
    confusion = Counter(zip(labels_1, labels_2))
    for left in sorted(ALLOWED_LABELS):
        for right in sorted(ALLOWED_LABELS):
            confusion_rows.append(
                {"rater_1_label": left, "rater_2_label": right, "count": confusion[(left, right)]}
            )
    write_csv(args.output_root / "reviewer_confusion_matrix.csv", confusion_rows)

    agreement_summary = [
        {
            "reviewed_episodes": len(anonymous_ids),
            "raw_agreement": raw_agreement,
            "chance_expected_agreement": expected_agreement,
            "cohen_kappa": kappa,
            "crawling_positive_agreement": positive_agreement,
            "disagreement_count": sum(left != right for left, right in zip(labels_1, labels_2)),
        }
    ]
    write_csv(args.output_root / "reviewer_agreement.csv", agreement_summary)

    adjudication_lookup: dict[str, dict[str, str]] = {}
    if args.adjudication:
        adjudication_lookup = {row["anonymous_id"]: row for row in read_csv(args.adjudication)}
    adjudication_rows: list[dict[str, Any]] = []
    resolved_labels: dict[str, str] = {}
    unresolved = 0
    for anonymous_id, left, right in zip(anonymous_ids, labels_1, labels_2):
        agreement = left == right
        adjudicated_label = left if agreement else ""
        reason = "independent_rater_agreement" if agreement else ""
        if not agreement and anonymous_id in adjudication_lookup:
            source = adjudication_lookup[anonymous_id]
            adjudicated_label = source.get("adjudicated_label", "").strip().lower()
            reason = source.get("adjudication_reason", "").strip()
            if adjudicated_label not in ALLOWED_LABELS:
                raise ValueError(f"Invalid adjudicated label for {anonymous_id}: {adjudicated_label!r}")
            if not reason:
                raise ValueError(f"Missing adjudication reason for {anonymous_id}")
        if not adjudicated_label:
            unresolved += 1
        else:
            resolved_labels[anonymous_id] = adjudicated_label
        adjudication_rows.append(
            {
                "anonymous_id": anonymous_id,
                "rater_1_label": left,
                "rater_2_label": right,
                "agreement": agreement,
                "adjudicated_label": adjudicated_label,
                "adjudication_reason": reason,
            }
        )
    write_csv(args.output_root / "adjudication.csv", adjudication_rows)

    status = {
        "status": "complete" if unresolved == 0 else "pending_adjudication",
        "reviewed_episodes": len(anonymous_ids),
        "unresolved_disagreements": unresolved,
        "raw_agreement": raw_agreement,
        "cohen_kappa": kappa,
        "crawling_positive_agreement": positive_agreement,
    }
    if unresolved:
        (args.output_root / "BLIND_REVIEW_STATUS.json").write_text(
            json.dumps(status, indent=2), encoding="utf-8"
        )
        print(json.dumps(status, indent=2))
        return

    key_rows = read_csv(args.private_key)
    key = {row["anonymous_id"]: row for row in key_rows}
    features = {row["episode_id"]: row for row in read_csv(args.episode_features)}
    final_rows: list[dict[str, Any]] = []
    for anonymous_id in anonymous_ids:
        identity = key[anonymous_id]
        feature = features[identity["episode_id"]]
        human_label = resolved_labels[anonymous_id]
        automatic_label = feature["automated_gait_label"]
        if automatic_label == "formal_rolling":
            final_label = "formal_rolling"
            reason = "unchanged_formal_rolling_gate"
        elif automatic_label in {"partial_roll", "rocking"}:
            final_label = automatic_label
            reason = "automatic_rotation_exclusion_gate"
        elif automatic_label == "crawling_candidate" and human_label == "crawling":
            final_label = "confirmed_crawling"
            reason = "automatic_candidate_and_human_consensus"
        elif human_label == "sliding":
            final_label = "sliding"
            reason = "human_consensus"
        else:
            final_label = "failed_or_other"
            reason = "not_jointly_confirmed_as_crawling"
        final_rows.append(
            {
                **feature,
                "anonymous_id": anonymous_id,
                "rater_1_label": rater_1[anonymous_id]["rater_label"].strip().lower(),
                "rater_2_label": rater_2[anonymous_id]["rater_label"].strip().lower(),
                "adjudicated_human_label": human_label,
                "final_gait_label": final_label,
                "final_label_reason": reason,
            }
        )
    write_csv(args.output_root / "episode_classification_final.csv", final_rows)
    (args.output_root / "BLIND_REVIEW_STATUS.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
