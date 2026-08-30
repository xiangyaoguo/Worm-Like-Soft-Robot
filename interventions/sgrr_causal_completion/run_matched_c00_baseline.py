"""Run the prospectively frozen, main-seed-matched C00/R0 baseline.

This is a supplemental evaluation-only reference for the eight already frozen
``C00_PAIR_SUFF_Jxx`` interventions.  It does not add a condition to, remove a
condition from, or otherwise edit the 113-condition causal-completion contract.
It never trains, resumes, or writes a checkpoint.  All outputs are confined to
``matched_c00`` below this script's directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


SOURCE_ROOT = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("THESIS_SGRR_OUTPUT", str(SOURCE_ROOT))).resolve()
AMENDMENT_PATH = SOURCE_ROOT / "MATCHED_C00_BASELINE_AMENDMENT.json"
sys.path.insert(0, str(SOURCE_ROOT))
import run_causal_completion as completion  # type: ignore  # noqa: E402


CONFIG = completion.CONFIG
CONTRACT = completion.CONTRACT
OUTPUT_ROOT = ROOT / "matched_c00"
RESULT_ROOT = OUTPUT_ROOT / "results"
TRACE_ROOT = OUTPUT_ROOT / "traces"
MATCHED_CONDITION = completion.Condition(
    "C00",
    "supplemental_matched_c00_reference",
    (
        "Complete R0 K1 and K2 evaluated on the exact main causal initial-state "
        "set used by C00_PAIR_SUFF_J01..J08."
    ),
    {"op": "matched_c00_baseline", "source": "complete_R0_K1_and_K2"},
)


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


MATCHED_CONDITION_SHA256 = _canonical_sha256(MATCHED_CONDITION.to_dict())


class MatchedC00Runtime(completion.CompletionRuntime):
    """Completion runtime whose sole supplemental action is the full R0 action."""

    def apply_completion_condition(
        self,
        condition: completion.Condition,
        r0: torch.Tensor,
        roll: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        if condition.spec.get("op") != "matched_c00_baseline":
            return super().apply_completion_condition(condition, r0, roll, step)
        result = r0.clone()
        if tuple(result.shape) != (1, 8, 2):
            raise RuntimeError(f"Matched C00 action shape drift: {tuple(result.shape)}")
        if not bool(torch.isfinite(result).all().item()):
            raise RuntimeError("Matched C00 action contains NaN/Inf")
        if not bool(torch.equal(result, r0)):
            raise RuntimeError("Matched C00 action is not bitwise equal to full R0 K1+K2")
        return result


def _load_amendment() -> dict[str, Any]:
    if not AMENDMENT_PATH.is_file():
        raise FileNotFoundError(f"Missing prospective amendment: {AMENDMENT_PATH}")
    payload = completion.load_json(AMENDMENT_PATH)
    if payload.get("schema") != "obs2_v2_1_k_causal_completion/matched_c00_amendment/v1":
        raise RuntimeError("Matched-C00 amendment schema drift")
    if payload.get("parent_study_id") != CONFIG["study_id"]:
        raise RuntimeError("Matched-C00 amendment parent study drift")
    if payload.get("status") != "prospectively_frozen_before_joint_pair_sufficiency_outcomes":
        raise RuntimeError("Matched-C00 amendment was not prospectively frozen")
    return payload


def _resolve_frozen_source(relative_path: str) -> Path:
    archive_root = Path(
        os.environ.get("THESIS_SGRR_ARCHIVE_SOURCE", str(SOURCE_ROOT))
    ).resolve()
    candidate = (archive_root / relative_path).resolve()
    try:
        candidate.relative_to(archive_root)
    except ValueError as exc:
        raise RuntimeError(f"Frozen source escapes archive root: {relative_path}") from exc
    return candidate


def verify_amendment() -> dict[str, Any]:
    """Fail closed on amendment, source, contract, or seed-set drift."""

    amendment = _load_amendment()
    failures: list[str] = []
    verified: list[dict[str, str]] = []
    for relative_path, expected in amendment["source_sha256"].items():
        path = _resolve_frozen_source(relative_path)
        if not path.is_file():
            failures.append(f"missing: {path}")
            continue
        actual = completion.sha256_file(path)
        if actual.lower() != str(expected).lower():
            failures.append(f"hash drift: {path}: {actual} != {expected}")
        verified.append({"path": relative_path, "sha256": actual})
    if failures:
        raise RuntimeError("Matched-C00 frozen-source audit failed:\n" + "\n".join(failures))

    baseline = amendment["matched_baseline"]
    expected_seeds = [int(value) for value in CONFIG["training_seeds"]]
    expected_base = int(CONFIG["main_evaluation"]["base_seed"])
    expected_episodes = int(CONFIG["main_evaluation"]["episodes"])
    expected_steps = int(CONFIG["main_evaluation"]["steps"])
    checks = {
        "training_seeds": baseline.get("training_seeds") == expected_seeds,
        "evaluation_base_seed": baseline.get("evaluation_base_seed") == expected_base,
        "episodes": baseline.get("episodes_per_training_seed") == expected_episodes,
        "steps": baseline.get("steps_per_episode") == expected_steps,
        "condition": baseline.get("condition") == MATCHED_CONDITION.to_dict(),
        "condition_sha256": baseline.get("condition_sha256") == MATCHED_CONDITION_SHA256,
        "parent_condition_count": amendment.get("parent_condition_count_unchanged")
        == 113,
        "parent_condition_sha256": amendment.get("parent_condition_sha256_unchanged")
        == completion.canonical_condition_sha256(),
        "success_thresholds": amendment.get("episode_success")
        == CONFIG["episode_success"],
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    if failed_checks:
        raise RuntimeError(
            "Matched-C00 contract checks failed: " + ", ".join(failed_checks)
        )
    parent_audit = completion.verify_source_manifest()
    return {
        "passed": True,
        "amendment_sha256": completion.sha256_file(AMENDMENT_PATH),
        "matched_condition_sha256": MATCHED_CONDITION_SHA256,
        "verified_source_count": len(verified),
        "verified_sources": verified,
        "parent_source_audit_passed": bool(parent_audit["passed"]),
        "parent_condition_count_unchanged": len(completion.CONDITIONS),
        "parent_condition_sha256_unchanged": completion.canonical_condition_sha256(),
    }


def _expected_evaluation_seeds() -> list[int]:
    base = int(CONFIG["main_evaluation"]["base_seed"])
    episodes = int(CONFIG["main_evaluation"]["episodes"])
    return list(range(base, base + episodes))


def _trace_path(seed: int, episode_seed: int) -> Path:
    return (
        TRACE_ROOT
        / f"seed{seed}"
        / f"{MATCHED_CONDITION.id}__evalseed{episode_seed}.npz"
    )


def _result_path(seed: int) -> Path:
    return RESULT_ROOT / f"seed{seed}" / "C00.json"


def _validate_existing_result(seed: int) -> dict[str, Any] | None:
    path = _result_path(seed)
    if not path.is_file():
        return None
    payload = completion.load_json(path)
    amendment_hash = completion.sha256_file(AMENDMENT_PATH)
    expected_seeds = _expected_evaluation_seeds()
    valid = (
        payload.get("schema")
        == "obs2_v2_1_k_causal_completion/matched_c00_seed/v1"
        and payload.get("parent_study_id") == CONFIG["study_id"]
        and payload.get("training_seed") == seed
        and payload.get("condition") == MATCHED_CONDITION.to_dict()
        and payload.get("condition_sha256") == MATCHED_CONDITION_SHA256
        and payload.get("amendment_sha256") == amendment_hash
        and payload.get("parent_condition_sha256")
        == completion.canonical_condition_sha256()
        and payload.get("evaluation_base_seed")
        == int(CONFIG["main_evaluation"]["base_seed"])
        and payload.get("evaluation_episodes")
        == int(CONFIG["main_evaluation"]["episodes"])
        and payload.get("evaluation_steps")
        == int(CONFIG["main_evaluation"]["steps"])
        and [item.get("seed") for item in payload.get("episodes", [])]
        == expected_seeds
        and payload.get("success_episodes")
        == sum(bool(item.get("success")) for item in payload.get("episodes", []))
        and payload.get("checkpoint_sha256")
        == CONTRACT["checkpoint_sha256"][str(seed)]
    )
    if not valid:
        raise RuntimeError(f"Refusing incompatible existing matched-C00 result: {path}")
    traces = payload.get("trace_files", [])
    if len(traces) != 1:
        raise RuntimeError(f"Matched-C00 trace receipt count drift: {path}")
    trace = traces[0]
    trace_path = ROOT / trace["path"]
    if not trace_path.is_file() or completion.sha256_file(trace_path) != trace["sha256"]:
        raise RuntimeError(f"Matched-C00 trace missing or hash-drifted: {trace_path}")
    immutability = payload.get("immutability", {})
    if not (
        immutability.get("checkpoints_before_after_equal") is True
        and immutability.get("policies_before_after_equal") is True
    ):
        raise RuntimeError(f"Matched-C00 stored immutability gate is not passed: {path}")
    return payload


def run_seed(seed: int) -> dict[str, Any]:
    """Evaluate 20 matched main initial states for one immutable training seed."""

    allowed = {int(value) for value in CONFIG["training_seeds"]}
    if seed not in allowed:
        raise ValueError(f"Training seed is outside frozen contract: {seed}")
    source_audit = verify_amendment()
    existing = _validate_existing_result(seed)
    if existing is not None:
        return {
            "training_seed": seed,
            "status": "already_complete_and_hash_validated",
            "success_episodes": int(existing["success_episodes"]),
            "result_path": str(_result_path(seed).relative_to(ROOT)),
            "result_sha256": completion.sha256_file(_result_path(seed)),
        }

    completion._set_torch_threads()
    runtime = MatchedC00Runtime(seed, "main", environment_arm="Rroll")
    completion._validate_runtime_hashes(runtime)
    expected_seeds = _expected_evaluation_seeds()
    records: list[dict[str, Any]] = []
    trace_files: list[dict[str, str]] = []
    try:
        for index, episode_seed in enumerate(expected_seeds):
            metrics, trace = completion.run_rollout(
                runtime,
                MATCHED_CONDITION,
                episode_seed,
                capture_trace=index == 0,
            )
            if metrics["condition_id"] != MATCHED_CONDITION.id:
                raise RuntimeError("Frozen evaluator returned the wrong condition identity")
            records.append(metrics)
            if trace is not None:
                trace.update(
                    {
                        "amendment_sha256": np.asarray(
                            source_audit["amendment_sha256"]
                        ),
                        "matched_condition_sha256": np.asarray(
                            MATCHED_CONDITION_SHA256
                        ),
                        "parent_condition_sha256": np.asarray(
                            completion.canonical_condition_sha256()
                        ),
                    }
                )
                path = _trace_path(seed, episode_seed)
                completion.atomic_npz(path, **trace)
                trace_files.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "sha256": completion.sha256_file(path),
                    }
                )
        if len(records) != 20 or len(trace_files) != 1:
            raise RuntimeError("Matched-C00 episode or trace count gate failed")
        if [int(item["seed"]) for item in records] != expected_seeds:
            raise RuntimeError("Matched-C00 main initial-state pairing gate failed")
        immutability = runtime.verify_unchanged()
        payload = {
            "schema": "obs2_v2_1_k_causal_completion/matched_c00_seed/v1",
            "parent_study_id": CONFIG["study_id"],
            "supplemental_study_id": _load_amendment()["supplemental_study_id"],
            "training_seed": seed,
            "condition": MATCHED_CONDITION.to_dict(),
            "condition_sha256": MATCHED_CONDITION_SHA256,
            "amendment_sha256": source_audit["amendment_sha256"],
            "parent_condition_count": len(completion.CONDITIONS),
            "parent_condition_sha256": completion.canonical_condition_sha256(),
            "evaluation_base_seed": int(CONFIG["main_evaluation"]["base_seed"]),
            "evaluation_episodes": int(CONFIG["main_evaluation"]["episodes"]),
            "evaluation_steps": runtime.steps,
            "evaluation_seeds": expected_seeds,
            "environment_arm": "Rroll",
            "applied_controller": "complete_R0_K1_and_K2",
            "success_episodes": int(sum(bool(item["success"]) for item in records)),
            "episode_success": CONFIG["episode_success"],
            "episodes": records,
            "trace_files": trace_files,
            "checkpoint_sha256": runtime.checkpoint_hashes_before,
            "policy_state_sha256": runtime.policy_hashes_before,
            "frozen_evaluator_sha256": completion.sha256_file(
                runtime.frozen_eval_path
            ),
            "source_audit": source_audit,
            "immutability": immutability,
        }
        completion.atomic_json(_result_path(seed), payload)
        return {
            "training_seed": seed,
            "status": "complete",
            "success_episodes": int(payload["success_episodes"]),
            "result_path": str(_result_path(seed).relative_to(ROOT)),
            "result_sha256": completion.sha256_file(_result_path(seed)),
        }
    finally:
        runtime.close()


def summarize_available() -> dict[str, Any]:
    """Write a hash index; scientific paired contrasts are computed elsewhere."""

    audit = verify_amendment()
    rows: list[dict[str, Any]] = []
    for seed in [int(value) for value in CONFIG["training_seeds"]]:
        payload = _validate_existing_result(seed)
        if payload is None:
            continue
        rows.append(
            {
                "training_seed": seed,
                "success_episodes": int(payload["success_episodes"]),
                "evaluation_episodes": int(payload["evaluation_episodes"]),
                "result_path": str(_result_path(seed).relative_to(ROOT)),
                "result_sha256": completion.sha256_file(_result_path(seed)),
                "trace_path": payload["trace_files"][0]["path"],
                "trace_sha256": payload["trace_files"][0]["sha256"],
                "R0_checkpoint_sha256": payload["checkpoint_sha256"]["R0"],
                "R0_policy_state_sha256": payload["policy_state_sha256"]["R0"],
                "immutability": payload["immutability"],
            }
        )
    summary = {
        "schema": "obs2_v2_1_k_causal_completion/matched_c00_index/v1",
        "parent_study_id": CONFIG["study_id"],
        "supplemental_study_id": _load_amendment()["supplemental_study_id"],
        "amendment_sha256": audit["amendment_sha256"],
        "parent_condition_count_unchanged": len(completion.CONDITIONS),
        "parent_condition_sha256_unchanged": completion.canonical_condition_sha256(),
        "training_seed_rows": rows,
        "completed_training_seeds": len(rows),
        "expected_training_seeds": len(CONFIG["training_seeds"]),
        "complete": len(rows) == len(CONFIG["training_seeds"]),
        "total_episodes": int(
            sum(int(item["evaluation_episodes"]) for item in rows)
        ),
        "total_successes": int(sum(int(item["success_episodes"]) for item in rows)),
    }
    completion.atomic_json(RESULT_ROOT / "MATCHED_C00_INDEX.json", summary)
    if summary["complete"]:
        gate = {
            "schema": "obs2_v2_1_k_causal_completion/matched_c00_complete/v1",
            "study_id": CONFIG["study_id"],
            "supplemental_study_id": summary["supplemental_study_id"],
            "status": "complete",
            "condition_id": "C00",
            "applied_controller": "complete_R0_K1_and_K2",
            "environment_arm": "Rroll",
            "training_seeds": [int(value) for value in CONFIG["training_seeds"]],
            "evaluation_seeds": _expected_evaluation_seeds(),
            "episodes_per_training_seed": int(
                CONFIG["main_evaluation"]["episodes"]
            ),
            "steps_per_episode": int(CONFIG["main_evaluation"]["steps"]),
            "total_episodes": summary["total_episodes"],
            "total_successes": summary["total_successes"],
            "amendment_sha256": summary["amendment_sha256"],
            "parent_condition_count_unchanged": summary[
                "parent_condition_count_unchanged"
            ],
            "parent_condition_sha256_unchanged": summary[
                "parent_condition_sha256_unchanged"
            ],
            "seed_results": rows,
            "all_checkpoint_and_policy_immutability_gates_passed": all(
                row["immutability"].get("checkpoints_before_after_equal") is True
                and row["immutability"].get("policies_before_after_equal") is True
                for row in rows
            ),
        }
        if not gate["all_checkpoint_and_policy_immutability_gates_passed"]:
            raise RuntimeError("Refusing matched-C00 completion gate: immutability failed")
        completion.atomic_json(OUTPUT_ROOT / "MATCHED_C00_COMPLETE.json", gate)
    return summary


def _worker(seed: int) -> dict[str, Any]:
    return run_seed(seed)


def _parallel(seeds: Sequence[int], workers: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_worker, seed): seed for seed in seeds}
        try:
            for future in as_completed(futures):
                results.append(future.result())
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return sorted(results, key=lambda item: int(item["training_seed"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", required=True, choices=("verify", "run", "summarize", "all")
    )
    parser.add_argument("--training-seed", type=int, action="append", default=[])
    parser.add_argument("--workers", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_seeds = [int(value) for value in CONFIG["training_seeds"]]
    seeds = list(dict.fromkeys(args.training_seed)) if args.training_seed else all_seeds
    if any(seed not in all_seeds for seed in seeds):
        raise ValueError(f"Training seeds outside frozen contract: {seeds}")
    if args.workers < 1 or args.workers > len(all_seeds):
        raise ValueError("--workers must be between 1 and 5")
    if args.stage == "verify":
        payload: Any = verify_amendment()
    elif args.stage == "summarize":
        payload = summarize_available()
    else:
        results = _parallel(seeds, min(args.workers, len(seeds)))
        payload = {"runs": results, "summary": summarize_available()}
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
