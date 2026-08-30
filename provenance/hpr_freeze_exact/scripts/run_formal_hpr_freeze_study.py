from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


STUDY_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = STUDY_ROOT / "data"
FORMAL_ROOT = Path(
    r"C:\Users\PUBLIC_USER\CloudStorage\Desktop\finalproject\job\roll_learning"
    r"\obs2_roll_repro_v2_1_formal_20260803_r2"
)
FORMAL_CONFIG = FORMAL_ROOT / "_control" / "experiment_config.json"
MECHANISM_RUNTIME_ROOT = Path(
    r"C:\Users\PUBLIC_USER\Documents\Graduate_Thesis_Project"
    r"\obs2_v2_1_k_mechanism_20260804"
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


formal_config = load_json(FORMAL_CONFIG)
SITE_PACKAGES = Path(formal_config["runtime"]["site_packages"])
for import_path in (SITE_PACKAGES, MECHANISM_RUNTIME_ROOT):
    value = str(import_path)
    if value not in sys.path:
        sys.path.insert(0, value)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from mechanism_rollout import SeedRuntime  # noqa: E402


torch.set_num_threads(1)
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

EXPECTED_CHECKPOINT_HASHES = {
    9201: "b697425ffa994ccc4ce32db29573e6f87576c3c16ce39ea35a50a38a183e7ea6",
    9205: "0edb440aa10351b24f8208880b208fc66ebf03cb548e803a7e45f85062eb9383",
}
BASE_RESET_SEED = 20264101
EPISODES = 20
STEPS = 1000
IDENTITY_TOLERANCE = 1e-6


@dataclass(frozen=True)
class FreezeCondition:
    id: str
    family: str
    description: str
    op: str
    joint: int | None = None

    @property
    def spec(self) -> dict[str, Any]:
        return {"op": self.op, "joint": self.joint}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "family": self.family,
            "description": self.description,
            "op": self.op,
            "joint_zero_based": self.joint,
            "joint_one_based": None if self.joint is None else self.joint + 1,
        }


def build_conditions(matrix: str) -> list[FreezeCondition]:
    conditions = [
        FreezeCondition("BASELINE", "global", "Unmodified learned HPR outputs", "baseline"),
        FreezeCondition("GLOBAL_K1_OFF", "global", "Disable learned K1 at all joints", "global_k1_off"),
        FreezeCondition("GLOBAL_K2_OFF", "global", "Disable learned K2 at all joints", "global_k2_off"),
        FreezeCondition("GLOBAL_BOTH_OFF", "global", "Disable learned K1 and K2 at all joints", "global_both_off"),
    ]
    for joint in range(8):
        conditions.append(
            FreezeCondition(
                f"J{joint + 1:02d}_BOTH_OFF",
                "whole_joint_ablation",
                f"Disable both learned channels at J{joint + 1:02d}",
                "joint_both_off",
                joint,
            )
        )
    for joint in range(8):
        conditions.append(
            FreezeCondition(
                f"J{joint + 1:02d}_ONLY",
                "single_joint_retention",
                f"Retain only J{joint + 1:02d}; disable both learned channels at the other joints",
                "joint_only",
                joint,
            )
        )
    if matrix == "full":
        for joint in range(8):
            conditions.append(
                FreezeCondition(
                    f"J{joint + 1:02d}_K1_OFF",
                    "joint_k1_ablation",
                    f"Disable only learned K1 at J{joint + 1:02d}",
                    "joint_k1_off",
                    joint,
                )
            )
        for joint in range(8):
            conditions.append(
                FreezeCondition(
                    f"J{joint + 1:02d}_K2_OFF",
                    "joint_k2_ablation",
                    f"Disable only learned K2 at J{joint + 1:02d}",
                    "joint_k2_off",
                    joint,
                )
            )
    expected = 36 if matrix == "full" else 20
    if len(conditions) != expected or len({condition.id for condition in conditions}) != expected:
        raise RuntimeError(f"Condition matrix construction failed: {len(conditions)} != {expected}")
    return conditions


class HprFreezeRuntime(SeedRuntime):
    """Formal runtime with interventions applied only to the HPR actor output."""

    def actor_actions(self, td: Any) -> tuple[torch.Tensor, torch.Tensor, float]:
        action_td = self.choose_action(
            self.r0_policy, td.clone(recurse=True), "deterministic"
        )
        action = action_td["agents", "action"].detach().clone()
        if tuple(action.shape) != (1, 8, 2):
            raise RuntimeError(f"Unexpected HPR action shape: {tuple(action.shape)}")
        if not bool(torch.isfinite(action).all().item()):
            raise RuntimeError("HPR actor output contains NaN or Inf")
        # The inherited rollout computes action deltas against its second return.
        # Returning a clone of the HPR baseline makes that diagnostic HPR-relative.
        return action, action.clone(), 0.0

    def apply_condition(
        self,
        condition: FreezeCondition,
        r0: torch.Tensor,
        roll: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        del roll, step
        result = r0.clone()
        op = condition.op
        joint = condition.joint
        if op == "baseline":
            pass
        elif op == "global_k1_off":
            result[..., :, 0] = 0.0
        elif op == "global_k2_off":
            result[..., :, 1] = 0.0
        elif op == "global_both_off":
            result[..., :, :] = 0.0
        elif op == "joint_k1_off":
            result[..., int(joint), 0] = 0.0
        elif op == "joint_k2_off":
            result[..., int(joint), 1] = 0.0
        elif op == "joint_both_off":
            result[..., int(joint), :] = 0.0
        elif op == "joint_only":
            retained = result[..., int(joint), :].clone()
            result[..., :, :] = 0.0
            result[..., int(joint), :] = retained
        else:
            raise ValueError(f"Unknown intervention operation: {op}")
        if tuple(result.shape) != (1, 8, 2) or not bool(torch.isfinite(result).all().item()):
            raise RuntimeError(f"Applied-action contract failed for {condition.id}")
        return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def kinematic_success(metrics: dict[str, Any]) -> bool:
    return bool(
        float(metrics["desired_net_rotation_degrees"]) >= 360.0
        and float(metrics["desired_active_rotation_fraction"]) >= 0.70
        and float(metrics["forward_body_lengths"]) >= 1.0
    )


def historical_one_turn_success(metrics: dict[str, Any]) -> bool:
    return bool(float(metrics["desired_net_rotation_degrees"]) >= 360.0)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def compare_baseline_to_official(training_seed: int, episodes: list[dict[str, Any]]) -> dict[str, Any]:
    official_path = (
        FORMAL_ROOT
        / "formal"
        / "evaluations"
        / f"formal__seed{training_seed}__R0__eval_attempt1.json"
    )
    official_payload = load_json(official_path)
    official_episodes = official_payload["results"][0]["episodes"]
    if len(official_episodes) != EPISODES or len(episodes) != EPISODES:
        raise RuntimeError("Baseline identity episode count mismatch")
    official_by_seed = {int(row["seed"]): row for row in official_episodes}
    fields = (
        "initial_body_length",
        "forward_displacement",
        "forward_body_lengths",
        "net_best_fit_rotation_degrees",
        "desired_net_rotation_degrees",
        "desired_active_rotation_fraction",
    )
    maximum_error = {field: 0.0 for field in fields}
    for row in episodes:
        reset_seed = int(row["reset_seed"])
        expected = official_by_seed.get(reset_seed)
        if expected is None:
            raise RuntimeError(f"Official baseline missing reset seed {reset_seed}")
        for field in fields:
            error = abs(float(row[field]) - float(expected[field]))
            maximum_error[field] = max(maximum_error[field], error)
            if error > IDENTITY_TOLERANCE:
                raise RuntimeError(
                    f"Baseline identity mismatch for seed {training_seed}, reset {reset_seed}, "
                    f"{field}: error={error:.3g}"
                )
    if sum(bool(row["success_kinematic"]) for row in episodes) != 20:
        raise RuntimeError(f"Formal HPR baseline for seed {training_seed} did not reproduce 20/20")
    return {
        "official_evaluation": str(official_path),
        "episode_count": EPISODES,
        "all_common_criterion_success": True,
        "absolute_tolerance": IDENTITY_TOLERANCE,
        "maximum_absolute_error": maximum_error,
    }


def metric_row(
    training_seed: int,
    condition: FreezeCondition,
    episode_index: int,
    reset_seed: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "training_seed": training_seed,
        "condition_id": condition.id,
        "condition_family": condition.family,
        "condition_description": condition.description,
        "joint_zero_based": condition.joint,
        "joint_one_based": None if condition.joint is None else condition.joint + 1,
        "episode_index": episode_index,
        "reset_seed": reset_seed,
        "success_kinematic": kinematic_success(metrics),
        "success_rotation_only_sensitivity": historical_one_turn_success(metrics),
        "initial_body_length": float(metrics["initial_body_length"]),
        "forward_displacement": float(metrics["forward_displacement"]),
        "forward_body_lengths": float(metrics["forward_body_lengths"]),
        "net_best_fit_rotation_degrees": float(metrics["net_best_fit_rotation_degrees"]),
        "desired_net_rotation_degrees": float(metrics["desired_net_rotation_degrees"]),
        "desired_revolutions": float(metrics["desired_net_rotation_degrees"]) / 360.0,
        "desired_active_rotation_fraction": float(metrics["desired_active_rotation_fraction"]),
        "roll_pulse_count_diagnostic_only": int(metrics["roll_pulse_count"]),
        "contact_metric_source_diagnostic_only": str(metrics["contact_metric_source"]),
        "mean_abs_k1": float(np.mean(np.asarray(metrics["joint_summary"]["K_abs_mean"])[:, 0])),
        "mean_abs_k2": float(np.mean(np.asarray(metrics["joint_summary"]["K_abs_mean"])[:, 1])),
        "mean_torque_boundary_saturation_fraction": float(
            np.mean(metrics["joint_summary"]["torque_boundary_saturation_fraction"])
        ),
    }


def condition_summary(condition: FreezeCondition, episodes: list[dict[str, Any]], seconds: float) -> dict[str, Any]:
    def mean(field: str) -> float:
        return float(np.mean([float(row[field]) for row in episodes]))

    def sample_sd(field: str) -> float:
        return float(np.std([float(row[field]) for row in episodes], ddof=1))

    return {
        **condition.to_dict(),
        "episode_count": len(episodes),
        "kinematic_success_count": int(sum(bool(row["success_kinematic"]) for row in episodes)),
        "kinematic_success_rate": float(np.mean([bool(row["success_kinematic"]) for row in episodes])),
        "rotation_only_success_count": int(
            sum(bool(row["success_rotation_only_sensitivity"]) for row in episodes)
        ),
        "desired_revolutions_mean": mean("desired_revolutions"),
        "desired_revolutions_sample_sd": sample_sd("desired_revolutions"),
        "direction_fraction_mean": mean("desired_active_rotation_fraction"),
        "direction_fraction_sample_sd": sample_sd("desired_active_rotation_fraction"),
        "forward_body_lengths_mean": mean("forward_body_lengths"),
        "forward_body_lengths_sample_sd": sample_sd("forward_body_lengths"),
        "mean_abs_k1": mean("mean_abs_k1"),
        "mean_abs_k2": mean("mean_abs_k2"),
        "mean_torque_boundary_saturation_fraction": mean(
            "mean_torque_boundary_saturation_fraction"
        ),
        "wall_seconds": seconds,
    }


def run_training_seed(training_seed: int, matrix: str, pilot: bool = False) -> dict[str, Any]:
    conditions = build_conditions(matrix)
    if pilot:
        conditions = conditions[:1]
    seed_dir = DATA_ROOT / "raw" / f"seed{training_seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    runtime = HprFreezeRuntime(training_seed, stage="identity", environment_arm="R0")
    expected_checkpoint_hash = EXPECTED_CHECKPOINT_HASHES[training_seed]
    actual_checkpoint_hash = sha256_file(runtime.r0_checkpoint)
    if actual_checkpoint_hash != expected_checkpoint_hash:
        runtime.close()
        raise RuntimeError(
            f"Formal checkpoint hash drift for seed {training_seed}: {actual_checkpoint_hash}"
        )

    started = time.perf_counter()
    summaries: list[dict[str, Any]] = []
    identity_audit: dict[str, Any] | None = None
    try:
        for condition_index, condition in enumerate(conditions, start=1):
            output_path = seed_dir / f"{condition.id}.json"
            condition_started = time.perf_counter()
            episode_rows: list[dict[str, Any]] = []
            for episode_index in range(1, EPISODES + 1):
                reset_seed = BASE_RESET_SEED + episode_index - 1
                metrics, _ = runtime.run_episode(condition, reset_seed)
                row = metric_row(
                    training_seed, condition, episode_index, reset_seed, metrics
                )
                if not all(
                    math.isfinite(float(row[field]))
                    for field in (
                        "forward_body_lengths",
                        "desired_net_rotation_degrees",
                        "desired_active_rotation_fraction",
                        "mean_abs_k1",
                        "mean_abs_k2",
                    )
                ):
                    raise RuntimeError(
                        f"Non-finite metric in {condition.id}, reset {reset_seed}"
                    )
                episode_rows.append(row)
            elapsed = time.perf_counter() - condition_started
            summary = condition_summary(condition, episode_rows, elapsed)
            if condition.id == "BASELINE":
                identity_audit = compare_baseline_to_official(training_seed, episode_rows)
                identity_audit["checkpoint_sha256"] = actual_checkpoint_hash
                atomic_json(seed_dir / "BASELINE_IDENTITY_AUDIT.json", identity_audit)
            elif identity_audit is None:
                raise RuntimeError("Intervention evaluated before baseline identity gate")
            atomic_json(
                output_path,
                {
                    "schema": "formal_hpr_freeze_validation/condition/v1",
                    "training_seed": training_seed,
                    "condition": condition.to_dict(),
                    "summary": summary,
                    "episodes": episode_rows,
                },
            )
            summaries.append(summary)
            print(
                f"[seed {training_seed}] {condition_index:02d}/{len(conditions):02d} "
                f"{condition.id}: {summary['kinematic_success_count']}/{EPISODES}, "
                f"{elapsed:.1f} s",
                flush=True,
            )
        integrity = runtime.verify_unchanged()
    finally:
        runtime.close()

    seed_manifest = {
        "schema": "formal_hpr_freeze_validation/seed_manifest/v1",
        "training_seed": training_seed,
        "matrix": matrix,
        "pilot": pilot,
        "formal_checkpoint": str(runtime.r0_checkpoint),
        "checkpoint_sha256": actual_checkpoint_hash,
        "base_reset_seed": BASE_RESET_SEED,
        "episode_count_per_condition": EPISODES,
        "steps_per_episode": STEPS,
        "condition_count": len(conditions),
        "total_rollouts": len(conditions) * EPISODES,
        "baseline_identity_audit": identity_audit,
        "post_run_integrity": integrity,
        "wall_seconds": time.perf_counter() - started,
        "condition_summaries": summaries,
    }
    atomic_json(seed_dir / "SEED_MANIFEST.json", seed_manifest)
    return seed_manifest


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
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_outputs(training_seeds: list[int], matrix: str, pilot: bool) -> dict[str, Any]:
    episode_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    seed_manifests: list[dict[str, Any]] = []
    conditions = build_conditions(matrix)
    if pilot:
        conditions = conditions[:1]
    for training_seed in training_seeds:
        seed_dir = DATA_ROOT / "raw" / f"seed{training_seed}"
        seed_manifests.append(load_json(seed_dir / "SEED_MANIFEST.json"))
        for condition in conditions:
            payload = load_json(seed_dir / f"{condition.id}.json")
            summary_rows.append(
                {"training_seed": training_seed, **payload["summary"]}
            )
            episode_rows.extend(payload["episodes"])
    write_csv(DATA_ROOT / "episode_results.csv", episode_rows)
    write_csv(DATA_ROOT / "condition_policy_summary.csv", summary_rows)
    result = {
        "schema": "formal_hpr_freeze_validation/study_manifest/v1",
        "matrix": matrix,
        "pilot": pilot,
        "training_seeds": training_seeds,
        "condition_count": len(conditions),
        "episode_count_per_condition": EPISODES,
        "total_rollouts": len(episode_rows),
        "steps_per_rollout": STEPS,
        "common_kinematic_criterion": {
            "minimum_desired_net_rotation_degrees": 360.0,
            "minimum_desired_active_rotation_fraction": 0.70,
            "minimum_forward_body_lengths": 1.0,
            "pulse_or_contact_gate_used": False,
        },
        "seed_manifests": seed_manifests,
    }
    atomic_json(DATA_ROOT / ("PILOT_MANIFEST.json" if pilot else "STUDY_MANIFEST.json"), result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[9201, 9205])
    parser.add_argument("--matrix", choices=("core", "full"), default="full")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(seed) for seed in args.training_seeds]
    if any(seed not in EXPECTED_CHECKPOINT_HASHES for seed in seeds):
        raise ValueError(f"Only formal HPR seeds 9201 and 9205 are permitted: {seeds}")
    started = time.perf_counter()
    manifests: list[dict[str, Any]] = []
    if args.workers <= 1 or len(seeds) == 1:
        for seed in seeds:
            manifests.append(run_training_seed(seed, args.matrix, args.pilot))
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(seeds))) as pool:
            futures = {
                pool.submit(run_training_seed, seed, args.matrix, args.pilot): seed
                for seed in seeds
            }
            for future in as_completed(futures):
                manifests.append(future.result())
    result = aggregate_outputs(seeds, args.matrix, args.pilot)
    result["wall_seconds_current_execution"] = time.perf_counter() - started
    result["worker_manifests"] = manifests
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
