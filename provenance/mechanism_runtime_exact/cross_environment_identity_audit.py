from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from mechanism_rollout import (
    ROOT,
    SeedRuntime,
    atomic_json,
    compare_metric_value,
    episode_success,
    load_json,
    metric_projection,
    policy_sha256,
    sha256_file,
)


HARD_TOLERANCE = 1.0e-10
POLICY_ARMS = ("R0", "Rroll")
ENVIRONMENT_ARMS = ("native_R0", "canonical_Rroll")
EVALUATION_SETS = (
    ("identity", "identity_gate", 20264101, 20),
    ("main", "main_evaluation", 20264201, 20),
)


def _main_result_binding(
    seeds: list[int], payloads: list[dict[str, Any]]
) -> dict[str, Any]:
    """Bind canonical main-set replays to the stored C00/C11 endpoint files."""

    by_seed = {int(item["training_seed"]): item for item in payloads}
    result_file_sha256: dict[str, str] = {}
    mismatches: list[str] = []
    checked_episodes = 0
    arm_to_condition = {"R0": "C00", "Rroll": "C11"}
    for seed in seeds:
        seed_payload = by_seed.get(seed)
        if seed_payload is None:
            mismatches.append(f"missing cross-environment payload for seed{seed}")
            continue
        for policy_arm, condition_id in arm_to_condition.items():
            relative = Path("results") / f"seed{seed}" / f"{condition_id}.json"
            path = ROOT / relative
            if not path.is_file():
                mismatches.append(f"missing stored endpoint result: {relative.as_posix()}")
                continue
            result_file_sha256[relative.as_posix()] = sha256_file(path)
            stored = load_json(path)
            stored_episodes = {
                int(item["seed"]): item for item in stored.get("episodes", [])
            }
            if len(stored_episodes) != 20:
                mismatches.append(
                    f"{relative.as_posix()} does not contain 20 unique evaluation seeds"
                )
                continue
            replay = seed_payload["arms"][policy_arm]["eval_sets"]["main"]
            for record in replay.get("episode_results", []):
                eval_seed = int(record["evaluation_seed"])
                expected = stored_episodes.get(eval_seed)
                if expected is None:
                    mismatches.append(
                        f"{relative.as_posix()} missing evaluation seed {eval_seed}"
                    )
                    continue
                actual = record["recomputed"]["canonical_Rroll"]["env_support"]
                try:
                    compare_metric_value(
                        bool(actual["success"]),
                        expected["success"],
                        HARD_TOLERANCE,
                        f"{relative.as_posix()}.seed{eval_seed}.success",
                    )
                    for key, value in actual["metrics"].items():
                        compare_metric_value(
                            value,
                            expected.get(key),
                            HARD_TOLERANCE,
                            f"{relative.as_posix()}.seed{eval_seed}.{key}",
                        )
                except Exception as exc:  # fail closed and retain every mismatch
                    mismatches.append(str(exc))
                checked_episodes += 1
    return {
        "passed": not mismatches and checked_episodes == len(seeds) * 2 * 20,
        "checked_episodes": checked_episodes,
        "expected_episodes": len(seeds) * 2 * 20,
        "result_file_sha256": result_file_sha256,
        "mismatches": mismatches,
    }


def _nested_checkpoint_hashes(snapshot: dict[str, Any], seeds: list[int]) -> dict[str, Any]:
    hashes = snapshot.get("sha256", {})
    return {
        str(seed): {
            arm: hashes.get(f"seed{seed}_{arm}") for arm in POLICY_ARMS
        }
        for seed in seeds
    }


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _capture_rng_state() -> tuple[object, tuple[Any, ...], torch.Tensor]:
    numpy_state = np.random.get_state()
    copied_numpy_state = (
        numpy_state[0],
        numpy_state[1].copy(),
        numpy_state[2],
        numpy_state[3],
        numpy_state[4],
    )
    return random.getstate(), copied_numpy_state, torch.random.get_rng_state().clone()


def _restore_rng_state(
    state: tuple[object, tuple[Any, ...], torch.Tensor]
) -> None:
    python_state, numpy_state, torch_state = state
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.random.set_rng_state(torch_state)


def _paired_call(
    native_call: Callable[[], Any], canonical_call: Callable[[], Any]
) -> tuple[Any, Any]:
    """Call both branches from the same process-level RNG state."""

    state = _capture_rng_state()
    _restore_rng_state(state)
    native_value = native_call()
    _restore_rng_state(state)
    canonical_value = canonical_call()
    return native_value, canonical_value


def _to_array(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def _positions(runtime: SeedRuntime) -> np.ndarray:
    return _to_array(runtime.frozen_eval._positions(runtime.env))


def _velocity(runtime: SeedRuntime) -> np.ndarray | None:
    value = getattr(runtime.env, "vel", None)
    return None if value is None else _to_array(value)


def _observation(td: Any) -> np.ndarray:
    return _to_array(td["agents", "observation"])


def _policy_action(
    runtime: SeedRuntime, td: Any, policy_arm: str
) -> torch.Tensor:
    if policy_arm == "R0":
        policy = runtime.r0_policy
    elif policy_arm == "Rroll":
        policy = runtime.roll_policy
    else:
        raise ValueError(f"Unsupported policy arm: {policy_arm}")
    result = runtime.choose_action(
        policy, td.clone(recurse=True), "deterministic"
    )["agents", "action"].detach().clone()
    if tuple(result.shape) != (1, 8, 2):
        raise RuntimeError(
            f"{policy_arm} action shape {tuple(result.shape)} != (1, 8, 2)"
        )
    if not bool(torch.isfinite(result).all().item()):
        raise RuntimeError(f"{policy_arm} action contains NaN/Inf")
    return result


def _max_abs_error(
    left: Any, right: Any
) -> tuple[float | None, str | None]:
    left_array = _to_array(left)
    right_array = _to_array(right)
    if left_array.shape != right_array.shape:
        return None, f"shape mismatch: {left_array.shape} != {right_array.shape}"
    if not np.isfinite(left_array).all() or not np.isfinite(right_array).all():
        return None, "non-finite value"
    if np.array_equal(left_array, right_array):
        return 0.0, None
    return float(np.max(np.abs(left_array - right_array))), None


def _compare_required(
    maxima: dict[str, float | None],
    issues: dict[str, str],
    label: str,
    left: Any,
    right: Any,
    context: str,
) -> None:
    error, issue = _max_abs_error(left, right)
    if issue is not None:
        issues.setdefault(label, f"{context}: {issue}")
        return
    assert error is not None
    current = maxima[label]
    maxima[label] = error if current is None else max(current, error)


def _compare_velocity(
    maxima: dict[str, float | None],
    issues: dict[str, str],
    availability: dict[str, int],
    native_velocity: np.ndarray | None,
    canonical_velocity: np.ndarray | None,
    context: str,
) -> None:
    if native_velocity is None and canonical_velocity is None:
        availability["both_absent_samples"] += 1
        return
    if native_velocity is None or canonical_velocity is None:
        availability["availability_mismatch_samples"] += 1
        issues.setdefault(
            "velocity",
            f"{context}: env.vel availability mismatch "
            f"(native={native_velocity is not None}, "
            f"canonical={canonical_velocity is not None})",
        )
        return
    availability["compared_samples"] += 1
    _compare_required(
        maxima,
        issues,
        "velocity",
        native_velocity,
        canonical_velocity,
        context,
    )


def _safe_hash(path: Path) -> tuple[str | None, str | None]:
    try:
        return sha256_file(path), None
    except (OSError, RuntimeError) as error:
        return None, f"{type(error).__name__}: {error}"


def _runtime_immutability(runtime: SeedRuntime) -> dict[str, Any]:
    checkpoint_after: dict[str, str | None] = {}
    checkpoint_errors: dict[str, str] = {}
    for arm, path in (
        ("R0", runtime.r0_checkpoint),
        ("Rroll", runtime.roll_checkpoint),
    ):
        digest, error = _safe_hash(path)
        checkpoint_after[arm] = digest
        if error is not None:
            checkpoint_errors[arm] = error

    policy_after: dict[str, str | None] = {}
    policy_errors: dict[str, str] = {}
    for arm, policy in (
        ("R0", runtime.r0_policy),
        ("Rroll", runtime.roll_policy),
    ):
        try:
            policy_after[arm] = policy_sha256(policy)
        except (OSError, RuntimeError, ValueError) as error:
            policy_after[arm] = None
            policy_errors[arm] = f"{type(error).__name__}: {error}"

    checkpoints_unchanged = (
        not checkpoint_errors
        and checkpoint_after == runtime.checkpoint_hashes_before
    )
    policies_unchanged = (
        not policy_errors and policy_after == runtime.policy_hashes_before
    )
    return {
        "checkpoint_sha256_before": runtime.checkpoint_hashes_before,
        "checkpoint_sha256_after": checkpoint_after,
        "checkpoint_hash_errors": checkpoint_errors,
        "checkpoints_unchanged": checkpoints_unchanged,
        "policy_state_sha256_before": runtime.policy_hashes_before,
        "policy_state_sha256_after": policy_after,
        "policy_hash_errors": policy_errors,
        "policy_states_unchanged": policies_unchanged,
    }


def _checkpoint_paths(config: dict[str, Any], seed: int) -> dict[str, Path]:
    run_root = Path(config["formal_root"]).resolve() / "formal" / "runs"
    return {
        arm: run_root / f"formal__seed{seed}__{arm}" / "checkpoint_1500.pt"
        for arm in POLICY_ARMS
    }


def _checkpoint_snapshot(paths: dict[str, Path]) -> dict[str, Any]:
    hashes: dict[str, str | None] = {}
    errors: dict[str, str] = {}
    for arm, path in paths.items():
        digest, error = _safe_hash(path)
        hashes[arm] = digest
        if error is not None:
            errors[arm] = error
    return {"sha256": hashes, "errors": errors}


def _checkpoint_snapshots_equal(
    before: dict[str, Any], after: dict[str, Any]
) -> bool:
    return (
        not before["errors"]
        and not after["errors"]
        and before["sha256"] == after["sha256"]
    )


def _metric_recomputations(
    runtime: SeedRuntime,
    trajectory: list[np.ndarray],
    support: list[float | None],
    contact: list[float | None],
) -> dict[str, Any]:
    legacy_metrics = runtime.frozen_eval._episode_metrics(
        trajectory,
        "right",
        "left",
        runtime.metric_args,
    )
    env_support_metrics = runtime.frozen_eval._episode_metrics(
        trajectory,
        "right",
        "left",
        runtime.metric_args,
        support,
        contact,
    )
    criteria = runtime.config["episode_success"]
    return {
        "legacy": {
            "metrics": metric_projection(legacy_metrics),
            "success": episode_success(legacy_metrics, criteria),
        },
        "env_support": {
            "metrics": metric_projection(env_support_metrics),
            "success": episode_success(env_support_metrics, criteria),
            "support_samples_available": int(
                sum(value is not None for value in support)
            ),
            "contact_samples_available": int(
                sum(value is not None for value in contact)
            ),
        },
    }


def _failure_reasons(
    maxima: dict[str, float | None],
    issues: dict[str, str],
    velocity_compared: bool,
    checkpoints_unchanged: bool,
) -> list[str]:
    reasons = list(issues.values())
    required_labels = ["position", "observation", "action"]
    if velocity_compared:
        required_labels.append("velocity")
    for label in required_labels:
        value = maxima[label]
        if value is None:
            reasons.append(f"{label} maximum error is unavailable")
        elif value > HARD_TOLERANCE:
            reasons.append(
                f"{label} maximum error {value:.17g} exceeds {HARD_TOLERANCE:.1e}"
            )
    if not checkpoints_unchanged:
        reasons.append("checkpoint SHA-256 changed or could not be verified")
    return list(dict.fromkeys(reasons))


def _run_paired_evaluation_set(
    training_seed: int,
    policy_arm: str,
    eval_set: str,
    config_key: str,
    expected_base_seed: int,
    expected_episodes: int,
) -> dict[str, Any]:
    native = SeedRuntime(training_seed, eval_set, environment_arm="R0")
    canonical = SeedRuntime(training_seed, eval_set, environment_arm="Rroll")
    config = native.config
    section = config[config_key]
    base_seed = int(section["base_seed"])
    episodes = int(section["episodes"])
    steps = int(section["steps"])
    if base_seed != expected_base_seed or episodes != expected_episodes:
        native.close()
        canonical.close()
        raise RuntimeError(
            f"Frozen {eval_set} set drift: base={base_seed}, episodes={episodes}; "
            f"expected {expected_base_seed}/{expected_episodes}"
        )
    if native.steps != steps or canonical.steps != steps:
        native.close()
        canonical.close()
        raise RuntimeError(f"Runtime step-count drift for {eval_set}")

    selected_checkpoint_pair = {
        "native_R0_sha256": native.checkpoint_hashes_before[policy_arm],
        "canonical_Rroll_sha256": canonical.checkpoint_hashes_before[policy_arm],
        "same_sha256": (
            native.checkpoint_hashes_before[policy_arm]
            == canonical.checkpoint_hashes_before[policy_arm]
        ),
    }
    selected_policy_pair = {
        "native_R0_state_sha256": native.policy_hashes_before[policy_arm],
        "canonical_Rroll_state_sha256": canonical.policy_hashes_before[policy_arm],
        "same_state_sha256": (
            native.policy_hashes_before[policy_arm]
            == canonical.policy_hashes_before[policy_arm]
        ),
    }
    episode_records: list[dict[str, Any]] = []
    set_maxima: dict[str, float | None] = {
        "position": 0.0,
        "velocity": None,
        "observation": 0.0,
        "action": 0.0,
    }
    set_issues: dict[str, str] = {}
    velocity_availability = {
        "compared_samples": 0,
        "both_absent_samples": 0,
        "availability_mismatch_samples": 0,
    }
    success_counts = {
        environment: {"legacy": 0, "env_support": 0}
        for environment in ENVIRONMENT_ARMS
    }

    try:
        for episode_index in range(episodes):
            episode_seed = base_seed + episode_index
            _seed_everything(episode_seed)
            native_td = native.env.reset()
            _seed_everything(episode_seed)
            canonical_td = canonical.env.reset()

            trajectories = {
                "native_R0": [_positions(native)],
                "canonical_Rroll": [_positions(canonical)],
            }
            supports = {
                "native_R0": [
                    native.frozen_eval._log_info_scalar(
                        native_td, "fast_forward_support_index"
                    )
                ],
                "canonical_Rroll": [
                    canonical.frozen_eval._log_info_scalar(
                        canonical_td, "fast_forward_support_index"
                    )
                ],
            }
            contacts = {
                "native_R0": [
                    native.frozen_eval._log_info_scalar(
                        native_td, "fast_forward_ground_contact_strength"
                    )
                ],
                "canonical_Rroll": [
                    canonical.frozen_eval._log_info_scalar(
                        canonical_td, "fast_forward_ground_contact_strength"
                    )
                ],
            }
            episode_maxima: dict[str, float | None] = {
                "position": 0.0,
                "velocity": None,
                "observation": 0.0,
                "action": 0.0,
            }
            episode_issues: dict[str, str] = {}
            episode_velocity_availability = {
                "compared_samples": 0,
                "both_absent_samples": 0,
                "availability_mismatch_samples": 0,
            }

            _compare_required(
                episode_maxima,
                episode_issues,
                "position",
                trajectories["native_R0"][-1],
                trajectories["canonical_Rroll"][-1],
                "reset",
            )
            _compare_velocity(
                episode_maxima,
                episode_issues,
                episode_velocity_availability,
                _velocity(native),
                _velocity(canonical),
                "reset",
            )
            _compare_required(
                episode_maxima,
                episode_issues,
                "observation",
                _observation(native_td),
                _observation(canonical_td),
                "reset",
            )

            for step in range(steps):
                native_action, canonical_action = _paired_call(
                    lambda: _policy_action(native, native_td, policy_arm),
                    lambda: _policy_action(canonical, canonical_td, policy_arm),
                )
                _compare_required(
                    episode_maxima,
                    episode_issues,
                    "action",
                    native_action,
                    canonical_action,
                    f"step {step} action",
                )

                native_action_td = native_td.clone(recurse=True)
                native_action_td["agents", "action"] = native_action
                canonical_action_td = canonical_td.clone(recurse=True)
                canonical_action_td["agents", "action"] = canonical_action
                native_td, canonical_td = _paired_call(
                    lambda: native.env.step(native_action_td)["next"],
                    lambda: canonical.env.step(canonical_action_td)["next"],
                )

                trajectories["native_R0"].append(_positions(native))
                trajectories["canonical_Rroll"].append(_positions(canonical))
                supports["native_R0"].append(
                    native.frozen_eval._log_info_scalar(
                        native_td, "fast_forward_support_index"
                    )
                )
                supports["canonical_Rroll"].append(
                    canonical.frozen_eval._log_info_scalar(
                        canonical_td, "fast_forward_support_index"
                    )
                )
                contacts["native_R0"].append(
                    native.frozen_eval._log_info_scalar(
                        native_td, "fast_forward_ground_contact_strength"
                    )
                )
                contacts["canonical_Rroll"].append(
                    canonical.frozen_eval._log_info_scalar(
                        canonical_td, "fast_forward_ground_contact_strength"
                    )
                )
                _compare_required(
                    episode_maxima,
                    episode_issues,
                    "position",
                    trajectories["native_R0"][-1],
                    trajectories["canonical_Rroll"][-1],
                    f"step {step + 1} state",
                )
                _compare_velocity(
                    episode_maxima,
                    episode_issues,
                    episode_velocity_availability,
                    _velocity(native),
                    _velocity(canonical),
                    f"step {step + 1} state",
                )
                _compare_required(
                    episode_maxima,
                    episode_issues,
                    "observation",
                    _observation(native_td),
                    _observation(canonical_td),
                    f"step {step + 1} observation",
                )

            recomputed = {
                "native_R0": _metric_recomputations(
                    native,
                    trajectories["native_R0"],
                    supports["native_R0"],
                    contacts["native_R0"],
                ),
                "canonical_Rroll": _metric_recomputations(
                    canonical,
                    trajectories["canonical_Rroll"],
                    supports["canonical_Rroll"],
                    contacts["canonical_Rroll"],
                ),
            }
            for environment in ENVIRONMENT_ARMS:
                for metric_mode in ("legacy", "env_support"):
                    success_counts[environment][metric_mode] += int(
                        bool(recomputed[environment][metric_mode]["success"])
                    )

            velocity_compared = (
                episode_velocity_availability["compared_samples"] > 0
            )
            episode_reasons = _failure_reasons(
                episode_maxima,
                episode_issues,
                velocity_compared,
                checkpoints_unchanged=True,
            )
            episode_records.append(
                {
                    "episode": episode_index + 1,
                    "evaluation_seed": episode_seed,
                    "max_errors": episode_maxima,
                    "velocity_comparison": episode_velocity_availability,
                    "within_numeric_tolerance": not episode_reasons,
                    "numeric_failure_reasons": episode_reasons,
                    "recomputed": recomputed,
                }
            )

            for label, value in episode_maxima.items():
                if value is not None:
                    current = set_maxima[label]
                    set_maxima[label] = (
                        value if current is None else max(current, value)
                    )
            for label, issue in episode_issues.items():
                set_issues.setdefault(label, f"episode {episode_index + 1}: {issue}")
            for key, value in episode_velocity_availability.items():
                velocity_availability[key] += value

        immutability = {
            "native_R0": _runtime_immutability(native),
            "canonical_Rroll": _runtime_immutability(canonical),
        }
        checkpoints_unchanged = (
            selected_checkpoint_pair["same_sha256"]
            and all(
                bool(item["checkpoints_unchanged"])
                for item in immutability.values()
            )
        )
        velocity_compared = velocity_availability["compared_samples"] > 0
        reasons = _failure_reasons(
            set_maxima,
            set_issues,
            velocity_compared,
            checkpoints_unchanged,
        )
        return {
            "schema": "obs2_v2_1_k_cross_environment_policy_eval_set/v2",
            "training_seed": training_seed,
            "policy_arm": policy_arm,
            "eval_set": eval_set,
            "base_seed": base_seed,
            "episodes": episodes,
            "steps": steps,
            "environment_pair": {
                "native": "R0",
                "canonical": "Rroll",
                "closed_loop_per_environment": True,
                "common_rng_state_per_paired_action_and_step": True,
            },
            "hard_tolerance": HARD_TOLERANCE,
            "max_errors": set_maxima,
            "velocity_comparison": velocity_availability,
            "selected_checkpoint_pair": selected_checkpoint_pair,
            "selected_policy_pair": selected_policy_pair,
            "success_episodes": success_counts,
            "checkpoint_immutability": immutability,
            "passed": not reasons,
            "failure_reasons": reasons,
            "episode_results": episode_records,
        }
    finally:
        native.close()
        canonical.close()


def run_seed(training_seed: int) -> dict[str, Any]:
    config = load_json(ROOT / "study_config.json")
    checkpoint_paths = _checkpoint_paths(config, training_seed)
    checkpoints_before = _checkpoint_snapshot(checkpoint_paths)
    result: dict[str, Any] = {
        "schema": "obs2_v2_1_k_cross_environment_identity_seed/v2",
        "study_id": config["study_id"],
        "training_seed": training_seed,
        "hard_tolerance": HARD_TOLERANCE,
        "arms": {},
    }
    for policy_arm in POLICY_ARMS:
        eval_sets: dict[str, Any] = {}
        for eval_set, config_key, base_seed, episodes in EVALUATION_SETS:
            eval_sets[eval_set] = _run_paired_evaluation_set(
                training_seed,
                policy_arm,
                eval_set,
                config_key,
                base_seed,
                episodes,
            )
        result["arms"][policy_arm] = {
            "passed": all(bool(item["passed"]) for item in eval_sets.values()),
            "eval_sets": eval_sets,
        }

    checkpoints_after = _checkpoint_snapshot(checkpoint_paths)
    checkpoints_unchanged = _checkpoint_snapshots_equal(
        checkpoints_before, checkpoints_after
    )
    result["checkpoint_immutability_across_seed_audit"] = {
        "before": checkpoints_before,
        "after": checkpoints_after,
        "checkpoints_unchanged": checkpoints_unchanged,
    }
    result["passed"] = checkpoints_unchanged and all(
        bool(item["passed"]) for item in result["arms"].values()
    )
    result["failure_reasons"] = []
    if not checkpoints_unchanged:
        result["failure_reasons"].append(
            "checkpoint SHA-256 changed or could not be verified across seed audit"
        )
    for policy_arm, arm_result in result["arms"].items():
        for eval_set, eval_result in arm_result["eval_sets"].items():
            if not eval_result["passed"]:
                result["failure_reasons"].append(
                    f"{policy_arm}/{eval_set} failed its hard gate"
                )
    atomic_json(
        ROOT / "cross_environment_audit" / f"seed{training_seed}.json", result
    )
    return result


def child(seed: int) -> None:
    payload = run_seed(seed)
    print(
        json.dumps(
            {
                "seed": seed,
                "passed": payload["passed"],
                "arms": {
                    arm: {
                        eval_set: result["passed"]
                        for eval_set, result in arm_result["eval_sets"].items()
                    }
                    for arm, arm_result in payload["arms"].items()
                },
            },
            ensure_ascii=False,
        )
    )


def _eval_set_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "passed": payload["passed"],
        "base_seed": payload["base_seed"],
        "episodes": payload["episodes"],
        "steps": payload["steps"],
        "max_errors": payload["max_errors"],
        "velocity_comparison": payload["velocity_comparison"],
        "success_episodes": payload["success_episodes"],
        "selected_checkpoint_pair": payload["selected_checkpoint_pair"],
        "failure_reasons": payload["failure_reasons"],
    }


def _global_maxima(payloads: list[dict[str, Any]]) -> dict[str, float | None]:
    result: dict[str, float | None] = {
        "position": None,
        "velocity": None,
        "observation": None,
        "action": None,
    }
    for seed_payload in payloads:
        for arm_payload in seed_payload["arms"].values():
            for eval_payload in arm_payload["eval_sets"].values():
                for label, value in eval_payload["max_errors"].items():
                    if value is not None:
                        current = result[label]
                        result[label] = value if current is None else max(current, value)
    return result


def orchestrate() -> None:
    config = load_json(ROOT / "study_config.json")
    seeds = [int(value) for value in config["training_seeds"]]
    all_checkpoint_paths = {
        f"seed{seed}_{arm}": path
        for seed in seeds
        for arm, path in _checkpoint_paths(config, seed).items()
    }
    checkpoints_before = _checkpoint_snapshot(all_checkpoint_paths)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(seeds)) as executor:
        futures: list[tuple[int, Path, Any]] = []
        for seed in seeds:
            stdout = ROOT / "logs" / f"cross_environment_seed{seed}.stdout.log"
            stderr = ROOT / "logs" / f"cross_environment_seed{seed}.stderr.log"
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
                        env=dict(os.environ),
                        stdout=out,
                        stderr=err,
                        check=False,
                    ).returncode

            futures.append((seed, stderr, executor.submit(invoke)))
        for seed, stderr, future in futures:
            if future.result() != 0:
                raise RuntimeError(
                    f"Cross-environment seed {seed} failed:\n"
                    + stderr.read_text(encoding="utf-8", errors="replace")[-5000:]
                )

    payloads = [
        load_json(ROOT / "cross_environment_audit" / f"seed{seed}.json")
        for seed in seeds
    ]
    checkpoints_after = _checkpoint_snapshot(all_checkpoint_paths)
    checkpoints_unchanged = _checkpoint_snapshots_equal(
        checkpoints_before, checkpoints_after
    )
    result_binding = _main_result_binding(seeds, payloads)
    passed = (
        checkpoints_unchanged
        and bool(result_binding["passed"])
        and all(bool(seed_payload["passed"]) for seed_payload in payloads)
    )
    failure_reasons: list[str] = []
    if not checkpoints_unchanged:
        failure_reasons.append(
            "checkpoint SHA-256 changed or could not be verified across full audit"
        )
    if not result_binding["passed"]:
        failure_reasons.append(
            "canonical main-set replay did not exactly reproduce stored C00/C11 metrics"
        )
    failure_reasons.extend(
        f"seed{seed_payload['training_seed']} failed its hard gate"
        for seed_payload in payloads
        if not seed_payload["passed"]
    )
    payload = {
        "schema": "obs2_v2_1_k_cross_environment_identity_audit/v2",
        "study_id": config["study_id"],
        "training_seeds": seeds,
        "hard_gate": {
            "absolute_tolerance": HARD_TOLERANCE,
            "required_stepwise_comparisons": [
                "positions",
                "env.vel_if_available",
                "observation",
                "action",
            ],
            "checkpoint_sha256_must_remain_unchanged": True,
        },
        "results_by_seed": {
            str(seed_payload["training_seed"]): {
                "passed": seed_payload["passed"],
                "arms": {
                    arm: {
                        "passed": arm_payload["passed"],
                        "eval_sets": {
                            eval_set: _eval_set_summary(eval_payload)
                            for eval_set, eval_payload in arm_payload[
                                "eval_sets"
                            ].items()
                        },
                    }
                    for arm, arm_payload in seed_payload["arms"].items()
                },
                "checkpoint_immutability_across_seed_audit": seed_payload[
                    "checkpoint_immutability_across_seed_audit"
                ],
                "failure_reasons": seed_payload["failure_reasons"],
            }
            for seed_payload in payloads
        },
        "global_max_errors": _global_maxima(payloads),
        "checkpoint_immutability_across_full_audit": {
            "before": checkpoints_before,
            "after": checkpoints_after,
            "checkpoints_unchanged": checkpoints_unchanged,
        },
        "checkpoint_sha256": {
            "before": _nested_checkpoint_hashes(checkpoints_before, seeds),
            "after": _nested_checkpoint_hashes(checkpoints_after, seeds),
        },
        "stored_main_result_binding": result_binding,
        "result_file_sha256": result_binding["result_file_sha256"],
        "passed": passed,
        "failure_reasons": failure_reasons,
        "audit_source_sha256": sha256_file(Path(__file__).resolve()),
        "source_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(ROOT / "CROSS_ENVIRONMENT_IDENTITY_AUDIT.json", payload)
    marker = ROOT / (
        "CROSS_ENVIRONMENT_IDENTITY_AUDIT_PASS.json"
        if passed
        else "CROSS_ENVIRONMENT_IDENTITY_AUDIT_FAIL.json"
    )
    stale = ROOT / (
        "CROSS_ENVIRONMENT_IDENTITY_AUDIT_FAIL.json"
        if passed
        else "CROSS_ENVIRONMENT_IDENTITY_AUDIT_PASS.json"
    )
    if stale.exists():
        stale.unlink()
    atomic_json(marker, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.seed is None:
        orchestrate()
    else:
        config = load_json(ROOT / "study_config.json")
        allowed = {int(value) for value in config["training_seeds"]}
        if args.seed not in allowed:
            raise ValueError(f"Training seed is outside frozen contract: {args.seed}")
        child(args.seed)


if __name__ == "__main__":
    main()
