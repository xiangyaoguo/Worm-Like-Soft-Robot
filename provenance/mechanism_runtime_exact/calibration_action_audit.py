from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from condition_matrix import build_conditions
from mechanism_rollout import ROOT, SeedRuntime, atomic_json, load_json, sha256_file


def _new_accumulator() -> dict[str, Any]:
    return {
        "calls": 0,
        "action_digest": hashlib.sha256(),
        "clipped_torque_digest": hashlib.sha256(),
        "abs_action_delta_C00": np.zeros((8, 2), dtype=np.float64),
        "abs_action_delta_C11": np.zeros((8, 2), dtype=np.float64),
        "action_changed_C00": np.zeros((8, 2), dtype=np.float64),
        "action_changed_C11": np.zeros((8, 2), dtype=np.float64),
        "abs_torque_delta_C00": np.zeros(8, dtype=np.float64),
        "abs_torque_delta_C11": np.zeros(8, dtype=np.float64),
        "torque_changed_C00": np.zeros(8, dtype=np.float64),
        "torque_changed_C11": np.zeros(8, dtype=np.float64),
        "saturated": np.zeros(8, dtype=np.float64),
    }


def run_seed(seed: int) -> dict[str, Any]:
    runtime = SeedRuntime(seed, "main", environment_arm="Rroll")
    config = runtime.config
    conditions = build_conditions()
    by_id = {condition.id: condition for condition in conditions}
    base_seed = int(config["identity_gate"]["base_seed"])
    episodes = int(config["identity_gate"]["episodes"])
    accumulators = {condition.id: _new_accumulator() for condition in conditions}
    scale = float(runtime.env.k_action_scale)
    gain = float(getattr(runtime.env, "feedback_gain", 1.0))
    limit = float(runtime.env.max_torque)
    try:
        for episode in range(episodes):
            torch.manual_seed(base_seed + episode)
            np.random.seed(base_seed + episode)
            td = runtime.env.reset()
            for step in range(runtime.steps):
                obs = td["agents", "observation"][0].detach().cpu().numpy().astype(np.float64)
                r0, roll, same_observation_error = runtime.actor_actions(td)
                if same_observation_error != 0.0:
                    raise RuntimeError("Actors did not receive the same observation")
                c00 = runtime.apply_condition(by_id["C00"], r0, roll, step)[0]
                c11 = runtime.apply_condition(by_id["C11"], r0, roll, step)[0]

                def clipped(action: torch.Tensor) -> np.ndarray:
                    value = action.detach().cpu().numpy().astype(np.float64)
                    physical = scale * value
                    preclip = (
                        physical[:, 0] * obs[:, 0]
                        + physical[:, 1] * gain * obs[:, 1]
                    )
                    return np.clip(preclip, -limit, limit)

                c00_np = c00.detach().cpu().numpy().astype(np.float64)
                c11_np = c11.detach().cpu().numpy().astype(np.float64)
                c00_tau = clipped(c00)
                c11_tau = clipped(c11)
                for condition in conditions:
                    applied = runtime.apply_condition(condition, r0, roll, step)[0]
                    applied_np = applied.detach().cpu().numpy().astype(np.float64)
                    applied_tau = clipped(applied)
                    acc = accumulators[condition.id]
                    acc["calls"] += 1
                    acc["action_digest"].update(applied_np.astype(np.float32).tobytes())
                    acc["clipped_torque_digest"].update(
                        applied_tau.astype(np.float32).tobytes()
                    )
                    delta00 = np.abs(applied_np - c00_np)
                    delta11 = np.abs(applied_np - c11_np)
                    tau_delta00 = np.abs(applied_tau - c00_tau)
                    tau_delta11 = np.abs(applied_tau - c11_tau)
                    acc["abs_action_delta_C00"] += delta00
                    acc["abs_action_delta_C11"] += delta11
                    acc["action_changed_C00"] += delta00 > 1e-7
                    acc["action_changed_C11"] += delta11 > 1e-7
                    acc["abs_torque_delta_C00"] += tau_delta00
                    acc["abs_torque_delta_C11"] += tau_delta11
                    acc["torque_changed_C00"] += tau_delta00 > 1e-6
                    acc["torque_changed_C11"] += tau_delta11 > 1e-6
                    acc["saturated"] += np.abs(applied_tau) >= limit - 1e-7

                action_td = td.clone(recurse=True)
                action_td["agents", "action"] = roll
                td = runtime.env.step(action_td)["next"]

        rows: list[dict[str, Any]] = []
        expected_calls = episodes * runtime.steps
        for condition in conditions:
            acc = accumulators[condition.id]
            if acc["calls"] != expected_calls:
                raise RuntimeError(f"Audit call count mismatch for {condition.id}")
            rows.append(
                {
                    "condition_id": condition.id,
                    "module": condition.module,
                    "family": condition.family,
                    "calls": expected_calls,
                    "action_digest": acc["action_digest"].hexdigest(),
                    "clipped_torque_digest": acc["clipped_torque_digest"].hexdigest(),
                    "mean_abs_action_delta_from_C00": (
                        acc["abs_action_delta_C00"] / expected_calls
                    ).tolist(),
                    "mean_abs_action_delta_from_C11": (
                        acc["abs_action_delta_C11"] / expected_calls
                    ).tolist(),
                    "action_changed_fraction_from_C00": (
                        acc["action_changed_C00"] / expected_calls
                    ).tolist(),
                    "action_changed_fraction_from_C11": (
                        acc["action_changed_C11"] / expected_calls
                    ).tolist(),
                    "mean_abs_clipped_torque_delta_from_C00": (
                        acc["abs_torque_delta_C00"] / expected_calls
                    ).tolist(),
                    "mean_abs_clipped_torque_delta_from_C11": (
                        acc["abs_torque_delta_C11"] / expected_calls
                    ).tolist(),
                    "clipped_torque_changed_fraction_from_C00": (
                        acc["torque_changed_C00"] / expected_calls
                    ).tolist(),
                    "clipped_torque_changed_fraction_from_C11": (
                        acc["torque_changed_C11"] / expected_calls
                    ).tolist(),
                    "clipped_torque_saturation_fraction": (
                        acc["saturated"] / expected_calls
                    ).tolist(),
                }
            )
        payload = {
            "schema": "obs2_v2_1_k_calibration_action_audit_seed/v1",
            "training_seed": seed,
            "calibration_base_seed": base_seed,
            "calibration_episodes": episodes,
            "steps": runtime.steps,
            "condition_count": len(conditions),
            "results": rows,
            "immutability": runtime.verify_unchanged(),
        }
        atomic_json(ROOT / "calibration_audit" / f"seed{seed}.json", payload)
        return payload
    finally:
        runtime.close()


def aggregate() -> dict[str, Any]:
    config = load_json(ROOT / "study_config.json")
    seed_payloads = [
        load_json(ROOT / "calibration_audit" / f"seed{seed}.json")
        for seed in config["training_seeds"]
    ]
    condition_ids = [condition.id for condition in build_conditions()]
    duplicate_groups: list[list[str]] = []
    fingerprints: dict[tuple[tuple[str, str], ...], list[str]] = {}
    for condition_id in condition_ids:
        fingerprint = tuple(
            (
                next(row for row in payload["results"] if row["condition_id"] == condition_id)[
                    "action_digest"
                ],
                next(row for row in payload["results"] if row["condition_id"] == condition_id)[
                    "clipped_torque_digest"
                ],
            )
            for payload in seed_payloads
        )
        fingerprints.setdefault(fingerprint, []).append(condition_id)
    duplicate_groups = [group for group in fingerprints.values() if len(group) > 1]

    weak_conditions: list[dict[str, Any]] = []
    for condition_id in condition_ids:
        rows = [
            next(row for row in payload["results"] if row["condition_id"] == condition_id)
            for payload in seed_payloads
        ]
        changed_vs_c00 = float(
            np.mean(
                [
                    np.mean(row["clipped_torque_changed_fraction_from_C00"])
                    for row in rows
                ]
            )
        )
        changed_vs_c11 = float(
            np.mean(
                [
                    np.mean(row["clipped_torque_changed_fraction_from_C11"])
                    for row in rows
                ]
            )
        )
        if max(changed_vs_c00, changed_vs_c11) < 0.01:
            weak_conditions.append(
                {
                    "condition_id": condition_id,
                    "mean_changed_fraction_vs_C00": changed_vs_c00,
                    "mean_changed_fraction_vs_C11": changed_vs_c11,
                }
            )
    payload = {
        "schema": "obs2_v2_1_k_calibration_action_audit/v1",
        "passed_technical_execution": True,
        "calibration_only": True,
        "main_evaluation_states_used": False,
        "condition_count": len(condition_ids),
        "exact_duplicate_condition_groups": duplicate_groups,
        "weak_postclip_separation_conditions": weak_conditions,
        "drop_or_modify_conditions_based_on_audit": False,
        "interpretation": "Weak or duplicate post-clip interventions remain in the frozen matrix and are flagged as limited-identifiability results, never silently removed.",
        "seed_files": [
            {
                "training_seed": payload["training_seed"],
                "path": str(ROOT / "calibration_audit" / f"seed{payload['training_seed']}.json"),
                "sha256": sha256_file(
                    ROOT / "calibration_audit" / f"seed{payload['training_seed']}.json"
                ),
            }
            for payload in seed_payloads
        ],
        "audit_source_sha256": sha256_file(Path(__file__).resolve()),
    }
    atomic_json(ROOT / "CALIBRATION_ACTION_AUDIT.json", payload)
    return payload


def run_child(seed: int) -> None:
    payload = run_seed(seed)
    print(json.dumps({"seed": seed, "conditions": payload["condition_count"]}))


def orchestrate() -> None:
    config = load_json(ROOT / "study_config.json")
    seeds = [int(value) for value in config["training_seeds"]]
    environment = dict(os.environ)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(seeds)) as executor:
        futures = []
        for seed in seeds:
            stdout = ROOT / "logs" / f"calibration_audit_seed{seed}.stdout.log"
            stderr = ROOT / "logs" / f"calibration_audit_seed{seed}.stderr.log"
            stdout.parent.mkdir(parents=True, exist_ok=True)
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
                        env=environment,
                        stdout=out,
                        stderr=err,
                        check=False,
                    ).returncode

            futures.append((seed, stderr, executor.submit(invoke)))
        for seed, stderr, future in futures:
            code = future.result()
            if code != 0:
                raise RuntimeError(
                    f"Calibration audit seed {seed} failed:\n"
                    + stderr.read_text(encoding="utf-8", errors="replace")[-5000:]
                )
    print(json.dumps(aggregate(), ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.seed is None:
        orchestrate()
    else:
        run_child(args.seed)


if __name__ == "__main__":
    main()
