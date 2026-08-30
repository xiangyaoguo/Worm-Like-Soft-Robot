from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from condition_matrix import build_conditions, canonical_json


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "study_config.json"
RUNNER_PATH = ROOT / "mechanism_rollout.py"
CONDITION_PATH = ROOT / "condition_matrix.py"
SMOKE_PATH = ROOT / "smoke_test.py"
STATE_PATH = ROOT / "orchestrator_state.json"
EVENTS_PATH = ROOT / "events.jsonl"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def event(name: str, **payload: Any) -> None:
    EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVENTS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": name, **payload}, ensure_ascii=False) + "\n")


def protected_files(config: dict[str, Any]) -> dict[str, Path]:
    formal_root = Path(config["formal_root"]).resolve()
    paths: dict[str, Path] = {
        "formal_config": formal_root / "_control" / "experiment_config.json",
        "formal_result": formal_root / "FORMAL_RESULT.json",
        "formal_source_manifest": formal_root / "_control" / "source_manifest.json",
        "frozen_evaluator": (
            formal_root
            / "_control"
            / "code_snapshot"
            / "training"
            / "evaluate_fast_forward_roll.py"
        ),
    }
    for seed in config["training_seeds"]:
        for arm in ("R0", "Rroll"):
            paths[f"checkpoint_seed{seed}_{arm}"] = (
                formal_root
                / "formal"
                / "runs"
                / f"formal__seed{seed}__{arm}"
                / "checkpoint_1500.pt"
            )
    return paths


def protected_hashes(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, path in protected_files(config).items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing protected source {name}: {path}")
        result[name] = {
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return result


def validate_formal_result(config: dict[str, Any]) -> None:
    formal_root = Path(config["formal_root"]).resolve()
    formal_config = load_json(formal_root / "_control" / "experiment_config.json")
    formal_result = load_json(formal_root / "FORMAL_RESULT.json")
    if formal_config.get("study_id") != config["formal_study_id"]:
        raise RuntimeError("Formal study identity drift")
    if formal_result.get("study_id") != config["formal_study_id"]:
        raise RuntimeError("Formal result identity drift")
    if formal_result.get("execution_status") != "complete":
        raise RuntimeError("Formal experiment is not complete")
    if formal_result.get("robust") is not True:
        raise RuntimeError("Formal rolling condition was not robust")
    if formal_result.get("historical_checkpoint_loaded") is not False:
        raise RuntimeError("Formal result violates from-scratch contract")
    expected = [int(value) for value in config["training_seeds"]]
    actual = [int(value) for value in formal_config["formal"]["training_seeds"]]
    if actual != expected:
        raise RuntimeError(f"Formal seed drift: {actual} != {expected}")


def frozen_contract(config: dict[str, Any]) -> dict[str, Any]:
    conditions = list(build_conditions())
    counts: dict[str, int] = {}
    for condition in conditions:
        counts[condition.module] = counts.get(condition.module, 0) + 1
    if counts != {"A": 4, "B": 32, "C": 13, "D": 10}:
        raise RuntimeError(f"Condition count contract failed: {counts}")
    if len(conditions) != int(config["main_evaluation"]["condition_count"]):
        raise RuntimeError("Study config condition count drift")
    canonical = canonical_json()
    sources = {
        path.name: {"path": str(path), "sha256": sha256_file(path), "size": path.stat().st_size}
        for path in (
            CONFIG_PATH,
            CONDITION_PATH,
            RUNNER_PATH,
            SMOKE_PATH,
            Path(__file__).resolve(),
        )
    }
    amendments = sorted(ROOT.glob("TECHNICAL_AMENDMENT_*.json"))
    return {
        "schema": "obs2_v2_1_k_frozen_contract/v1",
        "study_id": config["study_id"],
        "analysis_type": config["analysis_type"],
        "condition_count": len(conditions),
        "condition_counts_by_module": counts,
        "conditions_sha256": sha256_text(canonical),
        "conditions": [condition.to_dict() for condition in conditions],
        "training_seeds": config["training_seeds"],
        "identity_gate": config["identity_gate"],
        "main_evaluation": config["main_evaluation"],
        "episode_success": config["episode_success"],
        "locked_contract": config["locked_contract"],
        "source_files": sources,
        "protected_formal_files": protected_hashes(config),
        "technical_amendments": [
            {"path": str(path), "sha256": sha256_file(path)} for path in amendments
        ],
        "scientific_boundaries": {
            "primary": "A and B are the preregistered causal core.",
            "secondary": "C tests prespecified K1 subsets and fixed-sign spatial constraints.",
            "exploratory": "D tests K2 dose, region, and calibration-derived temporal structure.",
            "training_seed_inference": "n=5 supports effect sizes and direction consistency; a two-sided exact sign-flip p<0.05 is mathematically impossible.",
            "fixed_policy_initial_state_inference": "The 20 paired main initial states support conditional claims for these five frozen policy pairs only.",
            "temporal_information_sets": "D static uses no online input; template and shuffled-template use a global step clock; none may be described as the original local feedback information set.",
        },
    }


def freeze_or_validate_contract(config: dict[str, Any]) -> dict[str, Any]:
    validate_formal_result(config)
    payload = frozen_contract(config)
    original_path = ROOT / "FROZEN_CONTRACT.json"
    amendments = sorted(ROOT.glob("TECHNICAL_AMENDMENT_*.json"))
    revision = len(amendments)
    path = original_path if revision == 0 else ROOT / f"FROZEN_CONTRACT_R{revision}.json"
    if path.is_file():
        existing = load_json(path)
        if existing != payload:
            raise RuntimeError("Frozen K-mechanism contract/source/checkpoint drift")
    else:
        if (ROOT / "results").exists() or (ROOT / "identity").exists():
            raise RuntimeError("Results exist before the contract was frozen")
        if amendments:
            amendment = load_json(amendments[-1])
            evidence_state = amendment.get("evidence_state", {})
            if evidence_state.get("scientific_outcomes_observed") is True:
                raise RuntimeError("Technical amendment was made after scientific outcomes")
            previous_path = (
                original_path
                if revision == 1
                else ROOT / f"FROZEN_CONTRACT_R{revision - 1}.json"
            )
            expected_previous_hash = amendment.get(
                "previous_contract_sha256", amendment.get("original_contract_sha256")
            )
            if not previous_path.is_file() or sha256_file(previous_path) != expected_previous_hash:
                raise RuntimeError("Preserved previous contract hash does not match amendment")
        atomic_json(path, payload)
    return payload


def python_environment(config: dict[str, Any]) -> dict[str, str]:
    formal_config = load_json(
        Path(config["formal_root"]) / "_control" / "experiment_config.json"
    )
    site_packages = str(formal_config["runtime"]["site_packages"])
    environment = dict(os.environ)
    old_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = site_packages + (os.pathsep + old_pythonpath if old_pythonpath else "")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["OMP_NUM_THREADS"] = "1"
    environment["MKL_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    environment["NUMEXPR_NUM_THREADS"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
    return environment


def run_seed_process(stage: str, seed: int, config: dict[str, Any]) -> dict[str, Any]:
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stdout_path = logs / f"{stage}_seed{seed}.stdout.log"
    stderr_path = logs / f"{stage}_seed{seed}.stderr.log"
    command = [
        sys.executable,
        str(RUNNER_PATH),
        "--stage",
        stage,
        "--training-seed",
        str(seed),
    ]
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            env=python_environment(config),
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if completed.returncode != 0:
        tail = stderr_path.read_text(encoding="utf-8", errors="replace")[-6000:]
        raise RuntimeError(
            f"{stage} seed {seed} failed with exit {completed.returncode}:\n{tail}"
        )
    return {
        "seed": seed,
        "stage": stage,
        "exit_code": completed.returncode,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def run_parallel(stage: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    seeds = [int(value) for value in config["training_seeds"]]
    workers = min(
        len(seeds), int(config["main_evaluation"]["max_parallel_seed_workers"])
    )
    event("stage_started", stage=stage, seeds=seeds, workers=workers)
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_seed_process, stage, seed, config): seed for seed in seeds
        }
        for future in concurrent.futures.as_completed(futures):
            seed = futures[future]
            receipt = future.result()
            results.append(receipt)
            event("seed_stage_completed", stage=stage, seed=seed)
    results.sort(key=lambda item: int(item["seed"]))
    event("stage_completed", stage=stage, seeds=seeds)
    return results


def aggregate_identity(config: dict[str, Any], receipts: list[dict[str, Any]]) -> dict[str, Any]:
    seeds = [int(value) for value in config["training_seeds"]]
    results = [load_json(ROOT / "identity" / f"seed{seed}.json") for seed in seeds]
    for item in results:
        if item.get("passed") is not True:
            raise RuntimeError(f"Identity seed failed: {item.get('training_seed')}")
    payload = {
        "schema": "obs2_v2_1_k_identity_gate/v1",
        "study_id": config["study_id"],
        "passed": True,
        "environment_identity": {
            "C00": "native R0 environment exactly matched frozen formal R0 metrics",
            "C11": "native Rroll environment exactly matched frozen formal Rroll metrics",
            "main_interventions": "all 59 conditions use one canonical Rroll-metadata environment and env_fast_forward_log_info",
        },
        "R0_success_episodes_by_seed": {
            str(item["training_seed"]): item["arms"]["R0"]["success_episodes"]
            for item in results
        },
        "Rroll_success_episodes_by_seed": {
            str(item["training_seed"]): item["arms"]["Rroll"]["success_episodes"]
            for item in results
        },
        "calibration_sha256": {
            str(seed): sha256_file(ROOT / "calibration" / f"seed{seed}.npz")
            for seed in seeds
        },
        "receipts": receipts,
    }
    atomic_json(ROOT / "IDENTITY_GATE_PASS.json", payload)
    return payload


def verify_protected_unchanged(contract: dict[str, Any]) -> None:
    for name, evidence in contract["protected_formal_files"].items():
        path = Path(evidence["path"])
        if sha256_file(path) != evidence["sha256"]:
            raise RuntimeError(f"Protected formal file changed: {name}")


def run_identity(config: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    smoke_path = ROOT / "SMOKE_TEST_PASS.json"
    if not smoke_path.is_file():
        raise RuntimeError("Identity gate is blocked until the frozen smoke test passes")
    smoke = load_json(smoke_path)
    if (
        smoke.get("passed") is not True
        or smoke.get("condition_matrix_sha256") != contract["conditions_sha256"]
    ):
        raise RuntimeError("Smoke-test evidence does not match the frozen condition matrix")
    if (ROOT / "results").exists():
        raise RuntimeError("Main results exist before identity gate")
    gate = ROOT / "IDENTITY_GATE_PASS.json"
    if gate.is_file():
        payload = load_json(gate)
        if payload.get("passed") is not True:
            raise RuntimeError("Existing identity gate is not passed")
        return payload
    receipts = run_parallel("identity", config)
    payload = aggregate_identity(config, receipts)
    verify_protected_unchanged(contract)
    return payload


def run_main(config: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    gate = ROOT / "IDENTITY_GATE_PASS.json"
    if not gate.is_file() or load_json(gate).get("passed") is not True:
        raise RuntimeError("Main stage requires passed identity gate")
    receipts = run_parallel("main", config)
    condition_ids = [condition.id for condition in build_conditions()]
    inventory: dict[str, list[str]] = {}
    for seed in config["training_seeds"]:
        result_dir = ROOT / "results" / f"seed{seed}"
        actual = sorted(path.stem for path in result_dir.glob("*.json"))
        if actual != sorted(condition_ids):
            raise RuntimeError(f"Incomplete result inventory for seed {seed}")
        inventory[str(seed)] = actual
    verify_protected_unchanged(contract)
    payload = {
        "schema": "obs2_v2_1_k_execution_complete/v1",
        "study_id": config["study_id"],
        "status": "complete",
        "training_seed_count": len(config["training_seeds"]),
        "condition_count": len(condition_ids),
        "policy_condition_cases": len(config["training_seeds"]) * len(condition_ids),
        "episodes": int(config["main_evaluation"]["total_episodes"]),
        "inventory": inventory,
        "receipts": receipts,
        "protected_formal_files_unchanged": True,
    }
    atomic_json(ROOT / "MAIN_EXECUTION_COMPLETE.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen K1/K2 mechanism study")
    parser.add_argument("--stage", choices=("freeze", "identity", "main", "all"), default="all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json(CONFIG_PATH)
    contract = freeze_or_validate_contract(config)
    state: dict[str, Any] = {
        "schema": "obs2_v2_1_k_orchestrator_state/v1",
        "study_id": config["study_id"],
        "status": "running",
        "requested_stage": args.stage,
        "pid": os.getpid(),
    }
    atomic_json(STATE_PATH, state)
    try:
        if args.stage in {"identity", "all"}:
            state["phase"] = "identity"
            atomic_json(STATE_PATH, state)
            state["identity_gate"] = run_identity(config, contract)
        if args.stage in {"main", "all"}:
            state["phase"] = "main"
            atomic_json(STATE_PATH, state)
            state["main"] = run_main(config, contract)
        state["phase"] = "complete" if args.stage != "freeze" else "frozen"
        state["status"] = "complete"
        atomic_json(STATE_PATH, state)
        event("orchestrator_completed", requested_stage=args.stage)
    except Exception as error:
        state["status"] = "failed"
        state["error"] = f"{type(error).__name__}: {error}"
        atomic_json(STATE_PATH, state)
        event("orchestrator_failed", error=state["error"])
        raise
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
