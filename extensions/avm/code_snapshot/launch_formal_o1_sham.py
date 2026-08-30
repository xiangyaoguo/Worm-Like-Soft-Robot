"""Formal launcher for the five seed-paired HPR/O1-sham training runs.

No full training is started unless ``--phase launch`` is given and the frozen
manifest, all five preflights, and a manifest-bound approval marker pass.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from verify_extension import verify


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "extension_config.json"
REFERENCE_PATH = HERE / "reference_initializations.json"
AUDIT_WRAPPER = HERE / "audited_train_o1_sham.py"
TRAINER_SHIM = HERE / "actor_observation_shim.py"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_documents() -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        json.loads(CONFIG_PATH.read_text(encoding="utf-8")),
        json.loads(REFERENCE_PATH.read_text(encoding="utf-8")),
    )


def code_files() -> list[Path]:
    return sorted(
        path for path in HERE.iterdir()
        if path.is_file() and path.suffix.lower() in {".py", ".json", ".md"}
    )


def current_code_hashes() -> dict[str, str]:
    return {path.name: sha256_file(path) for path in code_files()}


def environment(config: dict[str, Any]) -> dict[str, str]:
    value = os.environ.copy()
    runtime = config["runtime"]
    value["PYTHONPATH"] = runtime["site_packages"]
    value["CUDA_VISIBLE_DEVICES"] = runtime["cuda_visible_devices"]
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        value[name] = str(runtime["thread_limit_per_process"])
    value["PYTHONDONTWRITEBYTECODE"] = "1"
    return value


def base_trainer_args(config: dict[str, Any], seed: int, run_name: str, results_dir: Path) -> list[str]:
    training = config["training_contract"]
    return [
        "--robot", "crawler", "--terrain", "flat", "--terrain-contact-mode", "legacy_flat",
        "--num-particles", "10", "--channel", "action",
        "--observation-func", "dth_tot_plus_friction_thdot", "--control-mode", "formula",
        "--feedback-gain", "1.0", "--max-control-gain", "9.0", "--no-fix-k1", "--no-fix-k2",
        "--k-action-scale", "100.0", "--passive-kappa", "4.0", "--per-joint-k1-k2",
        "--no-share-policy", "--share-critic", "--centralised-critic", "--algorithm", "ppo",
        "--policy-depth", "2", "--policy-cells", "256", "--normal-scale-lb", "0.0001",
        "--episode-steps", str(training["episode_steps"]),
        "--frames-per-batch", str(training["frames_per_batch"]),
        "--memory-size", str(training["memory_size"]),
        "--minibatch-size", str(training["minibatch_size"]),
        "--optim-steps", str(training["optim_steps"]), "--lr", str(training["learning_rate"]),
        "--weight-decay", str(training["weight_decay"]), "--max-grad-norm", str(training["max_grad_norm"]),
        "--gamma", str(training["gamma"]), "--clip-epsilon", str(training["clip_epsilon"]),
        "--lambda-gae", str(training["lambda_gae"]), "--entropy-eps", str(training["entropy_epsilon"]),
        "--no-ppo-normalize-advantage", "--ppo-target-kl", "0.0",
        "--init-pos-randomness", str(training["init_pos_randomness"]),
        "--init-angle-range-degrees", str(training["init_angle_range_degrees"]),
        "--init-height-jitter", str(training["init_height_jitter"]),
        "--action-smoothness-weight", "0.0", "--policy-anchor-coeff", "0.0",
        "--policy-anchor-anneal-batches", "0", "--bc-steps", "0", "--bc-epochs", "0",
        "--rolling-direction", "right", "--rolling-curl-episodes", "300",
        "--rolling-transition-episodes", "300", "--rolling-reward-scale", "3.0",
        "--tail-side", "left", "--tail-roll-init-assist-degrees", "0.0",
        "--tail-roll-init-assist-episodes", "0", "--no-rolling-observation",
        "--no-tail-roll-observation", "--no-fast-forward-observation",
        "--fast-forward-event-degrees", "60.0", "--fast-forward-event-forward-fraction", "0.08",
        "--fast-forward-event-contact-nodes", "1.5", "--fast-forward-direction-fraction", "0.65",
        "--fast-forward-event-target-steps", "250", "--fast-forward-launch-lift", "0.2",
        "--fast-forward-launch-forward", "0.1", "--fast-forward-launch-curl", "0.12",
        "--fast-forward-launch-head-contact", "0.5", "--fast-forward-launch-hold-steps", "8",
        "--fast-forward-stall-steps", "150", "--fast-forward-rotation-step-ref-degrees", "2.0",
        "--fast-forward-translation-step-ref", "0.002", "--no-pretrained-policy-only",
        "--no-compatible-input-expansion", "--buffer-storage", "tensor", "--no-auto-analysis",
        "--episodes", "1500", "--save-every", "100", "--reward-func", "horizontal_speed",
        "--seed", str(seed), "--results-dir", str(results_dir), "--run-name", run_name,
        "--actor-observation-mode", "spatial_only_sham",
    ]


def case(config: dict[str, Any], references: dict[str, Any], seed: int, preflight: bool) -> dict[str, Any]:
    parent = Path(config["parent_root"])
    control = Path(config["control_root"])
    run_name = config["formal_runs"]["run_name_template"].format(seed=seed)
    extension = parent / Path(config["formal_runs"]["formal_extension_relative"])
    if preflight:
        results_dir = control / "_control" / "preflight" / "scratch_runs"
        audit_path = control / "_control" / "preflight" / "audits" / f"seed{seed}.json"
        command_run_name = f"preflight__seed{seed}__HPR__O1sham"
    else:
        results_dir = extension / "runs"
        audit_path = extension / "initialization" / f"{run_name}.json"
        command_run_name = run_name
    reference = references["references"][str(seed)]
    command = [
        config["runtime"]["python"], str(AUDIT_WRAPPER), "--audit-output", str(audit_path),
        "--case-id", run_name, "--expected-seed", str(seed),
        "--reference-audit", str(parent / Path(reference["relative_path"])),
        "--reference-audit-sha256", reference["file_sha256"], "--trainer", str(TRAINER_SHIM),
    ]
    if preflight:
        command.append("--preflight-only")
    command.extend(base_trainer_args(config, seed, command_run_name, results_dir))
    return {
        "seed": seed, "run_name": run_name, "results_dir": results_dir,
        "run_dir": results_dir / command_run_name, "audit_path": audit_path, "command": command,
        "stdout": control / "logs" / ("preflight" if preflight else "training") / f"seed{seed}.stdout.log",
        "stderr": control / "logs" / ("preflight" if preflight else "training") / f"seed{seed}.stderr.log",
    }


def execute(item: dict[str, Any], env: dict[str, str]) -> dict[str, Any]:
    if item["audit_path"].exists():
        raise FileExistsError(f"Refusing to overwrite seed {item['seed']} audit")
    if item["run_dir"].exists() and any(item["run_dir"].iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty seed {item['seed']} run directory")
    item["stdout"].parent.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    with item["stdout"].open("w", encoding="utf-8") as out, item["stderr"].open("w", encoding="utf-8") as err:
        code = subprocess.run(item["command"], cwd=str(HERE), env=env, stdout=out, stderr=err).returncode
    return {"seed": item["seed"], "exit_code": int(code), "started_at_utc": started,
            "finished_at_utc": utc_now(), "audit": str(item["audit_path"]),
            "run_dir": str(item["run_dir"]), "stdout": str(item["stdout"]), "stderr": str(item["stderr"])}


def require_frozen(control: Path) -> dict[str, Any]:
    path = control / "_control" / "FROZEN_EXTENSION_MANIFEST.json"
    if not path.is_file():
        raise RuntimeError("Run --phase freeze first")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["code_sha256"] != current_code_hashes():
        raise RuntimeError("Extension code changed after freeze")
    return value


def require_preflights(config: dict[str, Any]) -> None:
    control = Path(config["control_root"])
    for seed in config["formal_runs"]["internal_seeds"]:
        path = control / "_control" / "preflight" / "audits" / f"seed{seed}.json"
        if not path.is_file():
            raise RuntimeError(f"Missing preflight audit for seed {seed}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("status") != "preflight_passed" or not value.get("initialization_comparison", {}).get("all_match"):
            raise RuntimeError(f"Preflight did not pass for seed {seed}")


def require_approval(config: dict[str, Any], frozen_path: Path) -> None:
    approval_path = Path(config["control_root"]) / "_control" / "FORMAL_APPROVAL.json"
    if not approval_path.is_file():
        raise RuntimeError(f"Explicit approval marker is missing: {approval_path}")
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    if approval.get("statement") != config["formal_approval_statement"]:
        raise RuntimeError("Formal approval statement mismatch")
    if approval.get("frozen_manifest_sha256") != sha256_file(frozen_path):
        raise RuntimeError("Approval is not bound to the current frozen manifest")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["commands", "freeze", "preflight", "launch", "status"], required=True)
    args = ap.parse_args()
    config, references = load_documents()
    verify()
    control = Path(config["control_root"])
    frozen_path = control / "_control" / "FROZEN_EXTENSION_MANIFEST.json"
    if args.phase == "commands":
        for seed in config["formal_runs"]["internal_seeds"]:
            print(subprocess.list2cmdline(case(config, references, seed, False)["command"]))
        return 0
    if args.phase == "freeze":
        if frozen_path.exists():
            raise FileExistsError(f"Refusing to overwrite {frozen_path}")
        atomic_json(frozen_path, {"schema": "formal_o1_sham_frozen_manifest/v1", "created_at_utc": utc_now(),
                                  "study_id": config["study_id"], "code_sha256": current_code_hashes(),
                                  "parent_verification": verify()})
        print(f"Frozen manifest: {frozen_path}\nSHA-256: {sha256_file(frozen_path)}")
        return 0
    if args.phase == "status":
        extension = Path(config["parent_root"]) / Path(config["formal_runs"]["formal_extension_relative"])
        for seed in config["formal_runs"]["internal_seeds"]:
            name = config["formal_runs"]["run_name_template"].format(seed=seed)
            checkpoint = extension / "runs" / name / "checkpoint_1500.pt"
            print(seed, "complete" if checkpoint.is_file() else "not complete")
        return 0
    require_frozen(control)
    if args.phase == "preflight":
        items = [case(config, references, seed, True) for seed in config["formal_runs"]["internal_seeds"]]
        results = [execute(item, environment(config)) for item in items]
        atomic_json(control / "_control" / "preflight" / "PRECHECK_RESULT.json", results)
        if any(item["exit_code"] != 0 for item in results):
            return 1
        require_preflights(config)
        return 0
    require_preflights(config)
    require_approval(config, frozen_path)
    items = [case(config, references, seed, False) for seed in config["formal_runs"]["internal_seeds"]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(execute, item, environment(config)) for item in items]
        results = [future.result() for future in futures]
    atomic_json(control / "receipts" / "FORMAL_TRAINING_EXECUTION.json", results)
    return 1 if any(item["exit_code"] != 0 for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
