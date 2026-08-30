from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PYTHON = Path(r"C:\Users\PUBLIC_USER\AppData\Local\Programs\Python\Python311\python.exe")
SITE_PACKAGES = Path(
    r"C:\Users\PUBLIC_USER\CloudStorage\Desktop\finalproject\job\RLMetamaterialLocomotion-main"
    r"\RLMetamaterialLocomotion-main\.venv\Lib\site-packages"
)
SNAPSHOT = Path(
    r"C:\Users\PUBLIC_USER\CloudStorage\Desktop\finalproject\job\roll_learning"
    r"\obs2_roll_repro_v2_1_formal_20260803_r2\_control\code_snapshot"
)
TRAINER = SNAPSHOT / "training" / "train_metamaterial.py"
FORMAL_RESULTS = Path(r"E:\FormalTrainingResults\runs")
CONTROL_DIR = Path(__file__).resolve().parent
LOG_DIR = CONTROL_DIR / "logs"
STATUS_PATH = CONTROL_DIR / "orchestrator_status.json"
MANIFEST_PATH = CONTROL_DIR / "launch_manifest.json"
COMPLETE_PATH = CONTROL_DIR / "TRAINING_COMPLETE.json"

SEEDS = (9201, 9202, 9203, 9204, 9205)
ARMS: tuple[dict[str, str], ...] = (
    {
        "tag": "DTH",
        "channel": "dth",
        "observation": "dth_neighbours",
        "control_mode": "direct",
    },
    {
        "tag": "THDOT",
        "channel": "thdot",
        "observation": "dth_neighbours_plus_thdot",
        "control_mode": "direct",
    },
    {
        "tag": "OBS",
        "channel": "obs",
        "observation": "dth_tot",
        "control_mode": "direct",
    },
    {
        "tag": "HPR__O2shared",
        "channel": "action",
        "observation": "dth_tot_plus_friction_thdot",
        "control_mode": "formula",
    },
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def base_args(*, episodes: int, save_every: int, results_dir: Path) -> list[str]:
    return [
        str(TRAINER),
        "--robot", "crawler",
        "--terrain", "flat",
        "--terrain-contact-mode", "legacy_flat",
        "--num-particles", "10",
        "--feedback-gain", "1.0",
        "--max-control-gain", "9.0",
        "--no-fix-k1",
        "--no-fix-k2",
        "--k-action-scale", "100.0",
        "--passive-kappa", "4.0",
        "--share-policy",
        "--share-critic",
        "--centralised-critic",
        "--algorithm", "ppo",
        "--policy-depth", "2",
        "--policy-cells", "256",
        "--normal-scale-lb", "0.0001",
        "--episode-steps", "1000",
        "--frames-per-batch", "10000",
        "--memory-size", "1000000",
        "--minibatch-size", "128",
        "--optim-steps", "10",
        "--lr", "0.0003",
        "--weight-decay", "0.0001",
        "--max-grad-norm", "1.0",
        "--gamma", "0.99",
        "--clip-epsilon", "0.2",
        "--lambda-gae", "0.9",
        "--entropy-eps", "0.0001",
        "--no-ppo-normalize-advantage",
        "--ppo-target-kl", "0.0",
        "--init-pos-randomness", "0.01",
        "--init-angle-range-degrees", "0.0",
        "--init-height-jitter", "0.0",
        "--action-smoothness-weight", "0.0",
        "--policy-anchor-coeff", "0.0",
        "--policy-anchor-anneal-batches", "0",
        "--bc-steps", "0",
        "--bc-epochs", "0",
        "--rolling-direction", "right",
        "--rolling-curl-episodes", "300",
        "--rolling-transition-episodes", "300",
        "--rolling-reward-scale", "3.0",
        "--tail-side", "left",
        "--tail-roll-init-assist-degrees", "0.0",
        "--tail-roll-init-assist-episodes", "0",
        "--no-rolling-observation",
        "--no-tail-roll-observation",
        "--no-fast-forward-observation",
        "--fast-forward-event-degrees", "60.0",
        "--fast-forward-event-forward-fraction", "0.08",
        "--fast-forward-event-contact-nodes", "1.5",
        "--fast-forward-direction-fraction", "0.65",
        "--fast-forward-event-target-steps", "250",
        "--fast-forward-launch-lift", "0.2",
        "--fast-forward-launch-forward", "0.1",
        "--fast-forward-launch-curl", "0.12",
        "--fast-forward-launch-head-contact", "0.5",
        "--fast-forward-launch-hold-steps", "8",
        "--fast-forward-stall-steps", "150",
        "--fast-forward-rotation-step-ref-degrees", "2.0",
        "--fast-forward-translation-step-ref", "0.002",
        "--no-pretrained-policy-only",
        "--no-compatible-input-expansion",
        "--buffer-storage", "tensor",
        "--no-auto-analysis",
        "--episodes", str(episodes),
        "--save-every", str(save_every),
        "--reward-func", "horizontal_speed",
        "--results-dir", str(results_dir),
    ]


def build_case(
    arm: dict[str, str],
    seed: int,
    *,
    results_dir: Path,
    episodes: int,
    save_every: int,
    smoke: bool,
) -> dict[str, Any]:
    if smoke:
        run_name = f"smoke__seed{seed}__{arm['tag']}"
    else:
        run_name = f"formal__seed{seed}__{arm['tag']}"
    args = base_args(episodes=episodes, save_every=save_every, results_dir=results_dir)
    args += [
        "--channel", arm["channel"],
        "--observation-func", arm["observation"],
        "--control-mode", arm["control_mode"],
        "--seed", str(seed),
        "--run-name", run_name,
    ]
    return {
        "run_name": run_name,
        "seed": seed,
        "tag": arm["tag"],
        "channel": arm["channel"],
        "observation_func": arm["observation"],
        "control_mode": arm["control_mode"],
        "results_dir": str(results_dir),
        "run_dir": str(results_dir / run_name),
        "command": [str(PYTHON), *args],
        "state": "pending",
        "pid": None,
        "started_at_utc": None,
        "finished_at_utc": None,
        "exit_code": None,
        "progress_batches": 0,
        "latest_checkpoint": None,
        "latest_reward_mean": None,
        "latest_speed_x100": None,
        "issue": None,
    }


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8:strict",
            "PYGAME_HIDE_SUPPORT_PROMPT": "1",
            "MPLBACKEND": "Agg",
            "PYTHONPATH": str(SITE_PACKAGES),
            "CUDA_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "8",
            "NUMEXPR_NUM_THREADS": "8",
            "CUDA_MODULE_LOADING": "LAZY",
        }
    )
    return env


def validate_paths(cases: list[dict[str, Any]], *, allow_existing: bool = False) -> None:
    for required in (PYTHON, SITE_PACKAGES, SNAPSHOT, TRAINER):
        if not required.exists():
            raise RuntimeError(f"Required path does not exist: {required}")
    run_names = [case["run_name"] for case in cases]
    if len(run_names) != len(set(run_names)):
        raise RuntimeError("Duplicate run names in launch plan")
    collisions = [case["run_dir"] for case in cases if Path(case["run_dir"]).exists()]
    if collisions and not allow_existing:
        raise RuntimeError(
            "Refusing to reuse existing result directories:\n" + "\n".join(collisions)
        )


def update_progress(case: dict[str, Any]) -> None:
    run_dir = Path(case["run_dir"])
    csv_path = run_dir / "training_log.csv"
    if csv_path.exists():
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            case["progress_batches"] = len(rows)
            if rows:
                last = rows[-1]
                case["latest_reward_mean"] = float(last["reward_mean"])
                case["latest_speed_x100"] = float(last["speed_x100"])
        except (OSError, ValueError, KeyError):
            pass
    checkpoints: list[tuple[int, Path]] = []
    for path in run_dir.glob("checkpoint_*.pt"):
        try:
            checkpoints.append((int(path.stem.split("_")[-1]), path))
        except ValueError:
            continue
    if checkpoints:
        case["latest_checkpoint"] = max(checkpoints)[0]


def verify_complete(case: dict[str, Any], expected_episodes: int) -> tuple[bool, str | None]:
    run_dir = Path(case["run_dir"])
    checkpoint = run_dir / f"checkpoint_{expected_episodes}.pt"
    metadata_path = run_dir / "metadata.json"
    summary_path = run_dir / "training_summary.json"
    if not checkpoint.exists() or not metadata_path.exists() or not summary_path.exists():
        return False, "missing final checkpoint, metadata, or training summary"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"unreadable completion metadata: {exc}"
    checks = {
        "seed": metadata.get("training_args", {}).get("seed") == case["seed"],
        "channel": metadata.get("channel") == case["channel"],
        "observation": metadata.get("observation_func") == case["observation_func"],
        "control_mode": metadata.get("control_mode") == case["control_mode"],
        "reward": metadata.get("reward_func") == "horizontal_speed",
        "share_policy": metadata.get("share_policy") is True,
        "per_joint_k1_k2": metadata.get("per_joint_k1_k2") is False,
        "status": summary.get("status") == "complete",
        "episodes": summary.get("episodes") == expected_episodes,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        return False, "post-run contract mismatch: " + ", ".join(failed)
    return True, None


def status_payload(
    *,
    campaign_started: str,
    mode: str,
    max_workers: int,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = {state: sum(case["state"] == state for case in cases) for state in (
        "pending", "running", "complete", "failed"
    )}
    return {
        "schema": "formal_four_channel_orchestrator/v1",
        "mode": mode,
        "campaign_started_at_utc": campaign_started,
        "updated_at_utc": utc_now(),
        "max_workers": max_workers,
        "counts": counts,
        "all_finished": counts["pending"] == 0 and counts["running"] == 0,
        "runs": cases,
    }


def run_campaign(
    cases: list[dict[str, Any]],
    *,
    mode: str,
    expected_episodes: int,
    max_workers: int,
) -> int:
    campaign_started = utc_now()
    running: dict[str, dict[str, Any]] = {}
    env = environment()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    atomic_json(
        MANIFEST_PATH,
        {
            "schema": "formal_four_channel_launch_manifest/v1",
            "mode": mode,
            "created_at_utc": campaign_started,
            "python": str(PYTHON),
            "python_sha256": sha256(PYTHON),
            "site_packages": str(SITE_PACKAGES),
            "snapshot": str(SNAPSHOT),
            "trainer": str(TRAINER),
            "trainer_sha256": sha256(TRAINER),
            "rlmm_common_sha256": sha256(SNAPSHOT / "training" / "rlmm_common.py"),
            "environment_sha256": sha256(
                SNAPSHOT / "metamaterial_envs" / "metamaterial_envs" / "env" / "metamaterial.py"
            ),
            "seeds": list(SEEDS if mode == "formal" else (9201,)),
            "max_workers": max_workers,
            "results_dir": cases[0]["results_dir"] if cases else None,
            "runs": [
                {
                    key: case[key]
                    for key in (
                        "run_name", "seed", "tag", "channel", "observation_func",
                        "control_mode", "run_dir", "command"
                    )
                }
                for case in cases
            ],
        },
    )

    while True:
        while len(running) < max_workers:
            next_case = next((case for case in cases if case["state"] == "pending"), None)
            if next_case is None:
                break
            stdout_path = LOG_DIR / f"{next_case['run_name']}.stdout.log"
            stderr_path = LOG_DIR / f"{next_case['run_name']}.stderr.log"
            if stdout_path.exists() or stderr_path.exists():
                next_case["state"] = "failed"
                next_case["issue"] = "refusing to overwrite an existing launcher log"
                next_case["finished_at_utc"] = utc_now()
                continue
            stdout_handle = stdout_path.open("xb")
            stderr_handle = stderr_path.open("xb")
            try:
                process = subprocess.Popen(
                    next_case["command"],
                    cwd=SNAPSHOT,
                    env=env,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    creationflags=creation_flags,
                )
            except Exception:
                stdout_handle.close()
                stderr_handle.close()
                raise
            next_case["state"] = "running"
            next_case["pid"] = process.pid
            next_case["started_at_utc"] = utc_now()
            running[next_case["run_name"]] = {
                "case": next_case,
                "process": process,
                "stdout": stdout_handle,
                "stderr": stderr_handle,
            }

        for run_name, entry in list(running.items()):
            case = entry["case"]
            update_progress(case)
            exit_code = entry["process"].poll()
            if exit_code is None:
                continue
            entry["stdout"].close()
            entry["stderr"].close()
            case["exit_code"] = exit_code
            case["finished_at_utc"] = utc_now()
            update_progress(case)
            valid, issue = verify_complete(case, expected_episodes)
            if exit_code == 0 and valid:
                case["state"] = "complete"
            else:
                case["state"] = "failed"
                case["issue"] = issue or f"trainer exited with code {exit_code}"
            del running[run_name]

        payload = status_payload(
            campaign_started=campaign_started,
            mode=mode,
            max_workers=max_workers,
            cases=cases,
        )
        atomic_json(STATUS_PATH, payload)
        if payload["all_finished"]:
            atomic_json(COMPLETE_PATH, payload)
            return 0 if payload["counts"]["failed"] == 0 else 1
        time.sleep(15)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args()
    if args.max_workers not in (1, 2):
        raise RuntimeError("The audited single-GPU concurrency limit is 2")

    if args.mode == "smoke":
        smoke_root = CONTROL_DIR / "smoke_runs"
        cases = [
            build_case(
                arm,
                9201,
                results_dir=smoke_root,
                episodes=1,
                save_every=1,
                smoke=True,
            )
            for arm in ARMS
        ]
        expected_episodes = 1
    else:
        cases = [
            build_case(
                arm,
                seed,
                results_dir=FORMAL_RESULTS,
                episodes=1500,
                save_every=100,
                smoke=False,
            )
            for arm in ARMS
            for seed in SEEDS
        ]
        expected_episodes = 1500

    validate_paths(cases)
    return run_campaign(
        cases,
        mode=args.mode,
        expected_episodes=expected_episodes,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        atomic_json(
            CONTROL_DIR / "ORCHESTRATOR_FATAL.json",
            {
                "schema": "formal_four_channel_orchestrator_fatal/v1",
                "time_utc": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        print(f"FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
