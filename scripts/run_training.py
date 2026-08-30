"""Portable launcher for the six thesis arms and the AVM extension."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from project_config import (
    PROJECT_ROOT,
    configured_python,
    ensure_output_path,
    load_json,
    load_paths,
    subprocess_environment,
    write_json,
)


TRAINING_CONFIG = PROJECT_ROOT / "configs" / "formal_training.json"
MAIN_ARMS = (
    "HPR-DTH-PS",
    "HPR-THDOT-PS",
    "HPR-OBS-PS",
    "HPR-O2-PS",
    "HPR-O2-JS",
    "SGRR-O2-JS",
)
AVM_ARM = "HPR-O2-AVM-JS"


def common_args(common: dict[str, Any], *, mode: str, output_root: Path) -> list[str]:
    if mode == "formal":
        batches, save_every = 1500, 100
        episode_steps = int(common["episode_steps"])
        frames_per_batch = int(common["frames_per_batch"])
    else:
        batches, save_every = 1, 1
        episode_steps, frames_per_batch = 100, 1000
    return [
        "--robot", str(common["robot"]),
        "--terrain", str(common["terrain"]),
        "--terrain-contact-mode", str(common["terrain_contact_mode"]),
        "--num-particles", str(common["num_particles"]),
        "--feedback-gain", "1.0",
        "--max-control-gain", str(common["max_active_torque"]),
        "--no-fix-k1", "--no-fix-k2",
        "--k-action-scale", str(common["k_action_scale"]),
        "--passive-kappa", str(common["passive_kappa"]),
        "--share-critic", "--centralised-critic",
        "--algorithm", str(common["algorithm"]),
        "--policy-depth", str(common["policy_depth"]),
        "--policy-cells", str(common["policy_cells"]),
        "--normal-scale-lb", str(common["normal_scale_lb"]),
        "--episode-steps", str(episode_steps),
        "--frames-per-batch", str(frames_per_batch),
        "--memory-size", str(common["memory_size"]),
        "--minibatch-size", str(common["minibatch_size"]),
        "--optim-steps", str(common["optim_steps"]),
        "--lr", str(common["learning_rate"]),
        "--weight-decay", str(common["weight_decay"]),
        "--max-grad-norm", str(common["max_grad_norm"]),
        "--gamma", str(common["gamma"]),
        "--clip-epsilon", str(common["clip_epsilon"]),
        "--lambda-gae", str(common["lambda_gae"]),
        "--entropy-eps", str(common["entropy_epsilon"]),
        "--no-ppo-normalize-advantage", "--ppo-target-kl", "0.0",
        "--init-pos-randomness", str(common["init_pos_randomness"]),
        "--init-angle-range-degrees", str(common["init_angle_range_degrees"]),
        "--init-height-jitter", str(common["init_height_jitter"]),
        "--action-smoothness-weight", "0.0",
        "--policy-anchor-coeff", "0.0",
        "--policy-anchor-anneal-batches", "0",
        "--bc-steps", "0", "--bc-epochs", "0",
        "--rolling-direction", "right",
        "--rolling-curl-episodes", "300",
        "--rolling-transition-episodes", "300",
        "--rolling-reward-scale", "3.0",
        "--tail-side", "left",
        "--tail-roll-init-assist-degrees", "0.0",
        "--tail-roll-init-assist-episodes", "0",
        "--no-rolling-observation", "--no-tail-roll-observation", "--no-fast-forward-observation",
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
        "--no-pretrained-policy-only", "--no-compatible-input-expansion",
        "--buffer-storage", "tensor", "--no-auto-analysis",
        "--episodes", str(batches), "--save-every", str(save_every),
        "--results-dir", str(output_root),
    ]


def select_arms(requested: list[str]) -> list[str]:
    if "all" in requested:
        return [*MAIN_ARMS, AVM_ARM]
    if "main6" in requested:
        requested = [value for value in requested if value != "main6"] + list(MAIN_ARMS)
    result: list[str] = []
    for value in requested:
        if value not in result:
            result.append(value)
    return result


def build_case(
    arm_name: str,
    arm: dict[str, Any],
    seed: int,
    mode: str,
    output_root: Path,
    python: Path,
    common: dict[str, Any],
) -> dict[str, Any]:
    prefix = "formal" if mode == "formal" else "smoke"
    run_name = f"{prefix}__seed{seed}__{arm['archive_tag']}"
    trainer = PROJECT_ROOT / "training" / "train_metamaterial.py"
    env_extra: dict[str, str] = {}
    args = common_args(common, mode=mode, output_root=output_root)
    if arm_name == AVM_ARM:
        trainer = PROJECT_ROOT / "extensions" / "avm" / "code_snapshot" / "actor_observation_shim.py"
        args.extend(["--actor-observation-mode", "spatial_only_sham"])
        env_extra["FORMAL_PARENT_TRAINER"] = str(
            PROJECT_ROOT / "training" / "train_metamaterial.py"
        )
    args.extend(
        [
            "--channel", str(arm["channel"]),
            "--observation-func", str(arm["observation_func"]),
            "--control-mode", str(arm["control_mode"]),
            "--reward-func", str(arm["reward_func"]),
            "--share-policy" if arm["share_policy"] else "--no-share-policy",
            "--seed", str(seed),
            "--run-name", run_name,
        ]
    )
    if arm["per_joint_k1_k2"]:
        args.append("--per-joint-k1-k2")
    return {
        "arm": arm_name,
        "seed": seed,
        "run_name": run_name,
        "run_dir": output_root / run_name,
        "command": [str(python), str(trainer), *args],
        "env_extra": env_extra,
    }


def execute_case(case: dict[str, Any], base_env: dict[str, str], log_root: Path) -> dict[str, Any]:
    run_dir = Path(case["run_dir"])
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty run: {run_dir}")
    log_root.mkdir(parents=True, exist_ok=True)
    stdout = log_root / f"{case['run_name']}.stdout.log"
    stderr = log_root / f"{case['run_name']}.stderr.log"
    env = dict(base_env)
    env.update(case["env_extra"])
    started = datetime.now(timezone.utc).isoformat()
    with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
        code = subprocess.run(
            case["command"], cwd=str(PROJECT_ROOT), env=env, stdout=out, stderr=err
        ).returncode
    return {
        "arm": case["arm"],
        "seed": case["seed"],
        "run_name": case["run_name"],
        "exit_code": int(code),
        "started_at_utc": started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "stdout": str(stdout),
        "stderr": str(stderr),
    }


def main() -> int:
    training = load_json(TRAINING_CONFIG)
    available = tuple(training["arms"])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("commands", "smoke", "formal"), default="commands")
    parser.add_argument(
        "--commands-for",
        choices=("smoke", "formal"),
        default="formal",
        help="Parameter set printed when --mode commands is selected.",
    )
    parser.add_argument(
        "--arm", nargs="+", choices=("main6", "all", *available), default=["main6"]
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=training["training_seeds"])
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()

    invalid = sorted(set(args.seeds) - set(training["training_seeds"]))
    if invalid:
        parser.error(f"Only preregistered seeds are allowed: {invalid}")
    arm_names = select_arms(args.arm)
    config = load_paths()
    python = configured_python(config)
    output_root = Path(config["training_output_root"])
    if args.mode != "commands":
        output_root = ensure_output_path(output_root)
    mode_for_args = args.commands_for if args.mode == "commands" else args.mode
    cases = [
        build_case(
            arm_name,
            training["arms"][arm_name],
            seed,
            mode_for_args,
            output_root,
            python,
            training["common"],
        )
        for arm_name in arm_names
        for seed in args.seeds
    ]
    for case in cases:
        print(subprocess.list2cmdline(case["command"]))
    if args.mode == "commands":
        return 0

    manifest = {
        "schema": "thesis_portable_training_launch/v1",
        "mode": args.mode,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_config": str(TRAINING_CONFIG),
        "cases": [
            {**case, "run_dir": str(case["run_dir"])} for case in cases
        ],
    }
    write_json(output_root / f"LAUNCH_MANIFEST_{args.mode}.json", manifest)
    workers = args.workers or int(config.get("maximum_workers", 2))
    base_env = subprocess_environment(config)
    log_root = output_root / "_logs" / args.mode
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(
            executor.map(lambda case: execute_case(case, base_env, log_root), cases)
        )
    write_json(output_root / f"LAUNCH_RESULT_{args.mode}.json", results)
    return 1 if any(item["exit_code"] != 0 for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
