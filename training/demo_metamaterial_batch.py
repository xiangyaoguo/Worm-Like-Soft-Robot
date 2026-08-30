r"""Batch demo/evaluation runner for trained metamaterial checkpoints.

This script lets one or more trained checkpoints be placed into one or more
simulation terrains without editing the checkpoint metadata.

Examples
--------
Run one checkpoint on all demo terrains and save motion-frame figures:
    python .\training\demo_metamaterial_batch.py --checkpoint .\results\run\checkpoint_500.pt --terrain all --mode frames

Run three checkpoints in three different environments, one after another:
    python .\training\demo_metamaterial_batch.py --checkpoint ckptA.pt ckptB.pt ckptC.pt --terrain flat stairs tunnel --mode human

Evaluate multiple checkpoints on the tunnel environment:
    python .\training\demo_metamaterial_batch.py --checkpoint ckptA.pt ckptB.pt --terrain tunnel --mode evaluate
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from rlmm_common import find_latest_checkpoint, find_project_root, load_checkpoint, to_plain
from analyze_training_results import (
    TerrainArgs,
    evaluate_checkpoint_on_terrains,
    metadata_from_checkpoint,
    save_motion_contact_sheet,
    safe_json_dump,
)

PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)


VALID_TERRAINS = {"checkpoint", "training", "flat", "stairs", "tunnel", "all"}


def resolve_checkpoint_path(value: str | Path) -> Path:
    if str(value).lower() == "latest":
        return find_latest_checkpoint(PROJECT_ROOT).resolve()
    return Path(value).resolve()


def expand_jobs(checkpoints: list[Path], terrains: list[str]) -> list[tuple[Path, str]]:
    terrain_list = [str(t).strip().lower() for t in terrains if str(t).strip()]
    if not terrain_list:
        terrain_list = ["checkpoint"]
    for terrain in terrain_list:
        if terrain not in VALID_TERRAINS:
            raise ValueError(f"Unsupported terrain {terrain!r}. Use checkpoint/training/flat/stairs/tunnel/all.")

    if terrain_list == ["all"]:
        return [(ckpt, terrain) for ckpt in checkpoints for terrain in ["flat", "stairs", "tunnel"]]
    if len(checkpoints) == 1 and len(terrain_list) > 1:
        return [(checkpoints[0], terrain) for terrain in terrain_list]
    if len(terrain_list) == 1:
        return [(ckpt, terrain_list[0]) for ckpt in checkpoints]
    if len(terrain_list) == len(checkpoints):
        return list(zip(checkpoints, terrain_list))
    raise ValueError(
        "Cannot map checkpoints to terrains. Use one terrain for all checkpoints, "
        "one checkpoint with multiple terrains, --terrain all, or the same number of checkpoints and terrains."
    )


def run_human_demo(job: tuple[Path, str], args: argparse.Namespace) -> None:
    checkpoint, terrain = job
    demo_terrain = "checkpoint" if terrain == "training" else terrain
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "training" / "demo_metamaterial.py"),
        "--checkpoint",
        str(checkpoint),
        "--terrain",
        demo_terrain,
        "--policy-mode",
        args.policy_mode,
        "--max-steps",
        str(args.max_steps),
        "--window-width",
        str(args.window_width),
        "--window-height",
        str(args.window_height),
        "--print-every",
        str(args.print_every),
    ]
    if args.follow_camera:
        cmd.append("--follow-camera")
    if args.no_pause:
        cmd.append("--no-pause")
    if terrain == "tunnel":
        cmd += [
            "--tunnel-start", str(args.tunnel_start),
            "--tunnel-slope", str(args.tunnel_slope),
            "--tunnel-slope-height", str(args.tunnel_slope_height),
            "--tunnel-length", str(args.tunnel_length),
            "--tunnel-height", str(args.tunnel_height),
        ]
    if terrain == "stairs":
        cmd += [
            "--start-stairs", str(args.start_stairs),
            "--step-width", str(args.step_width),
            "--step-height", str(args.step_height),
            "--steps", str(args.steps),
        ]

    print("\nRunning demo:")
    print(" ", " ".join(f'"{part}"' if " " in part else part for part in cmd))
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if completed.returncode != 0:
        raise RuntimeError(f"Demo failed for {checkpoint} on {terrain} with exit code {completed.returncode}")


def safe_name(path: Path) -> str:
    return f"{path.parent.name}_{path.stem}".replace(" ", "_").replace("/", "_").replace("\\", "_")


def run_frames_and_eval(jobs: list[tuple[Path, str]], args: argparse.Namespace) -> list[Path]:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    terrain_args = TerrainArgs.from_namespace(args)
    saved: list[Path] = []
    eval_rows: list[dict[str, Any]] = []

    for checkpoint, terrain in jobs:
        metadata = metadata_from_checkpoint(checkpoint)
        tag = f"{safe_name(checkpoint)}__demo_{terrain}"
        if args.mode in {"frames", "all"}:
            path = save_motion_contact_sheet(
                checkpoint,
                metadata,
                [terrain],
                terrain_args,
                output_dir / f"{tag}_motion_frames.png",
                steps=args.max_steps,
                frames_per_terrain=args.motion_frames,
                policy_mode=args.policy_mode,
                follow_camera=args.follow_camera,
                dpi=args.dpi,
            )
            if path is not None:
                saved.append(path)
                print("Saved motion frames:", path)
        if args.mode in {"evaluate", "all"}:
            rows = evaluate_checkpoint_on_terrains(
                checkpoint,
                metadata,
                [terrain],
                terrain_args,
                episodes=args.eval_episodes,
                steps=args.eval_steps,
                policy_mode=args.policy_mode,
            )
            for row in rows:
                row["checkpoint_name"] = safe_name(checkpoint)
            eval_rows.extend(rows)

    if eval_rows:
        csv_path = output_dir / "batch_demo_evaluation.csv"
        fieldnames = sorted({key for row in eval_rows for key in row.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(eval_rows)
        saved.append(csv_path)
        print("Saved evaluation CSV:", csv_path)

    summary_path = safe_json_dump(
        {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": args.mode,
            "jobs": [{"checkpoint": str(c), "terrain": t} for c, t in jobs],
            "terrain_args": asdict(terrain_args),
            "policy_mode": args.policy_mode,
            "max_steps": args.max_steps,
            "eval_episodes": args.eval_episodes,
            "eval_steps": args.eval_steps,
            "saved_files": [str(p) for p in saved],
        },
        output_dir / "batch_demo_summary.json",
    )
    saved.append(summary_path)
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one or more checkpoints in one or more demo terrains.")
    parser.add_argument("--checkpoint", nargs="+", required=True, help="One or more checkpoint paths, or latest.")
    parser.add_argument("--terrain", nargs="+", default=["checkpoint"], help="checkpoint/training, flat, stairs, tunnel, or all.")
    parser.add_argument("--mode", choices=["human", "frames", "evaluate", "all"], default="human")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "demo_outputs")

    parser.add_argument("--policy-mode", choices=["sample", "deterministic"], default="deterministic")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--motion-frames", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--follow-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--window-width", type=int, default=1000)
    parser.add_argument("--window-height", type=int, default=500)
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--no-pause", action="store_true", help="In human mode, close each demo without asking for Enter.")

    parser.add_argument("--start-stairs", type=float, default=5)
    parser.add_argument("--step-width", type=float, default=5)
    parser.add_argument("--step-height", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--tunnel-start", type=float, default=10)
    parser.add_argument("--tunnel-slope", type=float, default=5)
    parser.add_argument("--tunnel-slope-height", type=float, default=1)
    parser.add_argument("--tunnel-length", type=float, default=10)
    parser.add_argument("--tunnel-height", type=float, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints = [resolve_checkpoint_path(value) for value in args.checkpoint]
    jobs = expand_jobs(checkpoints, args.terrain)

    print("Batch demo jobs:")
    for idx, (checkpoint, terrain) in enumerate(jobs, start=1):
        print(f"  {idx:02d}. {checkpoint} -> {terrain}")

    if args.mode == "human":
        for job in jobs:
            run_human_demo(job, args)
        print("All human demos finished.")
    else:
        saved = run_frames_and_eval(jobs, args)
        print("Batch demo outputs saved:")
        for path in saved:
            print(" ", path)


if __name__ == "__main__":
    main()
