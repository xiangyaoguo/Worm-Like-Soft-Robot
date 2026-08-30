"""Replay any bundled formal endpoint with checkpoint metadata as authority."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from project_config import PROJECT_ROOT, configured_python, load_paths, subprocess_environment


ARM_TAGS = {
    "HPR-DTH-PS": "DTH",
    "HPR-THDOT-PS": "THDOT",
    "HPR-OBS-PS": "OBS",
    "HPR-O2-PS": "HPR__O2shared",
    "HPR-O2-JS": "R0",
    "SGRR-O2-JS": "Rroll",
    "HPR-O2-AVM-JS": "HPR__O1sham",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=tuple(ARM_TAGS), default="SGRR-O2-JS")
    parser.add_argument("--seed", type=int, choices=range(9201, 9206), default=9201)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--sample", action="store_true", help="Sample actions instead of deterministic replay.")
    parser.add_argument("--preflight", action="store_true", help="Run one headless step and exit.")
    parser.add_argument("--no-follow-camera", action="store_true")
    args, passthrough = parser.parse_known_args()

    config = load_paths()
    tag = ARM_TAGS[args.arm]
    if args.arm == "HPR-O2-AVM-JS":
        root = Path(config["avm_checkpoint_root"])
    else:
        root = Path(config["formal_checkpoint_root"])
    checkpoint = root / f"formal__seed{args.seed}__{tag}" / "checkpoint_1500.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    command = [
        str(configured_python(config)),
        str(PROJECT_ROOT / "training" / "demo_metamaterial.py"),
        "--checkpoint",
        str(checkpoint),
        "--policy-mode",
        "sample" if args.sample else "deterministic",
        "--max-steps",
        str(args.steps),
        "--no-pause",
    ]
    if args.preflight:
        command.append("--preflight")
    elif not args.no_follow_camera:
        command.append("--follow-camera")
    command.extend(passthrough)
    print(subprocess.list2cmdline(command))
    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=subprocess_environment(config),
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
