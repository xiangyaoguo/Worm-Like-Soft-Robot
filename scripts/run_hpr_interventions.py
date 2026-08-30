"""Run the 36-condition frozen HPR matrix for thesis runs 0, 2, and 4."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from project_config import (
    PROJECT_ROOT,
    configured_python,
    ensure_output_path,
    load_paths,
    subprocess_environment,
)


HPR_ROOT = PROJECT_ROOT / "interventions" / "hpr_freeze"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[9201, 9203, 9205])
    parser.add_argument("--workers", type=int)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    paths = load_paths()
    output = ensure_output_path(Path(paths["hpr_output"]))
    env = subprocess_environment(paths)
    env["THESIS_HPR_OUTPUT"] = str(output)
    python = str(configured_python(paths))

    if not args.summarize_only:
        command = [
            python,
            str(HPR_ROOT / "scripts" / "run_formal_hpr_freeze_study.py"),
            "--training-seeds", *[str(seed) for seed in args.training_seeds],
            "--matrix", "full",
            "--workers", str(args.workers or int(paths.get("maximum_workers", 2))),
        ]
        if args.pilot:
            command.append("--pilot")
        print(subprocess.list2cmdline(command))
        code = subprocess.run(command, cwd=str(PROJECT_ROOT), env=env, check=False).returncode
        if code or args.pilot:
            return code

    summary = [
        python,
        str(HPR_ROOT / "summarize_three_run_study.py"),
        "--data-root", str(output / "data"),
    ]
    print(subprocess.list2cmdline(summary))
    return subprocess.run(summary, cwd=str(PROJECT_ROOT), env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
