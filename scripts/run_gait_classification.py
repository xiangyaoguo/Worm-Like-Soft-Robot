"""Run the automated six-configuration gait classifier with safe defaults."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, help="Completed six-config evaluator output.")
    parser.add_argument("--freeze-root", type=Path, help="HPR freeze root containing data/trajectories.")
    parser.add_argument(
        "--portable-new-results",
        action="store_true",
        help="Do not require the archived thesis label counts for newly trained policies.",
    )
    parser.add_argument("--random-seed", type=int, default=20260824)
    args = parser.parse_args()

    config = load_paths()
    input_root = (args.input_root or Path(config["formal_evaluation_output"])).resolve()
    freeze_root = (
        args.freeze_root
        or (PROJECT_ROOT / "reference_results" / "hpr_freeze")
    ).resolve()
    output_root = ensure_output_path(Path(config["gait_output"]))
    command = [
        str(configured_python(config)),
        str(PROJECT_ROOT / "evaluation" / "gait_classification" / "run_gait_classification.py"),
        "--input-root", str(input_root),
        "--freeze-root", str(freeze_root),
        "--output-root", str(output_root),
        "--review-all",
        "--random-seed", str(args.random_seed),
    ]
    if args.portable_new_results:
        command.append("--portable-new-results")
    print(subprocess.list2cmdline(command))
    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=subprocess_environment(config),
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
