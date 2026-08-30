"""Single front door for the portable thesis reproducibility package."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COMMANDS = {
    "configure": ROOT / "scripts" / "configure_paths.py",
    "verify": ROOT / "scripts" / "verify_install.py",
    "simulate": ROOT / "scripts" / "run_simulation.py",
    "train": ROOT / "scripts" / "run_training.py",
    "evaluate": ROOT / "scripts" / "run_formal_evaluation.py",
    "evaluate-avm": ROOT / "scripts" / "run_avm_endpoint_evaluation.py",
    "classify-gait": ROOT / "scripts" / "run_gait_classification.py",
    "intervene-hpr": ROOT / "scripts" / "run_hpr_interventions.py",
    "intervene-sgrr": ROOT / "scripts" / "run_sgrr_interventions.py",
    "figure-response": ROOT / "scripts" / "run_response_figure.py",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=tuple(COMMANDS))
    if len(sys.argv) == 1 or sys.argv[1] in {"-h", "--help"}:
        parser.print_help()
        return 0
    command_name = sys.argv[1]
    if command_name not in COMMANDS:
        parser.error(
            f"argument command: invalid choice: {command_name!r} "
            f"(choose from {', '.join(COMMANDS)})"
        )
    command = [sys.executable, str(COMMANDS[command_name]), *sys.argv[2:]]
    print(subprocess.list2cmdline(command), flush=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return subprocess.run(command, cwd=str(ROOT), env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
