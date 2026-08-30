r"""Train with training/train_metamaterial.py, then generate policy heatmaps automatically.

Put this file in the training/ directory and run it exactly like train_metamaterial.py:

    python ./training/train_then_analyze.py --robot crawler --terrain stairs --episodes 500 --episode-steps 400
    python ./training/train_then_analyze.py --robot ring --terrain tunnel --episodes 500 --episode-steps 500 --tunnel-height 5

All arguments before `--` are passed to train_metamaterial.py. Optional analysis arguments can
be placed after `--`, for example:

    python ./training/train_then_analyze.py --robot ring --terrain flat --episodes 200 -- --grid-size 151
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    train_script = here / "train_metamaterial.py"
    analysis_script = here / "analyze_policy_heatmaps.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Cannot find {train_script}")
    if not analysis_script.exists():
        raise FileNotFoundError(f"Cannot find {analysis_script}")

    args = sys.argv[1:]
    if "--" in args:
        split = args.index("--")
        train_args = args[:split]
        analysis_args = args[split + 1 :]
    else:
        train_args = args
        analysis_args = []

    print("[1/2] Training...")
    subprocess.run([sys.executable, str(train_script), *train_args], check=True)

    print("[2/2] Generating policy heatmap analysis from newest checkpoint...")
    subprocess.run([sys.executable, str(analysis_script), "--checkpoint", "latest", *analysis_args], check=True)


if __name__ == "__main__":
    main()
