"""Rebuild the thesis HPR/SGRR K1/K2 shared-scale response figure."""

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
    parser.parse_args()
    paths = load_paths()
    output = ensure_output_path(Path(paths["figure_output"]))
    env = subprocess_environment(paths)
    env["THESIS_FIGURE_OUTPUT"] = str(output)
    command = [
        str(configured_python(paths)),
        str(
            PROJECT_ROOT
            / "analysis"
            / "response_surfaces"
            / "build_shared_scale_response_surfaces.py"
        ),
    ]
    print(subprocess.list2cmdline(command), flush=True)
    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=env,
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
