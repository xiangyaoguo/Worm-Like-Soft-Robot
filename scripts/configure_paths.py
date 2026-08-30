"""Create or inspect the single machine-local path configuration file."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from project_config import DEFAULT_PATHS, LOCAL_PATHS, PROJECT_ROOT, load_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--create", action="store_true", help="Create configs/paths.local.json.")
    parser.add_argument("--force", action="store_true", help="Replace an existing local config.")
    parser.add_argument("--show", action="store_true", help="Print resolved paths.")
    args = parser.parse_args()

    if args.create:
        if LOCAL_PATHS.exists() and not args.force:
            print(f"Keeping existing path config: {LOCAL_PATHS}")
        else:
            shutil.copyfile(DEFAULT_PATHS, LOCAL_PATHS)
            print(f"Created: {LOCAL_PATHS}")

    config = load_paths()
    if args.show or not args.create:
        printable = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in config.items()
        }
        printable["project_root"] = str(PROJECT_ROOT)
        printable["running_python"] = sys.executable
        print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
