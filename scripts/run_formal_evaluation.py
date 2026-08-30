"""Prepare and run the fail-closed six-configuration endpoint evaluator."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from project_config import (
    PROJECT_ROOT,
    configured_python,
    ensure_output_path,
    load_json,
    load_paths,
    python_site_packages,
    sha256_file,
    subprocess_environment,
    write_json,
)


EVALUATOR_DIR = PROJECT_ROOT / "evaluation" / "formal_endpoint"
ORIGINAL_CONFIG = EVALUATOR_DIR / "evaluator_config.json"
GENERATED_CONFIG = PROJECT_ROOT / "configs" / "generated" / "formal_evaluator.json"


def build_portable_config(paths: dict, workers: int) -> dict:
    config = load_json(ORIGINAL_CONFIG)
    python = configured_python(paths)
    if not python.is_file():
        raise FileNotFoundError(python)
    output = ensure_output_path(Path(paths["formal_evaluation_output"]))
    config["status"] = "portable_paths_derived_from_frozen_evaluator_config"
    config["runtime"].update(
        {
            "python": str(python),
            "python_sha256": sha256_file(python),
            "site_packages": str(python_site_packages(python)),
            "maximum_workers": int(workers),
            "torch_num_threads_per_worker": int(paths.get("threads_per_worker", 1)),
        }
    )
    config["formal_runs_root"] = str(Path(paths["formal_checkpoint_root"]))
    config["source_snapshot_root"] = str(PROJECT_ROOT / "provenance" / "formal_core_exact")
    config["official_parent_formal_root"] = str(
        PROJECT_ROOT / "provenance" / "formal_parent_exact"
    )
    config["immutable_analysis_extension"]["rotation_span_helper"]["absolute"] = str(
        EVALUATOR_DIR / "rotation_span_helper.py"
    )
    config["output_root"] = str(output)
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("contract", "self-test", "process-smoke", "execute", "validate"),
        default="contract",
    )
    parser.add_argument("--workers", type=int)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    paths = load_paths()
    workers = args.workers or int(paths.get("maximum_workers", 2))
    config = build_portable_config(paths, workers)
    write_json(GENERATED_CONFIG, config)
    print(f"Generated evaluator config: {GENERATED_CONFIG}")
    if args.prepare_only:
        return 0

    flag = {
        "contract": "--contract-only",
        "self-test": "--self-test",
        "process-smoke": "--process-smoke",
        "execute": "--execute",
        "validate": "--validate-only",
    }[args.mode]
    command = [
        str(configured_python(paths)),
        str(EVALUATOR_DIR / "six_config_endpoint_evaluator.py"),
        "--config",
        str(GENERATED_CONFIG),
        flag,
    ]
    if args.mode == "execute":
        command.extend(["--workers", str(workers)])
    print(subprocess.list2cmdline(command))
    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=subprocess_environment(paths),
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
