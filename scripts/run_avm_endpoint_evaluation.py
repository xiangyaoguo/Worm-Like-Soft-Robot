"""Evaluate the bundled O2 versus AVM checkpoint-1500 endpoints.

This portable mode evaluates only the thesis endpoint.  The immutable archived
AVM program under provenance/avm_exact retains the original 15-checkpoint
learning-curve contract.
"""

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
    subprocess_environment,
    write_json,
)


AVM_ROOT = PROJECT_ROOT / "extensions" / "avm" / "code_snapshot"
EVALUATOR = AVM_ROOT / "evaluator" / "frozen_evaluator.py"
ORIGINAL_CONFIG = AVM_ROOT / "evaluator" / "evaluator_config.json"
GENERATED_CONFIG = PROJECT_ROOT / "configs" / "generated" / "avm_endpoint_evaluator.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("contract", "execute"), default="contract")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()

    paths = load_paths()
    workers = args.workers or int(paths.get("maximum_workers", 2))
    python = configured_python(paths)
    output = ensure_output_path(Path(paths["avm_evaluation_output"]))
    config = load_json(ORIGINAL_CONFIG)
    config.update(
        {
            "status": "portable_endpoint_only_derived_from_frozen_config",
            "portable_evaluation_mode": "endpoint_only",
            "parent_formal_root": str(PROJECT_ROOT / "provenance" / "formal_parent_exact"),
            "output_root": str(output),
            "checkpoint_batches": [1500],
        }
    )
    config["runtime"].update(
        {
            "python": str(python),
            "site_packages": str(python_site_packages(python)),
            "maximum_workers": workers,
            "torch_num_threads_per_worker": int(paths.get("threads_per_worker", 1)),
        }
    )
    config["evaluation"]["trace_export"] = "all_episodes_endpoint_only"
    config["arms"]["O2"]["run_absolute_template"] = str(
        Path(paths["formal_checkpoint_root"]) / "formal__seed{seed}__R0"
    )
    config["arms"]["O1_sham"]["run_absolute_template"] = str(
        Path(paths["avm_checkpoint_root"]) / "formal__seed{seed}__HPR__O1sham"
    )
    write_json(GENERATED_CONFIG, config)
    command = [str(python), str(EVALUATOR), "--config", str(GENERATED_CONFIG)]
    if args.mode == "execute":
        command.extend(["--execute", "--workers", str(workers)])
    else:
        command.append("--contract-only")
    print(subprocess.list2cmdline(command))
    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=subprocess_environment(paths),
        check=False,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
