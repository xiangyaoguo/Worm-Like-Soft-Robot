"""Portable entry point for the SGRR 113-condition intervention programs."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from project_config import (
    PROJECT_ROOT,
    configured_python,
    ensure_output_path,
    load_paths,
    sha256_file,
    subprocess_environment,
    write_json,
)


SOURCE = PROJECT_ROOT / "interventions" / "sgrr_causal_completion"
MECHANISM = PROJECT_ROOT / "interventions" / "mechanism_runtime"
ARCHIVE_SOURCE = PROJECT_ROOT / "provenance" / "sgrr_causal_completion_exact"


def portable_manifest(output: Path, checkpoint_root: Path) -> Path:
    files: list[dict[str, str]] = []

    def add(role: str, path: Path) -> None:
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append({"role": role, "path": str(path), "sha256": sha256_file(path)})

    add("portable_causal_runner", SOURCE / "run_causal_completion.py")
    add("portable_matched_c00_runner", SOURCE / "run_matched_c00_baseline.py")
    add("portable_study_config", SOURCE / "study_config.json")
    add("frozen_study_contract", SOURCE / "study_contract.json")
    add("frozen_matched_c00_amendment", SOURCE / "MATCHED_C00_BASELINE_AMENDMENT.json")
    add("readme", SOURCE / "README.md")
    add("portable_mechanism_runtime", MECHANISM / "mechanism_rollout.py")
    add("legacy_condition_reference", MECHANISM / "condition_matrix.py")
    add(
        "formal_frozen_evaluator",
        PROJECT_ROOT / "provenance" / "formal_core_exact" / "training" / "evaluate_fast_forward_roll.py",
    )
    for name in (
        "run_causal_completion.py",
        "run_matched_c00_baseline.py",
        "study_config.json",
        "study_contract.json",
        "SOURCE_MANIFEST.json",
        "MATCHED_C00_BASELINE_AMENDMENT.json",
    ):
        add(f"archive_exact_{name}", ARCHIVE_SOURCE / name)
    for seed in range(9201, 9206):
        for arm in ("R0", "Rroll"):
            add(
                f"checkpoint_{seed}_{arm}",
                checkpoint_root / f"formal__seed{seed}__{arm}" / "checkpoint_1500.pt",
            )
    manifest = {
        "schema": "obs2_v2_1_k_causal_completion/portable_source_manifest/v1",
        "study_id": "obs2_v2_1_k_causal_completion_20260804",
        "note": "Paths/runtime were rebound locally; frozen scientific contract and checkpoints are unchanged.",
        "files": files,
    }
    path = output / "PORTABLE_SOURCE_MANIFEST.json"
    write_json(path, manifest)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("causal", "matched", "analysis"), default="causal")
    parser.add_argument("--stage", help="Stage accepted by the selected original program.")
    parser.add_argument("--training-seed", type=int, action="append", default=[])
    parser.add_argument("--condition", action="append", default=[])
    parser.add_argument("--workers", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    paths = load_paths()
    output = ensure_output_path(Path(paths["sgrr_output"]))
    manifest = portable_manifest(output, Path(paths["formal_checkpoint_root"]))
    workers = args.workers or min(5, int(paths.get("maximum_workers", 2)))
    env = subprocess_environment(paths)
    env["THESIS_SGRR_OUTPUT"] = str(output)
    env["THESIS_SGRR_MANIFEST"] = str(manifest)
    env["THESIS_SGRR_ARCHIVE_SOURCE"] = str(ARCHIVE_SOURCE)
    env["THESIS_SGRR_LEGACY_ROOT"] = str(Path(paths["sgrr_legacy_root"]))

    if args.component == "causal":
        stage = args.stage or "verify"
        command = [
            str(configured_python(paths)),
            str(SOURCE / "run_causal_completion.py"),
            "--stage", stage,
            "--workers", str(workers),
        ]
        if len(args.training_seed) > 1:
            parser.error("causal calibration/main/smoke accepts one --training-seed")
        if args.training_seed:
            command.extend(["--training-seed", str(args.training_seed[0])])
        for condition in args.condition:
            command.extend(["--condition", condition])
    elif args.component == "matched":
        stage = args.stage or "verify"
        command = [
            str(configured_python(paths)),
            str(SOURCE / "run_matched_c00_baseline.py"),
            "--stage", stage,
            "--workers", str(workers),
        ]
        for seed in args.training_seed:
            command.extend(["--training-seed", str(seed)])
    else:
        command = [
            str(configured_python(paths)),
            str(SOURCE / "analyze_causal_completion_results.py"),
        ]
        if args.validate_only:
            command.append("--validate-only")

    print(subprocess.list2cmdline(command))
    return subprocess.run(command, cwd=str(PROJECT_ROOT), env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
