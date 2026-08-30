"""Verify the Python environment, release layout, and bundled endpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

from project_config import PROJECT_ROOT, configured_python, load_paths, subprocess_environment


EXPECTED_MODULES = (
    "torch",
    "torchrl",
    "tensordict",
    "numpy",
    "numba",
    "gymnasium",
    "pygame",
    "matplotlib",
    "pandas",
    "scipy",
    "tqdm",
    "PIL",
    "metamaterial_envs",
)
FORMAL_TAGS = ("DTH", "THDOT", "OBS", "HPR__O2shared", "R0", "Rroll")
SEEDS = (9201, 9202, 9203, 9204, 9205)
CHECKPOINT_MANIFEST = PROJECT_ROOT / "CHECKPOINT_SHA256.csv"


def module_version(name: str, module: object) -> str:
    if name == "PIL":
        return str(getattr(module, "__version__", "unknown"))
    return str(getattr(module, "__version__", "installed"))


def check_run(run_dir: Path) -> list[str]:
    issues: list[str] = []
    required = ("checkpoint_1500.pt", "metadata.json", "training_log.csv", "training_summary.json")
    for name in required:
        path = run_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            issues.append(f"missing/empty: {path}")
    summary = run_dir / "training_summary.json"
    if summary.is_file():
        value = json.loads(summary.read_text(encoding="utf-8-sig"))
        if str(value.get("status", "")).lower() != "complete":
            issues.append(f"training status is not complete: {summary}")
    return issues


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_contract() -> dict[str, tuple[int, str]]:
    if not CHECKPOINT_MANIFEST.is_file():
        raise FileNotFoundError(CHECKPOINT_MANIFEST)
    with CHECKPOINT_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 35:
        raise RuntimeError(f"Expected 35 checkpoint hashes, found {len(rows)}")
    result = {
        row["relative_path"]: (int(row["bytes"]), row["sha256"].lower())
        for row in rows
    }
    if len(result) != 35:
        raise RuntimeError("Checkpoint hash manifest contains duplicate paths")
    return result


def check_checkpoint_hash(path: Path, contract: dict[str, tuple[int, str]]) -> list[str]:
    relative = path.resolve().relative_to(PROJECT_ROOT).as_posix()
    expected = contract.get(relative)
    if expected is None:
        return [f"checkpoint is absent from hash manifest: {relative}"]
    expected_bytes, expected_hash = expected
    if path.stat().st_size != expected_bytes:
        return [f"checkpoint byte-size mismatch: {relative}"]
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        return [f"checkpoint SHA-256 mismatch: {relative}: {actual_hash}"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Skip policy reconstruction preflight.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Checkpoint used by the full preflight (default: SGRR seed 9201).",
    )
    args = parser.parse_args()

    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(
            f"Python 3.11 is required for the tested environment; found {sys.version.split()[0]}"
        )

    # Make the bundled simulator importable even before an editable install is
    # registered in the active environment. The setup scripts still install it
    # normally; this fallback keeps a copied release self-contained for checks.
    local_simulator_root = PROJECT_ROOT / "packages" / "metamaterial_envs"
    if local_simulator_root.is_dir():
        local_simulator_text = str(local_simulator_root)
        if local_simulator_text not in sys.path:
            sys.path.insert(0, local_simulator_text)

    versions: dict[str, str] = {}
    for name in EXPECTED_MODULES:
        module = importlib.import_module(name)
        versions[name] = module_version(name, module)

    config = load_paths()
    issues: list[str] = []
    hash_contract = checkpoint_contract()
    formal_root = Path(config["formal_checkpoint_root"])
    avm_root = Path(config["avm_checkpoint_root"])
    for seed in SEEDS:
        for tag in FORMAL_TAGS:
            run_dir = formal_root / f"formal__seed{seed}__{tag}"
            issues.extend(check_run(run_dir))
            issues.extend(check_checkpoint_hash(run_dir / "checkpoint_1500.pt", hash_contract))
        avm_run = avm_root / f"formal__seed{seed}__HPR__O1sham"
        issues.extend(check_run(avm_run))
        issues.extend(check_checkpoint_hash(avm_run / "checkpoint_1500.pt", hash_contract))
    if issues:
        raise RuntimeError("Release asset check failed:\n- " + "\n- ".join(issues))

    print(json.dumps({"python": sys.version.split()[0], "packages": versions}, indent=2))
    print("Release layout: 30 formal + 5 AVM endpoint runs are complete and SHA-256 verified.")

    if not args.quick:
        checkpoint = args.checkpoint or (
            formal_root / "formal__seed9201__Rroll" / "checkpoint_1500.pt"
        )
        command = [
            str(configured_python(config)),
            str(PROJECT_ROOT / "training" / "demo_metamaterial.py"),
            "--checkpoint",
            str(checkpoint),
            "--policy-mode",
            "deterministic",
            "--preflight",
        ]
        subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=subprocess_environment(config),
            check=True,
        )
        print(f"Policy reconstruction preflight passed: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
