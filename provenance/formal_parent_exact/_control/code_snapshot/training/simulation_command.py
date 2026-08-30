"""Create copy-paste PowerShell command files for trained checkpoints.

Each UTF-8 text file contains one complete PowerShell command.  The command
uses absolute paths for the Python interpreter, demo program, dependency
directory, and checkpoint so it can be copied into any PowerShell window.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


COMMAND_FILENAME = "simulation_command.txt"
CHECKPOINT_PATTERN = re.compile(r"^checkpoint_(\d+)\.pt$", re.IGNORECASE)


@dataclass(frozen=True)
class CommandWriteResult:
    """Outcome of creating or refreshing one simulation command file."""

    path: Path
    checkpoint: Path
    action: str


def _powershell_single_quoted(value: str) -> str:
    """Quote a literal for a PowerShell single-quoted string."""

    return "'" + value.replace("'", "''") + "'"


def build_simulation_command_text(
    checkpoint_path: Path,
    project_root: Path,
    python_executable: Path,
) -> str:
    """Return one directly copyable PowerShell command line."""

    checkpoint_path = checkpoint_path.resolve()
    project_root = project_root.resolve()
    python_executable = python_executable.resolve()
    demo_script = project_root / "training" / "demo_metamaterial.py"
    site_packages = project_root / ".venv" / "Lib" / "site-packages"

    return (
        f"$env:PYTHONPATH={_powershell_single_quoted(str(site_packages))}; "
        f"& {_powershell_single_quoted(str(python_executable))} "
        f"{_powershell_single_quoted(str(demo_script))} "
        f"--checkpoint {_powershell_single_quoted(str(checkpoint_path))} "
        "--terrain checkpoint --policy-mode deterministic --follow-camera\n"
    )


def _read_existing_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8-sig")


def _atomic_write_utf8_bom(path: Path, content: str) -> None:
    """Atomically write a Windows-friendly UTF-8-with-BOM text file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8-sig",
            newline=None,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)


def write_simulation_command(
    run_dir: Path | str,
    checkpoint_path: Path | str,
    project_root: Path | str,
    *,
    filename: str = COMMAND_FILENAME,
    python_executable: Path | str = sys.executable,
    dry_run: bool = False,
) -> CommandWriteResult:
    """Create or refresh the command text file in one result directory."""

    run_dir = Path(run_dir).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    project_root = Path(project_root).resolve()
    python_executable = Path(python_executable).resolve()

    if checkpoint_path.parent != run_dir:
        raise ValueError(
            f"Checkpoint must be directly inside its result directory: {checkpoint_path}"
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not (project_root / "training" / "demo_metamaterial.py").is_file():
        raise FileNotFoundError(
            f"demo_metamaterial.py not found under project root: {project_root}"
        )

    output_path = run_dir / filename
    content = build_simulation_command_text(
        checkpoint_path,
        project_root,
        python_executable,
    )
    existing = _read_existing_text(output_path)
    if existing == content:
        action = "unchanged"
    elif existing is None:
        action = "would_create" if dry_run else "created"
    else:
        action = "would_update" if dry_run else "updated"

    if not dry_run and existing != content:
        _atomic_write_utf8_bom(output_path, content)

    return CommandWriteResult(output_path, checkpoint_path, action)


def latest_checkpoint_in(run_dir: Path) -> Path | None:
    """Find a normalized final checkpoint or the highest numbered checkpoint."""

    normalized = run_dir / "checkpoint_final.pt"
    if normalized.is_file():
        return normalized

    numbered: list[tuple[int, Path]] = []
    for candidate in run_dir.glob("checkpoint_*.pt"):
        match = CHECKPOINT_PATTERN.fullmatch(candidate.name)
        if match and candidate.is_file():
            numbered.append((int(match.group(1)), candidate))
    return max(numbered, default=(0, None), key=lambda item: item[0])[1]


def discover_run_directories(results_root: Path) -> Iterable[Path]:
    """Yield result directories identified by their top-level metadata file."""

    for metadata_path in sorted(results_root.rglob("metadata.json")):
        if metadata_path.is_file():
            yield metadata_path.parent


def generate_for_tree(
    results_root: Path | str,
    project_root: Path | str,
    *,
    filename: str = COMMAND_FILENAME,
    python_executable: Path | str = sys.executable,
    dry_run: bool = False,
) -> tuple[list[CommandWriteResult], list[Path]]:
    """Generate launchers for every checkpoint-bearing run below a root."""

    results_root = Path(results_root).resolve()
    if not results_root.is_dir():
        raise NotADirectoryError(f"Results root not found: {results_root}")

    results: list[CommandWriteResult] = []
    skipped: list[Path] = []
    for run_dir in discover_run_directories(results_root):
        checkpoint = latest_checkpoint_in(run_dir)
        if checkpoint is None:
            skipped.append(run_dir)
            continue
        results.append(
            write_simulation_command(
                run_dir,
                checkpoint,
                project_root,
                filename=filename,
                python_executable=python_executable,
                dry_run=dry_run,
            )
        )
    return results, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate one PowerShell simulation command TXT per training result."
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Source tree containing training/demo_metamaterial.py.",
    )
    parser.add_argument("--filename", default=COMMAND_FILENAME)
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=Path(sys.executable),
        help="Python executable embedded in each launcher (default: this interpreter).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results, skipped = generate_for_tree(
        args.results_root,
        args.project_root,
        filename=args.filename,
        python_executable=args.python_executable,
        dry_run=args.dry_run,
    )
    counts = Counter(result.action for result in results)
    for result in results:
        print(f"[{result.action}] {result.path} -> {result.checkpoint.name}")
    for run_dir in skipped:
        print(f"[skipped:no-checkpoint] {run_dir}")
    print(
        "Summary:",
        f"runs={len(results)}",
        f"skipped={len(skipped)}",
        " ".join(f"{key}={value}" for key, value in sorted(counts.items())),
    )


if __name__ == "__main__":
    main()
