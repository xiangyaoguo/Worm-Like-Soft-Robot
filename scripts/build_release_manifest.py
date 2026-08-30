"""Build or verify the checkpoint and full-file SHA-256 release manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_MANIFEST = PROJECT_ROOT / "CHECKPOINT_SHA256.csv"
RELEASE_MANIFEST = PROJECT_ROOT / "RELEASE_MANIFEST.csv"
SEEDS = tuple(range(9201, 9206))
FORMAL_ARMS = {
    "DTH": "HPR-DTH-PS",
    "THDOT": "HPR-THDOT-PS",
    "OBS": "HPR-OBS-PS",
    "HPR__O2shared": "HPR-O2-PS",
    "R0": "HPR-O2-JS",
    "Rroll": "SGRR-O2-JS",
}
EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
    "data_external",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def excluded(path: Path) -> bool:
    relative = path.relative_to(PROJECT_ROOT)
    parts = relative.parts
    if any(part in EXCLUDED_DIR_NAMES for part in parts):
        return True
    relative_text = relative.as_posix()
    if relative_text in {"RELEASE_MANIFEST.csv", "configs/paths.local.json"}:
        return True
    if relative_text.startswith("configs/generated/"):
        return True
    if relative_text.startswith("outputs/") or relative_text.startswith("logs/"):
        return True
    if relative_text.startswith("analysis/response_surfaces/generated/"):
        return True
    if path.suffix.lower() in {".pyc", ".pyo"}:
        return True
    if path.name.endswith((".tmp", ".log")):
        return True
    if path.name == "PRIVATE_unblinded_code_key.csv":
        return True
    if ".egg-info" in parts:
        return True
    return False


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    formal_root = PROJECT_ROOT / "checkpoints" / "formal_six" / "runs"
    for seed in SEEDS:
        for archive_tag, paper_arm in FORMAL_ARMS.items():
            path = formal_root / f"formal__seed{seed}__{archive_tag}" / "checkpoint_1500.pt"
            if not path.is_file():
                raise FileNotFoundError(path)
            rows.append(
                {
                    "study": "formal_six",
                    "paper_arm": paper_arm,
                    "archive_tag": archive_tag,
                    "training_seed": seed,
                    "formal_run_index": seed - 9201,
                    "relative_path": normalized_relative(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    avm_root = PROJECT_ROOT / "checkpoints" / "avm" / "runs"
    for seed in SEEDS:
        archive_tag = "HPR__O1sham"
        path = avm_root / f"formal__seed{seed}__{archive_tag}" / "checkpoint_1500.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "study": "avm_endpoint",
                "paper_arm": "HPR-O2-AVM-JS",
                "archive_tag": archive_tag,
                "training_seed": seed,
                "formal_run_index": seed - 9201,
                "relative_path": normalized_relative(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if len(rows) != 35 or len({row["relative_path"] for row in rows}) != 35:
        raise RuntimeError("Checkpoint manifest must contain exactly 35 unique endpoints")
    return rows


def role_for(relative_path: str) -> str:
    top = relative_path.split("/", 1)[0]
    return {
        "checkpoints": "checkpoint_bundle",
        "provenance": "translated_provenance",
        "reference_results": "reference_result",
        "data": "bundled_data",
        "docs": "documentation",
        "tests": "test",
        "configs": "configuration",
    }.get(top, "active_source")


def release_rows() -> list[dict[str, object]]:
    files = sorted(
        (path for path in PROJECT_ROOT.rglob("*") if path.is_file() and not excluded(path)),
        key=lambda value: normalized_relative(value).lower(),
    )
    return [
        {
            "relative_path": normalized_relative(path),
            "role": role_for(normalized_relative(path)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]


def build() -> None:
    checkpoints = checkpoint_rows()
    write_csv(
        CHECKPOINT_MANIFEST,
        [
            "study",
            "paper_arm",
            "archive_tag",
            "training_seed",
            "formal_run_index",
            "relative_path",
            "bytes",
            "sha256",
        ],
        checkpoints,
    )
    rows = release_rows()
    write_csv(
        RELEASE_MANIFEST,
        ["relative_path", "role", "bytes", "sha256"],
        rows,
    )
    total_bytes = sum(int(row["bytes"]) for row in rows)
    print(
        f"Built {CHECKPOINT_MANIFEST.name} ({len(checkpoints)} endpoints) and "
        f"{RELEASE_MANIFEST.name} ({len(rows)} files, {total_bytes} bytes)."
    )


def verify() -> None:
    if not CHECKPOINT_MANIFEST.is_file() or not RELEASE_MANIFEST.is_file():
        raise FileNotFoundError("Run --build before --verify")
    with CHECKPOINT_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        recorded_checkpoints = list(csv.DictReader(handle))
    actual_checkpoints = checkpoint_rows()
    actual_by_path = {str(row["relative_path"]): row for row in actual_checkpoints}
    if len(recorded_checkpoints) != 35:
        raise RuntimeError(f"Checkpoint manifest row drift: {len(recorded_checkpoints)}")
    for row in recorded_checkpoints:
        actual = actual_by_path.get(row["relative_path"])
        if actual is None:
            raise RuntimeError(f"Unexpected checkpoint manifest path: {row['relative_path']}")
        if int(row["bytes"]) != int(actual["bytes"]) or row["sha256"] != actual["sha256"]:
            raise RuntimeError(f"Checkpoint hash/size mismatch: {row['relative_path']}")

    with RELEASE_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        recorded = {row["relative_path"]: row for row in csv.DictReader(handle)}
    actual = {row["relative_path"]: row for row in release_rows()}
    if set(recorded) != set(actual):
        missing = sorted(set(recorded) - set(actual))
        extra = sorted(set(actual) - set(recorded))
        raise RuntimeError(f"Release file-set drift; missing={missing}, extra={extra}")
    for relative_path, row in recorded.items():
        current = actual[relative_path]
        if int(row["bytes"]) != int(current["bytes"]) or row["sha256"] != current["sha256"]:
            raise RuntimeError(f"Release hash/size mismatch: {relative_path}")
    print(f"Verified 35 checkpoints and {len(recorded)} release files.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.build:
        build()
    else:
        verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
