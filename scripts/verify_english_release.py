"""Verify that the release has English-only names and readable metadata."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePath, WindowsPath
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HAN_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2FA1F),
)
HAN = re.compile(
    "[" + "".join(f"{chr(start)}-{chr(end)}" for start, end in HAN_RANGES) + "]"
)
UNICODE_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})|\\U([0-9a-fA-F]{8})")
TEXT_SUFFIXES = {
    "",
    ".bak",
    ".bak_abstract_methods",
    ".bak_dtype",
    ".bak_render_full",
    ".bak_render_methods",
    ".bak_wave_feedback",
    ".bat",
    ".before_fix",
    ".cfg",
    ".cff",
    ".cmd",
    ".csv",
    ".css",
    ".diff",
    ".html",
    ".ini",
    ".ipynb",
    ".js",
    ".json",
    ".log",
    ".md",
    ".patch",
    ".ps1",
    ".py",
    ".rst",
    ".sh",
    ".svg",
    ".tex",
    ".toml",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}

EXCLUDED_AUDIT_DIR_NAMES = {
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


def read_release_text(path: Path) -> str:
    raw = path.read_bytes()
    encodings = ("utf-8-sig", "utf-16", "gb18030")
    errors: list[str] = []
    for encoding in encodings:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise UnicodeError(f"Could not decode {path}: {'; '.join(errors)}")


def decode_unicode_escapes(text: str) -> str:
    def replacement(match: re.Match[str]) -> str:
        value = match.group(1) or match.group(2)
        return chr(int(value, 16))

    return UNICODE_ESCAPE.sub(replacement, text)


def scan_text_and_names() -> tuple[list[str], int]:
    failures: list[str] = []
    scanned = 0
    for path in sorted(PROJECT_ROOT.rglob("*")):
        relative_path = path.relative_to(PROJECT_ROOT)
        if any(part in EXCLUDED_AUDIT_DIR_NAMES for part in relative_path.parts):
            continue
        relative = relative_path.as_posix()
        name_match = HAN.search(relative)
        if name_match:
            failures.append(f"Non-English path name: {relative}")
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        scanned += 1
        try:
            text = read_release_text(path)
        except UnicodeError as exc:
            failures.append(str(exc))
            continue
        match = HAN.search(text)
        if match:
            line = text.count("\n", 0, match.start()) + 1
            failures.append(f"Han text: {relative}:{line}")
        escaped = decode_unicode_escapes(text)
        escaped_match = HAN.search(escaped)
        if escaped_match and not match:
            line = escaped.count("\n", 0, escaped_match.start()) + 1
            failures.append(f"Escaped Han text: {relative}:{line}")
    return failures, scanned


def excluded_from_translation_audit(relative: str) -> bool:
    path = Path(relative)
    if any(part in EXCLUDED_AUDIT_DIR_NAMES for part in path.parts):
        return True
    if relative in {
        "RELEASE_MANIFEST.csv",
        "TRANSLATION_AUDIT.csv",
        "configs/paths.local.json",
    }:
        return True
    if relative.startswith(("configs/generated/", "outputs/", "logs/")):
        return True
    if relative.startswith("analysis/response_surfaces/generated/"):
        return True
    if path.suffix.lower() in {".pyc", ".pyo"}:
        return True
    if path.name.endswith((".tmp", ".log")):
        return True
    if path.name == "PRIVATE_unblinded_code_key.csv" or ".egg-info" in path.parts:
        return True
    return False


def verify_translation_audit() -> tuple[list[str], int]:
    audit_path = PROJECT_ROOT / "TRANSLATION_AUDIT.csv"
    if not audit_path.is_file():
        return [f"Missing translation audit: {audit_path.name}"], 0
    with audit_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    failures: list[str] = []
    seen: set[str] = set()
    allowed_statuses = {
        "unchanged",
        "translated_or_sanitized",
        "renamed_and_translated",
        "english_release_only",
    }
    for row in rows:
        relative = row.get("english_relative_path", "")
        if not relative or relative in seen:
            failures.append(f"Invalid or duplicate translation-audit path: {relative!r}")
            continue
        seen.add(relative)
        if row.get("translation_status") not in allowed_statuses:
            failures.append(f"Invalid translation status: {relative}")
        path = PROJECT_ROOT / Path(relative)
        if not path.is_file():
            failures.append(f"Translation-audit file is missing: {relative}")
            continue
        if sha256_file(path) != row.get("english_file_sha256", "").lower():
            failures.append(f"Translation-audit hash mismatch: {relative}")
        original_hash = row.get("original_file_sha256", "")
        if original_hash and len(original_hash) != 64:
            failures.append(f"Invalid original hash in translation audit: {relative}")
        if row.get("translation_status") == "unchanged" and original_hash != row.get(
            "english_file_sha256", ""
        ).lower():
            failures.append(f"Unchanged audit row has different hashes: {relative}")

    expected = {
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in PROJECT_ROOT.rglob("*")
        if path.is_file()
        and not excluded_from_translation_audit(
            path.relative_to(PROJECT_ROOT).as_posix()
        )
    }
    if seen != expected:
        for relative in sorted(expected - seen):
            failures.append(f"File missing from translation audit: {relative}")
        for relative in sorted(seen - expected):
            failures.append(f"Unexpected translation-audit row: {relative}")
    return failures, len(rows)


def walk_readable_values(value: Any, object_path: str = "root"):
    if isinstance(value, PurePath):
        yield object_path, str(value)
        return
    if isinstance(value, str):
        yield object_path, value
        return
    if isinstance(value, bytes):
        try:
            yield object_path, value.decode("utf-8")
        except UnicodeDecodeError:
            pass
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from walk_readable_values(key, f"{object_path}.<key>")
            yield from walk_readable_values(item, f"{object_path}[{key!r}]")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            yield from walk_readable_values(item, f"{object_path}[{index}]")


def tensor_fingerprint(payload: Any) -> tuple[str, int]:
    import torch

    records: list[tuple[str, Any]] = []

    def collect(value: Any, object_path: str) -> None:
        if isinstance(value, torch.Tensor):
            records.append((object_path, value))
        elif isinstance(value, Mapping):
            for key in sorted(value, key=lambda item: repr(item)):
                collect(value[key], f"{object_path}[{key!r}]")
        elif isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for index, item in enumerate(value):
                collect(item, f"{object_path}[{index}]")

    collect(payload, "root")
    digest = hashlib.sha256()
    for object_path, tensor in sorted(records, key=lambda pair: pair[0]):
        dense = tensor.detach().cpu().contiguous()
        raw = dense.view(torch.uint8).numpy().tobytes()
        for text in (object_path, str(tensor.dtype), repr(tuple(tensor.shape))):
            encoded = text.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest(), len(records)


def verify_checkpoints() -> tuple[list[str], int]:
    import torch

    audit_path = PROJECT_ROOT / "CHECKPOINT_TRANSLATION_AUDIT.csv"
    if not audit_path.is_file():
        return [f"Missing checkpoint translation audit: {audit_path.name}"], 0
    with audit_path.open("r", encoding="utf-8-sig", newline="") as handle:
        audit = {row["relative_path"]: row for row in csv.DictReader(handle)}
    failures: list[str] = []
    paths = sorted((PROJECT_ROOT / "checkpoints").rglob("checkpoint_1500.pt"))
    if len(paths) != 35 or len(audit) != 35:
        failures.append(
            f"Expected 35 checkpoints and audit rows; found {len(paths)} and {len(audit)}"
        )
        return failures, len(paths)

    status_counts: dict[str, int] = {}
    for row in audit.values():
        status = row.get("status", "")
        status_counts[status] = status_counts.get(status, 0) + 1
        if status not in {
            "translated_and_normalized_results_dir",
            "normalized_results_dir",
        }:
            failures.append(f"Invalid checkpoint translation status: {status!r}")
    if status_counts != {
        "translated_and_normalized_results_dir": 20,
        "normalized_results_dir": 15,
    }:
        failures.append(f"Unexpected checkpoint translation status counts: {status_counts}")

    for path in paths:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        row = audit.get(relative)
        if row is None:
            failures.append(f"Missing checkpoint audit row: {relative}")
            continue
        current_sha = sha256_file(path)
        if current_sha != row["english_file_sha256"].lower():
            failures.append(f"Checkpoint file hash mismatch: {relative}")
        try:
            with torch.serialization.safe_globals([WindowsPath]):
                payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            failures.append(f"Restricted checkpoint load failed: {relative}: {exc}")
            continue
        for object_path, text in walk_readable_values(payload):
            if HAN.search(text) or HAN.search(decode_unicode_escapes(text)):
                failures.append(f"Han checkpoint metadata: {relative}:{object_path}")
        expected_results_dir = (
            "outputs/avm_training/runs"
            if relative.startswith("checkpoints/avm/")
            else "outputs/formal_training/runs"
        )
        actual_results_dir = (
            payload.get("metadata", {})
            .get("training_args", {})
            .get("results_dir")
        )
        if actual_results_dir != expected_results_dir:
            failures.append(
                "Non-portable checkpoint results_dir: "
                f"{relative}: {actual_results_dir!r} != {expected_results_dir!r}"
            )
        fingerprint, count = tensor_fingerprint(payload)
        if fingerprint != row["tensor_fingerprint_sha256"].lower():
            failures.append(f"Checkpoint tensor fingerprint mismatch: {relative}")
        if count != int(row["tensor_count"]):
            failures.append(f"Checkpoint tensor count mismatch: {relative}")
    return failures, len(paths)


def verify_numpy_files() -> tuple[list[str], int]:
    import numpy as np

    failures: list[str] = []
    paths = sorted(
        path
        for path in [*PROJECT_ROOT.rglob("*.npz"), *PROJECT_ROOT.rglob("*.npy")]
        if not any(
            part in EXCLUDED_AUDIT_DIR_NAMES
            for part in path.relative_to(PROJECT_ROOT).parts
        )
    )
    for path in paths:
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        try:
            payload = np.load(path, allow_pickle=False)
            if isinstance(payload, np.lib.npyio.NpzFile):
                try:
                    arrays = [(name, payload[name]) for name in payload.files]
                finally:
                    payload.close()
            else:
                arrays = [("array", payload)]
            for name, array in arrays:
                if array.dtype.kind not in {"U", "S"}:
                    continue
                for value in array.reshape(-1).tolist():
                    text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
                    if HAN.search(text) or HAN.search(decode_unicode_escapes(text)):
                        failures.append(f"Han NumPy metadata: {relative}:{name}")
        except Exception as exc:
            failures.append(f"Safe NumPy load failed: {relative}: {exc}")
    return failures, len(paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Skip structured checkpoint and NumPy metadata checks.",
    )
    args = parser.parse_args()
    failures, text_count = scan_text_and_names()
    audit_failures, audit_count = verify_translation_audit()
    failures.extend(audit_failures)
    checkpoint_count = 0
    numpy_count = 0
    if not args.text_only:
        checkpoint_failures, checkpoint_count = verify_checkpoints()
        numpy_failures, numpy_count = verify_numpy_files()
        failures.extend(checkpoint_failures)
        failures.extend(numpy_failures)
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        print(f"English-release verification failed with {len(failures)} issue(s).")
        return 1
    print(
        "English-release verification passed: "
        f"{text_count} text files, {audit_count} translation-audit rows, "
        f"{checkpoint_count} checkpoints, "
        f"and {numpy_count} NumPy files."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
