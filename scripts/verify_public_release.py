"""Fail closed when the repository is not safe and complete for public release.

The verifier is read-only: it writes no reports, caches, or generated metadata and
reports every result on stdout.  It supplements, rather than replaces, the
English-release integrity verifier.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from datetime import date
from pathlib import Path, WindowsPath
from typing import Iterable


sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GITHUB_LIMIT_BYTES = 100 * 1024 * 1024

REQUIRED_FILES = {"README.md", "CITATION.cff", ".gitattributes"}
SKIP_DIR_NAMES = {".git"}
FORBIDDEN_RUNTIME_DIR_NAMES = {
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".vscode",
    "__pycache__",
    "data_external",
    "logs",
    "outputs",
    "venv",
}
FORBIDDEN_RUNTIME_FILES = {
    "configs/paths.local.json",
    "PRIVATE_unblinded_code_key.csv",
}
FORBIDDEN_MATERIAL_NAMES = {
    "CHAT_EVIDENCE_REPORT_MANIFEST.json",
    "MENTOR_REQUIREMENTS_EVIDENCE.md",
    "build_chat_evidence_report.py",
    "chat_report_a11y.json",
    "chat_report_style_lint.json",
    "workspace.xml",
}
BACKUP_NAME = re.compile(
    r"(?i)(?:\.bak(?:$|[._-])|\.before_fix$|\.backup$|\.orig$|~$)"
)
SECRET_FILE_NAME = re.compile(
    r"(?i)^(?:\.env(?:\..+)?|\.npmrc|\.pypirc|credentials(?:\..+)?|"
    r"secrets?(?:\..+)?|id_rsa|id_ed25519)$"
)
SECRET_SUFFIXES = {".key", ".p12", ".pem", ".pfx"}

CHAT_ACCOUNT_PREFIX = "wx" + "id_"
XWECHAT_MARKER = "xwechat" + "_files"
WEIXIN_DOWNLOAD_MARKER = "Wei" + "xin Down" + "load"
CLIPBOARD_MARKER = "codex-" + "clipboard-"
CLOUD_PROVIDER_MARKER = "One" + "Drive"

PRIVATE_TEXT_PATTERNS = {
    "private Windows user path": re.compile(
        r"(?i)[A-Z]:[\\/]+Users[\\/]+(?!PUBLIC_USER(?:[\\/]|$))"
        r"[^\\/\s\"']+"
    ),
    "original numeric Windows user path": re.compile(
        r"(?i)C:(?:\\+|/)Users(?:\\+|/)42017(?=\\+|/|\b)"
    ),
    "WeChat identifier": re.compile(
        rf"(?i){re.escape(CHAT_ACCOUNT_PREFIX)}[a-z0-9_]+"
    ),
    "WeChat storage path": re.compile(
        "(?i)(?:"
        + re.escape(XWECHAT_MARKER)
        + "|"
        + re.escape(WEIXIN_DOWNLOAD_MARKER)
        + ")"
    ),
    "local clipboard artifact": re.compile(
        rf"(?i){re.escape(CLIPBOARD_MARKER)}[0-9a-f-]+"
    ),
    "cloud-provider path component": re.compile(
        rf"(?i)\b{re.escape(CLOUD_PROVIDER_MARKER)}\b"
    ),
}

# Construct fixed secret markers in pieces so this verifier never flags itself.
SECRET_TEXT_PATTERNS = {
    "private-key block": re.compile("-----BEGIN " + "PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,255}\b"),
    "GitHub fine-grained token": re.compile(
        r"\bgithub" + r"_pat_[A-Za-z0-9_]{40,255}\b"
    ),
    "AWS access key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "OpenAI-style secret": re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
}


def relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def iter_release_files() -> Iterable[Path]:
    """Yield publishable files while deliberately ignoring Git internals."""

    for directory, dir_names, file_names in os.walk(PROJECT_ROOT):
        dir_names[:] = sorted(name for name in dir_names if name not in SKIP_DIR_NAMES)
        base = Path(directory)
        for name in sorted(file_names):
            yield base / name


def verify_required_files() -> list[str]:
    failures: list[str] = []
    for name in sorted(REQUIRED_FILES):
        path = PROJECT_ROOT / name
        if not path.is_file() or path.stat().st_size == 0:
            failures.append(f"Missing or empty required file: {name}")
    readme = PROJECT_ROOT / "README.md"
    if readme.is_file() and readme.stat().st_size < 1000:
        failures.append("README.md is unexpectedly short for a reproducibility release")
    return failures


def verify_excluded_runtime_material() -> list[str]:
    failures: list[str] = []
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if ".git" in path.relative_to(PROJECT_ROOT).parts:
            continue
        if path.is_dir() and path.name in FORBIDDEN_RUNTIME_DIR_NAMES:
            failures.append(f"Runtime/local directory must not be published: {relative(path)}/")
        if path.is_file():
            rel = relative(path)
            if rel in FORBIDDEN_RUNTIME_FILES or path.name in FORBIDDEN_RUNTIME_FILES:
                failures.append(f"Runtime/private file must not be published: {rel}")
    generated = PROJECT_ROOT / "configs" / "generated"
    if generated.exists():
        failures.append("Generated runtime configuration must not be published: configs/generated/")

    manifest = PROJECT_ROOT / "RELEASE_MANIFEST.csv"
    if manifest.is_file():
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                rel = row.get("relative_path", "").replace("\\", "/")
                parts = Path(rel).parts
                if ".git" in parts or any(
                    part in FORBIDDEN_RUNTIME_DIR_NAMES for part in parts
                ):
                    failures.append(f"Excluded path appears in release manifest: {rel}")
    return failures


def verify_forbidden_material_and_backups() -> list[str]:
    failures: list[str] = []
    for path in iter_release_files():
        rel = relative(path)
        if path.name in FORBIDDEN_MATERIAL_NAMES:
            failures.append(f"Internal discussion/IDE material is forbidden: {rel}")
        if "Robot_Rolling_K1K2_Full_Chat" in path.name:
            failures.append(f"Internal chat report is forbidden: {rel}")
        if BACKUP_NAME.search(path.name):
            failures.append(f"Backup file is forbidden: {rel}")
    return failures


def verify_file_sizes() -> tuple[list[str], int, int]:
    failures: list[str] = []
    count = 0
    largest = 0
    for path in iter_release_files():
        size = path.stat().st_size
        count += 1
        largest = max(largest, size)
        if size >= GITHUB_LIMIT_BYTES:
            failures.append(
                f"File reaches GitHub's 100 MiB limit: {relative(path)} ({size} bytes)"
            )
    return failures, count, largest


def verify_gitattributes() -> list[str]:
    path = PROJECT_ROOT / ".gitattributes"
    if not path.is_file():
        return ["Missing .gitattributes"]
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    failures: list[str] = []
    if "* -text" not in lines:
        failures.append(".gitattributes must contain the exact byte-preservation rule: * -text")
    for line in lines:
        if "text" in line.lower() and line != "* -text":
            failures.append(f"Conflicting text-normalisation rule in .gitattributes: {line}")
    return failures


def top_level_cff_value(text: str, key: str) -> str | None:
    match = re.search(rf"(?m)^{re.escape(key)}:\s*[\"']?([^\"'\r\n]+)", text)
    return match.group(1).strip() if match else None


def verify_citation_cff() -> list[str]:
    path = PROJECT_ROOT / "CITATION.cff"
    if not path.is_file():
        return ["Missing CITATION.cff"]
    text = path.read_text(encoding="utf-8-sig")
    failures: list[str] = []
    expected = {
        "cff-version": "1.2.0",
        "type": "software",
        "title": "Reinforcement Learning for a Soft Multi-Joint Robot: Official Thesis Reproducibility Package",
        "version": "1.0.0",
        "date-released": "2026-08-30",
        "repository-code": "https://github.com/xiangyaoguo/Worm-Like-Soft-Robot",
    }
    for key, value in expected.items():
        if top_level_cff_value(text, key) != value:
            failures.append(f"CITATION.cff must set {key}: {value}")
    for key in ("message", "title", "version", "date-released", "repository-code"):
        if not top_level_cff_value(text, key):
            failures.append(f"CITATION.cff is missing {key}")
    if not re.search(r"(?m)^authors:\s*$", text):
        failures.append("CITATION.cff is missing authors")
    if not re.search(r"(?m)^\s+-\s+family-names:\s*[\"']?Guo[\"']?\s*$", text):
        failures.append("CITATION.cff must identify the family name as Guo")
    if not re.search(r"(?m)^\s+given-names:\s*[\"']?Xiangyao[\"']?\s*$", text):
        failures.append("CITATION.cff must identify the given name as Xiangyao")

    released = top_level_cff_value(text, "date-released")
    if released:
        try:
            date.fromisoformat(released)
        except ValueError:
            failures.append(f"CITATION.cff has an invalid ISO release date: {released}")
    for forbidden_key in ("doi", "orcid", "license"):
        if re.search(rf"(?im)^\s*-?\s*{forbidden_key}:\s*", text):
            failures.append(
                f"CITATION.cff must omit {forbidden_key} until it is confirmed"
            )
    if re.search(r"(?i)\b(?:TODO|TBD|CHANGEME|example\.com|XXXX+)\b", text):
        failures.append("CITATION.cff contains placeholder metadata")
    return failures


def verify_text_privacy_and_secrets(english) -> tuple[list[str], int]:
    failures: list[str] = []
    scanned = 0
    for path in iter_release_files():
        rel = relative(path)
        for label, pattern in PRIVATE_TEXT_PATTERNS.items():
            if pattern.search(rel):
                failures.append(f"{label} in relative file path: {rel}")
        if SECRET_FILE_NAME.match(path.name) or path.suffix.lower() in SECRET_SUFFIXES:
            failures.append(f"Possible secret/key file: {rel}")
        if path.suffix.lower() not in english.TEXT_SUFFIXES | {".cff"}:
            continue
        scanned += 1
        try:
            text = english.read_release_text(path)
        except UnicodeError as exc:
            failures.append(str(exc))
            continue
        for label, pattern in PRIVATE_TEXT_PATTERNS.items():
            match = pattern.search(text)
            if match:
                line = text.count("\n", 0, match.start()) + 1
                failures.append(f"{label}: {rel}:{line}")
        for label, pattern in SECRET_TEXT_PATTERNS.items():
            match = pattern.search(text)
            if match:
                line = text.count("\n", 0, match.start()) + 1
                failures.append(f"Possible {label}: {rel}:{line}")
    return failures, scanned


def verify_binary_metadata_privacy(english) -> tuple[list[str], int]:
    """Safely inspect checkpoint and NumPy string metadata for private markers."""

    failures: list[str] = []
    inspected = 0
    try:
        import numpy as np
        import torch
    except ImportError as exc:
        return [f"Binary privacy audit dependency is unavailable: {exc}"], inspected

    for path in sorted((PROJECT_ROOT / "checkpoints").rglob("checkpoint_1500.pt")):
        inspected += 1
        rel = relative(path)
        try:
            with torch.serialization.safe_globals([WindowsPath]):
                payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            failures.append(f"Restricted checkpoint privacy load failed: {rel}: {exc}")
            continue
        for object_path, text in english.walk_readable_values(payload):
            for label, pattern in PRIVATE_TEXT_PATTERNS.items():
                if pattern.search(text):
                    failures.append(f"{label} in checkpoint metadata: {rel}:{object_path}")

    numpy_paths = sorted([*PROJECT_ROOT.rglob("*.npz"), *PROJECT_ROOT.rglob("*.npy")])
    for path in numpy_paths:
        inspected += 1
        rel = relative(path)
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
                    for label, pattern in PRIVATE_TEXT_PATTERNS.items():
                        if pattern.search(text):
                            failures.append(f"{label} in NumPy metadata: {rel}:{name}")
        except Exception as exc:
            failures.append(f"Safe NumPy privacy load failed: {rel}: {exc}")
    return failures, inspected


def run_english_release_checks(english, text_only: bool) -> tuple[list[str], dict[str, int]]:
    failures, text_count = english.scan_text_and_names()
    audit_failures, audit_count = english.verify_translation_audit()
    failures.extend(audit_failures)
    counts = {
        "english_text": text_count,
        "translation_audit": audit_count,
        "checkpoints": 0,
        "numpy": 0,
    }
    if not text_only:
        checkpoint_failures, counts["checkpoints"] = english.verify_checkpoints()
        numpy_failures, counts["numpy"] = english.verify_numpy_files()
        failures.extend(checkpoint_failures)
        failures.extend(numpy_failures)
    return [f"English verifier: {failure}" for failure in failures], counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Skip checkpoint and NumPy integrity/privacy checks.",
    )
    args = parser.parse_args()

    import verify_english_release as english

    failures: list[str] = []
    failures.extend(verify_required_files())
    failures.extend(verify_excluded_runtime_material())
    failures.extend(verify_forbidden_material_and_backups())
    size_failures, file_count, largest = verify_file_sizes()
    failures.extend(size_failures)
    failures.extend(verify_gitattributes())
    failures.extend(verify_citation_cff())
    privacy_failures, privacy_text_count = verify_text_privacy_and_secrets(english)
    failures.extend(privacy_failures)
    english_failures, english_counts = run_english_release_checks(english, args.text_only)
    failures.extend(english_failures)

    binary_count = 0
    if not args.text_only:
        binary_failures, binary_count = verify_binary_metadata_privacy(english)
        failures.extend(binary_failures)

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        print(f"Public-release verification failed with {len(failures)} issue(s).")
        return 1

    print(
        "Public-release verification passed: "
        f"{file_count} files, largest file {largest} bytes, "
        f"{privacy_text_count} privacy-scanned text files, "
        f"{english_counts['translation_audit']} translation-audit rows, "
        f"{english_counts['checkpoints']} checkpoints, "
        f"{english_counts['numpy']} NumPy files, and "
        f"{binary_count} binary metadata artifacts privacy-scanned."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
