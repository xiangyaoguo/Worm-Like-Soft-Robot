"""Read-only validation for the translated English provenance snapshot."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


HAN_RE = re.compile(
    f"[{chr(0x3400)}-{chr(0x4DBF)}{chr(0x4E00)}-{chr(0x9FFF)}{chr(0xF900)}-{chr(0xFAFF)}]"
)
ESCAPED_UNICODE_RE = re.compile(r"\\u([0-9a-fA-F]{4})")
MARKDOWN_LINK_RE = re.compile(r"]\(([^)#]+)(?:#[^)]+)?\)")
TEXT_SUFFIXES = {
    ".bat",
    ".cfg",
    ".cmd",
    ".csv",
    ".css",
    ".diff",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".log",
    ".md",
    ".patch",
    ".ps1",
    ".py",
    ".sh",
    ".svg",
    ".toml",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decoded_escape_contains_han(text: str) -> bool:
    return any(HAN_RE.search(chr(int(match.group(1), 16))) for match in ESCAPED_UNICODE_RE.finditer(text))


def validate(root: Path, source_root: Path | None) -> list[str]:
    failures: list[str] = []
    files = [path for path in root.rglob("*") if path.is_file()]

    for path in [root, *root.rglob("*")]:
        if HAN_RE.search(path.name):
            failures.append(f"Han character in path: {path.relative_to(root)}")

    text_cache: dict[Path, str] = {}
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        relative = path.relative_to(root)
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as error:
            failures.append(f"UTF-8 decode failure: {relative}: {error}")
            continue
        text_cache[path] = text
        if HAN_RE.search(text):
            failures.append(f"Han character in text: {relative}")
        if decoded_escape_contains_han(text):
            failures.append(f"Escaped Han character in text: {relative}")

    for path, text in text_cache.items():
        relative = path.relative_to(root)
        suffix = path.suffix.lower()
        if suffix == ".py":
            try:
                tree = ast.parse(text, filename=str(relative))
            except SyntaxError as error:
                failures.append(f"Python syntax failure: {relative}: {error}")
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and HAN_RE.search(node.value):
                    failures.append(f"Decoded Han in Python string: {relative}:{getattr(node, 'lineno', '?')}")
        elif suffix == ".json":
            try:
                value = json.loads(text)
            except json.JSONDecodeError as error:
                failures.append(f"JSON parse failure: {relative}: {error}")
                continue
            if HAN_RE.search(json.dumps(value, ensure_ascii=False)):
                failures.append(f"Decoded Han in JSON value: {relative}")
        elif suffix in {".xml", ".svg"}:
            try:
                ET.fromstring(text)
            except ET.ParseError as error:
                failures.append(f"XML parse failure: {relative}: {error}")
        elif suffix in {".csv", ".tsv"}:
            try:
                list(csv.reader(text.splitlines(), delimiter="\t" if suffix == ".tsv" else ","))
            except csv.Error as error:
                failures.append(f"CSV parse failure: {relative}: {error}")

        if suffix == ".md":
            for target in MARKDOWN_LINK_RE.findall(text):
                target = target.strip().strip("<>")
                if not target or "://" in target or target.startswith(("#", "mailto:")):
                    continue
                if re.match(r"^[A-Za-z]:[\\/]", target) or any(char in target for char in "*?{}"):
                    continue
                candidate = (path.parent / target).resolve()
                if not candidate.exists():
                    failures.append(f"Missing relative Markdown target: {relative} -> {target}")

    if source_root is not None:
        for path in files:
            if path.suffix.lower() in TEXT_SUFFIXES:
                continue
            relative = path.relative_to(root)
            source = source_root / relative
            if not source.is_file():
                failures.append(f"Binary absent from source snapshot: {relative}")
            elif sha256(path) != sha256(source):
                failures.append(f"Binary changed during translation: {relative}")
        for source in source_root.rglob("*"):
            if not source.is_file() or source.suffix.lower() in TEXT_SUFFIXES:
                continue
            if source.suffix.lower() == ".pyc" or "__pycache__" in source.parts:
                continue
            relative = source.relative_to(source_root)
            if not (root / relative).is_file():
                failures.append(f"Source binary missing from English snapshot: {relative}")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--source-provenance", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    source = args.source_provenance.resolve() if args.source_provenance else None
    failures = validate(root, source)
    print(f"Validated provenance root: {root}")
    print(f"Failure count: {len(failures)}")
    for failure in failures:
        print(f"- {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
