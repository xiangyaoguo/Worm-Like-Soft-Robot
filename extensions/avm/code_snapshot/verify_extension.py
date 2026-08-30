"""Read-only verification of parent sources and paired initialisation records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify() -> dict:
    config = json.loads((HERE / "extension_config.json").read_text(encoding="utf-8"))
    references = json.loads((HERE / "reference_initializations.json").read_text(encoding="utf-8"))
    parent = Path(config["parent_root"])
    errors: list[str] = []
    checked: list[dict] = []
    for relative, expected in config["source_contract"].items():
        path = parent / Path(relative)
        actual = sha256_file(path) if path.is_file() else None
        checked.append({"path": str(path), "actual": actual, "expected": expected})
        if actual != expected:
            errors.append(f"source drift: {relative}: {actual} != {expected}")
    for seed in config["formal_runs"]["internal_seeds"]:
        reference = references["references"][str(seed)]
        path = parent / Path(reference["relative_path"])
        actual_file_sha = sha256_file(path) if path.is_file() else None
        if actual_file_sha != reference["file_sha256"]:
            errors.append(f"reference audit drift: seed {seed}")
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("pair_hash_bundle_sha256") != reference["pair_hash_bundle_sha256"]:
            errors.append(f"reference bundle digest drift: seed {seed}")
        for field in config["initialization_gate"]["required_fields"]:
            if value["pair_hash_bundle"].get(field) != reference.get(field):
                errors.append(f"reference bundle field drift: seed {seed}: {field}")
    if config["formal_runs"]["internal_seeds"] != [9201, 9202, 9203, 9204, 9205]:
        errors.append("formal internal seed order drifted")
    if config["formal_runs"]["paper_run_ids"] != [0, 1, 2, 3, 4]:
        errors.append("paper run identifiers drifted")
    if errors:
        raise RuntimeError("\n".join(errors))
    return {"status": "pass", "sources_checked": len(checked), "seeds_checked": 5}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
