from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ARMS = ("O1_sham", "O2")
RUNS = tuple(range(5))
SEED_MAP = {0: 9201, 1: 9202, 2: 9203, 3: 9204, 4: 9205}
CHECKPOINTS = tuple(range(100, 1501, 100))
RESET_SEEDS = tuple(range(20264101, 20264121))

ROTATION_MIN = 360.0
DIRECTION_MIN = 0.70
FORWARD_MIN = 1.0
DISCOVERY_MIN = 10


class ContractError(RuntimeError):
    """Raised when formal inputs do not satisfy the frozen analysis contract."""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, fieldnames: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fieldnames)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def common_success(rotation_deg: float, direction_fraction: float, forward_bl: float) -> bool:
    return bool(
        rotation_deg >= ROTATION_MIN
        and direction_fraction >= DIRECTION_MIN
        and forward_bl >= FORWARD_MIN
    )


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"not a boolean: {value!r}")


def as_float(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} is not numeric: {value!r}") from exc
    if number != number or number in (float("inf"), float("-inf")):
        raise ContractError(f"{name} must be finite: {value!r}")
    return number


def as_int(value: Any, name: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{name} is not an integer: {value!r}") from exc
    return number

