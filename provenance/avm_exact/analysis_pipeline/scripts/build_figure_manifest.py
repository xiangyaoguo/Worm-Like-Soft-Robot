from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import sha256_file, write_json


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--figure-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    figures = sorted(path for path in args.figure_root.resolve().rglob("*") if path.is_file())
    data = sorted(path for path in args.data_root.resolve().rglob("*") if path.is_file())
    if not figures:
        raise RuntimeError("No figure files found")
    manifest = {
        "schema": "o1_o2_formal_figure_hash_ledger/v1",
        "figure_root": str(args.figure_root.resolve()),
        "data_root": str(args.data_root.resolve()),
        "figures": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in figures],
        "source_data": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in data],
    }
    write_json(args.out.resolve(), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

