from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_json, sha256_file, write_json


def read(root: Path, name: str) -> tuple[pd.DataFrame, Path]:
    path = root / name
    if not path.exists():
        path = root / f"{name}.gz"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path), path


def assert_arm(frame: pd.DataFrame, arm: str, source: str) -> None:
    if "arm" not in frame or set(frame.arm.astype(str)) != {arm}:
        raise RuntimeError(f"{source} must contain only arm={arm}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--o2-import-root", type=Path, required=True,
                        help="Output from ingest_archived_o2.py")
    parser.add_argument("--o2-eval-root", type=Path, required=True,
                        help="Frozen all-checkpoint O2 evaluator/probe exports")
    parser.add_argument("--o1-root", type=Path, required=True,
                        help="Completed O1-sham run/evaluator/probe exports")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contract", type=Path,
                        default=Path(__file__).resolve().parents[1] / "study_contract.json")
    args = parser.parse_args()
    o2_import = args.o2_import_root.resolve()
    o2_eval = args.o2_eval_root.resolve()
    o1 = args.o1_root.resolve()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite assembled package: {out}")
    out.mkdir(parents=True)

    sources: dict[str, str] = {}
    o2_runs, p = read(o2_import, "run_manifest.csv"); sources[str(p)] = sha256_file(p)
    o2_training, p = read(o2_import, "training_metrics.csv"); sources[str(p)] = sha256_file(p)
    o2_endpoint, p = read(o2_import, "checkpoint_episode_metrics.csv"); sources[str(p)] = sha256_file(p)
    o1_runs, p = read(o1, "run_manifest.csv"); sources[str(p)] = sha256_file(p)
    o1_training, p = read(o1, "training_metrics.csv"); sources[str(p)] = sha256_file(p)
    assert_arm(o2_runs, "O2", "archived O2 run manifest")
    assert_arm(o2_training, "O2", "archived O2 training metrics")
    assert_arm(o2_endpoint, "O2", "archived O2 endpoint metrics")
    assert_arm(o1_runs, "O1_sham", "O1-sham run manifest")
    assert_arm(o1_training, "O1_sham", "O1-sham training metrics")

    evaluation_tables = {}
    for name in ("checkpoint_episode_metrics.csv", "trajectory_joint.csv", "trajectory_node.csv", "actor_probe.csv"):
        o2_frame, p2 = read(o2_eval, name); sources[str(p2)] = sha256_file(p2)
        o1_frame, p1 = read(o1, name); sources[str(p1)] = sha256_file(p1)
        assert_arm(o2_frame, "O2", f"O2 {name}")
        assert_arm(o1_frame, "O1_sham", f"O1-sham {name}")
        evaluation_tables[name] = pd.concat([o1_frame, o2_frame], ignore_index=True, sort=False)

    # The new all-checkpoint O2 evaluator must reproduce every archived endpoint metric.
    new_endpoint = evaluation_tables["checkpoint_episode_metrics.csv"]
    new_endpoint = new_endpoint[(new_endpoint.arm == "O2") & (new_endpoint.checkpoint.astype(int) == 1500)]
    keys = ["paper_run", "reset_seed"]
    metrics = ["desired_net_rotation_deg", "desired_direction_fraction", "forward_body_lengths"]
    left = o2_endpoint[keys + metrics].copy().sort_values(keys).reset_index(drop=True)
    right = new_endpoint[keys + metrics].copy().sort_values(keys).reset_index(drop=True)
    if len(left) != 100 or len(right) != 100 or not left[keys].equals(right[keys]):
        raise RuntimeError("All-checkpoint O2 endpoint rows do not align with the archived 100 endpoint rows")
    error = np.max(np.abs(left[metrics].astype(float).to_numpy() - right[metrics].astype(float).to_numpy()))
    if not np.isfinite(error) or error > 1e-6:
        raise RuntimeError(f"O2 evaluator regression failed; maximum endpoint metric error={error}")

    merged = {
        "run_manifest.csv": pd.concat([o1_runs, o2_runs], ignore_index=True, sort=False),
        "training_metrics.csv": pd.concat([o1_training, o2_training], ignore_index=True, sort=False),
        **evaluation_tables,
    }
    for name, frame in merged.items():
        frame.to_csv(out / name, index=False, encoding="utf-8-sig")
    contract = load_json(args.contract.resolve())
    write_json(out / "study_contract.json", contract)
    receipt = {
        "schema": "o1_o2_paired_package_assembly/v1",
        "status": "assembled_not_yet_validated",
        "max_archived_o2_endpoint_regression_error": float(error),
        "source_sha256": sources,
        "output_sha256": {name: sha256_file(out / name) for name in [*merged, "study_contract.json"]},
        "rows": {name: len(frame) for name, frame in merged.items()},
    }
    write_json(out / "ASSEMBLY_RECEIPT.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

