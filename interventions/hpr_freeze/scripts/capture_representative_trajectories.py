from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


STUDY_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_formal_hpr_freeze_study import (  # noqa: E402
    BASE_RESET_SEED,
    DATA_ROOT,
    EXPECTED_CHECKPOINT_HASHES,
    HprFreezeRuntime,
    build_conditions,
    kinematic_success,
    sha256_file,
)

import numpy as np  # noqa: E402
import torch  # noqa: E402


CAPTURE_CONDITIONS = (
    "BASELINE",
    "GLOBAL_K1_OFF",
    "GLOBAL_K2_OFF",
    "GLOBAL_BOTH_OFF",
)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def capture(training_seed: int, reset_seed: int) -> dict[str, Any]:
    runtime = HprFreezeRuntime(training_seed, stage="identity", environment_arm="R0")
    if sha256_file(runtime.r0_checkpoint) != EXPECTED_CHECKPOINT_HASHES[training_seed]:
        runtime.close()
        raise RuntimeError(f"Checkpoint hash drift for seed {training_seed}")
    conditions_by_id = {condition.id: condition for condition in build_conditions("full")}
    output_dir = DATA_ROOT / "trajectories"
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    initial_hash: str | None = None
    try:
        for condition_id in CAPTURE_CONDITIONS:
            condition = conditions_by_id[condition_id]
            torch.manual_seed(reset_seed)
            np.random.seed(reset_seed)
            td = runtime.env.reset()
            trajectory = [runtime.frozen_eval._positions(runtime.env)]
            for step in range(runtime.steps):
                r0, r0_clone, _ = runtime.actor_actions(td)
                applied = runtime.apply_condition(condition, r0, r0_clone, step)
                action_td = td.clone(recurse=True)
                action_td["agents", "action"] = applied
                td = runtime.env.step(action_td)["next"]
                trajectory.append(runtime.frozen_eval._positions(runtime.env))
            positions = np.asarray(trajectory, dtype=np.complex64)
            metrics = runtime.frozen_eval._episode_metrics(
                trajectory,
                "right",
                "left",
                runtime.metric_args,
                None,
                None,
            )
            current_initial_hash = array_sha256(positions[0])
            if initial_hash is None:
                initial_hash = current_initial_hash
            elif current_initial_hash != initial_hash:
                raise RuntimeError("Representative conditions did not share the same reset state")
            output_path = output_dir / f"seed{training_seed}__{condition_id}__reset{reset_seed}.npz"
            np.savez_compressed(
                output_path,
                positions=positions,
                reset_seed=np.asarray(reset_seed, dtype=np.int64),
                training_seed=np.asarray(training_seed, dtype=np.int64),
            )
            records.append(
                {
                    "training_seed": training_seed,
                    "condition_id": condition_id,
                    "reset_seed": reset_seed,
                    "path": str(output_path),
                    "trajectory_sha256": array_sha256(positions),
                    "initial_state_sha256": current_initial_hash,
                    "success_kinematic": kinematic_success(metrics),
                    "desired_net_rotation_degrees": float(
                        metrics["desired_net_rotation_degrees"]
                    ),
                    "desired_active_rotation_fraction": float(
                        metrics["desired_active_rotation_fraction"]
                    ),
                    "forward_body_lengths": float(metrics["forward_body_lengths"]),
                }
            )
        integrity = runtime.verify_unchanged()
    finally:
        runtime.close()
    manifest = {
        "schema": "formal_hpr_freeze_validation/representative_trajectories/v1",
        "training_seed": training_seed,
        "reset_seed": reset_seed,
        "capture_conditions": list(CAPTURE_CONDITIONS),
        "common_initial_state_sha256": initial_hash,
        "records": records,
        "post_capture_integrity": integrity,
    }
    atomic_json(output_dir / f"seed{training_seed}__manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[9201, 9205])
    parser.add_argument("--reset-seed", type=int, default=BASE_RESET_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifests = [capture(seed, int(args.reset_seed)) for seed in args.training_seeds]
    atomic_json(
        DATA_ROOT / "trajectories" / "TRAJECTORY_MANIFEST.json",
        {
            "schema": "formal_hpr_freeze_validation/representative_trajectories_all/v1",
            "manifests": manifests,
        },
    )
    print(json.dumps(manifests, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
