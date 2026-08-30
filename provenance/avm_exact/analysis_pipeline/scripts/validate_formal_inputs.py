from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    ARMS,
    CHECKPOINTS,
    ContractError,
    RESET_SEEDS,
    RUNS,
    SEED_MAP,
    common_success,
    load_json,
    parse_bool,
    sha256_file,
    write_json,
)


HEX64 = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_FILES = (
    "run_manifest.csv",
    "training_metrics.csv",
    "checkpoint_episode_metrics.csv",
    "trajectory_joint.csv",
    "trajectory_node.csv",
    "actor_probe.csv",
    "study_contract.json",
)


def _read_table(data_root: Path, name: str) -> pd.DataFrame:
    direct = data_root / name
    compressed = data_root / f"{name}.gz"
    if direct.exists() and compressed.exists():
        raise ContractError(f"Both {direct.name} and {compressed.name} exist; source is ambiguous")
    path = direct if direct.exists() else compressed
    if not path.exists():
        raise ContractError(f"Missing required formal input: {name} (or {name}.gz)")
    return pd.read_csv(path)


def _require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ContractError(f"{name} missing columns: {missing}")


def _finite(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            bad = frame.loc[values.isna() | ~np.isfinite(values), column].head(5).tolist()
            raise ContractError(f"{name}.{column} has missing/non-finite values, examples={bad}")


def _validate_labels(frame: pd.DataFrame, name: str) -> None:
    arms = set(frame["arm"].astype(str))
    if arms != set(ARMS):
        raise ContractError(f"{name} arms must be exactly {ARMS}, observed={sorted(arms)}")
    for row in frame[["paper_run", "internal_seed"]].drop_duplicates().itertuples(index=False):
        run = int(row.paper_run)
        seed = int(row.internal_seed)
        if run not in RUNS or SEED_MAP[run] != seed:
            raise ContractError(f"{name} invalid paper-run/internal-seed mapping: run={run}, seed={seed}")


def validate_manifest(frame: pd.DataFrame) -> dict:
    required = [
        "arm", "paper_run", "internal_seed", "status", "reward", "observation_mode",
        "actor_input_dim", "actor_total_numel", "critic_observation_mode", "training_steps",
        "checkpoint_1500", "checkpoint_1500_sha256", "actor_init_sha256", "critic_init_sha256",
        "optimizer_init_sha256", "torch_cpu_rng_sha256", "torch_cuda_rng_sha256",
        "numpy_rng_sha256", "python_rng_sha256",
    ]
    _require_columns(frame, required, "run_manifest.csv")
    _validate_labels(frame, "run_manifest.csv")
    if len(frame) != 10 or frame.duplicated(["arm", "paper_run"]).any():
        raise ContractError("run_manifest.csv must contain exactly one row for each of 2 arms x 5 runs")
    if set(frame["status"].astype(str)) != {"complete"}:
        raise ContractError("Every run manifest status must be complete")
    if set(frame["reward"].astype(str)) != {"horizontal_speed"}:
        raise ContractError("Both arms must use the archived HPR implementation label horizontal_speed")
    expected_mode = {"O2": "full_o2", "O1_sham": "geometry_only_sham"}
    for row in frame.itertuples(index=False):
        if str(row.observation_mode) != expected_mode[str(row.arm)]:
            raise ContractError(f"Unexpected observation_mode for {row.arm}: {row.observation_mode}")
        if int(row.actor_input_dim) != 2 or int(row.training_steps) != 15_000_000:
            raise ContractError(f"Capacity/budget mismatch for {row.arm}, run {row.paper_run}")
        if str(row.critic_observation_mode) != "full_o2":
            raise ContractError(f"Critic observation must remain full_o2 for {row.arm}, run {row.paper_run}")
        for column in (
            "checkpoint_1500_sha256", "actor_init_sha256", "critic_init_sha256",
            "optimizer_init_sha256", "torch_cpu_rng_sha256", "numpy_rng_sha256",
            "python_rng_sha256",
        ):
            if not HEX64.fullmatch(str(getattr(row, column))):
                raise ContractError(f"Invalid SHA-256 in {column} for {row.arm}, run {row.paper_run}")
        cuda_hashes = str(row.torch_cuda_rng_sha256).split(";")
        if not cuda_hashes or any(not HEX64.fullmatch(item) for item in cuda_hashes):
            raise ContractError(f"Invalid CUDA RNG hash list for {row.arm}, run {row.paper_run}")
        checkpoint = Path(str(row.checkpoint_1500))
        if not checkpoint.is_file():
            raise ContractError(f"Checkpoint does not exist: {checkpoint}")
        observed_hash = sha256_file(checkpoint)
        if observed_hash != str(row.checkpoint_1500_sha256):
            raise ContractError(f"Checkpoint SHA-256 mismatch: {checkpoint}")

    hash_columns = [
        "actor_init_sha256", "critic_init_sha256", "optimizer_init_sha256",
        "torch_cpu_rng_sha256", "torch_cuda_rng_sha256", "numpy_rng_sha256",
        "python_rng_sha256",
    ]
    pairing = {}
    for run in RUNS:
        pair = frame.loc[frame["paper_run"].astype(int) == run].set_index("arm")
        if int(pair.loc["O1_sham", "actor_total_numel"]) != int(pair.loc["O2", "actor_total_numel"]):
            raise ContractError(f"Actor parameter count mismatch for run {run}")
        equality = {column: str(pair.loc["O1_sham", column]) == str(pair.loc["O2", column]) for column in hash_columns}
        if not all(equality.values()):
            raise ContractError(f"Initialization pairing failed for run {run}: {equality}")
        pairing[str(run)] = equality
    return {"rows": len(frame), "pairing": pairing}


def validate_training(frame: pd.DataFrame) -> dict:
    required = [
        "arm", "paper_run", "internal_seed", "batch", "reward_mean", "speed_mean",
        "ppo_approx_kl", "ppo_updates_completed", "elapsed_sec",
    ]
    _require_columns(frame, required, "training_metrics.csv")
    _validate_labels(frame, "training_metrics.csv")
    _finite(frame, ["batch", "reward_mean", "speed_mean", "ppo_approx_kl", "ppo_updates_completed", "elapsed_sec"], "training_metrics.csv")
    if frame.duplicated(["arm", "paper_run", "batch"]).any():
        raise ContractError("Duplicate arm/run/batch rows in training_metrics.csv")
    for arm in ARMS:
        for run in RUNS:
            subset = frame[(frame["arm"] == arm) & (frame["paper_run"].astype(int) == run)]
            if set(subset["batch"].astype(int)) != set(range(1, 1501)):
                raise ContractError(f"Training batches are not exactly 1..1500 for {arm}, run {run}")
            if not subset["elapsed_sec"].astype(float).is_monotonic_increasing:
                raise ContractError(f"elapsed_sec is not monotonic for {arm}, run {run}")
    return {"rows": len(frame), "expected_rows": 15_000}


def validate_episode_metrics(frame: pd.DataFrame) -> dict:
    required = [
        "arm", "paper_run", "internal_seed", "checkpoint", "episode", "reset_seed",
        "desired_net_rotation_deg", "desired_direction_fraction", "forward_body_lengths",
        "success_common", "mean_reward", "mean_speed_x", "mean_abs_k1", "mean_abs_k2",
        "torque_saturation_fraction",
    ]
    _require_columns(frame, required, "checkpoint_episode_metrics.csv")
    _validate_labels(frame, "checkpoint_episode_metrics.csv")
    numeric = [
        "checkpoint", "episode", "reset_seed", "desired_net_rotation_deg",
        "desired_direction_fraction", "forward_body_lengths", "mean_reward", "mean_speed_x",
        "mean_abs_k1", "mean_abs_k2", "torque_saturation_fraction",
    ]
    _finite(frame, numeric, "checkpoint_episode_metrics.csv")
    if frame.duplicated(["arm", "paper_run", "checkpoint", "reset_seed"]).any():
        raise ContractError("Duplicate arm/run/checkpoint/reset rows in checkpoint_episode_metrics.csv")
    for arm in ARMS:
        for run in RUNS:
            for checkpoint in CHECKPOINTS:
                subset = frame[
                    (frame["arm"] == arm)
                    & (frame["paper_run"].astype(int) == run)
                    & (frame["checkpoint"].astype(int) == checkpoint)
                ]
                if len(subset) != 20 or set(subset["reset_seed"].astype(int)) != set(RESET_SEEDS):
                    raise ContractError(f"Expected 20 frozen resets for {arm}, run {run}, checkpoint {checkpoint}")
    direction = frame["desired_direction_fraction"].astype(float)
    saturation = frame["torque_saturation_fraction"].astype(float)
    if ((direction < 0) | (direction > 1)).any():
        raise ContractError("desired_direction_fraction must be within [0,1]")
    if ((saturation < 0) | (saturation > 1)).any():
        raise ContractError("torque_saturation_fraction must be within [0,1]")

    recomputed = [
        common_success(float(r), float(d), float(f))
        for r, d, f in zip(
            frame["desired_net_rotation_deg"],
            frame["desired_direction_fraction"],
            frame["forward_body_lengths"],
        )
    ]
    supplied = [parse_bool(value) for value in frame["success_common"]]
    mismatch = np.flatnonzero(np.asarray(recomputed) != np.asarray(supplied))
    if len(mismatch):
        raise ContractError(f"success_common disagrees with frozen criterion in {len(mismatch)} rows")

    endpoint = frame[frame["checkpoint"].astype(int) == 1500]
    o2_counts = endpoint[endpoint["arm"] == "O2"].groupby("paper_run")["success_common"].apply(
        lambda values: sum(parse_bool(value) for value in values)
    )
    expected = {0: 20, 1: 0, 2: 7, 3: 0, 4: 20}
    observed = {int(key): int(value) for key, value in o2_counts.items()}
    if observed != expected:
        raise ContractError(f"Archived O2 endpoint regression mismatch: expected={expected}, observed={observed}")
    return {"rows": len(frame), "expected_rows": 3_000, "o2_endpoint_counts": observed}


def validate_trajectories(joint: pd.DataFrame, node: pd.DataFrame, representative_reset: int) -> dict:
    joint_required = [
        "arm", "paper_run", "internal_seed", "checkpoint", "episode", "reset_seed", "step",
        "time_s", "joint", "spatial_difference", "angular_velocity", "k1", "k2",
        "k1_component", "k2_component", "torque", "torque_saturated", "com_x", "com_y",
        "body_rotation_deg", "forward_body_lengths", "direction_fraction",
    ]
    node_required = [
        "arm", "paper_run", "internal_seed", "checkpoint", "episode", "reset_seed", "step",
        "time_s", "node", "x", "y", "ground_height",
    ]
    _require_columns(joint, joint_required, "trajectory_joint.csv")
    _require_columns(node, node_required, "trajectory_node.csv")
    _validate_labels(joint, "trajectory_joint.csv")
    _validate_labels(node, "trajectory_node.csv")
    _finite(joint, [
        "checkpoint", "episode", "reset_seed", "step", "time_s", "joint", "spatial_difference",
        "angular_velocity", "k1", "k2", "k1_component", "k2_component", "torque", "com_x",
        "com_y", "body_rotation_deg", "forward_body_lengths", "direction_fraction",
    ], "trajectory_joint.csv")
    _finite(node, ["checkpoint", "episode", "reset_seed", "step", "time_s", "node", "x", "y", "ground_height"], "trajectory_node.csv")
    if joint.duplicated(["arm", "paper_run", "checkpoint", "reset_seed", "step", "joint"]).any():
        raise ContractError("Duplicate rows in trajectory_joint.csv")
    if node.duplicated(["arm", "paper_run", "checkpoint", "reset_seed", "step", "node"]).any():
        raise ContractError("Duplicate rows in trajectory_node.csv")
    for arm in ARMS:
        for run in RUNS:
            j = joint[(joint["arm"] == arm) & (joint["paper_run"].astype(int) == run)]
            n = node[(node["arm"] == arm) & (node["paper_run"].astype(int) == run)]
            if set(j["checkpoint"].astype(int)) != {1500} or set(n["checkpoint"].astype(int)) != {1500}:
                raise ContractError(f"Representative trajectories must be checkpoint 1500 for {arm}, run {run}")
            if set(j["reset_seed"].astype(int)) != {representative_reset} or set(n["reset_seed"].astype(int)) != {representative_reset}:
                raise ContractError(f"Representative reset mismatch for {arm}, run {run}")
            if set(j["step"].astype(int)) != set(range(1000)) or set(n["step"].astype(int)) != set(range(1000)):
                raise ContractError(f"Representative trajectory must contain steps 0..999 for {arm}, run {run}")
            if set(j["joint"].astype(int)) != set(range(1, 9)):
                raise ContractError(f"Expected joints 1..8 for {arm}, run {run}")
            if set(n["node"].astype(int)) != set(range(10)):
                raise ContractError(f"Expected nodes 0..9 for {arm}, run {run}")
            if len(j) != 8_000 or len(n) != 10_000:
                raise ContractError(f"Unexpected representative trajectory row count for {arm}, run {run}")
    supplied_sat = [parse_bool(value) for value in joint["torque_saturated"]]
    if not all(isinstance(value, bool) for value in supplied_sat):
        raise ContractError("torque_saturated must be boolean")
    return {"joint_rows": len(joint), "node_rows": len(node), "representative_reset": representative_reset}


def validate_actor_probe(frame: pd.DataFrame) -> dict:
    required = [
        "arm", "paper_run", "internal_seed", "checkpoint", "joint",
        "spatial_difference", "angular_velocity", "k1", "k2",
    ]
    _require_columns(frame, required, "actor_probe.csv")
    _validate_labels(frame, "actor_probe.csv")
    _finite(frame, ["checkpoint", "joint", "spatial_difference", "angular_velocity", "k1", "k2"], "actor_probe.csv")
    if set(frame["checkpoint"].astype(int)) != {1500} or set(frame["joint"].astype(int)) != set(range(1, 9)):
        raise ContractError("actor_probe.csv must cover checkpoint 1500 and joints 1..8")
    if frame.duplicated(["arm", "paper_run", "joint", "spatial_difference", "angular_velocity"]).any():
        raise ContractError("Duplicate actor-probe coordinates")
    coordinates = None
    for arm in ARMS:
        for run in RUNS:
            for joint in range(1, 9):
                subset = frame[
                    (frame["arm"] == arm)
                    & (frame["paper_run"].astype(int) == run)
                    & (frame["joint"].astype(int) == joint)
                ]
                observed = set(zip(subset["spatial_difference"].astype(float), subset["angular_velocity"].astype(float)))
                if not observed:
                    raise ContractError(f"Missing actor probe for {arm}, run {run}, joint {joint}")
                if coordinates is None:
                    coordinates = observed
                elif observed != coordinates:
                    raise ContractError("Actor-probe coordinates must be identical across arms, runs and joints")
    for (run, joint, spatial), subset in frame[frame["arm"] == "O1_sham"].groupby(
        ["paper_run", "joint", "spatial_difference"]
    ):
        if subset["k1"].astype(float).max() - subset["k1"].astype(float).min() > 1e-8:
            raise ContractError(f"O1-sham K1 responds to masked angular velocity: run={run}, joint={joint}, s={spatial}")
        if subset["k2"].astype(float).max() - subset["k2"].astype(float).min() > 1e-8:
            raise ContractError(f"O1-sham K2 responds to masked angular velocity: run={run}, joint={joint}, s={spatial}")
    return {"rows": len(frame), "probe_coordinates_per_actor": len(coordinates or ())}


def validate(data_root: Path, receipt_path: Path, contract_path: Path) -> dict:
    if receipt_path.exists():
        receipt_path.unlink()
    contract = load_json(contract_path)
    local_contract = load_json(data_root / "study_contract.json")
    if local_contract != contract:
        raise ContractError("Data-root study_contract.json differs from the preregistered contract")

    manifest = _read_table(data_root, "run_manifest.csv")
    training = _read_table(data_root, "training_metrics.csv")
    episodes = _read_table(data_root, "checkpoint_episode_metrics.csv")
    joint = _read_table(data_root, "trajectory_joint.csv")
    node = _read_table(data_root, "trajectory_node.csv")
    probe = _read_table(data_root, "actor_probe.csv")

    checks = {
        "run_manifest": validate_manifest(manifest),
        "training_metrics": validate_training(training),
        "checkpoint_episode_metrics": validate_episode_metrics(episodes),
        "trajectories": validate_trajectories(joint, node, int(contract["predeclared_representative_reset"])),
        "actor_probe": validate_actor_probe(probe),
    }
    inputs = {}
    for name in REQUIRED_FILES:
        path = data_root / name
        if not path.exists() and name.endswith(".csv"):
            path = data_root / f"{name}.gz"
        inputs[str(path)] = sha256_file(path)
    receipt = {
        "schema": "o1_o2_formal_analysis_validation/v1",
        "status": "pass",
        "data_root": str(data_root.resolve()),
        "contract_sha256": sha256_file(contract_path),
        "checks": checks,
        "input_sha256": inputs,
    }
    write_json(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--contract",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "study_contract.json",
    )
    args = parser.parse_args()
    try:
        receipt = validate(args.data_root.resolve(), args.receipt.resolve(), args.contract.resolve())
    except Exception as exc:
        if args.receipt.exists():
            args.receipt.unlink()
        print(f"FORMAL INPUT VALIDATION FAILED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

