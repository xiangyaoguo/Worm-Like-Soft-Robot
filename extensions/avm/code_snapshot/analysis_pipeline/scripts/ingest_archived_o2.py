from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from common import CHECKPOINTS, SEED_MAP, common_success, sha256_file, write_csv_rows, write_json


RUN_FIELDS = [
    "arm", "paper_run", "internal_seed", "status", "reward", "observation_mode",
    "actor_input_dim", "actor_total_numel", "critic_observation_mode", "training_steps",
    "checkpoint_1500", "checkpoint_1500_sha256", "actor_init_sha256", "critic_init_sha256",
    "optimizer_init_sha256", "torch_cpu_rng_sha256", "torch_cuda_rng_sha256",
    "numpy_rng_sha256", "python_rng_sha256", "initialization_audit", "source_run_dir",
]

TRAIN_FIELDS = [
    "arm", "paper_run", "internal_seed", "batch", "reward_mean", "speed_mean",
    "ppo_approx_kl", "ppo_updates_completed", "elapsed_sec",
]

EPISODE_FIELDS = [
    "arm", "paper_run", "internal_seed", "checkpoint", "episode", "reset_seed",
    "desired_net_rotation_deg", "desired_direction_fraction", "forward_body_lengths",
    "success_common", "mean_reward", "mean_speed_x", "mean_abs_k1", "mean_abs_k2",
    "torque_saturation_fraction", "source_json",
]


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ingest(formal_root: Path, data_root: Path) -> dict:
    runs_root = require(formal_root / "formal" / "runs")
    init_root = require(formal_root / "formal" / "initialization")
    eval_root = require(formal_root / "formal" / "evaluations")

    data_root.mkdir(parents=True, exist_ok=True)
    existing = [data_root / name for name in ("run_manifest.csv", "training_metrics.csv", "checkpoint_episode_metrics.csv")]
    if any(path.exists() for path in existing):
        raise FileExistsError(
            "Refusing to overwrite normalized data. Use a new empty data root; merge O1-sham with an audited merge step."
        )

    run_rows: list[dict] = []
    training_rows: list[dict] = []
    episode_rows: list[dict] = []

    for paper_run, seed in SEED_MAP.items():
        run_name = f"formal__seed{seed}__R0"
        run_dir = require(runs_root / run_name)
        init_path = require(init_root / f"{run_name}.json")
        checkpoint = require(run_dir / "checkpoint_1500.pt")
        init = load_json(init_path)
        if init.get("seed") != seed or init.get("reward") != "horizontal_speed":
            raise RuntimeError(f"Unexpected initialization audit for {run_name}")
        if init.get("runtime_args", {}).get("observation_func") != "dth_tot_plus_friction_thdot":
            raise RuntimeError(f"Archived O2 observation mismatch for {run_name}")

        cuda_hash = init.get("rng", {}).get("torch_cuda_sha256", [])
        run_rows.append({
            "arm": "O2",
            "paper_run": paper_run,
            "internal_seed": seed,
            "status": "complete",
            "reward": "horizontal_speed",
            "observation_mode": "full_o2",
            "actor_input_dim": 2,
            "actor_total_numel": init["actor"]["total_numel"],
            "critic_observation_mode": "full_o2",
            "training_steps": 15_000_000,
            "checkpoint_1500": str(checkpoint),
            "checkpoint_1500_sha256": sha256_file(checkpoint),
            "actor_init_sha256": init["actor"]["state_sha256"],
            "critic_init_sha256": init["critic"]["state_sha256"],
            "optimizer_init_sha256": init["optimizer_sha256"],
            "torch_cpu_rng_sha256": init["rng"]["torch_cpu_sha256"],
            "torch_cuda_rng_sha256": ";".join(cuda_hash),
            "numpy_rng_sha256": init["rng"]["numpy_sha256"],
            "python_rng_sha256": init["rng"]["python_sha256"],
            "initialization_audit": str(init_path),
            "source_run_dir": str(run_dir),
        })

        log_path = require(run_dir / "training_log.csv")
        with log_path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                batch = int(row["episode"])
                training_rows.append({
                    "arm": "O2",
                    "paper_run": paper_run,
                    "internal_seed": seed,
                    "batch": batch,
                    "reward_mean": row["reward_mean"],
                    "speed_mean": row["speed_mean"],
                    "ppo_approx_kl": row["ppo_approx_kl"],
                    "ppo_updates_completed": row["ppo_updates_completed"],
                    "elapsed_sec": row["elapsed_sec"],
                })
        batches = {int(row["batch"]) for row in training_rows if int(row["paper_run"]) == paper_run}
        if batches != set(range(1, 1501)):
            raise RuntimeError(f"Training log for {run_name} is not exactly batches 1..1500")

        eval_path = require(eval_root / f"{run_name}__eval_attempt1.json")
        evaluation = load_json(eval_path)
        results = evaluation.get("results", [])
        endpoint = [item for item in results if item.get("checkpoint_name") == "checkpoint_1500.pt"]
        if len(endpoint) != 1 or len(endpoint[0].get("episodes", [])) != 20:
            raise RuntimeError(f"Expected one 20-episode checkpoint-1500 result for {run_name}")
        for episode in endpoint[0]["episodes"]:
            rotation = float(episode["desired_net_rotation_degrees"])
            direction = float(episode["desired_active_rotation_fraction"])
            forward = float(episode["forward_body_lengths"])
            episode_rows.append({
                "arm": "O2",
                "paper_run": paper_run,
                "internal_seed": seed,
                "checkpoint": 1500,
                "episode": int(episode["episode"]),
                "reset_seed": int(episode["seed"]),
                "desired_net_rotation_deg": rotation,
                "desired_direction_fraction": direction,
                "forward_body_lengths": forward,
                "success_common": int(common_success(rotation, direction, forward)),
                "mean_reward": "",
                "mean_speed_x": "",
                "mean_abs_k1": "",
                "mean_abs_k2": "",
                "torque_saturation_fraction": "",
                "source_json": str(eval_path),
            })

    write_csv_rows(data_root / "run_manifest.csv", RUN_FIELDS, run_rows)
    write_csv_rows(data_root / "training_metrics.csv", TRAIN_FIELDS, training_rows)
    write_csv_rows(data_root / "checkpoint_episode_metrics.csv", EPISODE_FIELDS, episode_rows)

    counts = {
        str(run): sum(int(row["success_common"]) for row in episode_rows if row["paper_run"] == run)
        for run in range(5)
    }
    if counts != {"0": 20, "1": 0, "2": 7, "3": 0, "4": 20}:
        raise RuntimeError(f"Archived O2 common-criterion regression mismatch: {counts}")

    receipt = {
        "schema": "archived_o2_normalisation/v1",
        "status": "incomplete_paired_study",
        "reason": "Only archived O2 rows are present; O1-sham formal results must be appended before validation/plotting.",
        "formal_root": str(formal_root),
        "paper_run_to_internal_seed": {str(k): v for k, v in SEED_MAP.items()},
        "endpoint_success_counts": counts,
        "row_counts": {
            "run_manifest": len(run_rows),
            "training_metrics": len(training_rows),
            "checkpoint_episode_metrics": len(episode_rows),
        },
        "file_sha256": {
            name: sha256_file(data_root / name)
            for name in ("run_manifest.csv", "training_metrics.csv", "checkpoint_episode_metrics.csv")
        },
    }
    write_json(data_root / "ARCHIVED_O2_IMPORT_RECEIPT.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    args = parser.parse_args()
    receipt = ingest(args.formal_root.resolve(), args.data_root.resolve())
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

