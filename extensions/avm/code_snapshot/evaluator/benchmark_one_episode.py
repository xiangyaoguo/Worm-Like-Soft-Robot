"""One archived O2 rollout for runtime estimation; never writes formal data."""

from __future__ import annotations

import json
import time
from pathlib import Path

import frozen_evaluator as evaluator


HERE = Path(__file__).resolve().parent


def main() -> None:
    started = time.perf_counter()
    config = evaluator.read_json(HERE / "evaluator_config.json")
    task = evaluator.Task("O2", 9201, 100)
    checkpoint = evaluator.checkpoint_path(config, task)
    deps = evaluator.Dependencies(config)
    metadata = dict(deps.metadata_from_checkpoint(checkpoint))
    contract = evaluator.validate_metadata(metadata, config, task.arm_id, task.seed)
    env, *_ = deps.build_demo_env(
        metadata,
        "flat",
        deps.TerrainArgs(),
        max_steps=1000,
        render_mode="rgb_array",
        num_envs=1,
    )
    try:
        evaluator.validate_environment(env, config)
        policy = deps.load_policy_for_env(checkpoint, env, metadata)
        setup_seconds = time.perf_counter() - started
        rollout_started = time.perf_counter()
        metrics, arrays = evaluator.run_episode(
            deps,
            env,
            policy,
            config,
            contract["actor_observation_mode"],
            20264101,
        )
        rollout_seconds = time.perf_counter() - rollout_started
        print(
            json.dumps(
                {
                    "schema": "formal_hpr_o1_sham_frozen_evaluator/benchmark/v1",
                    "nonformal": True,
                    "setup_seconds": setup_seconds,
                    "one_1000_step_rollout_seconds": rollout_seconds,
                    "success_common_kinematic": metrics["success_common_kinematic"],
                    "trace_bytes_uncompressed": sum(value.nbytes for value in arrays.values()),
                    "passed": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    finally:
        evaluator.close_env(env)


if __name__ == "__main__":
    main()
