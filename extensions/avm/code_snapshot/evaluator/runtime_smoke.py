"""One-step, read-only runtime smoke test; this is not a formal evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import frozen_evaluator as evaluator


HERE = Path(__file__).resolve().parent


def main() -> None:
    config = evaluator.read_json(HERE / "evaluator_config.json")
    evaluator.validate_config(config)
    task = evaluator.Task("O2", 9201, 100)
    checkpoint = evaluator.checkpoint_path(config, task)
    deps = evaluator.Dependencies(config)
    metadata = dict(deps.metadata_from_checkpoint(checkpoint))
    contract = evaluator.validate_metadata(metadata, config, task.arm_id, task.seed)
    env, *_ = deps.build_demo_env(
        metadata,
        "flat",
        deps.TerrainArgs(),
        max_steps=1,
        render_mode="rgb_array",
        num_envs=1,
    )
    try:
        evaluator.validate_environment(env, config)
        policy = deps.load_policy_for_env(checkpoint, env, metadata)
        deps.torch.manual_seed(20264101)
        deps.np.random.seed(20264101)
        td = env.reset()
        raw, actor, actor_td = evaluator.actor_input(
            td, contract["actor_observation_mode"], deps.torch
        )
        chosen = deps.choose_action(policy, actor_td, "deterministic")
        action = chosen["agents", "action"]
        env_td = td.clone(recurse=True)
        env_td["agents", "action"] = action
        next_td = env.step(env_td)["next"]
        position = deps.metric._positions(env)
        result = {
            "schema": "formal_hpr_o1_sham_frozen_evaluator/runtime_smoke/v1",
            "nonformal": True,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": evaluator.sha256_file(checkpoint),
            "actor_observation_mode": contract["actor_observation_mode"],
            "raw_observation_shape": list(raw.shape),
            "actor_observation_shape": list(actor.shape),
            "action_shape": list(action.shape),
            "next_observation_shape": list(next_td["agents", "observation"].shape),
            "position_shape": list(position.shape),
            "passed": True,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        evaluator.close_env(env)


if __name__ == "__main__":
    main()
