"""Verify environment channels plus shared and per-joint K1/K2 actor modes."""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torchrl.modules import MultiAgentMLP

from rlmm_common import add_env_package_to_path, channel_config, find_project_root

PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
add_env_package_to_path(PROJECT_ROOT)
from metamaterial_envs.env import metamaterial  # noqa: E402


def main() -> None:
    expected = {
        "dth": ("dth", "dth_neighbours", "direct", 2),
        "thdot": ("thdot", "dth_neighbours_plus_thdot", "direct", 3),
    }
    for requested, (channel, observation, control, obs_dim) in expected.items():
        resolved = channel_config(requested)
        assert resolved == (channel, observation, control), (requested, resolved)
        env = metamaterial.env(
            num_envs=1,
            material_shape="ring",
            num_particles=10,
            max_steps=1,
            observation_func=observation,
            control_mode=control,
            feedback_gain=123.0,  # must be ignored by the environment
        )
        assert env.feedback_gain == 1.0
        assert env.background_friction == 0.0
        assert env.ground_stiffness == 1000.0
        assert env.ground_damping == 5.0
        assert env.observation_spec["agents", "observation"].shape[-1] == obs_dim
        assert env.action_spec["agents", "action"].shape[-1] == 1
        td = env.reset()
        td = env.rand_action(td)
        env.step(td)
        print(
            f"{requested}: observation={observation}, obs_dim={obs_dim}, "
            f"control={control}, feedback_gain={env.feedback_gain}, "
            f"ground_damping={env.ground_damping}"
        )
    formula_cases = [
        (
            "formula_learn_k1_k2",
            {"control_mode": "formula", "k1_min": -2.0, "k1_max": 3.0, "k2_min": -4.0, "k2_max": 5.0},
            ("k1", "k2"),
        ),
        (
            "formula_fix_k1",
            {"control_mode": "formula", "fix_k1": True, "fixed_k1": 1.25, "k2_min": -4.0, "k2_max": 5.0},
            ("k2",),
        ),
        (
            "formula_fix_k2",
            {"control_mode": "formula", "fix_k2": True, "fixed_k2": -0.75, "k1_min": -2.0, "k1_max": 3.0},
            ("k1",),
        ),
        (
            "formula_fix_k1_k2",
            {"control_mode": "formula", "fix_k1": True, "fixed_k1": 1.25, "fix_k2": True, "fixed_k2": -0.75},
            (),
        ),
    ]
    for name, kwargs, action_names in formula_cases:
        env = metamaterial.env(
            num_envs=1,
            material_shape="ring",
            num_particles=10,
            max_steps=1,
            observation_func="dth_tot",
            **kwargs,
        )
        expected_action_dim = len(action_names) if action_names else 1
        assert env.action_size == expected_action_dim
        assert env.action_spec["agents", "action"].shape[-1] == expected_action_dim
        assert env.formula_action_names == action_names
        td = env.reset()
        td = env.rand_action(td)
        env.step(td)
        print(f"{name}: action_dim={expected_action_dim}, action_names={action_names}")
    default_formula_env = metamaterial.env(
        num_envs=1,
        material_shape="ring",
        num_particles=10,
        max_steps=1,
        observation_func="dth_tot",
        control_mode="formula",
    )
    assert math.isinf(default_formula_env.k1_min) and default_formula_env.k1_min < 0
    assert math.isinf(default_formula_env.k1_max) and default_formula_env.k1_max > 0
    assert math.isinf(default_formula_env.k2_min) and default_formula_env.k2_min < 0
    assert math.isinf(default_formula_env.k2_max) and default_formula_env.k2_max > 0
    assert type(default_formula_env.action_spec["agents", "action"]).__name__.startswith("Unbounded")
    print("formula_default_unbounded: K1/K2 bounds are [-inf, inf]")

    crawler_formula_env = metamaterial.env(
        num_envs=3,
        material_shape="crawler",
        num_particles=13,
        max_steps=1,
        observation_func="dth_tot",
        control_mode="formula",
        k1_min=-9.0,
        k1_max=9.0,
        k2_min=-9.0,
        k2_max=9.0,
    )
    assert crawler_formula_env.num_agents == 11
    assert tuple(crawler_formula_env.action_spec["agents", "action"].shape) == (3, 11, 2)
    td = crawler_formula_env.reset()
    td = crawler_formula_env.rand_action(td)
    assert tuple(td["agents", "action"].shape) == (3, 11, 2)
    crawler_formula_env.step(td)
    print("formula_per_joint_action_shape: crawler action shape is [3, 11, 2]")

    torch.manual_seed(0)
    actor_kwargs = {
        "n_agent_inputs": default_formula_env.observation_spec["agents", "observation"].shape[-1],
        "n_agent_outputs": default_formula_env.action_spec["agents", "action"].shape[-1],
        "n_agents": default_formula_env.num_agents,
        "centralised": False,
        "device": "cpu",
        "depth": 1,
        "num_cells": 8,
        "activation_class": torch.nn.Tanh,
    }
    shared_actor = MultiAgentMLP(share_params=True, **actor_kwargs)
    independent_actor = MultiAgentMLP(share_params=False, **actor_kwargs)
    identical_observations = torch.zeros(
        3,
        default_formula_env.num_agents,
        default_formula_env.observation_spec["agents", "observation"].shape[-1],
    )
    with torch.no_grad():
        shared_output = shared_actor(identical_observations)
        independent_output = independent_actor(identical_observations)
    expected_output_shape = (3, default_formula_env.num_agents, 2)
    assert tuple(shared_output.shape) == expected_output_shape
    assert tuple(independent_output.shape) == expected_output_shape
    assert torch.allclose(shared_output[:, 0], shared_output[:, 1])
    assert not torch.allclose(independent_output[:, 0], independent_output[:, 1])
    shared_parameter_count = sum(parameter.numel() for parameter in shared_actor.parameters())
    independent_parameter_count = sum(parameter.numel() for parameter in independent_actor.parameters())
    assert independent_parameter_count == shared_parameter_count * default_formula_env.num_agents
    print(
        "policy_sharing: shared actor matches across identical observations; "
        "independent actor keeps one parameter set per joint"
    )
    print("All checks passed.")


if __name__ == "__main__":
    main()
