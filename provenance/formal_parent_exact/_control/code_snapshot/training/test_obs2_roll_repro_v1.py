from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "metamaterial_envs"))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from metamaterial_envs.env import metamaterial  # noqa: E402


def make_env(reward_func: str, *, randomness: float = 0.01):
    return metamaterial.env(
        num_envs=1,
        material_shape="crawler",
        num_particles=10,
        max_steps=1000,
        terrain_type="flat",
        terrain_contact_mode="legacy_flat",
        observation_func="dth_tot_plus_friction_thdot",
        reward_func=reward_func,
        control_mode="formula",
        feedback_gain=1.0,
        k_action_scale=100.0,
        passive_kappa=4.0,
        rolling_observation=False,
        tail_roll_observation=False,
        fast_forward_observation=False,
        rolling_direction="right",
        rolling_curl_episodes=300,
        rolling_transition_episodes=300,
        rolling_reward_scale=3.0,
        init_pos_randomness=randomness,
        init_angle_range_degrees=0.0,
        init_height_jitter=0.0,
        action_smoothness_weight=0.0,
        scratch_wr_v2=False,
    )


def assert_contract() -> None:
    baseline = make_env("horizontal_speed")
    reward_only = make_env("obs2_roll_repro_v1")
    try:
        np.random.seed(12345)
        baseline_td = baseline.reset()
        np.random.seed(12345)
        reward_td = reward_only.reset()

        baseline_obs = baseline_td[("agents", "observation")]
        reward_obs = reward_td[("agents", "observation")]
        assert tuple(baseline_obs.shape) == (1, 8, 2)
        assert tuple(reward_obs.shape) == (1, 8, 2)
        torch.testing.assert_close(baseline_obs, reward_obs, rtol=0.0, atol=0.0)
        assert baseline.rolling_observation_size == reward_only.rolling_observation_size == 0
        assert baseline.tail_roll_observation_size == reward_only.tail_roll_observation_size == 0
        assert baseline.fast_forward_observation_size == reward_only.fast_forward_observation_size == 0
        assert baseline.scratch_wr_v2_observation_size == reward_only.scratch_wr_v2_observation_size == 0
        assert tuple(reward_only.formula_action_names) == ("k1", "k2")
        action_shape = tuple(reward_only.action_spec[reward_only.action_key].shape)
        assert action_shape == (1, 8, 2), action_shape
        assert "Unbounded" in type(reward_only.action_spec[reward_only.action_key]).__name__

        generator = torch.Generator().manual_seed(24680)
        for step in range(10):
            action = torch.randn((1, 8, 2), generator=generator, dtype=torch.float32) * 0.05
            baseline_td.set(baseline.action_key, action.clone())
            reward_td.set(reward_only.action_key, action.clone())
            baseline_td = baseline.step(baseline_td)["next"]
            reward_td = reward_only.step(reward_td)["next"]
            np.testing.assert_allclose(baseline.pos, reward_only.pos, rtol=0.0, atol=0.0, err_msg=f"pos step {step}")
            np.testing.assert_allclose(baseline.vel, reward_only.vel, rtol=0.0, atol=0.0, err_msg=f"vel step {step}")
            np.testing.assert_allclose(baseline.thdot, reward_only.thdot, rtol=0.0, atol=0.0, err_msg=f"thdot step {step}")
            torch.testing.assert_close(
                baseline_td[("agents", "observation")],
                reward_td[("agents", "observation")],
                rtol=0.0,
                atol=0.0,
                msg=f"observation step {step}",
            )
    finally:
        baseline.close()
        reward_only.close()


def assert_parser_default() -> None:
    import train_metamaterial

    original = sys.argv[:]
    try:
        sys.argv = [
            "train_metamaterial.py",
            "--robot", "crawler",
            "--terrain", "flat",
            "--channel", "action",
            "--observation-func", "dth_tot_plus_friction_thdot",
            "--control-mode", "formula",
            "--reward-func", "obs2_roll_repro_v1",
            "--per-joint-k1-k2",
            "--no-rolling-observation",
            "--no-tail-roll-observation",
            "--no-fast-forward-observation",
        ]
        args = train_metamaterial.parse_args()
    finally:
        sys.argv = original
    assert args.reward_func == "obs2_roll_repro_v1"
    assert args.rolling_observation is False
    assert args.tail_roll_observation is False
    assert args.fast_forward_observation is False
    assert args.scratch_wr_v2 is False


def assert_curriculum() -> None:
    test_env = make_env("obs2_roll_repro_v1", randomness=0.0)
    try:
        test_env.reset()
        expected = {0: 0.0, 300: 0.0, 450: 0.5, 600: 1.0}
        for batch, value in expected.items():
            test_env.set_curriculum_episode(batch)
            test_env._rolling_metrics_cache = None
            actual = float(test_env._compute_rolling_metrics()["curriculum_progress"][0, 0])
            assert math.isclose(actual, value, rel_tol=0.0, abs_tol=1e-7), (batch, actual)
    finally:
        test_env.close()


def ring_shape(angle: float = 0.0, x_offset: float = 0.0, airborne: bool = False) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False, dtype=np.float32) + np.float32(angle)
    points = np.cos(theta) + 1j * np.sin(theta)
    # Grounded synthetic shapes intentionally touch y=0 so the reward's
    # material-contact detector is unambiguously active.
    target_min_y = np.float32(5.0 if airborne else 0.0)
    points = points + np.float32(x_offset) + 1j * (target_min_y - np.min(np.imag(points)))
    return np.asarray(points[None, :], dtype=np.complex64)


def straight_shape(x_offset: float = 0.0) -> np.ndarray:
    points = np.arange(10, dtype=np.float32) + np.float32(x_offset) + np.float32(0.0) * 1j
    return np.asarray(points[None, :], dtype=np.complex64)


def inject_shape(test_env, shape: np.ndarray) -> float:
    previous = np.asarray(test_env.pos, dtype=np.complex64).copy()
    test_env.pos = np.ascontiguousarray(shape, dtype=np.complex64)
    test_env.vel = np.ascontiguousarray(shape - previous, dtype=np.complex64)
    test_env.thdot = np.zeros(test_env.pos.shape, dtype=np.float32)
    test_env.mean_speed = np.asarray(
        np.real(np.mean(shape - previous, axis=1, keepdims=True)), dtype=np.float32
    )
    test_env.steps += 1
    test_env._rolling_metrics_cache = None
    test_env._tail_roll_metrics_cache = None
    test_env._fast_rollover_metrics_cache = None
    test_env._fast_forward_metrics_cache = None
    test_env._terrain_contact_cache = None
    return float(test_env._reward_func().detach().cpu().numpy()[0, 0])


def assert_reward_adversaries_and_event() -> None:
    # A static loop may earn a one-time launch milestone, never a recurring reward.
    static_env = make_env("obs2_roll_repro_v1", randomness=0.0)
    try:
        static_env.reset()
        static_env.set_curriculum_episode(0)
        static_rewards = [inject_shape(static_env, ring_shape()) for _ in range(20)]
        assert max(abs(value) for value in static_rewards[-5:]) < 1e-6, static_rewards
        fast = static_env._compute_fast_forward_roll_v2_metrics()
        assert float(fast["fast_forward_event_count"][0, 0]) == 0.0
    finally:
        static_env.close()

    # Straight translation and airborne spin cannot create a grounded roll event.
    straight_env = make_env("obs2_roll_repro_v1", randomness=0.0)
    try:
        straight_env.reset()
        straight_env.set_curriculum_episode(600)
        for index in range(20):
            inject_shape(straight_env, straight_shape(0.15 * (index + 1)))
        fast = straight_env._compute_fast_forward_roll_v2_metrics()
        assert float(fast["fast_forward_event_count"][0, 0]) == 0.0
    finally:
        straight_env.close()

    airborne_env = make_env("obs2_roll_repro_v1", randomness=0.0)
    try:
        airborne_env.reset()
        airborne_env.set_curriculum_episode(600)
        airborne_rewards = []
        for index in range(20):
            airborne_rewards.append(
                inject_shape(
                    airborne_env,
                    ring_shape(angle=-math.radians(20.0) * index, x_offset=0.15 * index, airborne=True),
                )
            )
        fast = airborne_env._compute_fast_forward_roll_v2_metrics()
        assert float(fast["fast_forward_event_count"][0, 0]) == 0.0
        assert max(airborne_rewards[-5:]) <= 1e-6, airborne_rewards
    finally:
        airborne_env.close()

    # A closed, grounded, clockwise, translating loop must trigger a real event.
    rolling_env = make_env("obs2_roll_repro_v1", randomness=0.0)
    try:
        rolling_env.reset()
        rolling_env.set_curriculum_episode(600)
        for _ in range(10):
            inject_shape(rolling_env, ring_shape())
        launch_metrics = rolling_env._compute_fast_forward_roll_v2_metrics()
        assert float(launch_metrics["fast_forward_phase"][0, 0]) == 1.0
        for index in range(1, 8):
            inject_shape(
                rolling_env,
                ring_shape(angle=-math.radians(20.0) * index, x_offset=0.15 * index),
            )
        fast = rolling_env._compute_fast_forward_roll_v2_metrics()
        assert float(fast["fast_forward_event_count"][0, 0]) >= 1.0, fast
        count_before = float(fast["fast_forward_event_count"][0, 0])
        rolling_env._reward_func()
        rolling_env._get_info()
        count_after = float(
            rolling_env._compute_fast_forward_roll_v2_metrics()["fast_forward_event_count"][0, 0]
        )
        assert count_after == count_before
        rolling_env.reset()
        assert float(rolling_env._fast_forward_event_count[0, 0]) == 0.0
        assert float(rolling_env._fast_forward_phase[0, 0]) == 0.0
    finally:
        rolling_env.close()


def main() -> None:
    assert_contract()
    assert_parser_default()
    assert_curriculum()
    assert_reward_adversaries_and_event()
    print("PASS: obs2_roll_repro_v1 reward-only contract and adversarial tests")


if __name__ == "__main__":
    main()
