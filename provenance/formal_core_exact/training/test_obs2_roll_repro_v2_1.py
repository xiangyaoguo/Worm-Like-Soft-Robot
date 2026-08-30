from __future__ import annotations

import ast
import math
import inspect
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "metamaterial_envs"))
sys.path.insert(0, str(PROJECT_ROOT / "training"))

from metamaterial_envs.env import metamaterial  # noqa: E402


def make_env(
    reward_func: str,
    *,
    randomness: float = 0.0,
    num_envs: int = 1,
):
    return metamaterial.env(
        num_envs=num_envs,
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


def clear_metric_caches(test_env) -> None:
    test_env._rolling_metrics_cache = None
    test_env._tail_roll_metrics_cache = None
    test_env._fast_rollover_metrics_cache = None
    test_env._fast_forward_metrics_cache = None
    test_env._terrain_contact_cache = None


def inject_shapes(test_env, shapes: np.ndarray) -> np.ndarray:
    shapes = np.ascontiguousarray(shapes, dtype=np.complex64)
    previous = np.asarray(test_env.pos, dtype=np.complex64).copy()
    test_env.pos = shapes
    test_env.vel = np.ascontiguousarray(shapes - previous, dtype=np.complex64)
    test_env.thdot = np.zeros(test_env.pos.shape, dtype=np.float32)
    test_env.mean_speed = np.asarray(
        np.real(np.mean(shapes - previous, axis=1, keepdims=True)),
        dtype=np.float32,
    )
    test_env.steps += 1
    clear_metric_caches(test_env)
    reward = test_env._reward_func().detach().cpu().numpy()
    assert reward.shape == (test_env.num_envs, 1)
    assert reward.dtype == np.float32
    assert np.all(np.isfinite(reward))
    return reward


def straight_shape(x_offset: float = 0.0) -> np.ndarray:
    points = np.arange(10, dtype=np.float32) + np.float32(x_offset)
    return np.asarray(points[None, :], dtype=np.complex64)


def ring_shape(
    angle_degrees: float = 0.0,
    x_offset: float = 0.0,
    floor_offset: float = 0.0,
) -> np.ndarray:
    theta = (
        np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False, dtype=np.float32)
        + np.float32(math.radians(angle_degrees))
    )
    points = np.cos(theta) + 1j * np.sin(theta)
    points = points + np.float32(x_offset)
    points = points + 1j * (
        np.float32(floor_offset) - np.min(np.imag(points))
    )
    return np.asarray(points[None, :], dtype=np.complex64)


def tail_first_shape(
    *,
    curvature_degrees: float = 10.0,
    rigid_rotation_degrees: float = 0.0,
    x_offset: float = 0.0,
    floor_offset: float = 0.0,
) -> np.ndarray:
    headings = np.deg2rad(
        np.float32(150.0)
        + np.float32(curvature_degrees) * np.arange(9, dtype=np.float32)
    )
    segments = np.cos(headings) + 1j * np.sin(headings)
    points = np.concatenate(
        (np.zeros(1, dtype=np.complex64), np.cumsum(segments).astype(np.complex64))
    )
    center = np.mean(points)
    rotation = np.exp(1j * np.float32(math.radians(rigid_rotation_degrees)))
    points = (points - center) * rotation + center
    points = points + np.float32(x_offset)
    points = points + 1j * (
        np.float32(floor_offset) - np.min(np.imag(points))
    )
    return np.asarray(points[None, :], dtype=np.complex64)


def assert_reward_only_contract_and_batch_independence() -> None:
    rewards = (
        "horizontal_speed",
        "obs2_roll_repro_v1",
        "obs2_roll_repro_v2",
        "obs2_roll_repro_v2_1",
    )
    envs = [make_env(name, randomness=0.01) for name in rewards]
    batch_zero = make_env("obs2_roll_repro_v2_1", randomness=0.01)
    batch_late = make_env("obs2_roll_repro_v2_1", randomness=0.01)
    try:
        reset_tds = []
        for test_env in (*envs, batch_zero, batch_late):
            np.random.seed(12345)
            reset_tds.append(test_env.reset())

        reference_obs = reset_tds[0][("agents", "observation")]
        assert tuple(reference_obs.shape) == (1, 8, 2)
        for test_env, tensor_dict in zip((*envs, batch_zero, batch_late), reset_tds):
            torch.testing.assert_close(
                reference_obs,
                tensor_dict[("agents", "observation")],
                rtol=0.0,
                atol=0.0,
            )
            assert test_env.rolling_observation_size == 0
            assert test_env.tail_roll_observation_size == 0
            assert test_env.fast_forward_observation_size == 0
            assert test_env.scratch_wr_v2_observation_size == 0
            assert tuple(test_env.formula_action_names) == ("k1", "k2")
            assert tuple(test_env.action_spec[test_env.action_key].shape) == (1, 8, 2)
            assert "Unbounded" in type(test_env.action_spec[test_env.action_key]).__name__

        batch_zero.set_curriculum_episode(0)
        batch_late.set_curriculum_episode(600)
        generator = torch.Generator().manual_seed(24680)
        for step in range(10):
            action = (
                torch.randn((1, 8, 2), generator=generator, dtype=torch.float32)
                * np.float32(0.05)
            )
            for test_env, tensor_dict in zip(
                (*envs, batch_zero, batch_late), reset_tds
            ):
                tensor_dict.set(test_env.action_key, action.clone())
            reset_tds = [
                test_env.step(tensor_dict)["next"]
                for test_env, tensor_dict in zip(
                    (*envs, batch_zero, batch_late), reset_tds
                )
            ]

            reference_env = envs[0]
            reference_obs = reset_tds[0][("agents", "observation")]
            for test_env, tensor_dict in zip(
                (*envs[1:], batch_zero, batch_late), reset_tds[1:]
            ):
                np.testing.assert_array_equal(reference_env.pos, test_env.pos)
                np.testing.assert_array_equal(reference_env.vel, test_env.vel)
                np.testing.assert_array_equal(reference_env.thdot, test_env.thdot)
                torch.testing.assert_close(
                    reference_obs,
                    tensor_dict[("agents", "observation")],
                    rtol=0.0,
                    atol=0.0,
                    msg=f"observation step {step}",
                )

            reward_zero = batch_zero._reward_func().detach().cpu().numpy()
            reward_late = batch_late._reward_func().detach().cpu().numpy()
            np.testing.assert_array_equal(reward_zero, reward_late)
    finally:
        for test_env in (*envs, batch_zero, batch_late):
            test_env.close()


def assert_parser_contract() -> None:
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
            "--reward-func", "obs2_roll_repro_v2_1",
            "--per-joint-k1-k2",
            "--no-rolling-observation",
            "--no-tail-roll-observation",
            "--no-fast-forward-observation",
        ]
        args = train_metamaterial.parse_args()
    finally:
        sys.argv = original
    assert args.reward_func == "obs2_roll_repro_v2_1"
    assert args.rolling_observation is False
    assert args.tail_roll_observation is False
    assert args.fast_forward_observation is False
    assert args.scratch_wr_v2 is False


def assert_tail_ability_and_launch_gate() -> None:
    test_env = make_env("obs2_roll_repro_v2_1")
    try:
        test_env.reset()
        initial_reward = test_env._reward_func().detach().cpu().numpy()
        np.testing.assert_array_equal(initial_reward, np.zeros((1, 1), dtype=np.float32))
        info_keys = set(test_env._get_info().keys())
        assert {
            "fast_forward_reverse_rotation_penalty",
            "fast_forward_backward_penalty",
            "effort_penalty",
        }.issubset(info_keys)

        # One strong dimension (forward) cannot bypass the weakest-link gate.
        partial = tail_first_shape(curvature_degrees=8.0)
        rewards = [float(inject_shapes(test_env, partial)[0, 0]) for _ in range(120)]
        partial_fast = test_env._compute_fast_forward_roll_v2_metrics()
        score = float(partial_fast["fast_forward_launch_progress"][0, 0])
        assert 0.25 < score < 0.75, score
        assert float(partial_fast["fast_forward_phase"][0, 0]) == 0.0
        assert max(abs(value) for value in rewards[-5:]) < 1e-6, rewards[-5:]

        # Translating forever at a sub-threshold posture must not become a
        # permanent phase-0 optimum: its discovery pressure expires after 100
        # steps without a new synchrony high-water mark.
        test_env.reset()
        moving_partial_rewards = []
        for index in range(120):
            moving_partial_rewards.append(
                float(
                    inject_shapes(
                        test_env,
                        tail_first_shape(
                            curvature_degrees=8.0,
                            x_offset=0.20 * (index + 1),
                        ),
                    )[0, 0]
                )
            )
        assert max(abs(value) for value in moving_partial_rewards[-5:]) < 1e-6
        assert int(test_env._fast_forward_phase[0, 0]) == 0

        # Even grounded forward+rotation cannot be harvested forever while
        # deliberately keeping curl just below the raw launch threshold.
        test_env.reset()
        rotating_partial_rewards = []
        for index in range(160):
            rotating_partial_rewards.append(
                float(
                    inject_shapes(
                        test_env,
                        tail_first_shape(
                            curvature_degrees=7.0,
                            rigid_rotation_degrees=-20.0 * (index + 1),
                            x_offset=0.20 * (index + 1),
                        ),
                    )[0, 0]
                )
            )
        assert max(abs(value) for value in rotating_partial_rewards[-18:]) < 1e-6
        assert int(test_env._fast_forward_phase[0, 0]) == 0

        test_env.reset()
        ready = tail_first_shape(curvature_degrees=10.0)
        for _ in range(4):
            inject_shapes(test_env, ready)
        assert int(test_env._fast_forward_launch_ready_steps[0, 0]) == 4
        inject_shapes(test_env, partial)
        assert int(test_env._fast_forward_launch_ready_steps[0, 0]) == 0
        for _ in range(7):
            inject_shapes(test_env, ready)
        assert int(test_env._fast_forward_phase[0, 0]) == 0
        assert int(test_env._fast_forward_launch_ready_steps[0, 0]) == 7
        launch_reward = float(inject_shapes(test_env, ready)[0, 0])
        assert int(test_env._fast_forward_phase[0, 0]) == 1
        assert launch_reward > 0.0
    finally:
        test_env.close()

    # Tail metrics can look launch-ready while the body has never supplied a
    # real ground-support anchor; that airborne pose must not change phase.
    airborne = make_env("obs2_roll_repro_v2_1")
    try:
        airborne.reset()
        body_length = np.float32(9.0)
        no_contact_offset = np.float32(airborne.particle_radius) + np.float32(0.03) * body_length
        airborne_ready = tail_first_shape(
            curvature_degrees=10.0,
            floor_offset=float(no_contact_offset),
        )
        airborne_rewards = [
            float(inject_shapes(airborne, airborne_ready)[0, 0])
            for _ in range(12)
        ]
        tail = airborne._compute_tail_roll_metrics()
        fast = airborne._compute_fast_forward_roll_v2_metrics()
        assert float(tail["head_contact_score"][0, 0]) >= 0.50
        assert float(fast["fast_forward_ground_contact_strength"][0, 0]) < 0.50
        assert float(fast["fast_forward_launch_progress"][0, 0]) == 0.0
        assert max(abs(value) for value in airborne_rewards) < 1e-6
        assert int(airborne._fast_forward_phase[0, 0]) == 0
        assert not bool(airborne._fast_forward_event_anchor_support_valid[0, 0])
    finally:
        airborne.close()

    # A real pre-launch support may be followed by a brief take-off.  The
    # cached support provenance allows phase transition, while a pulse still
    # requires a real current landing/contact frame.
    takeoff = make_env("obs2_roll_repro_v2_1")
    try:
        takeoff.reset()
        inject_shapes(takeoff, straight_shape())
        assert bool(takeoff._fast_forward_event_anchor_support_valid[0, 0])
        body_length = np.float32(9.0)
        takeoff_offset = np.float32(takeoff.particle_radius) + np.float32(0.03) * body_length
        takeoff_ready = tail_first_shape(
            curvature_degrees=10.0,
            floor_offset=float(takeoff_offset),
        )
        for _ in range(8):
            inject_shapes(takeoff, takeoff_ready)
        assert int(takeoff._fast_forward_phase[0, 0]) == 1
        assert float(takeoff._fast_forward_event_count[0, 0]) == 0.0
    finally:
        takeoff.close()


def assert_adversaries_valid_pulse_cache_and_reset() -> None:
    # Static closure, pure translation, and airborne rotation never create a pulse.
    cases = (
        ("static_ring", [ring_shape() for _ in range(24)]),
        (
            "straight_translation",
            [straight_shape(0.20 * (index + 1)) for index in range(24)],
        ),
        (
            "airborne_rotation",
            [
                ring_shape(
                    angle_degrees=-20.0 * index,
                    x_offset=0.20 * index,
                    floor_offset=5.0,
                )
                for index in range(24)
            ],
        ),
    )
    for name, sequence in cases:
        test_env = make_env("obs2_roll_repro_v2_1")
        try:
            test_env.reset()
            for shape in sequence:
                inject_shapes(test_env, shape)
            fast = test_env._compute_fast_forward_roll_v2_metrics()
            assert float(fast["fast_forward_event_count"][0, 0]) == 0.0, name
        finally:
            test_env.close()

    rolling = make_env("obs2_roll_repro_v2_1")
    try:
        rolling.reset()
        ready = tail_first_shape(curvature_degrees=10.0)
        for _ in range(8):
            inject_shapes(rolling, ready)
        assert int(rolling._fast_forward_phase[0, 0]) == 1

        for index in range(1, 13):
            moving = tail_first_shape(
                curvature_degrees=10.0,
                rigid_rotation_degrees=-20.0 * index,
                x_offset=0.25 * index,
            )
            inject_shapes(rolling, moving)
        fast = rolling._compute_fast_forward_roll_v2_metrics()
        assert float(fast["fast_forward_event_count"][0, 0]) >= 1.0, fast

        state_names = (
            "_fast_forward_phase",
            "_fast_forward_phase_steps",
            "_fast_forward_launch_high_water",
            "_fast_forward_launch_ready_steps",
            "_fast_forward_roll_high_water",
            "_fast_forward_event_count",
            "_fast_forward_event_anchor_rotation",
            "_fast_forward_event_anchor_com_x",
            "_fast_forward_event_anchor_support_index",
            "_fast_forward_event_anchor_support_valid",
            "_fast_forward_progress_age",
        )
        before = {
            name: np.asarray(getattr(rolling, name)).copy() for name in state_names
        }
        rolling._reward_func()
        rolling._get_info()
        rolling._reward_func()
        for name, expected in before.items():
            np.testing.assert_array_equal(getattr(rolling, name), expected, err_msg=name)

        rolling.reset()
        assert int(rolling._fast_forward_phase[0, 0]) == 0
        assert int(rolling._fast_forward_launch_ready_steps[0, 0]) == 0
        assert float(rolling._fast_forward_event_count[0, 0]) == 0.0
        assert float(rolling._fast_forward_roll_high_water[0, 0]) == 0.0
        assert int(rolling._fast_forward_progress_age[0, 0]) == 0
        assert not bool(rolling._fast_forward_event_anchor_support_valid[0, 0])
    finally:
        rolling.close()


def assert_multi_env_state_isolation() -> None:
    test_env = make_env("obs2_roll_repro_v2_1", num_envs=2)
    try:
        test_env.reset()
        ready = tail_first_shape(curvature_degrees=10.0)[0]
        straight = straight_shape()[0]
        shapes = np.stack((ready, straight), axis=0)
        for _ in range(8):
            inject_shapes(test_env, shapes)
        np.testing.assert_array_equal(
            test_env._fast_forward_phase[:, 0], np.asarray([1, 0], dtype=np.int32)
        )
        np.testing.assert_array_equal(
            test_env._fast_forward_event_count[:, 0], np.zeros(2, dtype=np.float32)
        )
    finally:
        test_env.close()


def assert_exact_v2_1_reward_delta_contract() -> None:
    source_v2 = inspect.getsource(metamaterial.reward_func_obs2_roll_repro_v2)
    source_v2_1 = inspect.getsource(metamaterial.reward_func_obs2_roll_repro_v2_1)
    normalized_v2_1 = source_v2_1.replace(
        "reward_func_obs2_roll_repro_v2_1", "reward_func_obs2_roll_repro_v2"
    ).replace(
        "obs2_roll_repro_v2_1 produced", "obs2_roll_repro_v2 produced"
    ).replace(
        "np.float32(0.16) * motion_quality",
        "np.float32(0.08) * motion_quality",
    )
    ast_v2 = ast.parse(textwrap.dedent(source_v2))
    ast_v2_1 = ast.parse(textwrap.dedent(normalized_v2_1))
    assert ast.dump(ast_v2, include_attributes=False) == ast.dump(
        ast_v2_1, include_attributes=False
    )
    assert source_v2_1.count("np.float32(0.16) * motion_quality") == 1

    def array(first: float, second: float) -> np.ndarray:
        return np.asarray([[first], [second]], dtype=np.float32)

    def evaluate(phase: float, launch_event: float):
        fast = {
            "fast_forward_launch_progress": array(0.55, 0.85),
            "fast_forward_ground_contact_strength": array(0.75, 0.90),
            "fast_forward_episode_direction_fraction": array(0.72, 0.91),
            "fast_forward_progress_age": array(20.0, 70.0),
            "fast_forward_phase": array(phase, phase),
            "fast_forward_launch_event": array(launch_event, launch_event),
            "fast_forward_progress_delta": array(0.02, 0.04),
            "fast_forward_event_bonus": array(0.0, 3.5),
            "effort_penalty": array(0.10, 0.25),
            "fast_forward_event_count": array(0.0, 1.0),
            "fast_forward_reverse_rotation_penalty": array(0.0, 0.07),
            "fast_forward_backward_penalty": array(0.02, 0.0),
            "action_smoothness_penalty": array(0.03, 0.04),
            "fast_forward_stall_penalty": array(0.0, 1.0),
        }
        roll = {
            "speed_score": array(0.45, 0.82),
            "rotation_score": array(0.62, 0.51),
            "slip_penalty": array(0.20, 0.05),
        }
        stub = SimpleNamespace(
            num_envs=2,
            rolling_reward_scale=np.float32(3.0),
            action_smoothness_weight=np.float32(0.0),
            device=torch.device("cpu"),
        )
        stub._smooth_step = lambda value: np.clip(
            np.float32(0.5) + np.float32(0.5) * value,
            np.float32(0.0),
            np.float32(1.0),
        ).astype(np.float32)
        stub._compute_fast_forward_roll_v2_metrics = lambda: fast
        stub._compute_rolling_metrics = lambda: roll
        reward_v2 = (
            metamaterial.reward_func_obs2_roll_repro_v2(stub)
            .detach()
            .cpu()
            .numpy()
        )
        reward_v2_1 = (
            metamaterial.reward_func_obs2_roll_repro_v2_1(stub)
            .detach()
            .cpu()
            .numpy()
        )
        direction_gate = stub._smooth_step(
            (fast["fast_forward_episode_direction_fraction"] - np.float32(0.50))
            / np.float32(0.08)
        )
        motion_quality = (
            (fast["fast_forward_ground_contact_strength"] >= np.float32(0.50)).astype(np.float32)
            * np.sqrt(np.maximum(roll["speed_score"] * roll["rotation_score"], np.float32(0.0)))
            * (
                np.float32(0.45) * roll["rotation_score"]
                + np.float32(0.25) * roll["speed_score"]
                + np.float32(0.15) * direction_gate
                + np.float32(0.15) * (np.float32(1.0) - roll["slip_penalty"])
            )
        ).astype(np.float32)
        return reward_v2, reward_v2_1, motion_quality

    prep_v2, prep_v2_1, _ = evaluate(phase=0.0, launch_event=0.0)
    launch_v2, launch_v2_1, _ = evaluate(phase=1.0, launch_event=1.0)
    roll_v2, roll_v2_1, quality = evaluate(phase=1.0, launch_event=0.0)
    np.testing.assert_array_equal(prep_v2, prep_v2_1)
    np.testing.assert_array_equal(launch_v2, launch_v2_1)
    np.testing.assert_allclose(
        roll_v2_1 - roll_v2,
        np.float32(3.0 * (0.16 - 0.08)) * quality,
        rtol=1e-5,
        atol=1e-6,
    )


def main() -> None:
    assert_exact_v2_1_reward_delta_contract()
    assert_reward_only_contract_and_batch_independence()
    assert_parser_contract()
    assert_tail_ability_and_launch_gate()
    assert_adversaries_valid_pulse_cache_and_reset()
    assert_multi_env_state_isolation()
    print("PASS: obs2_roll_repro_v2_1 exact-delta, reward-only contract and adversarial tests")


if __name__ == "__main__":
    main()
