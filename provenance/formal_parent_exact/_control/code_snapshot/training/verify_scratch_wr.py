"""Deterministic regression and one-batch smoke checks for Scratch-WR.

It is intentionally a plain-Python verifier so the project does not need a new
pytest dependency.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import tempfile
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


def _discover_project_root() -> Path:
    candidates = [Path(__file__).resolve(), *Path(__file__).resolve().parents]
    candidates.extend(
        [
            Path(
                r"C:\Users\PUBLIC_USER\CloudStorage\Desktop\finalproject\job\RLMetamaterialLocomotion-main"
                r"\RLMetamaterialLocomotion-main"
            )
        ]
    )
    for candidate in candidates:
        if (candidate / "training" / "train_metamaterial.py").is_file():
            return candidate
    raise RuntimeError("Cannot locate the RLMetamaterialLocomotion project root.")


PROJECT_ROOT = _discover_project_root()
sys.path.insert(0, str(PROJECT_ROOT / "training"))
sys.path.insert(0, str(PROJECT_ROOT / "metamaterial_envs"))

from metamaterial_envs.env import metamaterial  # noqa: E402
import rlmm_common  # noqa: E402
import train_metamaterial as trainer  # noqa: E402


WAVE_ACTION = np.asarray([1.65, 0.28, 0.11, 0.72, 9.0, 1.2], dtype=np.float32)
SCRATCH_LOG_KEYS = (
    "scratch_wr_alpha",
    "scratch_wr_wave_torque_rms",
    "scratch_wr_residual_torque_rms",
    "scratch_wr_applied_residual_torque_rms",
    "scratch_wr_total_torque_rms",
    "scratch_wr_torque_clip_fraction",
    "scratch_wr_residual_saturation_fraction",
)
SCRATCH_CSV_KEYS = (
    "scratch_wr_stage_id",
    "scratch_wr_learning_rate",
    *SCRATCH_LOG_KEYS,
)


def make_env(control_mode: str, *, alpha: float | None = None):
    kwargs = dict(
        num_envs=1,
        material_shape="crawler",
        num_particles=10,
        max_steps=20,
        terrain_type="flat",
        observation_func="dth_tot",
        reward_func="fast_forward_roll_v2",
        control_mode=control_mode,
        fast_forward_observation=True,
        rolling_direction="right",
        tail_side="left",
        init_pos_randomness=0.0,
        init_angle_range_degrees=0.0,
        init_height_jitter=0.0,
        action_smoothness_weight=0.0,
    )
    if alpha is not None:
        kwargs["scratch_wr_alpha"] = alpha
    return metamaterial.env(**kwargs)


def _action_tensor(values: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.float32).reshape(1, 1, -1)


def _step(env, td, values: np.ndarray):
    td.set(env.action_key, _action_tensor(values))
    return env.step(td)["next"]


def _assert_state_equal(tail_env, scratch_env, tail_td, scratch_td, label: str) -> None:
    np.testing.assert_allclose(scratch_env.pos, tail_env.pos, rtol=2e-6, atol=2e-6, err_msg=label)
    np.testing.assert_allclose(scratch_env.vel, tail_env.vel, rtol=2e-6, atol=2e-6, err_msg=label)
    np.testing.assert_allclose(scratch_env.thdot, tail_env.thdot, rtol=2e-6, atol=2e-6, err_msg=label)
    torch.testing.assert_close(
        scratch_td[("agents", "observation")],
        tail_td[("agents", "observation")],
        rtol=2e-6,
        atol=2e-6,
        msg=label,
    )
    torch.testing.assert_close(scratch_td["reward"], tail_td["reward"], rtol=2e-6, atol=2e-6, msg=label)


def _metric(td, name: str) -> float:
    return float(td[("log_info", name)].detach().mean().cpu().item())


def _close_envs(*envs) -> None:
    for environment in envs:
        try:
            environment.close()
        except Exception:
            pass


def test_legacy_modes_and_specs() -> None:
    canonical, observation, control = rlmm_common.channel_config("tail_wave")
    assert (canonical, observation, control) == ("tail_wave", "dth_tot", "tail_wave")
    for mode, expected_agents, expected_action_dim in (
        ("direct", 8, 1),
        ("formula", 8, 2),
        ("tail_wave", 1, 6),
    ):
        kwargs = {}
        if mode == "formula":
            kwargs.update(k1_min=-9.0, k1_max=9.0, k2_min=-9.0, k2_max=9.0)
        env = metamaterial.env(
            num_envs=1,
            material_shape="crawler",
            num_particles=10,
            max_steps=1,
            observation_func="dth_tot",
            control_mode=mode,
            **kwargs,
        )
        assert env.control_mode == mode
        assert env.num_agents == expected_agents
        assert tuple(env.action_spec["agents", "action"].shape) == (1, expected_agents, expected_action_dim)
        td = env.reset()
        env.step(env.rand_action(td))
        _close_envs(env)


def test_scratch_specs_and_setter() -> None:
    canonical, observation, control = rlmm_common.channel_config("scratch_wr")
    assert canonical == "tail_wave_residual"
    assert observation == "dth_tot"
    assert control == "tail_wave_residual"
    env = make_env("scratch_wr", alpha=0.0)
    assert env.control_mode == "tail_wave_residual"
    assert env.num_agents == 1
    assert env.num_controlled_joints == 8
    assert tuple(env.action_spec["agents", "action"].shape) == (1, 1, 22)
    assert env.observation_spec["agents", "observation"].shape[-1] == 16
    assert len(env.scratch_wr_action_names) == 22
    assert tuple(env.scratch_wr_action_names[:6]) == tuple(env.tail_wave_action_names)
    assert len(set(env.scratch_wr_action_names)) == 22
    assert np.asarray(env.scratch_wr_action_low).shape == (22,)
    assert np.asarray(env.scratch_wr_action_high).shape == (22,)
    assert np.isfinite(env.scratch_wr_action_low).all()
    assert np.isfinite(env.scratch_wr_action_high).all()
    assert np.all(np.asarray(env.scratch_wr_action_low)[6:] <= 0.0)
    assert np.all(np.asarray(env.scratch_wr_action_high)[6:] >= 0.0)
    env.set_scratch_wr_alpha(0.35)
    assert math.isclose(env.scratch_wr_alpha, 0.35, rel_tol=0.0, abs_tol=1e-7)
    for invalid in (-0.001, 1.001, float("nan"), float("inf")):
        try:
            env.set_scratch_wr_alpha(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"set_scratch_wr_alpha accepted invalid value {invalid!r}")
    _close_envs(env)


def _equivalence_rollout(*, alpha: float, residual: np.ndarray, expected_applied_zero: bool) -> None:
    tail_env = make_env("tail_wave")
    scratch_env = make_env("tail_wave_residual", alpha=alpha)
    tail_td = tail_env.reset()
    scratch_td = scratch_env.reset()
    torch.testing.assert_close(
        scratch_td[("agents", "observation")], tail_td[("agents", "observation")], rtol=0.0, atol=0.0
    )

    # First create non-zero curvature and angular velocity.  Testing only from
    # the straight rest pose would let a broken residual path pass because its
    # K1/K2 state features are initially zero.
    for center in (0.12, 0.28, 0.46):
        wave = WAVE_ACTION.copy()
        wave[1] = center
        tail_td = _step(tail_env, tail_td, wave)
        scratch_td = _step(scratch_env, scratch_td, np.concatenate((wave, np.zeros(16, dtype=np.float32))))
        _assert_state_equal(tail_env, scratch_env, tail_td, scratch_td, f"warmup center={center}")

    tail_td = _step(tail_env, tail_td, WAVE_ACTION)
    scratch_td = _step(scratch_env, scratch_td, np.concatenate((WAVE_ACTION, residual.astype(np.float32))))
    _assert_state_equal(tail_env, scratch_env, tail_td, scratch_td, f"alpha={alpha}, residual equivalence")
    for key in SCRATCH_LOG_KEYS:
        assert key in scratch_td["log_info"].keys(), key
        assert math.isfinite(_metric(scratch_td, key)), key
    applied = _metric(scratch_td, "scratch_wr_applied_residual_torque_rms")
    raw = _metric(scratch_td, "scratch_wr_residual_torque_rms")
    if expected_applied_zero:
        assert applied <= 1e-7, applied
    if np.any(residual != 0):
        assert raw > 1e-7, "The curved-state residual torque must be observably non-zero."
    _close_envs(tail_env, scratch_env)


def test_alpha_zero_and_zero_residual_equivalence() -> None:
    residual = np.tile(np.asarray([8.0, -7.0], dtype=np.float32), 8)
    _equivalence_rollout(alpha=0.0, residual=residual, expected_applied_zero=True)
    _equivalence_rollout(alpha=1.0, residual=np.zeros(16, dtype=np.float32), expected_applied_zero=True)


def test_common_torque_clip() -> None:
    wave = np.asarray([8.0, -8.0, 3.0, -3.0], dtype=np.float32)
    residual = np.asarray([8.0, -8.0, 20.0, -20.0], dtype=np.float32)
    actual = metamaterial._blend_scratch_wr_torque(wave, residual, 0.5, 9.0)
    expected = np.asarray([9.0, -9.0, 9.0, -9.0], dtype=np.float32)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    unchanged = metamaterial._blend_scratch_wr_torque(wave, residual, 0.0, 9.0)
    np.testing.assert_allclose(unchanged, wave, rtol=0.0, atol=0.0)


def test_alpha_zero_freezes_residual_distribution_gradients() -> None:
    torch.manual_seed(8127)
    raw = torch.randn(4, 1, 44, dtype=torch.float32, requires_grad=True)
    extractor = trainer.ScratchWRNormalParamExtractor(
        scale_lb=1e-4,
        wave_action_size=6,
        alpha=0.0,
    )
    loc, scale = extractor(raw)
    assert tuple(loc.shape) == (4, 1, 22)
    assert tuple(scale.shape) == (4, 1, 22)
    (loc.sum() + scale.sum()).backward()
    assert raw.grad is not None
    wave_grad = torch.cat((raw.grad[..., :6], raw.grad[..., 22:28]), dim=-1)
    residual_grad = torch.cat((raw.grad[..., 6:22], raw.grad[..., 28:44]), dim=-1)
    assert float(wave_grad.abs().sum().item()) > 0.0
    assert float(residual_grad.abs().max().item()) == 0.0
    assert torch.count_nonzero(loc[..., 6:]) == 0
    torch.testing.assert_close(
        scale[..., 6:],
        torch.full_like(scale[..., 6:], 1e-4),
        rtol=0.0,
        atol=0.0,
    )

    raw_live = torch.randn(4, 1, 44, dtype=torch.float32, requires_grad=True)
    extractor.set_alpha(0.1)
    loc_live, scale_live = extractor(raw_live)
    (loc_live.sum() + scale_live.sum()).backward()
    residual_live_grad = torch.cat(
        (raw_live.grad[..., 6:22], raw_live.grad[..., 28:44]), dim=-1
    )
    assert float(residual_live_grad.abs().sum().item()) > 0.0


def _parse(*arguments: str):
    with patch.object(sys, "argv", ["train_metamaterial.py", *arguments]):
        return trainer.parse_args()


def _expect_parse_failure(*arguments: str) -> None:
    try:
        with redirect_stderr(io.StringIO()):
            _parse(*arguments)
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError(f"Scratch-WR accepted forbidden CLI arguments: {arguments!r}")


def test_strict_scratch_cli() -> None:
    args = _parse("--channel", "scratch_wr", "--algorithm", "ppo", "--scratch-wr-alpha", "0")
    assert args.pretrained_model_path is None
    assert args.bc_teacher_checkpoint is None
    assert args.wave_bc_teacher_json is None
    assert args.bc_steps == 0 and args.bc_epochs == 0
    assert args.policy_anchor_coeff == 0.0
    _expect_parse_failure(
        "--channel", "scratch_wr", "--pretrained-model-path", "old.pt"
    )
    _expect_parse_failure(
        "--channel", "scratch_wr", "--bc-teacher-checkpoint", "teacher.pt", "--bc-steps", "1", "--bc-epochs", "1"
    )
    _expect_parse_failure(
        "--channel", "scratch_wr", "--wave-bc-teacher-json", "teacher.json", "--bc-steps", "1", "--bc-epochs", "1"
    )
    _expect_parse_failure("--channel", "scratch_wr", "--policy-anchor-coeff", "0.01")
    _expect_parse_failure("--channel", "scratch_wr", "--algorithm", "ddpg")


def run_one_batch_smoke() -> dict:
    with tempfile.TemporaryDirectory(prefix="scratch_wr_smoke_") as temp_dir:
        results = Path(temp_dir) / "results"
        run_name = "scratch_wr_one_batch_smoke"
        argv = [
            "train_metamaterial.py",
            "--robot", "crawler",
            "--terrain", "flat",
            "--num-particles", "10",
            "--channel", "scratch_wr",
            "--reward-func", "fast_forward_roll_v2",
            "--algorithm", "ppo",
            "--scratch-wr-alpha", "0.1",
            "--tail-roll-init-assist-degrees", "0",
            "--tail-roll-init-assist-episodes", "0",
            "--episodes", "1",
            "--episode-steps", "10",
            "--frames-per-batch", "20",
            "--minibatch-size", "20",
            "--memory-size", "40",
            "--optim-steps", "1",
            "--save-every", "1",
            "--force-cpu",
            "--no-auto-analysis",
            "--results-dir", str(results),
            "--run-name", run_name,
            "--seed", "9101",
        ]
        with patch.object(sys, "argv", argv):
            trainer.main()
        run_dir = results / run_name
        checkpoint_zero = run_dir / "checkpoint_0.pt"
        final_checkpoint = run_dir / "checkpoint_1.pt"
        assert checkpoint_zero.is_file(), "Scratch audit checkpoint_0.pt is missing."
        assert final_checkpoint.is_file(), "One-batch final checkpoint is missing."
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "training_summary.json").read_text(encoding="utf-8"))
        with (run_dir / "training_log.csv").open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1
        assert all(key in rows[0] for key in SCRATCH_CSV_KEYS)
        train_args = metadata["training_args"]
        assert train_args["pretrained_model_path"] is None
        assert train_args["bc_teacher_checkpoint"] is None
        assert train_args["wave_bc_teacher_json"] is None
        assert train_args["bc_steps"] == 0 and train_args["bc_epochs"] == 0
        assert train_args["policy_anchor_coeff"] == 0.0
        assert metadata["behavior_cloning"] is None
        assert summary["episodes"] == 1
        assert math.isclose(float(metadata["scratch_wr_current_alpha"]), 0.1, abs_tol=1e-7)

        # Verify the ordinary analysis/demo loader can consume the specialised
        # training head's state dict and restores non-zero residual authority
        # from checkpoint metadata before a deterministic physics step.
        import analyze_training_results as analysis

        replay_env, _, _, _ = analysis.build_demo_env(
            metadata,
            "training",
            analysis.TerrainArgs(),
            max_steps=2,
            render_mode=None,
        )
        assert math.isclose(float(replay_env.scratch_wr_alpha), 0.1, abs_tol=1e-7)
        replay_policy = analysis.load_policy_for_env(final_checkpoint, replay_env, metadata)
        replay_td = analysis.choose_action(replay_policy, replay_env.reset(), "deterministic")
        assert torch.isfinite(replay_td[replay_env.action_key]).all()
        replay_next = replay_env.step(replay_td)["next"]
        assert torch.isfinite(replay_next[("agents", "observation")]).all()
        assert torch.isfinite(replay_next["reward"]).all()
        _close_envs(replay_env)
        return {
            "checkpoint_0": str(checkpoint_zero),
            "final_checkpoint": str(final_checkpoint),
            "log_fields": list(rows[0]),
            "deterministic_checkpoint_replay": True,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-smoke", action="store_true", help="Also train one tiny PPO collector batch.")
    args = parser.parse_args()
    checks = [
        test_legacy_modes_and_specs,
        test_scratch_specs_and_setter,
        test_alpha_zero_and_zero_residual_equivalence,
        test_common_torque_clip,
        test_alpha_zero_freezes_residual_distribution_gradients,
        test_strict_scratch_cli,
    ]
    try:
        for check in checks:
            check()
            print(f"PASS {check.__name__}")
        if args.full_smoke:
            print(json.dumps(run_one_batch_smoke(), ensure_ascii=False, indent=2))
            print("PASS one-batch PPO smoke")
        print("All Scratch-WR checks passed.")
    finally:
        # pygame starts a background subsystem at import time in the legacy
        # simulator. Explicit shutdown keeps this verifier from lingering.
        metamaterial.pygame.quit()


if __name__ == "__main__":
    main()
