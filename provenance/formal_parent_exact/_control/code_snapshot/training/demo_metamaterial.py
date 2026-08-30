"""Render a trained crawler/ring policy on flat, stairs, or tunnel terrain.

By default the demo reads robot, terrain, channel, control mode, and PPO/DDPG
algorithm information from checkpoint metadata. Override these only when the
checkpoint dimensions are compatible.

Examples:
  python ./training/demo_metamaterial.py --checkpoint latest --follow-camera
  python ./training/demo_metamaterial.py --checkpoint results/run_name/checkpoint_500.pt --policy-mode deterministic
  python ./training/demo_metamaterial.py --checkpoint latest --terrain tunnel --channel checkpoint
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from tensordict.nn import TensorDictModule
from torchrl.modules import Delta, IndependentNormal, MultiAgentMLP, ProbabilisticActor, TanhDelta, TanhNormal

try:
    from torchrl.envs.utils import ExplorationType, set_exploration_type
except ImportError:  # TorchRL version compatibility
    from torchrl.envs.utils import ExplorationMode as ExplorationType, set_exploration_type

from rlmm_common import (
    BiasedNormalParamExtractor,
    CONTROL_MODE_CHOICES,
    FirstOrderGaussian,
    TERRAIN_CONTACT_MODE_CHOICES,
    add_env_package_to_path,
    channel_config,
    channel_label as make_channel_label,
    find_latest_checkpoint,
    find_project_root,
    load_checkpoint,
    infer_channel_from_metadata,
    normalise_control_mode,
    terrain_config,
    terrain_contact_mode_from_metadata,
    terrain_label,
    to_plain,
)

PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
add_env_package_to_path(PROJECT_ROOT)
from metamaterial_envs.env import metamaterial  # noqa: E402


PPO_ALGORITHMS = {"ppo", "ppo_noclip"}
DDPG_ALGORITHMS = {"ddpg", "ddpg_clip"}


class ScratchWRDeterministicParamExtractor(torch.nn.Module):
    """Rebuild the alpha-aware DDPG Scratch-WR checkpoint architecture."""

    def __init__(self, wave_action_size: int = 6, alpha: float = 0.0):
        super().__init__()
        self.wave_action_size = int(wave_action_size)
        self.register_buffer(
            "scratch_wr_alpha",
            torch.tensor(float(alpha), dtype=torch.float32),
            persistent=False,
        )
        self.set_alpha(alpha)

    def set_alpha(self, alpha: float) -> None:
        alpha = float(alpha)
        if not np.isfinite(alpha) or not (0.0 <= alpha <= 1.0):
            raise ValueError("Scratch-WR alpha must be a finite value in [0, 1].")
        self.scratch_wr_alpha.fill_(alpha)

    def forward(self, param: torch.Tensor) -> torch.Tensor:
        if param.shape[-1] <= self.wave_action_size:
            raise ValueError("Scratch-WR policy output does not contain residual dimensions.")
        residual_train_mask = (self.scratch_wr_alpha > 0).to(
            dtype=param.dtype,
            device=param.device,
        )
        residual_param = param[..., self.wave_action_size:] * residual_train_mask
        return torch.cat((param[..., :self.wave_action_size], residual_param), dim=-1)


def algorithm_family(metadata: dict) -> str:
    family = metadata.get("algorithm_family", None)
    if family in {"ppo", "ddpg"}:
        return str(family)
    algorithm = metadata.get("algorithm", "ppo")
    if algorithm in PPO_ALGORITHMS:
        return "ppo"
    if algorithm in DDPG_ALGORITHMS:
        return "ddpg"
    raise ValueError(f"Unsupported algorithm in checkpoint: {algorithm}")


def finite_bound_or_none(value):
    if value is None:
        return None
    parsed = float(value)
    return parsed if np.isfinite(parsed) else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a trained metamaterial policy.")
    parser.add_argument("--checkpoint", default="latest", help="Path to checkpoint_*.pt, or 'latest'.")

    parser.add_argument(
        "--robot",
        choices=["checkpoint", "crawler", "ring"],
        default="checkpoint",
        help="Use checkpoint robot by default. Overriding robot requires a compatible checkpoint.",
    )
    parser.add_argument(
        "--terrain",
        choices=["checkpoint", "flat", "stairs", "tunnel"],
        default="checkpoint",
        help="Use checkpoint terrain by default. You may override terrain for testing.",
    )
    parser.add_argument(
        "--terrain-contact-mode",
        choices=TERRAIN_CONTACT_MODE_CHOICES,
        default=None,
        help=(
            "Override terrain contact implementation. By default read it from checkpoint metadata; "
            "old checkpoints without the field use legacy_flat."
        ),
    )
    parser.add_argument("--num-particles", type=int, default=None, help="Override particle count; normally use checkpoint metadata.")
    parser.add_argument(
        "--channel",
        default="checkpoint",
        help=(
            "Use the checkpoint channel by default. Paper channels: 'dth' uses "
            "[dtheta_prev,dtheta_next], and 'thdot' adds theta_dot; both output torque directly. "
            "Project channels: 'obs', 'action', 'paper', 'k2_positive', and 'k2_negative'. "
            "Aliases include paper_dth, paper_thdot, theta, and formula."
        ),
    )
    parser.add_argument("--observation-func", default=None, help="Advanced override for raw observation function; normally use checkpoint metadata.")
    parser.add_argument(
        "--control-mode",
        choices=["checkpoint", "auto", *CONTROL_MODE_CHOICES],
        default="checkpoint",
        help="Use the checkpoint control mode by default. Override only with a compatible checkpoint.",
    )
    parser.add_argument(
        "--algorithm",
        choices=["checkpoint", "ppo", "ddpg", "ddpg_clip", "ppo_noclip"],
        default="checkpoint",
        help="Use the algorithm saved in the checkpoint by default. Override only for old checkpoints with missing metadata.",
    )
    parser.add_argument("--feedback-gain", type=float, default=None, help="Compatibility option. F is fixed to 1.0; any explicit different value is rejected.")
    parser.add_argument(
        "--coefficient-limit",
        "--max-control-gain",
        dest="max_control_gain",
        type=float,
        default=None,
        help="Override finite K1/K2 bounds for formula/action control; normally use checkpoint metadata.",
    )
    parser.add_argument("--fixed-k1", type=float, default=None, help="Override fixed k1 for signed-k2 channels; normally use checkpoint metadata.")
    parser.add_argument("--fix-k1", action=argparse.BooleanOptionalAction, default=None, help="Override whether formula/action control fixes K1.")
    parser.add_argument("--fixed-k2", type=float, default=None, help="Override fixed K2 for formula/action control; normally use checkpoint metadata.")
    parser.add_argument("--fix-k2", action=argparse.BooleanOptionalAction, default=None, help="Override whether formula/action control fixes K2.")
    parser.add_argument("--k1-min", type=float, default=None, help="Override learned K1 lower bound; normally use checkpoint metadata.")
    parser.add_argument("--k1-max", type=float, default=None, help="Override learned K1 upper bound; normally use checkpoint metadata.")
    parser.add_argument("--k2-min", type=float, default=None, help="Override learned K2 lower bound; normally use checkpoint metadata.")
    parser.add_argument("--k2-max", type=float, default=None, help="Override learned K2 upper bound; normally use checkpoint metadata.")
    parser.add_argument("--k-action-scale", type=float, default=None, help="Override formula/action K scale; normally use checkpoint metadata.")
    parser.add_argument(
        "--min-k2-magnitude",
        type=float,
        default=None,
        help="Override strict lower bound on |k2| for signed-k2 channels; normally use checkpoint metadata.",
    )
    parser.add_argument("--passive-kappa", type=float, default=None, help="Override passive stiffness kappa; normally use checkpoint metadata.")
    parser.add_argument(
        "--rolling-observation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the checkpoint rolling-observation augmentation only when using a compatible policy.",
    )

    parser.add_argument("--start-stairs", type=float, default=5)
    parser.add_argument("--step-width", type=float, default=5)
    parser.add_argument("--step-height", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=10, help="Number of stair steps.")
    parser.add_argument("--tunnel-start", type=float, default=10, help="World x-position where the tunnel ramp begins.")
    parser.add_argument("--tunnel-slope", type=float, default=5, help="Horizontal length of each tunnel entry/exit ramp.")
    parser.add_argument("--tunnel-slope-height", type=float, default=1, help="Vertical rise/fall of each tunnel ramp.")
    parser.add_argument("--tunnel-length", type=float, default=10, help="Horizontal length of the constant-height tunnel section.")
    parser.add_argument("--tunnel-height", type=float, default=5, help="Vertical clearance inside the tunnel; use about 2 for a narrow crawler tunnel and 4-5 for the default ring.")

    parser.add_argument("--policy-mode", choices=["sample", "deterministic"], default="sample")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--window-width", type=int, default=1000)
    parser.add_argument("--window-height", type=int, default=500)
    parser.add_argument("--follow-camera", action="store_true")
    parser.add_argument("--print-every", type=int, default=50)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--no-pause", action="store_true", help="Do not wait for Enter before closing the demo window. Useful for batch demos.")
    args = parser.parse_args()
    if args.feedback_gain is not None and not np.isclose(args.feedback_gain, 1.0):
        parser.error("--feedback-gain is fixed to 1.0 for this project.")
    args.feedback_gain = 1.0
    return args


def get_enum_value(*names):
    enums = [ExplorationType]
    try:
        from torchrl.envs.utils import ExplorationMode

        enums.append(ExplorationMode)
    except Exception:
        pass

    for enum in enums:
        for name in names:
            if hasattr(enum, name):
                return getattr(enum, name)
    return None


def build_policy(env, metadata: dict):
    algorithm = algorithm_family(metadata)
    policy_net_config = to_plain(metadata.get("policy_net_config", {"depth": 2, "num_cells": 256}))
    share_parameters_policy = bool(metadata.get("share_parameters_policy", metadata.get("share_policy", True)))
    gaussian_activation = bool(metadata.get("gaussian_activation", False))
    normal_scale_lb = float(metadata.get("normal_scale_lb", 1e-4))

    action_key = env.action_key
    n_action_outputs = env.full_action_spec[action_key].shape[-1]

    policy_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=n_action_outputs * (2 if algorithm == "ppo" else 1),
        n_agents=env.num_agents,
        centralised=False,
        share_params=share_parameters_policy,
        device="cpu",
        activation_class=FirstOrderGaussian if gaussian_activation else torch.nn.Tanh,
        **policy_net_config,
    )

    if algorithm == "ppo":
        policy_net = torch.nn.Sequential(policy_net, BiasedNormalParamExtractor(scale_lb=normal_scale_lb))
        temp_keys = [("agents", "loc"), ("agents", "scale")]
    else:
        if metadata.get("control_mode") == "tail_wave_residual":
            policy_net = torch.nn.Sequential(
                policy_net,
                ScratchWRDeterministicParamExtractor(
                    wave_action_size=6,
                    alpha=float(
                        metadata.get(
                            "scratch_wr_current_alpha",
                            metadata.get("scratch_wr_initial_alpha", 0.0),
                        )
                    ),
                ),
            )
        temp_keys = [("agents", "param")]

    policy_module = TensorDictModule(policy_net, in_keys=[("agents", "observation")], out_keys=temp_keys)
    action_leaf_spec = env.full_action_spec_unbatched[action_key]
    action_low = action_leaf_spec.space.low
    action_high = action_leaf_spec.space.high
    bounded_action_spec = type(action_leaf_spec).__name__.startswith("Bounded")
    finite_action_bounds = bounded_action_spec and bool(torch.isfinite(action_low).all().item() and torch.isfinite(action_high).all().item())
    if finite_action_bounds:
        distribution_class = TanhNormal if algorithm == "ppo" else TanhDelta
        distribution_kwargs = {"low": action_low, "high": action_high}
    else:
        distribution_class = IndependentNormal if algorithm == "ppo" else Delta
        distribution_kwargs = {}

    return ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec_unbatched,
        in_keys=temp_keys,
        out_keys=[action_key],
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=False,
    )


def choose_action(policy, td, mode: str):
    if mode == "deterministic":
        deterministic = get_enum_value("DETERMINISTIC", "MEAN")
        if deterministic is None:
            with torch.no_grad():
                return policy(td)
        with torch.no_grad(), set_exploration_type(deterministic):
            return policy(td)

    random_mode = get_enum_value("RANDOM", "EXPLORATION")
    if random_mode is None:
        with torch.no_grad():
            return policy(td)
    with torch.no_grad(), set_exploration_type(random_mode):
        return policy(td)


def enable_follow_camera(env):
    def camera_matrix(this):
        center_x = float(np.real(this.pos[0]).mean()) if hasattr(this, "pos") else 0.0
        x_offset = this.window_width / 2 - this.render_scale * center_x
        return np.array(
            [
                [this.render_scale, 0, x_offset],
                [0, -this.render_scale, this.window_height - this.render_scale],
                [0, 0, 1],
            ]
        )

    env._camera_matrix = MethodType(camera_matrix, env)


def main() -> None:
    args = parse_args()
    checkpoint_path = find_latest_checkpoint(PROJECT_ROOT) if args.checkpoint == "latest" else Path(args.checkpoint)
    print("Using checkpoint:", checkpoint_path)

    checkpoint = load_checkpoint(checkpoint_path)
    metadata = to_plain(checkpoint.get("metadata", {}))
    print("Checkpoint metadata:", metadata)
    terrain_contact_mode = terrain_contact_mode_from_metadata(
        metadata,
        args.terrain_contact_mode,
    )

    material_shape = metadata.get("scenario", "crawler") if args.robot == "checkpoint" else args.robot
    num_particles = int(metadata.get("n_particles", 13) if args.num_particles is None else args.num_particles)

    saved_channel = infer_channel_from_metadata(metadata)
    if args.channel == "checkpoint":
        canonical_channel = saved_channel
        _, default_observation_func, default_control_mode = channel_config(canonical_channel)
        observation_func = metadata.get("observation_func", default_observation_func)
        if args.observation_func is not None:
            observation_func = args.observation_func

        if args.control_mode in {"checkpoint", "auto"}:
            saved_control_raw = metadata.get("control_mode", metadata.get("control_channel"))
            control_mode = normalise_control_mode(saved_control_raw) if saved_control_raw is not None else default_control_mode
        else:
            _, _, control_mode = channel_config(
                canonical_channel,
                observation_func=observation_func,
                control_mode=args.control_mode,
            )
    else:
        control_override = "auto" if args.control_mode == "checkpoint" else args.control_mode
        canonical_channel, observation_func, control_mode = channel_config(
            args.channel,
            observation_func=args.observation_func,
            control_mode=control_override,
        )

    resolved_channel_label = make_channel_label(canonical_channel, observation_func, control_mode)

    feedback_gain = 1.0
    max_control_gain = float(
        metadata.get("max_control_gain", metadata.get("coefficient_limit", 9.0))
        if args.max_control_gain is None
        else args.max_control_gain
    )
    k1_min = finite_bound_or_none(metadata.get("k1_min", None)) if args.k1_min is None else args.k1_min
    k1_max = finite_bound_or_none(metadata.get("k1_max", None)) if args.k1_max is None else args.k1_max
    k2_min = finite_bound_or_none(metadata.get("k2_min", None)) if args.k2_min is None else args.k2_min
    k2_max = finite_bound_or_none(metadata.get("k2_max", None)) if args.k2_max is None else args.k2_max
    fix_k1 = bool(
        metadata.get("fix_k1", metadata.get("formula_fix_k1", False))
        if args.fix_k1 is None
        else args.fix_k1
    )
    fixed_k1 = float(metadata.get("fixed_k1", -5.0) if args.fixed_k1 is None else args.fixed_k1)
    fix_k2 = bool(
        metadata.get("fix_k2", metadata.get("formula_fix_k2", False))
        if args.fix_k2 is None
        else args.fix_k2
    )
    fixed_k2 = float(metadata.get("fixed_k2", 0.0) if args.fixed_k2 is None else args.fixed_k2)
    k_action_scale = float(
        metadata.get("k_action_scale", metadata.get("formula_action_scale", 1.0))
        if args.k_action_scale is None
        else args.k_action_scale
    )
    min_k2_magnitude = float(
        metadata.get("min_k2_magnitude", 1e-3)
        if args.min_k2_magnitude is None
        else args.min_k2_magnitude
    )
    passive_kappa = float(metadata.get("passive_kappa", 4.0) if args.passive_kappa is None else args.passive_kappa)
    rolling_observation = bool(
        metadata.get("rolling_observation", False)
        if args.rolling_observation is None
        else args.rolling_observation
    )
    reward_func = str(metadata.get("reward_func", "horizontal_speed"))
    rolling_direction = str(metadata.get("rolling_direction", "right"))
    rolling_curl_episodes = int(metadata.get("rolling_curl_episodes", 500))
    rolling_transition_episodes = int(metadata.get("rolling_transition_episodes", 300))
    rolling_speed_ref_x100 = float(metadata.get("rolling_speed_ref_x100", 2.0))
    rolling_omega_ref = float(metadata.get("rolling_omega_ref", 1.0))
    rolling_reward_scale = float(metadata.get("rolling_reward_scale", 3.0))
    action_smoothness_weight = float(metadata.get("action_smoothness_weight", 0.0))
    tail_roll_observation = bool(metadata.get("tail_roll_observation", False))
    tail_side = str(metadata.get("tail_side", "left"))
    tail_curl_sign = metadata.get("tail_curl_sign", "auto")
    tail_roll_stage = int(metadata.get("tail_roll_stage", 0))
    tail_roll_reward_scale = float(metadata.get("tail_roll_reward_scale", 3.0))
    tail_roll_potential_gamma = float(metadata.get("tail_roll_potential_gamma", 1.0))
    tail_roll_contact_margin = float(metadata.get("tail_roll_contact_margin", 0.05))
    tail_roll_curl_reference = np.deg2rad(
        float(metadata.get("tail_roll_curl_reference_degrees", 60.0))
    )
    # These fields are deliberately metadata-only.  Old checkpoints do not
    # contain them and therefore retain the exact legacy observation shape.
    fast_forward_observation = bool(metadata.get("fast_forward_observation", False))
    fast_forward_reward_scale = float(metadata.get("fast_forward_reward_scale", 1.0))
    fast_forward_event_degrees = float(metadata.get("fast_forward_event_degrees", 60.0))
    fast_forward_event_forward_fraction = float(
        metadata.get("fast_forward_event_forward_fraction", 0.08)
    )
    fast_forward_event_contact_nodes = float(
        metadata.get("fast_forward_event_contact_nodes", 1.5)
    )
    fast_forward_direction_fraction = float(
        metadata.get("fast_forward_direction_fraction", 0.65)
    )
    fast_forward_event_target_steps = int(
        metadata.get("fast_forward_event_target_steps", 250)
    )
    fast_forward_launch_lift = float(metadata.get("fast_forward_launch_lift", 0.20))
    fast_forward_launch_forward = float(
        metadata.get("fast_forward_launch_forward", 0.10)
    )
    fast_forward_launch_curl = float(metadata.get("fast_forward_launch_curl", 0.12))
    fast_forward_launch_head_contact = float(
        metadata.get("fast_forward_launch_head_contact", 0.50)
    )
    fast_forward_launch_hold_steps = int(
        metadata.get("fast_forward_launch_hold_steps", 8)
    )
    fast_forward_stall_steps = int(metadata.get("fast_forward_stall_steps", 150))
    fast_forward_rotation_step_ref_degrees = float(
        metadata.get("fast_forward_rotation_step_ref_degrees", 2.0)
    )
    fast_forward_translation_step_ref = float(
        metadata.get("fast_forward_translation_step_ref", 0.002)
    )
    scratch_wr_alpha = float(
        metadata.get("scratch_wr_current_alpha", metadata.get("scratch_wr_initial_alpha", 0.0))
    )
    scratch_wr_v2 = bool(metadata.get("scratch_wr_v2", False))
    scratch_wr_v2_sync_dense_weight = float(
        metadata.get("scratch_wr_v2_sync_dense_weight", 0.02)
    )
    scratch_wr_v2_penalty_start_scale = float(
        metadata.get("scratch_wr_v2_penalty_start_scale", 0.20)
    )
    scratch_wr_v2_penalty_anneal_batches = int(
        metadata.get("scratch_wr_v2_penalty_anneal_batches", 200)
    )
    scratch_wr_v2_wave_ema_beta = float(
        metadata.get("scratch_wr_v2_wave_ema_beta", 0.90)
    )
    init_pos_randomness = float(metadata.get("init_pos_randomness", 0.01))
    init_angle_range_degrees = float(metadata.get("init_angle_range_degrees", 0.0))
    init_height_jitter = float(metadata.get("init_height_jitter", 0.0))

    algorithm = metadata.get("algorithm", "ppo") if args.algorithm == "checkpoint" else args.algorithm
    metadata = dict(metadata)
    metadata["algorithm"] = algorithm
    if args.algorithm != "checkpoint":
        metadata["algorithm_family"] = algorithm_family(metadata)

    if args.terrain == "checkpoint":
        terrain_type = metadata.get("terrain_type", "flat")
        terrain_settings = to_plain(metadata.get("terrain_settings", None))
    else:
        terrain_type, terrain_settings = terrain_config(
            args.terrain,
            start_stairs=args.start_stairs,
            step_width=args.step_width,
            step_height=args.step_height,
            steps=args.steps,
            tunnel_start=args.tunnel_start,
            tunnel_slope=args.tunnel_slope,
            tunnel_slope_height=args.tunnel_slope_height,
            tunnel_length=args.tunnel_length,
            tunnel_height=args.tunnel_height,
        )

    if args.robot != "checkpoint" and args.robot != metadata.get("scenario"):
        print(
            "WARNING: You are overriding the robot shape from the checkpoint. "
            "Policy loading will only work if observation/action dimensions match. "
            "Usually you should train a separate checkpoint for crawler and ring."
        )
    if (
        args.channel != "checkpoint"
        or args.control_mode != "checkpoint"
        or args.algorithm != "checkpoint"
        or args.observation_func is not None
        or args.rolling_observation is not None
        or args.terrain_contact_mode is not None
    ):
        print(
            "WARNING: You are overriding checkpoint metadata. Loading will only work when "
            "the observation size, action size, algorithm, and network dimensions remain compatible."
        )

    print("Demo environment:")
    print("  robot            =", material_shape)
    print("  num_particles    =", num_particles)
    print("  channel          =", resolved_channel_label)
    print("  observation_func =", observation_func)
    print("  control_mode     =", control_mode)
    print("  algorithm        =", algorithm)
    print("  feedback_gain    =", feedback_gain, "(fixed)")
    print("  max_control_gain =", max_control_gain)
    print("  k1_range         =", (k1_min, k1_max))
    print("  k2_range         =", (k2_min, k2_max))
    print("  k_action_scale   =", k_action_scale)
    print("  fix_k1           =", fix_k1)
    print("  fixed_k1         =", fixed_k1)
    print("  fix_k2           =", fix_k2)
    print("  fixed_k2         =", fixed_k2)
    print("  min_k2_magnitude =", min_k2_magnitude)
    print("  passive_kappa    =", passive_kappa)
    print("  reward_func      =", reward_func)
    print("  rolling_obs      =", rolling_observation)
    print("  fast_forward_obs =", fast_forward_observation)
    print("  terrain_type     =", terrain_type)
    print("  terrain_settings =", terrain_settings)
    print("  terrain_contact  =", terrain_contact_mode)
    print(
        "  policy_sharing   =",
        metadata.get(
            "policy_parameter_sharing",
            "shared" if metadata.get("share_parameters_policy", metadata.get("share_policy", True)) else "independent_per_joint",
        ),
    )
    print("  policy_mode      =", args.policy_mode)

    meta_env = metamaterial.env(
        num_envs=args.num_envs,
        material_shape=material_shape,
        max_steps=args.max_steps,
        num_particles=num_particles,
        render_mode="human",
        observation_func=observation_func,
        window_width=args.window_width,
        window_height=args.window_height,
        terrain_type=terrain_type,
        terrain_settings=terrain_settings,
        terrain_contact_mode=terrain_contact_mode,
        control_mode=control_mode,
        feedback_gain=feedback_gain,
        max_control_gain=max_control_gain,
        k1_min=k1_min,
        k1_max=k1_max,
        k2_min=k2_min,
        k2_max=k2_max,
        k_action_scale=k_action_scale,
        fix_k1=fix_k1,
        fixed_k1=fixed_k1,
        fix_k2=fix_k2,
        fixed_k2=fixed_k2,
        min_k2_magnitude=min_k2_magnitude,
        passive_kappa=passive_kappa,
        scratch_wr_alpha=scratch_wr_alpha,
        scratch_wr_v2=scratch_wr_v2,
        scratch_wr_v2_sync_dense_weight=scratch_wr_v2_sync_dense_weight,
        scratch_wr_v2_penalty_start_scale=scratch_wr_v2_penalty_start_scale,
        scratch_wr_v2_penalty_anneal_batches=scratch_wr_v2_penalty_anneal_batches,
        scratch_wr_v2_wave_ema_beta=scratch_wr_v2_wave_ema_beta,
        reward_func=reward_func,
        rolling_observation=rolling_observation,
        rolling_direction=rolling_direction,
        rolling_curl_episodes=rolling_curl_episodes,
        rolling_transition_episodes=rolling_transition_episodes,
        rolling_speed_ref_x100=rolling_speed_ref_x100,
        rolling_omega_ref=rolling_omega_ref,
        rolling_reward_scale=rolling_reward_scale,
        action_smoothness_weight=action_smoothness_weight,
        tail_roll_observation=tail_roll_observation,
        tail_side=tail_side,
        tail_curl_sign=tail_curl_sign,
        tail_roll_stage=tail_roll_stage,
        tail_roll_reward_scale=tail_roll_reward_scale,
        tail_roll_potential_gamma=tail_roll_potential_gamma,
        tail_roll_contact_margin=tail_roll_contact_margin,
        tail_roll_curl_reference=tail_roll_curl_reference,
        fast_forward_observation=fast_forward_observation,
        fast_forward_reward_scale=fast_forward_reward_scale,
        fast_forward_event_degrees=fast_forward_event_degrees,
        fast_forward_event_forward_fraction=fast_forward_event_forward_fraction,
        fast_forward_event_contact_nodes=fast_forward_event_contact_nodes,
        fast_forward_direction_fraction=fast_forward_direction_fraction,
        fast_forward_event_target_steps=fast_forward_event_target_steps,
        fast_forward_launch_lift=fast_forward_launch_lift,
        fast_forward_launch_forward=fast_forward_launch_forward,
        fast_forward_launch_curl=fast_forward_launch_curl,
        fast_forward_launch_head_contact=fast_forward_launch_head_contact,
        fast_forward_launch_hold_steps=fast_forward_launch_hold_steps,
        fast_forward_stall_steps=fast_forward_stall_steps,
        fast_forward_rotation_step_ref_degrees=fast_forward_rotation_step_ref_degrees,
        fast_forward_translation_step_ref=fast_forward_translation_step_ref,
        init_pos_randomness=init_pos_randomness,
        init_angle_range_degrees=init_angle_range_degrees,
        init_height_jitter=init_height_jitter,
        render_text_lines=["trained policy demo"],
    )
    if hasattr(meta_env, "set_curriculum_episode"):
        meta_env.set_curriculum_episode(int(metadata.get("scratch_wr_current_batch", 0)))

    if args.follow_camera:
        enable_follow_camera(meta_env)

    policy = build_policy(meta_env, metadata)
    try:
        policy.load_state_dict(checkpoint["policy"], strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load the policy into this demo environment. This usually means the checkpoint was trained "
            "with a different robot shape, number of particles, observation function, control mode, "
            "algorithm, or policy network size. Use a checkpoint trained with the same configuration."
        ) from exc
    policy.eval()
    print("Loaded trained policy.")

    td = meta_env.reset()
    if not args.no_warmup:
        for _ in range(10):
            action_td = choose_action(policy, td, args.policy_mode)
            td = meta_env.step(action_td)["next"]

    td = meta_env.reset()
    speeds: list[float] = []
    t_start = time.time()

    for step in range(args.max_steps):
        action_td = choose_action(policy, td, args.policy_mode)
        mean_abs_action = float(action_td["agents", "action"].abs().mean().item())
        td = meta_env.step(action_td)["next"]
        speed = float(td["log_info", "speed"].mean().item())
        speeds.append(speed)

        if hasattr(meta_env, "render_text_lines"):
            meta_env.render_text_lines = [
                f"robot: {material_shape} | terrain: {terrain_label(terrain_type, terrain_settings)} | alg: {algorithm} | control: {control_mode}",
                f"channel: {resolved_channel_label} | policy mode: {args.policy_mode}",
                f"step: {step + 1}/{args.max_steps}",
                f"speed: {speed:.5f}",
                f"mean|policy output|: {mean_abs_action:.5f}",
            ]
        meta_env.render()

        if args.print_every > 0 and (step + 1) % args.print_every == 0:
            print(f"step={step + 1:5d} speed={speed:.6f} mean|policy_output|={mean_abs_action:.6f}")

    t_end = time.time()
    print("mean speed:", float(np.mean(speeds)) if speeds else 0.0)
    print("fps:", 999 if t_end == t_start else args.max_steps / (t_end - t_start))

    if not args.no_pause:
        input("Run complete. Press Enter to close the window...")
    if hasattr(meta_env, "close"):
        meta_env.close()


if __name__ == "__main__":
    main()
