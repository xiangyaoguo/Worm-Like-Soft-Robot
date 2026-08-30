r"""Generate policy-analysis figures for trained metamaterial checkpoints.

This script creates figures similar to the thesis heatmaps:
  - torque/action as a function of neighbouring deflections δθ_{i+1}, δθ_{i-1}
  - if the observation includes rotational velocity, slices over θdot are plotted
  - optionally, one heatmap per agent/node is saved
  - a simple non-reciprocity baseline heatmap is also saved

Put this file in:
    RLMetamaterialLocomotion-main/RLMetamaterialLocomotion-main/training/analyze_policy_heatmaps.py

Examples:
    python .\training\analyze_policy_heatmaps.py --checkpoint latest
    python .\training\analyze_policy_heatmaps.py --checkpoint .\results\20260611_153000_ring_stairs\checkpoint_500.pt
    python .\training\analyze_policy_heatmaps.py --checkpoint latest --grid-size 151 --theta-dot-slices 9
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import Delta, IndependentNormal, MultiAgentMLP, ProbabilisticActor, TanhDelta, TanhNormal

try:
    from torchrl.envs.utils import ExplorationType, set_exploration_type
except ImportError:  # older/newer TorchRL naming compatibility
    from torchrl.envs.utils import ExplorationMode as ExplorationType, set_exploration_type

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def find_project_root(start: Path) -> Path:
    start = start.resolve()
    for p in [start] + list(start.parents):
        if (p / "metamaterial_envs").exists() and (p / "training").exists():
            return p
    raise RuntimeError("Cannot find project root. Put this file inside the project folder, preferably in training/.")


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
ENV_PACKAGE_ROOT = PROJECT_ROOT / "metamaterial_envs"
if str(ENV_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(ENV_PACKAGE_ROOT))

from metamaterial_envs.env import metamaterial  # noqa: E402


PPO_ALGORITHMS = {"ppo", "ppo_noclip"}
DDPG_ALGORITHMS = {"ddpg", "ddpg_clip"}


def algorithm_family(metadata: dict[str, Any]) -> str:
    family = metadata.get("algorithm_family", None)
    if family in {"ppo", "ddpg"}:
        return str(family)
    algorithm = metadata.get("algorithm", "ppo")
    if algorithm in PPO_ALGORITHMS:
        return "ppo"
    if algorithm in DDPG_ALGORITHMS:
        return "ddpg"
    raise ValueError(f"Unsupported algorithm in checkpoint metadata: {algorithm!r}")


def to_plain(obj: Any) -> Any:
    if isinstance(obj, OrderedDict):
        return OrderedDict((k, to_plain(v)) for k, v in obj.items())
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if hasattr(obj, "items") and not isinstance(obj, (str, bytes)):
        try:
            return {k: to_plain(v) for k, v in obj.items()}
        except Exception:
            pass
    if isinstance(obj, list):
        return [to_plain(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_plain(v) for v in obj)
    return obj


def finite_bound_or_none(value: Any) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    return parsed if np.isfinite(parsed) else None


class FirstOrderGaussian(torch.nn.Module):
    def __init__(self, std: float = 1.0):
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.pi * x * torch.exp((-x**2) / (2 * self.std**2)) / self.std


class BiasedNormalParamExtractor(NormalParamExtractor):
    def forward(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        tensor, *others = tensors
        loc, scale = tensor.chunk(2, -1)
        scale = self.scale_mapping(scale) + self.scale_lb
        return (loc, scale, *others)


def deterministic_exploration_type():
    if hasattr(ExplorationType, "DETERMINISTIC"):
        return ExplorationType.DETERMINISTIC
    try:
        from torchrl.envs.utils import ExplorationMode
        if hasattr(ExplorationMode, "DETERMINISTIC"):
            return ExplorationMode.DETERMINISTIC
    except Exception:
        pass
    if hasattr(ExplorationType, "MEAN"):
        return ExplorationType.MEAN
    raise RuntimeError("Cannot find deterministic TorchRL exploration type.")


def find_latest_checkpoint(project_root: Path) -> Path:
    candidates: list[Path] = []
    for root in [project_root / "results", project_root / "training" / "results", project_root]:
        if root.exists():
            candidates.extend(root.rglob("checkpoint_*.pt"))
    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No checkpoint_*.pt found. Train first or pass --checkpoint PATH.")
    return candidates[0]


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_env_from_metadata(metadata: dict[str, Any], max_steps: int = 1000):
    material_shape = metadata.get("scenario", metadata.get("robot", "crawler"))
    num_particles = int(metadata.get("n_particles", metadata.get("num_particles", 13)))
    observation_func = metadata.get("observation_func", "dth_tot")
    terrain_type = metadata.get("terrain_type", "flat")
    terrain_settings = to_plain(metadata.get("terrain_settings", None))
    control_mode = metadata.get("control_mode", metadata.get("control_channel", "direct"))
    feedback_gain = 1.0
    max_control_gain = float(metadata.get("max_control_gain", metadata.get("coefficient_limit", 9.0)))
    k1_min = finite_bound_or_none(metadata.get("k1_min", None))
    k1_max = finite_bound_or_none(metadata.get("k1_max", None))
    k2_min = finite_bound_or_none(metadata.get("k2_min", None))
    k2_max = finite_bound_or_none(metadata.get("k2_max", None))
    fix_k1 = bool(metadata.get("fix_k1", metadata.get("formula_fix_k1", False)))
    fixed_k1 = float(metadata.get("fixed_k1", -5.0))
    fix_k2 = bool(metadata.get("fix_k2", metadata.get("formula_fix_k2", False)))
    fixed_k2 = float(metadata.get("fixed_k2", 0.0))
    k_action_scale = float(metadata.get("k_action_scale", metadata.get("formula_action_scale", 1.0)))
    min_k2_magnitude = float(metadata.get("min_k2_magnitude", 1e-3))
    passive_kappa = float(metadata.get("passive_kappa", 4.0))
    reward_func = str(metadata.get("reward_func", "horizontal_speed"))
    rolling_observation = bool(metadata.get("rolling_observation", False))
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
    return metamaterial.env(
        num_envs=1,
        material_shape=material_shape,
        num_particles=num_particles,
        max_steps=max_steps,
        observation_func=observation_func,
        terrain_type=terrain_type,
        terrain_settings=terrain_settings,
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
        render_mode="rgb_array",
        window_width=1000,
        window_height=500,
        render_text_lines=[],
    )


def build_policy(env, metadata: dict[str, Any]) -> ProbabilisticActor:
    algorithm = algorithm_family(metadata)
    policy_net_config = to_plain(metadata.get("policy_net_config", {"depth": 2, "num_cells": 256}))
    share_parameters_policy = bool(metadata.get("share_parameters_policy", metadata.get("share_policy", True)))
    gaussian_activation = bool(metadata.get("gaussian_activation", False))
    normal_scale_lb = float(metadata.get("normal_scale_lb", 1e-4))

    action_key = env.action_key
    n_action_outputs = env.full_action_spec[action_key].shape[-1]
    obs_dim = env.observation_spec["agents", "observation"].shape[-1]

    policy_net = MultiAgentMLP(
        n_agent_inputs=obs_dim,
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
    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec_unbatched,
        in_keys=temp_keys,
        out_keys=[action_key],
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=False,
    )
    return policy


def observation_from_grid(
    observation_func: str,
    x_next: np.ndarray,
    y_prev: np.ndarray,
    theta_dot: float | np.ndarray,
    obs_dim: int,
    friction_gain: float = 1.0,
    own_deflection: float = 0.0,
) -> np.ndarray:
    """Map a conceptual grid to the actual observation vector used by the policy.

    x_next: δθ_{i+1}
    y_prev: δθ_{i-1}
    theta_dot: θdot_i slice value
    """
    dth_tot = x_next - y_prev
    theta_dot_arr = np.zeros_like(x_next, dtype=np.float32) + np.float32(theta_dot)
    own_arr = np.zeros_like(x_next, dtype=np.float32) + np.float32(own_deflection)
    name = observation_func

    if name == "dth_neighbours":
        obs = np.stack([x_next, y_prev], axis=-1)
    elif name == "dth_neighbours_plus_thdot":
        obs = np.stack([x_next, y_prev, theta_dot_arr], axis=-1)
    elif name == "dth_neighbours_plus_own":
        obs = np.stack([x_next, own_arr, y_prev], axis=-1)
    elif name == "dth_tot":
        obs = np.expand_dims(dth_tot, axis=-1)
    elif name == "dth_tot_plus_own":
        obs = np.stack([dth_tot, own_arr], axis=-1)
    elif name in {"dth_tot_plus_friction_thdot", "dth_tot_plus_feedback_thdot"}:
        obs = np.stack([dth_tot, np.float32(friction_gain) * theta_dot_arr], axis=-1)
    elif name == "dth_wave_feedback":
        obs = np.expand_dims(dth_tot + np.float32(friction_gain) * theta_dot_arr, axis=-1)
    else:
        # Generic fallback based on dimensionality. This keeps the script useful for custom obs functions.
        if obs_dim == 1:
            obs = np.expand_dims(dth_tot, axis=-1)
        elif obs_dim == 2:
            obs = np.stack([x_next, y_prev], axis=-1)
        elif obs_dim == 3:
            obs = np.stack([x_next, y_prev, theta_dot_arr], axis=-1)
        else:
            obs = np.zeros((*x_next.shape, obs_dim), dtype=np.float32)
            obs[..., 0] = dth_tot
            if obs_dim >= 2:
                obs[..., 1] = theta_dot_arr

    if obs.shape[-1] < obs_dim:
        padding = np.zeros((*obs.shape[:-1], obs_dim - obs.shape[-1]), dtype=np.float32)
        obs = np.concatenate((obs, padding), axis=-1)
    elif obs.shape[-1] > obs_dim:
        obs = obs[..., :obs_dim]
    return np.ascontiguousarray(obs, dtype=np.float32)


def evaluate_policy_grid(
    policy: ProbabilisticActor,
    env,
    obs_grid: np.ndarray,
    agent_index: int | None = 0,
    all_agents: bool = False,
    batch_size: int = 8192,
    dth_tot_grid: np.ndarray | None = None,
    theta_dot_grid: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate deterministic torque/action on an observation grid.

    In direct mode, the policy output is already the torque.
    In formula mode, the policy output is [k1, k2], so the plotted value is
        tau = k1 * (theta(i+1)-theta(i-1)) + k2 * F * theta_dot
    clipped to the environment actuator limit. This makes formula-channel and
    obs-channel heatmaps comparable.

    In paper/nonreciprocity mode, the policy output is kappa_alpha and the
    plotted active torque is kappa_alpha * (theta(i+1)-theta(i-1)). The fixed
    passive term is zero on these plots because own deflection is held at zero.

    In signed-k2 modes, the policy output is k2 and the plotted active torque is
        fixed_k1 * (theta(i+1)-theta(i-1)) + k2 * F * theta_dot.

    Returns shape:
      - all_agents=False: [num_points]
      - all_agents=True:  [num_agents, num_points]
    """
    obs_flat = obs_grid.reshape(-1, obs_grid.shape[-1])
    num_points = obs_flat.shape[0]
    num_agents = env.num_agents
    action_key = env.action_key
    det_mode = deterministic_exploration_type()
    control_mode = getattr(env, "control_mode", "direct")

    if dth_tot_grid is None:
        dth_tot_flat = obs_flat[:, 0].astype(np.float32, copy=False)
    else:
        dth_tot_flat = np.asarray(dth_tot_grid, dtype=np.float32).reshape(-1)
    if theta_dot_grid is None:
        theta_dot_flat = np.zeros((num_points,), dtype=np.float32)
    else:
        theta_dot_flat = np.asarray(theta_dot_grid, dtype=np.float32).reshape(-1)
    if dth_tot_flat.shape[0] != num_points or theta_dot_flat.shape[0] != num_points:
        raise ValueError("dth_tot_grid and theta_dot_grid must have the same number of points as obs_grid")

    if all_agents:
        values = np.zeros((num_agents, num_points), dtype=np.float32)
    else:
        if agent_index is None:
            agent_index = 0
        if not (0 <= agent_index < num_agents):
            raise ValueError(f"--agent-index must be in [0, {num_agents - 1}], got {agent_index}")
        values = np.zeros((num_points,), dtype=np.float32)

    policy.eval()
    with torch.no_grad(), set_exploration_type(det_mode):
        for start in range(0, num_points, batch_size):
            end = min(start + batch_size, num_points)
            obs_chunk = torch.as_tensor(obs_flat[start:end], dtype=torch.float32)
            # Feed the same local observation to every agent. With independent policies this lets us
            # compare how each node maps the same local state to torque.
            obs_all = obs_chunk[:, None, :].repeat(1, num_agents, 1)
            if bool(getattr(env, "rolling_observation", False)) and obs_all.shape[-1] >= 6:
                # Hold global rolling state at a neutral value, while preserving each joint's phase code.
                phase = 2.0 * np.pi * (np.arange(num_agents, dtype=np.float32) + 0.5) / float(num_agents)
                obs_all[:, :, -2] = torch.as_tensor(np.sin(phase), dtype=torch.float32)
                obs_all[:, :, -1] = torch.as_tensor(np.cos(phase), dtype=torch.float32)
            td = TensorDict(
                {
                    "agents": TensorDict(
                        {"observation": obs_all},
                        batch_size=[end - start, num_agents],
                    )
                },
                batch_size=[end - start],
            )
            out = policy(td)
            raw_action = out[action_key].detach().cpu().numpy()
            if control_mode == "formula":
                formula_fix_k1 = bool(getattr(env, "formula_fix_k1", getattr(env, "fix_k1", False)))
                formula_fix_k2 = bool(getattr(env, "formula_fix_k2", getattr(env, "fix_k2", False)))
                k_action_scale = float(getattr(env, "k_action_scale", getattr(env, "formula_action_scale", 1.0)))
                action_index = 0
                if formula_fix_k1:
                    k1 = np.zeros(raw_action.shape[:-1], dtype=np.float32) + float(getattr(env, "fixed_k1", -5.0))
                else:
                    k1 = np.clip(
                        raw_action[..., action_index] * k_action_scale,
                        float(getattr(env, "k1_min", -getattr(env, "max_control_gain", 9.0))),
                        float(getattr(env, "k1_max", getattr(env, "max_control_gain", 9.0))),
                    )
                    action_index += 1
                if formula_fix_k2:
                    k2 = np.zeros(raw_action.shape[:-1], dtype=np.float32) + float(getattr(env, "fixed_k2", 0.0))
                else:
                    k2 = np.clip(
                        raw_action[..., action_index] * k_action_scale,
                        float(getattr(env, "k2_min", -getattr(env, "max_control_gain", 9.0))),
                        float(getattr(env, "k2_max", getattr(env, "max_control_gain", 9.0))),
                    )
                local_dth = dth_tot_flat[start:end][:, None]
                local_thdot = theta_dot_flat[start:end][:, None]
                act = k1 * local_dth + k2 * float(getattr(env, "feedback_gain", 1.0)) * local_thdot
                act = np.clip(act, -float(getattr(env, "max_torque", 9.0)), float(getattr(env, "max_torque", 9.0)))
            elif control_mode == "nonreciprocity":
                kappa_alpha = raw_action[..., 0]
                local_dth = dth_tot_flat[start:end][:, None]
                act = kappa_alpha * local_dth
                act = np.clip(act, -float(getattr(env, "max_torque", 9.0)), float(getattr(env, "max_torque", 9.0)))
            elif control_mode in {"fixed_k1_k2_positive", "fixed_k1_k2_negative"}:
                k2 = np.clip(
                    raw_action[..., 0],
                    float(getattr(env, "k2_min", getattr(env, "min_k2_magnitude", 1e-3))),
                    float(getattr(env, "k2_max", getattr(env, "max_control_gain", 9.0))),
                )
                local_dth = dth_tot_flat[start:end][:, None]
                local_thdot = theta_dot_flat[start:end][:, None]
                act = float(getattr(env, "fixed_k1", -5.0)) * local_dth + k2 * float(getattr(env, "feedback_gain", 1.0)) * local_thdot
                act = np.clip(act, -float(getattr(env, "max_torque", 9.0)), float(getattr(env, "max_torque", 9.0)))
            else:
                act = raw_action[..., 0]
            if all_agents:
                values[:, start:end] = act.T
            else:
                values[start:end] = act[:, agent_index]

    return values


def make_grid(theta_min: float, theta_max: float, grid_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(theta_min, theta_max, grid_size, dtype=np.float32)
    ys = np.linspace(theta_min, theta_max, grid_size, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)
    return xs, ys, X.astype(np.float32), Y.astype(np.float32)


def save_single_heatmap(
    Z: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    output_path: Path,
    title: str,
    vlim: float,
    dpi: int,
    xlabel: str = r"$\delta\theta_{i+1}$",
    ylabel: str = r"$\delta\theta_{i-1}$",
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    im = ax.imshow(
        Z,
        origin="lower",
        extent=[float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())],
        cmap="RdBu",
        vmin=-vlim,
        vmax=vlim,
        aspect="auto",
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks([-math.pi, 0, math.pi], labels=[r"$-\pi$", "0", r"$\pi$"])
    ax.set_yticks([-math.pi, 0, math.pi], labels=[r"$-\pi$", "0", r"$\pi$"])
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("torque / action")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def save_theta_dot_slices(
    Zs: list[np.ndarray],
    theta_dots: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    output_path: Path,
    title: str,
    vlim: float,
    dpi: int,
) -> None:
    n = len(Zs)
    fig_width = max(10, 1.55 * n + 2.0)
    fig, axes = plt.subplots(1, n, figsize=(fig_width, 3.6), sharex=True, sharey=True, constrained_layout=True)
    if n == 1:
        axes = [axes]
    im = None
    for ax, Z, td in zip(axes, Zs, theta_dots):
        im = ax.imshow(
            Z,
            origin="lower",
            extent=[float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())],
            cmap="RdBu",
            vmin=-vlim,
            vmax=vlim,
            aspect="auto",
        )
        ax.set_title(f"{td:.2f}", fontsize=9)
        ax.set_xticks([-math.pi, 0, math.pi], labels=[r"$-\pi$", "0", r"$\pi$"], fontsize=8)
        ax.set_yticks([-math.pi, 0, math.pi], labels=[r"$-\pi$", "0", r"$\pi$"], fontsize=8)
        ax.tick_params(length=2)
    axes[0].set_ylabel(r"$\delta\theta_{i-1}$")
    for ax in axes:
        ax.set_xlabel(r"$\delta\theta_{i+1}$", fontsize=8)
    fig.suptitle(title + "\n" + r"slice value: $\dot{\theta}_i$", fontsize=12)
    assert im is not None
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("torque / action")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def save_all_agents_heatmap(
    values: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    output_path: Path,
    title: str,
    vlim: float,
    dpi: int,
) -> None:
    num_agents = values.shape[0]
    grid_size = len(xs)
    cols = min(num_agents, 10)
    rows = int(math.ceil(num_agents / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.55 + 1.1, rows * 1.55 + 1.0), sharex=True, sharey=True)
    axes_arr = np.asarray(axes).reshape(-1)
    im = None
    for i, ax in enumerate(axes_arr):
        if i >= num_agents:
            ax.axis("off")
            continue
        Z = values[i].reshape(grid_size, grid_size)
        im = ax.imshow(
            Z,
            origin="lower",
            extent=[float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())],
            cmap="RdBu",
            vmin=-vlim,
            vmax=vlim,
            aspect="auto",
        )
        ax.set_title(rf"$i={i}$", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=12)
    assert im is not None
    fig.colorbar(im, ax=axes_arr.tolist(), fraction=0.025, pad=0.02)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_nonreciprocity_baseline(xs: np.ndarray, ys: np.ndarray, X: np.ndarray, Y: np.ndarray, output_dir: Path, args) -> Path:
    # Benchmark active torque: τ = κ^α(δθ_{i+1} - δθ_{i-1}), clipped to actuator limits.
    Z = np.float32(args.kappa_alpha) * (X - Y)
    if args.clip_baseline:
        Z = np.clip(Z, -args.vlim, args.vlim)
    path = output_dir / "baseline_simple_nonreciprocity.png"
    save_single_heatmap(
        Z,
        xs,
        ys,
        path,
        title=rf"Simple non-reciprocity baseline, $\kappa^\alpha={args.kappa_alpha:g}$",
        vlim=args.vlim,
        dpi=args.dpi,
    )
    return path


def safe_json_dump(data: dict[str, Any], path: Path) -> None:
    def convert(x):
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.float32, np.float64)):
            return float(x)
        if isinstance(x, (np.int32, np.int64)):
            return int(x)
        return str(x)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=convert)


def analyze_checkpoint(
    checkpoint: Path | str = "latest",
    output_dir: Path | None = None,
    grid_size: int = 121,
    theta_min: float = -math.pi,
    theta_max: float = math.pi,
    theta_dot_min: float = -20.0,
    theta_dot_max: float = 20.0,
    theta_dot_slices: int = 9,
    agent_index: int = 0,
    all_agents: bool = True,
    friction_gain: float = 1.0,
    own_deflection: float = 0.0,
    vlim: float = 9.0,
    dpi: int = 180,
    make_baseline: bool = True,
    clip_baseline: bool = True,
) -> list[Path]:
    checkpoint_path = find_latest_checkpoint(PROJECT_ROOT) if str(checkpoint).lower() == "latest" else Path(checkpoint)
    checkpoint_path = checkpoint_path.resolve()
    ckpt = load_checkpoint(checkpoint_path)
    metadata = to_plain(ckpt.get("metadata", {}))

    env = build_env_from_metadata(metadata)
    policy = build_policy(env, metadata)
    policy.load_state_dict(ckpt["policy"], strict=True)
    policy.eval()

    if output_dir is None:
        output_dir = checkpoint_path.parent / "analysis"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    observation_func = str(metadata.get("observation_func", "dth_tot"))
    obs_dim = int(env.observation_spec["agents", "observation"].shape[-1])
    xs, ys, X, Y = make_grid(theta_min, theta_max, grid_size)

    saved: list[Path] = []
    base_title = (
        "Torque/action at node i as a function of neighbouring deflections"
        f"\n{metadata.get('scenario', 'robot')} / {metadata.get('algorithm', 'policy')} / {observation_func}"
    )

    # Main selected-agent plot. If obs includes theta-dot, create slices; otherwise create one heatmap.
    uses_theta_dot = getattr(env, "control_mode", "direct") in {
        "formula",
        "fixed_k1_k2_positive",
        "fixed_k1_k2_negative",
    } or observation_func in {
        "dth_neighbours_plus_thdot",
        "dth_tot_plus_friction_thdot",
        "dth_tot_plus_feedback_thdot",
        "dth_wave_feedback",
    } or (obs_dim >= 3 and "thdot" in observation_func)

    if uses_theta_dot:
        theta_dots = np.linspace(theta_dot_min, theta_dot_max, theta_dot_slices, dtype=np.float32)
        Zs = []
        for td_val in theta_dots:
            obs_grid = observation_from_grid(
                observation_func, X, Y, td_val, obs_dim, friction_gain=friction_gain, own_deflection=own_deflection
            )
            vals = evaluate_policy_grid(
                policy,
                env,
                obs_grid,
                agent_index=agent_index,
                all_agents=False,
                dth_tot_grid=X - Y,
                theta_dot_grid=np.zeros_like(X, dtype=np.float32) + np.float32(td_val),
            )
            Zs.append(vals.reshape(grid_size, grid_size))
        path = output_dir / f"policy_heatmap_agent{agent_index}_theta_dot_slices.png"
        save_theta_dot_slices(Zs, theta_dots, xs, ys, path, base_title, vlim=vlim, dpi=dpi)
        saved.append(path)

        # Also save a single θdot=0 heatmap for convenience.
        obs_grid_zero = observation_from_grid(
            observation_func, X, Y, 0.0, obs_dim, friction_gain=friction_gain, own_deflection=own_deflection
        )
        vals_zero = evaluate_policy_grid(
            policy,
            env,
            obs_grid_zero,
            agent_index=agent_index,
            all_agents=False,
            dth_tot_grid=X - Y,
            theta_dot_grid=np.zeros_like(X, dtype=np.float32),
        )
        path_zero = output_dir / f"policy_heatmap_agent{agent_index}_theta_dot_0.png"
        save_single_heatmap(vals_zero.reshape(grid_size, grid_size), xs, ys, path_zero, base_title + r" ($\dot{\theta}_i=0$)", vlim, dpi)
        saved.append(path_zero)
    else:
        obs_grid = observation_from_grid(
            observation_func, X, Y, 0.0, obs_dim, friction_gain=friction_gain, own_deflection=own_deflection
        )
        vals = evaluate_policy_grid(
            policy,
            env,
            obs_grid,
            agent_index=agent_index,
            all_agents=False,
            dth_tot_grid=X - Y,
            theta_dot_grid=np.zeros_like(X, dtype=np.float32),
        )
        path = output_dir / f"policy_heatmap_agent{agent_index}.png"
        save_single_heatmap(vals.reshape(grid_size, grid_size), xs, ys, path, base_title, vlim=vlim, dpi=dpi)
        saved.append(path)

    if all_agents:
        obs_grid_agents = observation_from_grid(
            observation_func, X, Y, 0.0, obs_dim, friction_gain=friction_gain, own_deflection=own_deflection
        )
        vals_agents = evaluate_policy_grid(
            policy,
            env,
            obs_grid_agents,
            all_agents=True,
            dth_tot_grid=X - Y,
            theta_dot_grid=np.zeros_like(X, dtype=np.float32),
        )
        path_agents = output_dir / "policy_heatmap_all_agents_theta_dot_0.png"
        save_all_agents_heatmap(vals_agents, xs, ys, path_agents, base_title + r"; each node, $\dot{\theta}_i=0$", vlim, dpi)
        saved.append(path_agents)

    if make_baseline:
        class ArgsLike:
            pass
        a = ArgsLike()
        a.kappa_alpha = -50.0
        a.clip_baseline = clip_baseline
        a.vlim = vlim
        a.dpi = dpi
        path_base = save_nonreciprocity_baseline(xs, ys, X, Y, output_dir, a)
        saved.append(path_base)

    data_path = output_dir / "analysis_grid_data.npz"
    np.savez_compressed(
        data_path,
        x_delta_theta_next=xs,
        y_delta_theta_prev=ys,
        theta_min=theta_min,
        theta_max=theta_max,
        checkpoint=str(checkpoint_path),
    )
    saved.append(data_path)

    summary_path = output_dir / "analysis_summary.json"
    safe_json_dump(
        {
            "checkpoint": str(checkpoint_path),
            "metadata": metadata,
            "output_dir": str(output_dir),
            "grid_size": grid_size,
            "theta_range": [theta_min, theta_max],
            "theta_dot_range": [theta_dot_min, theta_dot_max],
            "theta_dot_slices": theta_dot_slices,
            "agent_index": agent_index,
            "all_agents": all_agents,
            "friction_gain": friction_gain,
            "own_deflection": own_deflection,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "saved_files": [str(p) for p in saved],
        },
        summary_path,
    )
    saved.append(summary_path)
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate policy heatmap analysis figures from a checkpoint.")
    parser.add_argument("--checkpoint", default="latest", help="Path to checkpoint_*.pt or 'latest'.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Default: checkpoint folder / analysis")
    parser.add_argument("--grid-size", type=int, default=121)
    parser.add_argument("--theta-min", type=float, default=-math.pi)
    parser.add_argument("--theta-max", type=float, default=math.pi)
    parser.add_argument("--theta-dot-min", type=float, default=-20.0)
    parser.add_argument("--theta-dot-max", type=float, default=20.0)
    parser.add_argument("--theta-dot-slices", type=int, default=9)
    parser.add_argument("--agent-index", type=int, default=0, help="Agent index for the main selected-agent plot.")
    parser.add_argument("--no-all-agents", action="store_true", help="Do not create the per-agent panel figure.")
    parser.add_argument("--friction-gain", type=float, default=1.0, help="Gain used for custom wave/friction observations.")
    parser.add_argument("--own-deflection", type=float, default=0.0, help="Own-deflection slice for observation functions that include own δθ.")
    parser.add_argument("--vlim", type=float, default=9.0, help="Symmetric colorbar limit for torque/action.")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--no-baseline", action="store_true", help="Do not create the simple non-reciprocity baseline figure.")
    parser.add_argument("--no-clip-baseline", action="store_true", help="Do not clip baseline to actuator range.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    saved = analyze_checkpoint(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        grid_size=args.grid_size,
        theta_min=args.theta_min,
        theta_max=args.theta_max,
        theta_dot_min=args.theta_dot_min,
        theta_dot_max=args.theta_dot_max,
        theta_dot_slices=args.theta_dot_slices,
        agent_index=args.agent_index,
        all_agents=not args.no_all_agents,
        friction_gain=args.friction_gain,
        own_deflection=args.own_deflection,
        vlim=args.vlim,
        dpi=args.dpi,
        make_baseline=not args.no_baseline,
        clip_baseline=not args.no_clip_baseline,
    )
    print("Analysis figures/data saved:")
    for p in saved:
        print(" ", p)


if __name__ == "__main__":
    main()
