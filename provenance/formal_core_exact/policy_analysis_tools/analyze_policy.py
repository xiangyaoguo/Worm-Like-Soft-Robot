"""
Generate paper-style analysis figures for trained RL metamaterial policies.

Place this file at:
    RLMetamaterialLocomotion-main/RLMetamaterialLocomotion-main/training/analyze_policy.py

Examples:
    python .\training\analyze_policy.py --checkpoint latest
    python .\training\analyze_policy.py --checkpoint .\results\20260611_230813_crawler_stairs\checkpoint_500.pt

Outputs are written next to the checkpoint, e.g.:
    results/<run_name>/analysis_checkpoint_500/
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

# Use a non-interactive backend so the script works from PyCharm/PowerShell without opening plot windows.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhDelta, TanhNormal

try:
    from torchrl.envs.utils import ExplorationType, set_exploration_type
except ImportError:  # older TorchRL naming
    from torchrl.envs.utils import ExplorationMode as ExplorationType, set_exploration_type


def find_project_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "metamaterial_envs").exists() and (p / "training").exists():
            return p
    raise RuntimeError("Cannot find project root. Put analyze_policy.py inside the project, preferably training/.")


PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(PROJECT_ROOT / "metamaterial_envs"))
from metamaterial_envs.env import metamaterial  # noqa: E402


class BiasedNormalParamExtractor(NormalParamExtractor):
    """Matches the normal-parameter extractor used by the training scripts."""

    def forward(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        tensor, *others = tensors
        loc, scale = tensor.chunk(2, -1)
        scale = self.scale_mapping(scale) + self.scale_lb
        return (loc, scale, *others)


class FirstOrderGaussian(torch.nn.Module):
    """Compatibility with older experiments trained with gaussian_activation=True."""

    def __init__(self, std: float = 1):
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.pi * x * torch.exp((-x ** 2) / (2 * self.std ** 2)) / self.std


def to_plain(obj: Any) -> Any:
    """Convert Sacred ReadOnlyDict / nested mappings into normal Python containers."""
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


def find_latest_checkpoint() -> Path:
    candidates: List[Path] = []
    for root in [PROJECT_ROOT / "results", PROJECT_ROOT / "training" / "results", PROJECT_ROOT]:
        if root.exists():
            candidates.extend(root.rglob("checkpoint_*.pt"))
    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No checkpoint_*.pt found. Train first or pass --checkpoint <path>.")
    return candidates[0]


def load_checkpoint(path: Path) -> Dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


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
    raise RuntimeError("Cannot find deterministic exploration mode in this TorchRL version.")


def build_policy(env, metadata: Dict[str, Any]) -> ProbabilisticActor:
    algorithm = str(metadata.get("algorithm", "ppo")).lower()
    policy_net_config = to_plain(metadata.get("policy_net_config", {"depth": 2, "num_cells": 256}))
    share_parameters_policy = bool(metadata.get("share_parameters_policy", True))
    gaussian_activation = bool(metadata.get("gaussian_activation", False))
    normal_scale_lb = float(metadata.get("normal_scale_lb", 1e-4))

    if algorithm not in {"ppo", "ddpg"}:
        raise ValueError(f"Unsupported algorithm in checkpoint: {algorithm}")

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
        distribution_class = TanhNormal
    else:
        temp_keys = [("agents", "param")]
        distribution_class = TanhDelta

    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation")],
        out_keys=temp_keys,
    )

    policy = ProbabilisticActor(
        module=policy_module,
        spec=env.action_spec_unbatched,
        in_keys=temp_keys,
        out_keys=[action_key],
        distribution_class=distribution_class,
        distribution_kwargs={
            "low": env.full_action_spec_unbatched[action_key].space.low,
            "high": env.full_action_spec_unbatched[action_key].space.high,
        },
        return_log_prob=False,
    )
    return policy


def make_env_from_metadata(metadata: Dict[str, Any], max_steps: int = 1000):
    material_shape = metadata.get("scenario", metadata.get("robot", "crawler"))
    n_particles = int(metadata.get("n_particles", 13))
    observation_func = metadata.get("observation_func", "dth_tot")
    terrain_type = metadata.get("terrain_type", "flat")
    terrain_settings = to_plain(metadata.get("terrain_settings", None))
    control_mode = metadata.get("control_mode", metadata.get("control_channel", "direct"))
    if metadata.get("terrain", None) == "stairs" and terrain_type == "flat":
        terrain_type = "mesh"
    return metamaterial.env(
        num_envs=1,
        material_shape=material_shape,
        num_particles=n_particles,
        max_steps=max_steps,
        render_mode="rgb_array",
        observation_func=observation_func,
        terrain_type=terrain_type,
        terrain_settings=terrain_settings,
        control_mode=control_mode,
        max_control_gain=float(metadata.get("max_control_gain", metadata.get("coefficient_limit", 9.0))),
        k1_min=metadata.get("k1_min", None),
        k1_max=metadata.get("k1_max", None),
        k2_min=metadata.get("k2_min", None),
        k2_max=metadata.get("k2_max", None),
        fix_k1=bool(metadata.get("fix_k1", metadata.get("formula_fix_k1", False))),
        fixed_k1=float(metadata.get("fixed_k1", -5.0)),
        fix_k2=bool(metadata.get("fix_k2", metadata.get("formula_fix_k2", False))),
        fixed_k2=float(metadata.get("fixed_k2", 0.0)),
        min_k2_magnitude=float(metadata.get("min_k2_magnitude", 1e-3)),
        passive_kappa=float(metadata.get("passive_kappa", 4.0)),
        render_text_lines=[],
    )


def parse_slices(text: str) -> List[float]:
    vals = []
    for part in text.replace(",", " ").split():
        vals.append(float(part))
    if not vals:
        raise ValueError("--theta-dot-slices must contain at least one number")
    return vals


def make_output_dir(checkpoint_path: Path, explicit: Optional[Path] = None) -> Path:
    if explicit is not None:
        out = explicit
    else:
        out = checkpoint_path.parent / f"analysis_{checkpoint_path.stem}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def make_grid(grid_size: int, angle_limit: float = math.pi) -> Tuple[np.ndarray, np.ndarray]:
    vals = np.linspace(-angle_limit, angle_limit, grid_size, dtype=np.float32)
    x, y = np.meshgrid(vals, vals)
    return x.astype(np.float32), y.astype(np.float32)


def observations_for_grid(
    x: np.ndarray,
    y: np.ndarray,
    obs_dim: int,
    observation_func: str,
    theta_dot: float = 0.0,
) -> np.ndarray:
    """
    Build artificial local observations for policy visualisation.

    Convention used for plots:
      x-axis = δθ_{i+1}
      y-axis = δθ_{i-1}

    For dth_tot, we use δθ_{i+1} - δθ_{i-1}.
    For neighbour observations, channels are [δθ_{i+1}, δθ_{i-1}, θdot].
    """
    flat_x = x.reshape(-1).astype(np.float32)
    flat_y = y.reshape(-1).astype(np.float32)
    n = flat_x.shape[0]
    obs = np.zeros((n, obs_dim), dtype=np.float32)

    name = observation_func.lower()
    if obs_dim == 1:
        obs[:, 0] = flat_x - flat_y
    elif obs_dim == 2:
        if "tot" in name and ("thdot" in name or "friction" in name or "wave" in name):
            obs[:, 0] = flat_x - flat_y
            obs[:, 1] = np.float32(theta_dot)
        elif "tot" in name:
            obs[:, 0] = flat_x - flat_y
            obs[:, 1] = 0.0
        else:
            obs[:, 0] = flat_x
            obs[:, 1] = flat_y
    else:
        obs[:, 0] = flat_x
        obs[:, 1] = flat_y
        obs[:, 2] = np.float32(theta_dot)
        # If the policy has more than 3 observation dimensions, leave extras at 0.
    return obs


def eval_policy_on_observations(
    policy: ProbabilisticActor,
    env,
    obs_np: np.ndarray,
    agent_index: int = 0,
    chunk_size: int = 16384,
) -> np.ndarray:
    policy.eval()
    mode = deterministic_exploration_type()
    out = []
    obs_dim = obs_np.shape[-1]
    for start in range(0, obs_np.shape[0], chunk_size):
        chunk = obs_np[start:start + chunk_size]
        obs = np.zeros((chunk.shape[0], env.num_agents, obs_dim), dtype=np.float32)
        obs[:, :, :] = chunk[:, None, :]
        td = TensorDict(
            {"agents": TensorDict({"observation": torch.as_tensor(obs, dtype=torch.float32)}, batch_size=[chunk.shape[0], env.num_agents])},
            batch_size=[chunk.shape[0]],
        )
        with torch.no_grad(), set_exploration_type(mode):
            try:
                action_td = policy(td)
            except RuntimeError as exc:
                # Some TorchRL/TanhNormal combinations may not support a deterministic mode.
                # Fall back to a single sample so the analysis script still produces a figure.
                print(f"Warning: deterministic policy evaluation failed ({exc}). Falling back to stochastic sample.")
                action_td = policy(td)
        action = action_td[env.action_key].detach().cpu().numpy()
        out.append(action[:, agent_index, 0])
    return np.concatenate(out, axis=0)


def add_colorbar(fig, im, label: str = "torque"):
    cbar = fig.colorbar(im, ax=fig.axes, shrink=0.8, pad=0.02)
    cbar.set_label(label)
    return cbar


def plot_heatmap_2d(policy, env, metadata: Dict[str, Any], out_dir: Path, grid_size: int, agent_index: int, vlim: float):
    obs_dim = env.observation_spec["agents", "observation"].shape[-1]
    observation_func = metadata.get("observation_func", "")
    x, y = make_grid(grid_size)
    obs = observations_for_grid(x, y, obs_dim, observation_func, theta_dot=0.0)
    z = eval_policy_on_observations(policy, env, obs, agent_index=agent_index).reshape(x.shape)

    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    im = ax.imshow(
        z,
        origin="lower",
        extent=[-math.pi, math.pi, -math.pi, math.pi],
        cmap="RdBu",
        vmin=-vlim,
        vmax=vlim,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_title(f"Torque at node i as a function of local observation\n{metadata.get('scenario', metadata.get('robot', ''))} / {metadata.get('terrain_type', '')}")
    if obs_dim == 1:
        ax.set_xlabel(r"$\delta\theta_{i+1}$")
        ax.set_ylabel(r"$\delta\theta_{i-1}$")
        ax.text(0.02, 0.02, r"policy input: $\delta\theta_{i+1}-\delta\theta_{i-1}$", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=9, bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"})
    elif obs_dim == 2 and ("thdot" in observation_func.lower() or "friction" in observation_func.lower()):
        ax.set_xlabel(r"synthetic $\delta\theta_{i+1}$")
        ax.set_ylabel(r"synthetic $\delta\theta_{i-1}$")
        ax.text(0.02, 0.02, r"second input fixed at $0$", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=9, bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"})
    else:
        ax.set_xlabel(r"$\delta\theta_{i+1}$")
        ax.set_ylabel(r"$\delta\theta_{i-1}$")
    ax.set_xticks([-math.pi, 0, math.pi], [r"$-\pi$", "0", r"$\pi$"])
    ax.set_yticks([-math.pi, 0, math.pi], [r"$-\pi$", "0", r"$\pi$"])
    add_colorbar(fig, im, "torque")
    path = out_dir / "policy_torque_heatmap_2d.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_theta_dot_slices(policy, env, metadata: Dict[str, Any], out_dir: Path, grid_size: int, agent_index: int, slices: List[float], vlim: float):
    obs_dim = env.observation_spec["agents", "observation"].shape[-1]
    observation_func = metadata.get("observation_func", "")
    if obs_dim < 3:
        return None
    x, y = make_grid(grid_size)
    n = len(slices)
    fig, axes = plt.subplots(1, n, figsize=(1.8 * n + 1.8, 3.0), constrained_layout=True, sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    last_im = None
    for ax, s in zip(axes, slices):
        obs = observations_for_grid(x, y, obs_dim, observation_func, theta_dot=s)
        z = eval_policy_on_observations(policy, env, obs, agent_index=agent_index).reshape(x.shape)
        last_im = ax.imshow(
            z,
            origin="lower",
            extent=[-math.pi, math.pi, -math.pi, math.pi],
            cmap="RdBu",
            vmin=-vlim,
            vmax=vlim,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(fr"$\dot{{\theta}}_i={s:g}$", fontsize=10)
        ax.set_xticks([-math.pi, 0, math.pi], [r"$-\pi$", "0", r"$\pi$"], fontsize=8)
        ax.set_yticks([-math.pi, 0, math.pi], [r"$-\pi$", "0", r"$\pi$"], fontsize=8)
    axes[0].set_ylabel(r"$\delta\theta_{i-1}$")
    for ax in axes:
        ax.set_xlabel(r"$\delta\theta_{i+1}$")
    fig.suptitle("Torque at node i as a function of neighbouring deflections and rotational velocity", y=1.05)
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes, shrink=0.86, pad=0.015)
        cbar.set_label("torque")
    path = out_dir / "policy_torque_slices_theta_dot.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_tot_thdot_heatmap(policy, env, metadata: Dict[str, Any], out_dir: Path, grid_size: int, agent_index: int, vlim: float):
    """For 2D observations like [dth_tot, F*thdot], plot torque over those two actual inputs."""
    obs_dim = env.observation_spec["agents", "observation"].shape[-1]
    observation_func = metadata.get("observation_func", "").lower()
    if obs_dim != 2 or not ("thdot" in observation_func or "friction" in observation_func or "wave" in observation_func):
        return None
    a = np.linspace(-math.pi, math.pi, grid_size, dtype=np.float32)
    b = np.linspace(-20.0, 20.0, grid_size, dtype=np.float32)
    x, y = np.meshgrid(a, b)
    obs = np.stack([x.reshape(-1), y.reshape(-1)], axis=1).astype(np.float32)
    z = eval_policy_on_observations(policy, env, obs, agent_index=agent_index).reshape(x.shape)
    fig, ax = plt.subplots(figsize=(6.4, 4.8), constrained_layout=True)
    im = ax.imshow(z, origin="lower", extent=[-math.pi, math.pi, -20, 20], cmap="RdBu", vmin=-vlim, vmax=vlim, aspect="auto")
    ax.set_title("Torque as a function of dth_tot and rotational-velocity feedback")
    ax.set_xlabel(r"$\delta\theta_{i+1}-\delta\theta_{i-1}$")
    ax.set_ylabel(r"velocity feedback input")
    ax.set_xticks([-math.pi, 0, math.pi], [r"$-\pi$", "0", r"$\pi$"])
    add_colorbar(fig, im, "torque")
    path = out_dir / "policy_torque_dthtot_velocity_feedback.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_nonreciprocity_baseline(out_dir: Path, grid_size: int, kappa_alpha: float, vlim: float):
    x, y = make_grid(grid_size)
    # Convention: simple non-reciprocity tau = kappa_alpha * (delta theta_{i+1} - delta theta_{i-1})
    z = np.clip(kappa_alpha * (x - y), -vlim, vlim)
    fig, ax = plt.subplots(figsize=(5.5, 4.8), constrained_layout=True)
    im = ax.imshow(z, origin="lower", extent=[-math.pi, math.pi, -math.pi, math.pi], cmap="RdBu", vmin=-vlim, vmax=vlim, aspect="equal")
    ax.set_title(fr"Simple non-reciprocity baseline, $\kappa^\alpha={kappa_alpha:g}$")
    ax.set_xlabel(r"$\delta\theta_{i+1}$")
    ax.set_ylabel(r"$\delta\theta_{i-1}$")
    ax.set_xticks([-math.pi, 0, math.pi], [r"$-\pi$", "0", r"$\pi$"])
    ax.set_yticks([-math.pi, 0, math.pi], [r"$-\pi$", "0", r"$\pi$"])
    add_colorbar(fig, im, "torque")
    path = out_dir / "simple_nonreciprocity_baseline.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def draw_terrain(ax, env, offset_y: float = 0.0, color: str = "0.25", lw: float = 1.0):
    if hasattr(env, "terrain_mesh"):
        mesh = env.terrain_mesh
        if isinstance(mesh, list):
            mesh = mesh[0]
        for a, b in mesh:
            ax.plot([np.real(a), np.real(b)], [np.imag(a) + offset_y, np.imag(b) + offset_y], color=color, lw=lw)
    else:
        ax.axhline(offset_y, color=color, lw=lw)


def draw_material(ax, pos: np.ndarray, material_shape: str, offset_y: float = 0.0, lw: float = 1.5, s: float = 18.0):
    xs = np.real(pos)
    ys = np.imag(pos) + offset_y
    segments = []
    for i in range(len(xs) - 1):
        segments.append([(xs[i], ys[i]), (xs[i + 1], ys[i + 1])])
    if material_shape == "ring" and len(xs) > 2:
        segments.append([(xs[-1], ys[-1]), (xs[0], ys[0])])
    lc = LineCollection(segments, colors="0.1", linewidths=lw, zorder=1)
    ax.add_collection(lc)
    colors = plt.cm.hsv(np.linspace(0, 1, len(xs), endpoint=False))
    ax.scatter(xs, ys, c=colors, s=s, edgecolors="black", linewidths=0.35, zorder=2)


def rollout_policy(policy, env, steps: int, warmup: int = 10):
    mode = deterministic_exploration_type()
    td = env.reset()
    for _ in range(warmup):
        with torch.no_grad(), set_exploration_type(mode):
            action_td = policy(td)
        td = env.step(action_td)["next"]
    td = env.reset()
    positions = []
    speeds = []
    for _ in range(steps):
        with torch.no_grad(), set_exploration_type(mode):
            action_td = policy(td)
        td = env.step(action_td)["next"]
        positions.append(np.array(env.pos[0], copy=True))
        speeds.append(float(td["log_info", "speed"].mean().item()))
    return np.array(positions), np.asarray(speeds, dtype=np.float32)


def plot_gait_snapshots(policy, env, metadata: Dict[str, Any], out_dir: Path, rollout_steps: int, n_snapshots: int):
    positions, speeds = rollout_policy(policy, env, rollout_steps)
    if len(positions) == 0:
        return None, None
    idxs = np.linspace(0, len(positions) - 1, n_snapshots).astype(int)
    row_gap = max(2.2, float(np.nanmax(np.imag(positions)) - np.nanmin(np.imag(positions)) + 1.2))
    fig, ax = plt.subplots(figsize=(10, max(4, n_snapshots * 0.65)), constrained_layout=True)
    for row, idx in enumerate(idxs):
        offset = row * row_gap
        draw_terrain(ax, env, offset_y=offset, lw=0.8)
        draw_material(ax, positions[idx], metadata.get("scenario", metadata.get("robot", "crawler")), offset_y=offset)
        ax.text(-9.5, offset + 0.3, f"t={idx}", fontsize=8, ha="left", va="bottom")
    ax.set_title("Motion gait snapshots of the trained policy")
    ax.set_xlabel("x")
    ax.set_ylabel("time snapshots")
    ax.set_yticks([])
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="datalim")
    gait_path = out_dir / "gait_snapshots.png"
    fig.savefig(gait_path, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 3.6), constrained_layout=True)
    ax.plot(np.arange(len(speeds)), speeds)
    ax.set_title("Horizontal speed during analysis rollout")
    ax.set_xlabel("step")
    ax.set_ylabel("speed")
    ax.grid(True, alpha=0.3)
    speed_path = out_dir / "rollout_speed.png"
    fig.savefig(speed_path, dpi=220)
    plt.close(fig)
    return gait_path, speed_path


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate policy analysis figures from a checkpoint.")
    parser.add_argument("--checkpoint", default="latest", help="Path to checkpoint_*.pt, or 'latest'.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Default: next to checkpoint.")
    parser.add_argument("--grid-size", type=int, default=121, help="Grid resolution for heatmaps.")
    parser.add_argument("--agent-index", type=int, default=0, help="Agent/node index to plot when parameters are independent.")
    parser.add_argument("--theta-dot-slices", default="-20 -15 -10 -5 0 5 10 15 20", help="Values for theta-dot slices.")
    parser.add_argument("--vlim", type=float, default=9.0, help="Color scale limit for torque heatmaps.")
    parser.add_argument("--baseline-kappa", type=float, default=-50.0, help="Kappa-alpha for simple non-reciprocity baseline.")
    parser.add_argument("--rollout-steps", type=int, default=600, help="Steps used for gait snapshots and speed curve.")
    parser.add_argument("--snapshots", type=int, default=10, help="Number of gait snapshots to save.")
    parser.add_argument("--no-gait", action="store_true", help="Skip gait snapshots and speed curve.")
    parser.add_argument("--no-baseline", action="store_true", help="Skip simple non-reciprocity baseline heatmap.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    checkpoint_path = find_latest_checkpoint() if args.checkpoint == "latest" else Path(args.checkpoint)
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    out_dir = make_output_dir(checkpoint_path, Path(args.out_dir).resolve() if args.out_dir else None)

    print("Using checkpoint:", checkpoint_path)
    print("Writing analysis figures to:", out_dir)
    checkpoint = load_checkpoint(checkpoint_path)
    metadata = to_plain(checkpoint.get("metadata", {}))
    print("Checkpoint metadata:", metadata)

    env = make_env_from_metadata(metadata, max_steps=max(args.rollout_steps + 50, 1000))
    policy = build_policy(env, metadata)
    policy.load_state_dict(checkpoint["policy"], strict=True)
    policy.eval()

    obs_dim = int(env.observation_spec["agents", "observation"].shape[-1])
    files: Dict[str, str] = {}
    files["policy_heatmap_2d"] = str(plot_heatmap_2d(policy, env, metadata, out_dir, args.grid_size, args.agent_index, args.vlim))
    if not args.no_baseline:
        files["simple_nonreciprocity_baseline"] = str(plot_nonreciprocity_baseline(out_dir, args.grid_size, args.baseline_kappa, args.vlim))
    slices = parse_slices(args.theta_dot_slices)
    slice_path = plot_theta_dot_slices(policy, env, metadata, out_dir, args.grid_size, args.agent_index, slices, args.vlim)
    if slice_path is not None:
        files["policy_theta_dot_slices"] = str(slice_path)
    feedback_path = plot_tot_thdot_heatmap(policy, env, metadata, out_dir, args.grid_size, args.agent_index, args.vlim)
    if feedback_path is not None:
        files["policy_dthtot_velocity_feedback"] = str(feedback_path)
    if not args.no_gait:
        gait_path, speed_path = plot_gait_snapshots(policy, env, metadata, out_dir, args.rollout_steps, args.snapshots)
        if gait_path is not None:
            files["gait_snapshots"] = str(gait_path)
        if speed_path is not None:
            files["rollout_speed"] = str(speed_path)

    summary = {
        "checkpoint": str(checkpoint_path),
        "metadata": metadata,
        "observation_dim": obs_dim,
        "agent_index": args.agent_index,
        "generated_files": files,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_path = out_dir / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Generated files:")
    for name, path in files.items():
        print(f"  {name}: {path}")
    print("Summary:", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
