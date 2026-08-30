"""Train crawler/ring metamaterial robots on flat, stairs, or tunnel terrain.

Primary switchable settings:
  --robot crawler|ring
  --terrain flat|stairs|tunnel
  --channel dth|thdot|obs|action|paper|k2_positive|k2_negative
  --algorithm ppo|ddpg|ddpg_clip|ppo_noclip

Main channel equations:
  dth:    obs_i = [delta_theta_(i-1), delta_theta_(i+1)],
          policy outputs active torque directly (paper Eq. 2.3).
  thdot:  obs_i = [delta_theta_(i-1), delta_theta_(i+1), theta_dot_i],
          policy outputs active torque directly (paper alternate observation).
  obs:    obs_i = delta_theta_(i+1)-delta_theta_(i-1), policy outputs torque directly.
  action: obs_i = theta(i+1)-theta(i-1), policy outputs [k1_i, k2_i] by default,
          tau_i = k1_i * [theta(i+1)-theta(i-1)] + k2_i * theta_dot_i, with F fixed to 1.
          Use --fix-k1/--fix-k2 plus --fixed-k1/--fixed-k2 to freeze either coefficient;
          use --k1-min/max and --k2-min/max to set coefficient ranges.
          Use --per-joint-k1-k2 to give every controlled joint an independent actor,
          allowing it to learn its own K1_i/K2_i mapping instead of sharing one mapping.
  paper:  policy outputs kappa_alpha_i; the simulator applies
          tau_i = -kappa*dtheta_i + kappa_alpha_i*(dtheta(i+1)-dtheta(i-1)).
  k2_positive / k2_negative:
          k1 is fixed, policy outputs only k2, and k2 is constrained to be
          strictly positive or strictly negative.

Examples:
  python ./training/train_metamaterial.py --robot crawler --terrain tunnel --channel obs --algorithm ppo --episodes 500 --tunnel-height 2
  python ./training/train_metamaterial.py --robot ring --terrain tunnel --channel action --algorithm ddpg --episodes 500 --tunnel-height 5
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import tempfile
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictSequential, set_composite_lp_aggregate
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import RandomSampler, SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyMemmapStorage, LazyTensorStorage
from torchrl.envs import RewardSum, TransformedEnv
try:
    from torchrl.envs.utils import ExplorationType, check_env_specs, set_exploration_type
except ImportError:
    from torchrl.envs.utils import ExplorationMode as ExplorationType, check_env_specs, set_exploration_type
from torchrl.modules import AdditiveGaussianModule, Delta, IndependentNormal, MultiAgentMLP, ProbabilisticActor, TanhDelta, TanhNormal
from torchrl.objectives import ClipPPOLoss, DDPGLoss, SoftUpdate, ValueEstimators
from tqdm import tqdm

from rlmm_common import (
    BiasedNormalParamExtractor,
    CONTROL_MODE_CHOICES,
    DEFAULT_TERRAIN_CONTACT_MODE,
    FirstOrderGaussian,
    SafeTanhNormal,
    TERRAIN_CONTACT_MODE_CHOICES,
    add_env_package_to_path,
    channel_config,
    channel_label,
    channel_slug,
    choose_device,
    find_project_root,
    terrain_config,
)
from simulation_command import write_simulation_command

PROJECT_ROOT = find_project_root(Path(__file__).resolve().parent)
add_env_package_to_path(PROJECT_ROOT)
from metamaterial_envs.env import metamaterial  # noqa: E402


PPO_ALGORITHMS = {"ppo", "ppo_noclip"}
DDPG_ALGORITHMS = {"ddpg", "ddpg_clip"}
SUPPORTED_ALGORITHMS = sorted(PPO_ALGORITHMS | DDPG_ALGORITHMS)
SCRATCH_WR_DDPG_REPLAY_KEYS = (
    ("agents", "observation"),
    ("agents", "action"),
    ("next", "agents", "observation"),
    ("next", "agents", "reward"),
    ("next", "agents", "done"),
    ("next", "agents", "terminated"),
)
_ACTIVE_COLLECTORS = []


def shutdown_collector(collector) -> None:
    """Idempotent collector cleanup used by success, stop, and exception paths."""
    try:
        if hasattr(collector, "shutdown"):
            collector.shutdown()
    finally:
        if collector in _ACTIVE_COLLECTORS:
            _ACTIVE_COLLECTORS.remove(collector)


class ScratchWRNormalParamExtractor(BiasedNormalParamExtractor):
    """PPO head that removes residual log-prob/entropy gradients at alpha=0.

    Scratch-WR uses one bounded 22-D action for a ten-particle crawler.  The
    first six dimensions are the wave controller and the remainder are
    residual K1/K2 pairs.  With zero residual authority, using a fixed residual
    distribution makes its old/new log-probability ratio exactly cancel while
    retaining gradients for the wave dimensions.
    """

    def __init__(self, scale_lb: float = 1e-4, wave_action_size: int = 6, alpha: float = 0.0):
        super().__init__(scale_lb=scale_lb)
        self.wave_action_size = int(wave_action_size)
        # Runtime curriculum state belongs to metadata/training_state, not the
        # actor state_dict, keeping deterministic demo loaders architecture-compatible.
        self.register_buffer(
            "scratch_wr_alpha",
            torch.tensor(float(alpha), dtype=torch.float32),
            persistent=False,
        )

    def set_alpha(self, alpha: float) -> None:
        alpha = float(alpha)
        if not np.isfinite(alpha) or not (0.0 <= alpha <= 1.0):
            raise ValueError("Scratch-WR alpha must be a finite value in [0, 1].")
        self.scratch_wr_alpha.fill_(alpha)

    def forward(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        tensor, *others = tensors
        loc, scale = tensor.chunk(2, -1)
        scale = self.scale_mapping(scale) + self.scale_lb
        if loc.shape[-1] <= self.wave_action_size:
            raise ValueError("Scratch-WR policy output does not contain residual dimensions.")
        # The binary mask intentionally freezes residual distribution gradients
        # only at exactly alpha=0. Any positive authority trains the full head.
        residual_train_mask = (self.scratch_wr_alpha > 0).to(dtype=loc.dtype, device=loc.device)
        residual_loc = loc[..., self.wave_action_size:] * residual_train_mask
        fixed_scale = torch.full_like(scale[..., self.wave_action_size:], float(self.scale_lb))
        residual_scale = fixed_scale + (
            scale[..., self.wave_action_size:] - fixed_scale
        ) * residual_train_mask
        loc = torch.cat((loc[..., :self.wave_action_size], residual_loc), dim=-1)
        scale = torch.cat((scale[..., :self.wave_action_size], residual_scale), dim=-1)
        return (loc, scale, *others)


class ScratchWRDeterministicParamExtractor(torch.nn.Module):
    """DDPG head that freezes neutral residual outputs while alpha is zero."""

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
        # TanhDelta maps a raw value of zero to the bounded action midpoint.
        # The binary mask also removes every actor gradient through residual
        # dimensions at alpha=0. Any positive authority trains the full actor.
        residual_train_mask = (self.scratch_wr_alpha > 0).to(
            dtype=param.dtype,
            device=param.device,
        )
        residual_param = param[..., self.wave_action_size:] * residual_train_mask
        return torch.cat((param[..., :self.wave_action_size], residual_param), dim=-1)


class ScratchWRExplorationActionMask(torch.nn.Module):
    """Keep DDPG replay actions neutral in residual dimensions at alpha=0."""

    def __init__(
        self,
        residual_neutral_action: torch.Tensor,
        wave_action_size: int = 6,
        alpha: float = 0.0,
    ):
        super().__init__()
        self.wave_action_size = int(wave_action_size)
        self.register_buffer(
            "residual_neutral_action",
            torch.as_tensor(residual_neutral_action, dtype=torch.float32).clone(),
            persistent=False,
        )
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

    def forward(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-1] <= self.wave_action_size:
            raise ValueError("Scratch-WR exploration action does not contain residual dimensions.")
        residual_active = (self.scratch_wr_alpha > 0).to(
            dtype=action.dtype,
            device=action.device,
        )
        neutral = self.residual_neutral_action.to(
            dtype=action.dtype,
            device=action.device,
        )
        residual_action = (
            action[..., self.wave_action_size:] * residual_active
            + neutral * (1.0 - residual_active)
        )
        return torch.cat((action[..., :self.wave_action_size], residual_action), dim=-1)


class ScratchWRNormalizedAdditiveGaussianModule(AdditiveGaussianModule):
    """Add Gaussian noise in normalized [-1, 1] action coordinates."""

    def __init__(
        self,
        *,
        action_low: torch.Tensor,
        action_high: torch.Tensor,
        **kwargs,
    ):
        super().__init__(**kwargs)
        action_low = torch.as_tensor(action_low, dtype=torch.float32).clone()
        action_high = torch.as_tensor(action_high, dtype=torch.float32).clone()
        if action_low.shape != action_high.shape:
            raise ValueError("Scratch-WR exploration action bounds must have equal shapes.")
        if not bool(torch.isfinite(action_low).all() and torch.isfinite(action_high).all()):
            raise ValueError("Scratch-WR exploration action bounds must be finite.")
        if not bool((action_high > action_low).all()):
            raise ValueError("Scratch-WR exploration action bounds must have positive spans.")
        self.register_buffer("action_low", action_low, persistent=False)
        self.register_buffer("action_high", action_high, persistent=False)

    def _add_noise(self, action: torch.Tensor) -> torch.Tensor:
        low = self.action_low.to(dtype=action.dtype, device=action.device)
        high = self.action_high.to(dtype=action.dtype, device=action.device)
        half_span = (high - low) * 0.5
        # sigma is therefore directly interpretable in normalized [-1, 1]
        # coordinates, independent of heterogeneous physical action ranges.
        noisy_action = action + torch.randn_like(action) * self.sigma * half_span
        return torch.minimum(torch.maximum(noisy_action, low), high)


class ScratchWRNormalizedCriticInput(torch.nn.Module):
    """Concatenate observations with actions normalized to [-1, 1]."""

    def __init__(self, action_low: torch.Tensor, action_high: torch.Tensor):
        super().__init__()
        action_low = torch.as_tensor(action_low, dtype=torch.float32).clone()
        action_high = torch.as_tensor(action_high, dtype=torch.float32).clone()
        if action_low.shape != action_high.shape:
            raise ValueError("Scratch-WR critic action bounds must have equal shapes.")
        if not bool(torch.isfinite(action_low).all() and torch.isfinite(action_high).all()):
            raise ValueError("Scratch-WR critic action bounds must be finite.")
        if not bool((action_high > action_low).all()):
            raise ValueError("Scratch-WR critic action bounds must have positive spans.")
        self.register_buffer("action_low", action_low, persistent=False)
        self.register_buffer("action_high", action_high, persistent=False)

    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        low = self.action_low.to(dtype=action.dtype, device=action.device)
        high = self.action_high.to(dtype=action.dtype, device=action.device)
        normalized_action = (2.0 * (action - low) / (high - low) - 1.0).clamp(
            -1.0,
            1.0,
        )
        return torch.cat((observation, normalized_action), dim=-1)


def set_scratch_wr_policy_alpha(policy: torch.nn.Module, alpha: float) -> None:
    """Update every alpha-aware Scratch-WR head/mask embedded in a policy."""
    found = False
    for module in policy.modules():
        if isinstance(
            module,
            (
                ScratchWRNormalParamExtractor,
                ScratchWRDeterministicParamExtractor,
                ScratchWRExplorationActionMask,
            ),
        ):
            module.set_alpha(alpha)
            found = True
    if not found:
        raise RuntimeError("Scratch-WR alpha-aware head was not found in the policy.")


def algorithm_family(algorithm: str) -> str:
    if algorithm in PPO_ALGORITHMS:
        return "ppo"
    if algorithm in DDPG_ALGORITHMS:
        return "ddpg"
    raise ValueError(f"Unsupported algorithm: {algorithm!r}")


def uses_ppo_update_clip(algorithm: str) -> bool:
    return algorithm == "ppo"


def uses_ddpg_policy_update_clip(algorithm: str) -> bool:
    return algorithm == "ddpg_clip"


def parameter_update_snapshot(params) -> list[tuple[torch.nn.Parameter, torch.Tensor]]:
    return [(p, p.detach().clone()) for p in params if p.requires_grad]


def parameter_displacement_norm(
    snapshot: list[tuple[torch.nn.Parameter, torch.Tensor]],
) -> float:
    """Return the absolute L2 displacement since a parameter snapshot."""
    if not snapshot:
        return 0.0
    squared_norm = torch.zeros((), device=snapshot[0][0].device)
    with torch.no_grad():
        for parameter, before in snapshot:
            squared_norm = squared_norm + torch.sum((parameter - before) ** 2)
    return float(torch.sqrt(squared_norm).detach().cpu().item())


def require_finite_tensor(value: torch.Tensor, label: str) -> None:
    """Fail before an optimizer step if a monitored tensor is non-finite."""
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"Non-finite Scratch-WR DDPG {label}.")


def require_finite_scalar(value: float, label: str) -> None:
    """Fail before continuing if a monitored scalar is non-finite."""
    if not np.isfinite(value):
        raise FloatingPointError(f"Non-finite Scratch-WR DDPG {label}.")


def clip_parameter_update(
    snapshot: list[tuple[torch.nn.Parameter, torch.Tensor]],
    max_relative_update: float,
) -> tuple[float, float]:
    """Limit one optimizer step by parameter-space relative displacement."""
    if not snapshot or max_relative_update <= 0:
        return 0.0, 1.0

    update_sq = torch.zeros((), device=snapshot[0][0].device)
    param_sq = torch.zeros((), device=snapshot[0][0].device)
    for param, old_value in snapshot:
        update = param.detach() - old_value
        update_sq = update_sq + update.pow(2).sum()
        param_sq = param_sq + old_value.pow(2).sum()

    update_norm = torch.sqrt(update_sq)
    param_norm = torch.sqrt(param_sq)
    max_update_norm = float(max_relative_update) * torch.clamp(param_norm, min=1.0)
    if update_norm <= max_update_norm or update_norm <= 0:
        return float(update_norm.item()), 1.0

    scale = max_update_norm / (update_norm + 1e-12)
    for param, old_value in snapshot:
        param.data.copy_(old_value + scale * (param.data - old_value))
    return float(update_norm.item()), float(scale.item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train metamaterial locomotion policy.")

    parser.add_argument("--robot", choices=["crawler", "ring"], default="crawler", help="Robot morphology.")
    parser.add_argument("--terrain", choices=["flat", "stairs", "tunnel"], default="flat", help="Training terrain.")
    parser.add_argument(
        "--terrain-contact-mode",
        choices=TERRAIN_CONTACT_MODE_CHOICES,
        default=DEFAULT_TERRAIN_CONTACT_MODE,
        help=(
            "Terrain contact implementation. 'legacy_flat' preserves historical checkpoints; "
            "'mesh_v1' enables mesh-aware terrain contact."
        ),
    )
    parser.add_argument("--num-particles", type=int, default=13, help="Number of particles/nodes.")
    parser.add_argument(
        "--channel",
        default="obs",
        help=(
            "Main channel preset. 'dth': paper observation [dtheta_prev,dtheta_next] with direct torque. "
            "'thdot': paper observation [dtheta_prev,dtheta_next,theta_dot] with direct torque. "
            "'obs': dtheta_next-dtheta_prev with direct torque. "
            "'action': policy outputs [k1,k2], and the simulator applies "
            "tau=k1*(dtheta_next-dtheta_prev)+k2*theta_dot with F fixed to 1. "
            "'paper': policy outputs kappa_alpha for the constrained non-reciprocity formula. "
            "'k2_positive'/'k2_negative': k1 is fixed and only signed k2 is learned. "
            "'tail_wave': one global policy outputs six tail-to-head curl-wave parameters. "
            "'tail_wave_residual': strict-scratch global policy outputs six wave parameters "
            "plus one residual K1/K2 pair per controlled joint. "
            "Aliases: paper_dth=dth, paper_thdot=thdot, theta=obs, formula=action."
        ),
    )
    parser.add_argument(
        "--observation-func",
        default=None,
        help="Advanced override for the raw observation function name. Overrides --channel when set.",
    )
    parser.add_argument(
        "--control-mode",
        choices=["auto", *CONTROL_MODE_CHOICES],
        default="auto",
        help=(
            "Advanced override. 'auto' uses the mode implied by --channel: "
            "dth/thdot/obs -> direct torque, action -> formula control, paper -> nonreciprocity, "
            "and the signed-k2 channels -> fixed-k1 control."
        ),
    )
    parser.add_argument("--feedback-gain", type=float, default=1.0, help="Compatibility option. F is fixed to 1.0 in every experiment; any other value is rejected.")
    parser.add_argument(
        "--max-control-gain",
        "--coefficient-limit",
        dest="max_control_gain",
        type=float,
        default=9.0,
        help=(
            "Default finite gain used by nonreciprocity and signed-K2 controls. "
            "In action/formula control, learned K1/K2 are unbounded unless "
            "--k1-min/max or --k2-min/max are provided."
        ),
    )
    parser.add_argument(
        "--fixed-k1",
        type=float,
        default=-5.0,
        help=(
            "K1 value used when K1 is fixed. It is always used by k2_positive/k2_negative, "
            "and is used by action/formula when --fix-k1 is set. In action/formula, passing "
            "--fixed-k1 explicitly also enables --fix-k1 for convenience."
        ),
    )
    parser.add_argument(
        "--fix-k1",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="In action/formula control, remove K1 from the policy output and use --fixed-k1.",
    )
    parser.add_argument(
        "--fixed-k2",
        type=float,
        default=0.0,
        help=(
            "K2 value used when --fix-k2 is set. In action/formula, passing --fixed-k2 "
            "explicitly also enables --fix-k2 for convenience."
        ),
    )
    parser.add_argument(
        "--fix-k2",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="In action/formula control, remove K2 from the policy output and use --fixed-k2.",
    )
    parser.add_argument("--k1-min", type=float, default=None, help="Lower bound for learned K1 in action/formula control.")
    parser.add_argument("--k1-max", type=float, default=None, help="Upper bound for learned K1 in action/formula control.")
    parser.add_argument("--k2-min", type=float, default=None, help="Lower bound for learned K2 in action/formula or signed-k2 control.")
    parser.add_argument("--k2-max", type=float, default=None, help="Upper bound for learned K2 in action/formula or signed-k2 control.")
    parser.add_argument(
        "--k-action-scale",
        "--formula-action-scale",
        dest="k_action_scale",
        type=float,
        default=1.0,
        help=(
            "Scale applied inside action/formula control: K1=s*u1 and K2=s*u2. "
            "The policy action u remains unbounded when K bounds are omitted, "
            "so K1/K2 are still unbounded while exploration starts at a useful magnitude."
        ),
    )
    parser.add_argument(
        "--min-k2-magnitude",
        type=float,
        default=1e-3,
        help="Strict lower bound on |k2| in the signed-k2 channels; must be smaller than --max-control-gain.",
    )
    parser.add_argument(
        "--passive-kappa",
        type=float,
        default=4.0,
        help="Passive torsional stiffness kappa in -kappa*dtheta_i. Default preserves the original simulator value.",
    )

    parser.add_argument(
        "--reward-func",
        choices=[
            "horizontal_speed",
            "rolling_curriculum",
            "obs2_roll_repro_v1",
            "obs2_roll_repro_v2",
            "obs2_roll_repro_v2_1",
            "tail_roll_curriculum",
            "fast_rollover",
            "fast_forward_roll_v2",
            "scratch_wr_fast_forward_v2",
        ],
        default="horizontal_speed",
        help="Reward preset. The default preserves the original horizontal-speed objective.",
    )
    parser.add_argument(
        "--rolling-observation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Append normalized COM speed, body angular velocity, closure, circularity, and joint phase encoding "
            "to every local actor observation. Disabled by default for checkpoint compatibility."
        ),
    )
    parser.add_argument("--rolling-direction", choices=["right", "left"], default="right")
    parser.add_argument("--rolling-curl-episodes", type=int, default=500)
    parser.add_argument("--rolling-transition-episodes", type=int, default=300)
    parser.add_argument("--rolling-speed-ref-x100", type=float, default=2.0)
    parser.add_argument("--rolling-omega-ref", type=float, default=1.0)
    parser.add_argument("--rolling-reward-scale", type=float, default=3.0)
    parser.add_argument("--init-pos-randomness", type=float, default=0.01)
    parser.add_argument("--init-angle-range-degrees", type=float, default=0.0)
    parser.add_argument("--init-height-jitter", type=float, default=0.0)
    parser.add_argument(
        "--action-smoothness-weight",
        type=float,
        default=0.0,
        help=(
            "Penalty weight for the normalized squared change between consecutive actions. "
            "The default 0 preserves every existing reward exactly."
        ),
    )
    parser.add_argument(
        "--scratch-wr-alpha",
        type=float,
        default=0.0,
        help="Residual torque authority in tail_wave_residual control; must be in [0,1].",
    )
    parser.add_argument(
        "--scratch-wr-v2",
        action="store_true",
        help=(
            "Enable the opt-in, strictly-from-zero Scratch-WR-v2 controller, "
            "synchronized Z0 shaping, and observable temporal wave filter."
        ),
    )
    parser.add_argument("--scratch-wr-v2-sync-dense-weight", type=float, default=0.02)
    parser.add_argument("--scratch-wr-v2-penalty-start-scale", type=float, default=0.20)
    parser.add_argument("--scratch-wr-v2-penalty-anneal-batches", type=int, default=200)
    parser.add_argument("--scratch-wr-v2-wave-ema-beta", type=float, default=0.90)
    parser.add_argument(
        "--scratch-wr-stage-id",
        default="Z0",
        help="Auditable Scratch-WR curriculum stage label written to metadata and training_log.csv.",
    )
    parser.add_argument(
        "--scratch-wr-control-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON IPC file with stage_id, alpha, stop_requested and evaluated_batch. "
            "It is read only at collector-batch boundaries."
        ),
    )
    parser.add_argument(
        "--scratch-wr-eval-sync-every",
        type=int,
        default=0,
        help="Pause after each Nth saved Scratch-WR batch until control-file evaluated_batch acknowledges it.",
    )
    parser.add_argument(
        "--scratch-wr-eval-sync-timeout-minutes",
        type=float,
        default=0.0,
        help="Maximum wait for one Scratch-WR evaluation acknowledgement; zero waits indefinitely.",
    )
    parser.add_argument(
        "--scratch-wr-control-read-retry-seconds",
        type=float,
        default=0.0,
        help=(
            "Bounded retry budget for transient Scratch-WR control-file reads. "
            "The default 0 preserves the legacy fail-fast behaviour."
        ),
    )
    parser.add_argument(
        "--scratch-wr-control-read-retry-initial-ms",
        type=float,
        default=25.0,
        help="Initial transient control-file read retry delay in milliseconds.",
    )
    parser.add_argument(
        "--resume-training-state",
        type=Path,
        default=None,
        help="Resume a full Scratch-WR lineage state (policy, critic, optimizer and RNG); never accepts a plain model checkpoint.",
    )
    parser.add_argument(
        "--tail-roll-observation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append tail/head, ordered-curl, contact-pivot, cumulative-rotation, and stage features.",
    )
    parser.add_argument("--tail-side", choices=["left", "right"], default="left")
    parser.add_argument("--tail-curl-sign", choices=["auto", "-1", "1"], default="auto")
    parser.add_argument("--tail-roll-stage", type=int, choices=[0, 1, 2, 3], default=0)
    parser.add_argument(
        "--tail-roll-auto-curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Promote stages using a rolling competence window. Has no effect for other reward presets.",
    )
    parser.add_argument("--tail-roll-competence-window", type=int, default=50)
    parser.add_argument("--tail-roll-competence-threshold", type=float, default=0.70)
    parser.add_argument("--tail-roll-min-stage-batches", type=int, default=20)
    parser.add_argument("--tail-roll-reward-scale", type=float, default=3.0)
    parser.add_argument(
        "--tail-roll-potential-gamma",
        type=float,
        default=1.0,
        help="Potential-difference factor. 1.0 avoids penalizing a maintained curl pose.",
    )
    parser.add_argument("--tail-roll-contact-margin", type=float, default=0.05)
    parser.add_argument("--tail-roll-curl-reference-degrees", type=float, default=60.0)
    parser.add_argument(
        "--tail-roll-init-assist-degrees",
        type=float,
        default=0.0,
        help="Optional stage-0 tail-up initial bend; zero preserves the original reset.",
    )
    parser.add_argument("--tail-roll-init-assist-segments", type=int, default=4)
    parser.add_argument(
        "--tail-roll-init-assist-episodes",
        type=int,
        default=0,
        help="Linearly anneal the initial bend to zero over this many episodes; zero keeps it constant.",
    )
    parser.add_argument("--fast-rollover-reward-scale", type=float, default=3.0)
    parser.add_argument(
        "--fast-rollover-flip-degrees", type=float, default=60.0,
        help="Minimum desired-direction rotation for one forward-flip event; no full revolution is required.",
    )
    parser.add_argument("--fast-rollover-forward-fraction", type=float, default=0.10)
    parser.add_argument("--fast-rollover-support-fraction", type=float, default=0.02)
    parser.add_argument("--fast-rollover-reset-open-ratio", type=float, default=0.70)
    parser.add_argument("--fast-rollover-cycle-target-steps", type=int, default=250)
    parser.add_argument(
        "--fast-forward-observation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Append the two-state launch/roll phase, event time, progress, and direction features. "
            "Defaults on for fast_forward_roll_v2 and off for every legacy reward."
        ),
    )
    parser.add_argument("--fast-forward-reward-scale", type=float, default=1.0)
    parser.add_argument("--fast-forward-event-degrees", type=float, default=60.0)
    parser.add_argument("--fast-forward-event-forward-fraction", type=float, default=0.08)
    parser.add_argument("--fast-forward-event-contact-nodes", type=float, default=1.5)
    parser.add_argument("--fast-forward-direction-fraction", type=float, default=0.65)
    parser.add_argument("--fast-forward-event-target-steps", type=int, default=250)
    parser.add_argument("--fast-forward-launch-lift", type=float, default=0.20)
    parser.add_argument("--fast-forward-launch-forward", type=float, default=0.10)
    parser.add_argument("--fast-forward-launch-curl", type=float, default=0.12)
    parser.add_argument("--fast-forward-launch-head-contact", type=float, default=0.50)
    parser.add_argument("--fast-forward-launch-hold-steps", type=int, default=8)
    parser.add_argument("--fast-forward-stall-steps", type=int, default=150)
    parser.add_argument("--fast-forward-rotation-step-ref-degrees", type=float, default=2.0)
    parser.add_argument("--fast-forward-translation-step-ref", type=float, default=0.002)

    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--episode-steps", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--frames-per-batch", type=int, default=10_000)
    parser.add_argument("--memory-size", type=int, default=1_000_000)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--optim-steps", type=int, default=10)

    parser.add_argument("--algorithm", choices=SUPPORTED_ALGORITHMS, default="ppo")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--polyak-tau", type=float, default=0.005)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument(
        "--ppo-noclip-epsilon",
        type=float,
        default=1e6,
        help="Large surrogate-ratio range used to approximate PPO without clipped updates.",
    )
    parser.add_argument(
        "--ddpg-policy-update-clip",
        type=float,
        default=0.02,
        help=(
            "For ddpg_clip, cap each actor optimizer step to this fraction of the actor "
            "parameter norm. A value of 0 disables this variant's policy-update clipping."
        ),
    )
    parser.add_argument("--lambda-gae", type=float, default=0.9)
    parser.add_argument("--entropy-eps", type=float, default=1e-4)
    parser.add_argument(
        "--ppo-exact-log-prob",
        action="store_true",
        default=argparse.SUPPRESS,
        help=(
            "Use TorchRL's finite exact TanhNormal log-probability instead of the legacy "
            "joint epsilon floor. Required by Scratch-WR v2; omitted legacy commands are unchanged."
        ),
    )
    parser.add_argument(
        "--ppo-normalize-advantage",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normalize PPO advantages. Off by default to preserve legacy runs.",
    )
    parser.add_argument(
        "--ppo-target-kl",
        type=float,
        default=0.0,
        help="Stop the current PPO minibatch epoch when approximate KL exceeds this value; 0 disables.",
    )
    parser.add_argument("--expl-noise-start", type=float, default=0.9)
    parser.add_argument("--expl-noise-end", type=float, default=0.1)

    parser.add_argument("--policy-depth", type=int, default=2)
    parser.add_argument("--policy-cells", type=int, default=256)
    parser.add_argument(
        "--share-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Share one actor network across all controlled joints (default). "
            "Use --no-share-policy for independent per-joint actors on any channel."
        ),
    )
    parser.add_argument(
        "--per-joint-k1-k2",
        "--joint-specific-k1-k2",
        dest="per_joint_k1_k2",
        action="store_true",
        help=(
            "Action/formula convenience mode: give each controlled joint its own actor "
            "parameters so it can learn a distinct state-dependent K1_i/K2_i mapping. "
            "Equivalent to --no-share-policy for the actor; the critic remains unchanged."
        ),
    )
    parser.add_argument("--share-critic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--centralised-critic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gaussian-activation", action="store_true")
    parser.add_argument("--normal-scale-lb", type=float, default=1e-4)

    parser.add_argument("--start-stairs", type=float, default=5)
    parser.add_argument("--step-width", type=float, default=5)
    parser.add_argument("--step-height", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=10, help="Number of stair steps.")

    parser.add_argument("--tunnel-start", type=float, default=10, help="Tunnel preset: x position where the tunnel obstacle starts.")
    parser.add_argument("--tunnel-slope", type=float, default=5, help="Tunnel preset: horizontal length of the entry/exit ramps.")
    parser.add_argument("--tunnel-slope-height", type=float, default=1, help="Tunnel preset: vertical height of the entry/exit ramps.")
    parser.add_argument("--tunnel-length", type=float, default=10, help="Tunnel preset: horizontal length of the enclosed section.")
    parser.add_argument("--tunnel-height", type=float, default=5, help="Tunnel preset: clearance between floor and ceiling inside the tunnel. Use 2 for a narrower crawler task.")

    parser.add_argument("--buffer-storage", choices=["tensor", "memmap"], default="tensor")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--render", action="store_true", help="Render during training; much slower.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrained-model-path", type=Path, default=None)
    parser.add_argument(
        "--compatible-input-expansion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow an old checkpoint's linear input weights to be zero-padded when new observation "
            "features are appended. Existing columns are copied exactly."
        ),
    )
    parser.add_argument(
        "--pretrained-policy-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Load only actor weights from --pretrained-model-path and leave the critic freshly initialized. "
            "Recommended when changing the reward or appending observations."
        ),
    )
    parser.add_argument(
        "--policy-anchor-coeff",
        type=float,
        default=0.0,
        help="L2 anchor coefficient to the initially loaded actor parameters; 0 disables.",
    )
    parser.add_argument(
        "--policy-anchor-anneal-batches",
        type=int,
        default=0,
        help="Linearly anneal the policy anchor to zero over this many collector batches; 0 keeps it constant.",
    )
    parser.add_argument(
        "--bc-teacher-checkpoint",
        type=Path,
        default=None,
        help="Optional formula/K1K2 checkpoint used to behavior-clone direct torque before RL training.",
    )
    parser.add_argument(
        "--wave-bc-teacher-json",
        type=Path,
        default=None,
        help="Optional open-loop tail-wave parameter-search result used as a behavior-cloning teacher.",
    )
    parser.add_argument("--bc-steps", type=int, default=0, help="Number of deterministic teacher steps collected for behavior cloning.")
    parser.add_argument("--bc-epochs", type=int, default=0, help="Supervised behavior-cloning passes over the collected dataset.")
    parser.add_argument("--bc-batch-size", type=int, default=256)
    parser.add_argument("--bc-lr", type=float, default=1e-4)
    parser.add_argument(
        "--bc-target-scale",
        type=float,
        default=0.1,
        help="Target PPO action standard deviation during behavior cloning; only used when BC is enabled.",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")

    parser.add_argument(
        "--auto-analysis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Automatically create thesis-style analysis figures after training finishes. "
            "Use --no-auto-analysis for very fast smoke tests."
        ),
    )
    parser.add_argument(
        "--analysis-every",
        type=int,
        default=0,
        help=(
            "Also analyze intermediate checkpoints every N episodes. "
            "0 means final checkpoint only; set this equal to --save-every to analyze every saved checkpoint."
        ),
    )
    parser.add_argument(
        "--analysis-terrains",
        nargs="+",
        default=["all"],
        help="Terrains used for automatic evaluation/motion figures: training, flat, stairs, tunnel, or all.",
    )
    parser.add_argument("--analysis-grid-size", type=int, default=81, help="Grid size for automatic policy heatmaps.")
    parser.add_argument("--analysis-theta-dot-slices", type=int, default=9, help="Number of theta-dot slices for formula/action heatmaps.")
    parser.add_argument("--analysis-eval-episodes", type=int, default=3, help="Episodes per terrain in automatic evaluation.")
    parser.add_argument("--analysis-eval-steps", type=int, default=300, help="Steps per episode in automatic evaluation.")
    parser.add_argument("--analysis-motion-steps", type=int, default=300, help="Rollout length for automatic motion-frame figures.")
    parser.add_argument("--analysis-motion-frames", type=int, default=8, help="Number of rendered snapshots per terrain.")
    parser.add_argument("--analysis-dpi", type=int, default=180, help="DPI for saved analysis figures.")
    parser.add_argument("--analysis-no-baseline", action="store_true", help="Skip simple non-reciprocity baseline during automatic analysis.")

    args = parser.parse_args()
    if not np.isclose(args.feedback_gain, 1.0):
        parser.error("--feedback-gain is fixed to 1.0 for this project.")
    if not np.isfinite(args.k_action_scale) or args.k_action_scale <= 0:
        parser.error("--k-action-scale must be a positive finite value.")
    if args.rolling_curl_episodes < 0 or args.rolling_transition_episodes < 0:
        parser.error("--rolling-curl-episodes and --rolling-transition-episodes must be non-negative.")
    if args.rolling_speed_ref_x100 <= 0 or args.rolling_omega_ref <= 0 or args.rolling_reward_scale <= 0:
        parser.error("rolling reference values and reward scale must be positive.")
    if args.init_pos_randomness < 0 or args.init_angle_range_degrees < 0 or args.init_height_jitter < 0:
        parser.error("initial-state randomization values must be non-negative.")
    if not np.isfinite(args.action_smoothness_weight) or args.action_smoothness_weight < 0:
        parser.error("--action-smoothness-weight must be a non-negative finite value.")
    if not np.isfinite(args.scratch_wr_alpha) or not (0.0 <= args.scratch_wr_alpha <= 1.0):
        parser.error("--scratch-wr-alpha must be a finite value in [0, 1].")
    if not np.isfinite(args.scratch_wr_v2_sync_dense_weight) or args.scratch_wr_v2_sync_dense_weight < 0:
        parser.error("--scratch-wr-v2-sync-dense-weight must be non-negative and finite.")
    if not np.isfinite(args.scratch_wr_v2_penalty_start_scale) or not (
        0.0 < args.scratch_wr_v2_penalty_start_scale <= 1.0
    ):
        parser.error("--scratch-wr-v2-penalty-start-scale must be in (0, 1].")
    if args.scratch_wr_v2_penalty_anneal_batches < 0:
        parser.error("--scratch-wr-v2-penalty-anneal-batches must be non-negative.")
    if not np.isfinite(args.scratch_wr_v2_wave_ema_beta) or not (
        0.0 <= args.scratch_wr_v2_wave_ema_beta < 1.0
    ):
        parser.error("--scratch-wr-v2-wave-ema-beta must be in [0, 1).")
    if args.scratch_wr_eval_sync_every < 0:
        parser.error("--scratch-wr-eval-sync-every must be non-negative.")
    if not np.isfinite(args.scratch_wr_eval_sync_timeout_minutes) or args.scratch_wr_eval_sync_timeout_minutes < 0:
        parser.error("--scratch-wr-eval-sync-timeout-minutes must be non-negative and finite.")
    if (
        not np.isfinite(args.scratch_wr_control_read_retry_seconds)
        or args.scratch_wr_control_read_retry_seconds < 0
    ):
        parser.error("--scratch-wr-control-read-retry-seconds must be non-negative and finite.")
    if (
        not np.isfinite(args.scratch_wr_control_read_retry_initial_ms)
        or args.scratch_wr_control_read_retry_initial_ms <= 0
    ):
        parser.error("--scratch-wr-control-read-retry-initial-ms must be positive and finite.")
    if args.tail_roll_competence_window <= 0 or args.tail_roll_min_stage_batches <= 0:
        parser.error("tail-roll competence window and minimum stage batches must be positive.")
    if not (0.0 < args.tail_roll_competence_threshold <= 1.0):
        parser.error("--tail-roll-competence-threshold must be in (0, 1].")
    if args.tail_roll_reward_scale <= 0 or not np.isfinite(args.tail_roll_reward_scale):
        parser.error("--tail-roll-reward-scale must be a positive finite value.")
    if not (0.0 < args.tail_roll_potential_gamma <= 1.0):
        parser.error("--tail-roll-potential-gamma must be in (0, 1].")
    if args.tail_roll_contact_margin < 0 or not np.isfinite(args.tail_roll_contact_margin):
        parser.error("--tail-roll-contact-margin must be a non-negative finite value.")
    if args.tail_roll_curl_reference_degrees <= 0 or not np.isfinite(args.tail_roll_curl_reference_degrees):
        parser.error("--tail-roll-curl-reference-degrees must be a positive finite value.")
    if not np.isfinite(args.tail_roll_init_assist_degrees) or not (0.0 <= args.tail_roll_init_assist_degrees < 180.0):
        parser.error("--tail-roll-init-assist-degrees must be in [0, 180).")
    if args.tail_roll_init_assist_segments < 1:
        parser.error("--tail-roll-init-assist-segments must be positive.")
    if args.tail_roll_init_assist_episodes < 0:
        parser.error("--tail-roll-init-assist-episodes must be non-negative.")
    if args.fast_rollover_reward_scale <= 0 or not np.isfinite(args.fast_rollover_reward_scale):
        parser.error("--fast-rollover-reward-scale must be a positive finite value.")
    if not (0.0 < args.fast_rollover_flip_degrees < 180.0):
        parser.error("--fast-rollover-flip-degrees must be in (0, 180).")
    if args.fast_rollover_forward_fraction < 0 or args.fast_rollover_support_fraction < 0:
        parser.error("fast-rollover displacement thresholds must be non-negative.")
    if not (0.0 < args.fast_rollover_reset_open_ratio <= 1.0):
        parser.error("--fast-rollover-reset-open-ratio must be in (0, 1].")
    if args.fast_rollover_cycle_target_steps <= 0:
        parser.error("--fast-rollover-cycle-target-steps must be positive.")
    if args.fast_forward_observation is None:
        args.fast_forward_observation = args.reward_func in {
            "fast_forward_roll_v2", "scratch_wr_fast_forward_v2"
        }
    if args.fast_forward_reward_scale <= 0 or not np.isfinite(args.fast_forward_reward_scale):
        parser.error("--fast-forward-reward-scale must be a positive finite value.")
    if not (0.0 < args.fast_forward_event_degrees < 180.0):
        parser.error("--fast-forward-event-degrees must be in (0, 180).")
    if args.fast_forward_event_forward_fraction <= 0 or args.fast_forward_event_contact_nodes <= 0:
        parser.error("fast-forward event translation/contact thresholds must be positive.")
    if not (0.5 <= args.fast_forward_direction_fraction <= 1.0):
        parser.error("--fast-forward-direction-fraction must be in [0.5, 1].")
    if (
        args.fast_forward_event_target_steps <= 0
        or args.fast_forward_launch_hold_steps <= 0
        or args.fast_forward_stall_steps <= 0
    ):
        parser.error("fast-forward event/launch-hold/stall step counts must be positive.")
    for option_name, option_value in (
        ("--fast-forward-launch-lift", args.fast_forward_launch_lift),
        ("--fast-forward-launch-forward", args.fast_forward_launch_forward),
        ("--fast-forward-launch-curl", args.fast_forward_launch_curl),
        ("--fast-forward-launch-head-contact", args.fast_forward_launch_head_contact),
    ):
        if not (0.0 <= option_value <= 1.0):
            parser.error(f"{option_name} must be in [0, 1].")
    if args.fast_forward_rotation_step_ref_degrees <= 0 or args.fast_forward_translation_step_ref <= 0:
        parser.error("fast-forward step references must be positive.")
    if args.ppo_target_kl < 0 or not np.isfinite(args.ppo_target_kl):
        parser.error("--ppo-target-kl must be a non-negative finite value.")
    if args.policy_anchor_coeff < 0 or not np.isfinite(args.policy_anchor_coeff):
        parser.error("--policy-anchor-coeff must be a non-negative finite value.")
    if args.policy_anchor_anneal_batches < 0:
        parser.error("--policy-anchor-anneal-batches must be non-negative.")
    if (
        args.reward_func in {
            "tail_roll_curriculum", "fast_rollover", "fast_forward_roll_v2",
            "obs2_roll_repro_v1", "obs2_roll_repro_v2", "obs2_roll_repro_v2_1",
            "scratch_wr_fast_forward_v2",
        }
        or args.fast_forward_observation
    ) and args.robot != "crawler":
        parser.error("tail-first rolling rewards are available only for --robot crawler.")
    if args.bc_steps < 0 or args.bc_epochs < 0 or args.bc_batch_size <= 0 or args.bc_lr <= 0 or args.bc_target_scale <= 0:
        parser.error("BC steps/epochs must be non-negative and BC batch size/lr must be positive.")
    if args.bc_teacher_checkpoint is not None and args.wave_bc_teacher_json is not None:
        parser.error("Use only one behavior-cloning teacher source.")
    has_bc_teacher = args.bc_teacher_checkpoint is not None or args.wave_bc_teacher_json is not None
    if not has_bc_teacher and (args.bc_steps > 0 or args.bc_epochs > 0):
        parser.error("--bc-steps/--bc-epochs require a checkpoint or wave JSON teacher.")
    if has_bc_teacher and (args.bc_steps <= 0 or args.bc_epochs <= 0):
        parser.error("Behavior cloning requires positive --bc-steps and --bc-epochs.")
    args.feedback_gain = 1.0
    argv = sys.argv[1:]
    args.fixed_k1_was_set = any(token == "--fixed-k1" or token.startswith("--fixed-k1=") for token in argv)
    args.fixed_k2_was_set = any(token == "--fixed-k2" or token.startswith("--fixed-k2=") for token in argv)
    args.no_fix_k1_was_set = any(token == "--no-fix-k1" for token in argv)
    args.no_fix_k2_was_set = any(token == "--no-fix-k2" for token in argv)
    share_policy_was_set = any(token == "--share-policy" for token in argv)
    if args.per_joint_k1_k2 and share_policy_was_set:
        parser.error("--per-joint-k1-k2 cannot be combined with --share-policy; omit --share-policy or use --no-share-policy.")
    if args.per_joint_k1_k2:
        try:
            _, _, requested_control_mode = channel_config(
                args.channel,
                observation_func=args.observation_func,
                control_mode=args.control_mode,
            )
        except ValueError as exc:
            parser.error(str(exc))
        if requested_control_mode != "formula":
            parser.error("--per-joint-k1-k2 requires --channel action (or --control-mode formula).")
        args.share_policy = False
    try:
        _, _, requested_control_mode = channel_config(
            args.channel,
            observation_func=args.observation_func,
            control_mode=args.control_mode,
        )
    except ValueError as exc:
        parser.error(str(exc))
    scratch_wr_requested = requested_control_mode == "tail_wave_residual"
    if scratch_wr_requested:
        # The mode never loads another checkpoint, so input-expansion
        # compatibility is disabled explicitly to make metadata unambiguous.
        args.compatible_input_expansion = False
        if args.pretrained_model_path is not None or args.pretrained_policy_only:
            parser.error("tail_wave_residual forbids pretrained policy or model loading.")
        if args.bc_teacher_checkpoint is not None or args.wave_bc_teacher_json is not None:
            parser.error("tail_wave_residual forbids teacher checkpoints and teacher parameter files.")
        if args.bc_steps != 0 or args.bc_epochs != 0:
            parser.error("tail_wave_residual forbids behavior cloning steps and epochs.")
        if args.policy_anchor_coeff != 0 or args.policy_anchor_anneal_batches != 0:
            parser.error("tail_wave_residual forbids policy anchoring.")
        if args.scratch_wr_eval_sync_every > 0 and args.scratch_wr_control_file is None:
            parser.error("--scratch-wr-eval-sync-every requires --scratch-wr-control-file.")
        if (
            algorithm_family(args.algorithm) == "ddpg"
            and args.resume_training_state is not None
        ):
            parser.error(
                "DDPG Scratch-WR resume is intentionally unsupported because replay, "
                "target-network, and exploration-noise state are not serialized; "
                "start a new strict-from-zero lineage instead."
            )
        if args.scratch_wr_v2:
            if args.reward_func != "scratch_wr_fast_forward_v2":
                parser.error("--scratch-wr-v2 requires --reward-func scratch_wr_fast_forward_v2.")
            if (
                algorithm_family(args.algorithm) == "ppo"
                and not getattr(args, "ppo_exact_log_prob", False)
            ):
                parser.error("--scratch-wr-v2 requires --ppo-exact-log-prob.")
        elif args.reward_func == "scratch_wr_fast_forward_v2":
            parser.error("scratch_wr_fast_forward_v2 requires --scratch-wr-v2.")
    elif (
        args.scratch_wr_control_file is not None
        or args.scratch_wr_eval_sync_every > 0
        or args.resume_training_state is not None
    ):
        parser.error("Scratch-WR IPC/resume options require --channel tail_wave_residual.")
    if args.scratch_wr_v2 and not scratch_wr_requested:
        parser.error("--scratch-wr-v2 requires --channel tail_wave_residual.")
    if (
        getattr(args, "ppo_exact_log_prob", False)
        and algorithm_family(args.algorithm) != "ppo"
    ):
        parser.error("--ppo-exact-log-prob is a PPO-only option and must be omitted for DDPG.")
    if getattr(args, "ppo_exact_log_prob", False) and not args.scratch_wr_v2:
        parser.error("--ppo-exact-log-prob is currently restricted to --scratch-wr-v2.")
    return args


def process_batch(batch: TensorDictBase) -> TensorDictBase:
    keys = list(batch.keys(True, True))
    group_shape = batch.get_item_shape("agents")
    for key in ["done", "reward", "terminated"]:
        nested_key = ("next", "agents", key)
        if nested_key not in keys:
            batch.set(nested_key, batch.get(("next", key)).unsqueeze(-1).expand((*group_shape, 1)))
    return batch


def scratch_wr_ddpg_replay_batch(batch: TensorDictBase) -> TensorDictBase:
    """Project a collector batch to only the tensors consumed by DDPGLoss."""
    return batch.select(*SCRATCH_WR_DDPG_REPLAY_KEYS, strict=True)


def replay_transition_bytes(batch: TensorDictBase) -> int:
    """Estimate allocated tensor bytes per replay transition."""
    transition_count = int(batch.numel())
    if transition_count <= 0:
        return 0
    total_bytes = 0
    for key in batch.keys(True, True):
        value = batch.get(key)
        if isinstance(value, torch.Tensor):
            total_bytes += int(value.numel() * value.element_size())
    return int((total_bytes + transition_count - 1) // transition_count)


def save_params(obj: dict, path: Path) -> None:
    serialised = {}
    for k, v in obj.items():
        serialised[k] = v if k == "metadata" else v.state_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(serialised, path)


_SCRATCH_WR_TRANSIENT_WINERRORS = frozenset({5, 32, 33})


class _ScratchWRTransientControlState(RuntimeError):
    """A well-formed but stale/conflicting control snapshot that may be replaced."""


def new_scratch_wr_control_cursor() -> dict:
    """Return mutable monotonic-read state shared by all reads in one trainer."""
    return {
        "generation_mode": False,
        "generation": None,
        "digest": None,
    }


def _scratch_wr_control_digest(document: dict) -> str:
    """Hash JSON semantics, independent of whitespace, key order, or an optional digest field."""
    semantic_document = dict(document)
    semantic_document.pop("digest", None)
    try:
        canonical = json.dumps(
            semantic_document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid Scratch-WR control-file JSON values: {exc}") from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _scratch_wr_generation(document: dict) -> int | None:
    if "generation" not in document:
        return None
    generation = document["generation"]
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise RuntimeError("Scratch-WR control-file generation must be a non-negative integer.")
    return generation


def _validate_scratch_wr_control_identity(document: dict) -> None:
    if "schema" in document and document["schema"] != "scratch_wr_control/v1":
        raise RuntimeError(
            "Scratch-WR control-file schema must be scratch_wr_control/v1."
        )
    if "session_id" in document and (
        not isinstance(document["session_id"], str) or not document["session_id"].strip()
    ):
        raise RuntimeError("Scratch-WR control-file session_id must be a non-empty string.")
    if "writer_pid" in document:
        writer_pid = document["writer_pid"]
        if isinstance(writer_pid, bool) or not isinstance(writer_pid, int) or writer_pid <= 0:
            raise RuntimeError("Scratch-WR control-file writer_pid must be a positive integer.")


def _advance_scratch_wr_control_cursor(
    cursor: dict | None,
    *,
    generation: int | None,
    digest: str,
) -> None:
    if cursor is None:
        return
    generation_mode = bool(cursor.get("generation_mode", False))
    if generation is None:
        if generation_mode:
            raise RuntimeError(
                "Scratch-WR control-file generation disappeared after monotonic mode was established."
            )
        return

    if not generation_mode:
        cursor.update(generation_mode=True, generation=generation, digest=digest)
        return

    previous_generation = cursor.get("generation")
    previous_digest = cursor.get("digest")
    if isinstance(previous_generation, bool) or not isinstance(previous_generation, int):
        raise RuntimeError("Scratch-WR control cursor is internally inconsistent.")
    if generation < previous_generation:
        raise _ScratchWRTransientControlState(
            f"Scratch-WR control generation regressed from {previous_generation} to {generation}."
        )
    if generation == previous_generation and digest != previous_digest:
        raise _ScratchWRTransientControlState(
            f"Scratch-WR control generation {generation} was reused with different content."
        )
    if generation > previous_generation:
        cursor.update(generation=generation, digest=digest)


def _is_transient_scratch_wr_control_error(exc: Exception) -> bool:
    if isinstance(exc, (PermissionError, FileNotFoundError, json.JSONDecodeError)):
        return True
    if isinstance(exc, _ScratchWRTransientControlState):
        return True
    return isinstance(exc, OSError) and getattr(exc, "winerror", None) in _SCRATCH_WR_TRANSIENT_WINERRORS


def _normalise_scratch_wr_control_document(
    document: dict,
    *,
    default_stage_id: str,
    default_alpha: float,
    default_learning_rate: float | None,
) -> dict:
    if not isinstance(document, dict):
        raise RuntimeError("Scratch-WR control file must contain one JSON object.")
    if "stage_id" in document and (
        not isinstance(document["stage_id"], str) or not document["stage_id"].strip()
    ):
        raise RuntimeError("Scratch-WR control-file stage_id must be a non-empty string.")
    if "alpha" in document and (
        isinstance(document["alpha"], bool) or not isinstance(document["alpha"], (int, float))
    ):
        raise RuntimeError("Scratch-WR control-file alpha must be numeric.")
    if "stop_requested" in document and not isinstance(document["stop_requested"], bool):
        raise RuntimeError("Scratch-WR control-file stop_requested must be boolean.")
    if "evaluated_batch" in document and (
        isinstance(document["evaluated_batch"], bool)
        or not isinstance(document["evaluated_batch"], int)
    ):
        raise RuntimeError("Scratch-WR control-file evaluated_batch must be an integer.")
    if "pause_reason" in document and document["pause_reason"] is not None and not isinstance(
        document["pause_reason"], str
    ):
        raise RuntimeError("Scratch-WR control-file pause_reason must be null or a string.")
    if "learning_rate" in document and document["learning_rate"] is not None and (
        isinstance(document["learning_rate"], bool)
        or not isinstance(document["learning_rate"], (int, float))
    ):
        raise RuntimeError("Scratch-WR control-file learning_rate must be numeric.")
    state = {
        "stage_id": str(default_stage_id),
        "alpha": float(default_alpha),
        "stop_requested": False,
        "evaluated_batch": -1,
        "pause_reason": None,
        "learning_rate": default_learning_rate,
    }
    state.update({key: document[key] for key in state if key in document})
    try:
        state["stage_id"] = str(state["stage_id"])
        state["alpha"] = float(state["alpha"])
        state["evaluated_batch"] = int(state["evaluated_batch"])
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(f"Invalid Scratch-WR control-file field type: {exc}") from exc
    if not np.isfinite(state["alpha"]) or not (0.0 <= state["alpha"] <= 1.0):
        raise RuntimeError("Scratch-WR control-file alpha must be finite and in [0,1].")
    if not isinstance(state["stop_requested"], bool):
        raise RuntimeError("Scratch-WR control-file stop_requested must be boolean.")
    if state["learning_rate"] is not None:
        try:
            state["learning_rate"] = float(state["learning_rate"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError("Scratch-WR control-file learning_rate must be numeric.") from exc
        if not np.isfinite(state["learning_rate"]) or state["learning_rate"] <= 0:
            raise RuntimeError("Scratch-WR control-file learning_rate must be positive and finite.")
    return state


def read_scratch_wr_control_file(
    path: Path | None,
    *,
    default_stage_id: str,
    default_alpha: float,
    default_learning_rate: float | None = None,
    cursor: dict | None = None,
    retry_seconds: float = 0.0,
    retry_initial_ms: float = 25.0,
) -> dict:
    """Read one Scratch-WR IPC document with bounded transient retry and monotonicity."""
    try:
        retry_seconds = float(retry_seconds)
        retry_initial_ms = float(retry_initial_ms)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Scratch-WR control-file retry settings must be numeric.") from exc
    if not np.isfinite(retry_seconds) or retry_seconds < 0:
        raise ValueError("Scratch-WR control-file retry seconds must be non-negative and finite.")
    if not np.isfinite(retry_initial_ms) or retry_initial_ms <= 0:
        raise ValueError("Scratch-WR control-file initial retry delay must be positive and finite.")

    default_state = _normalise_scratch_wr_control_document(
        {},
        default_stage_id=default_stage_id,
        default_alpha=default_alpha,
        default_learning_rate=default_learning_rate,
    )
    if path is None:
        return default_state
    path = Path(path)
    generation_mode = cursor is not None and bool(cursor.get("generation_mode", False))
    if retry_seconds == 0.0 and not generation_mode and not path.exists():
        # Preserve the legacy optional-control-file behaviour exactly unless a
        # caller explicitly enables retries or has already observed generation.
        return default_state

    started = time.monotonic()
    deadline = started + retry_seconds
    delay_seconds = retry_initial_ms / 1000.0
    attempts = 0
    last_observed_generation = cursor.get("generation") if cursor is not None else None
    while True:
        attempts += 1
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                raise RuntimeError("Scratch-WR control file must contain one JSON object.")
            _validate_scratch_wr_control_identity(document)
            generation = _scratch_wr_generation(document)
            last_observed_generation = generation
            digest = _scratch_wr_control_digest(document)
            state = _normalise_scratch_wr_control_document(
                document,
                default_stage_id=default_stage_id,
                default_alpha=default_alpha,
                default_learning_rate=default_learning_rate,
            )
            _advance_scratch_wr_control_cursor(
                cursor,
                generation=generation,
                digest=digest,
            )
            state["generation"] = generation
            state["control_digest"] = digest
            return state
        except Exception as exc:
            if not _is_transient_scratch_wr_control_error(exc):
                if isinstance(exc, RuntimeError):
                    raise
                raise RuntimeError(f"Cannot read Scratch-WR control file {path}: {exc}") from exc
            remaining = deadline - time.monotonic()
            if retry_seconds == 0.0 or remaining <= 0.0:
                elapsed = time.monotonic() - started
                error_type = type(exc).__name__
                errno_value = getattr(exc, "errno", None)
                winerror_value = getattr(exc, "winerror", None)
                raise RuntimeError(
                    f"Cannot read a stable Scratch-WR control file {path} "
                    f"after {attempts} attempts/{elapsed:.3f}s "
                    f"(budget={retry_seconds:g}s, last_generation={last_observed_generation}, "
                    f"last_error={error_type}, errno={errno_value}, winerror={winerror_value}): {exc}"
                ) from exc
            time.sleep(min(delay_seconds, remaining))
            delay_seconds = min(delay_seconds * 2.0, 1.0)


def wait_for_scratch_wr_evaluation(
    path: Path,
    *,
    current_batch: int,
    stage_id: str,
    alpha: float,
    timeout_minutes: float,
    learning_rate: float,
    control_cursor: dict,
    control_read_retry_seconds: float = 0.0,
    control_read_retry_initial_ms: float = 25.0,
) -> dict:
    """Pause without collecting until the controller acknowledges a checkpoint."""
    started = time.monotonic()
    while True:
        state = read_scratch_wr_control_file(
            path,
            default_stage_id=stage_id,
            default_alpha=alpha,
            default_learning_rate=learning_rate,
            cursor=control_cursor,
            retry_seconds=control_read_retry_seconds,
            retry_initial_ms=control_read_retry_initial_ms,
        )
        if state["stop_requested"] or state["evaluated_batch"] >= current_batch:
            return state
        if timeout_minutes > 0 and time.monotonic() - started > timeout_minutes * 60.0:
            raise TimeoutError(
                f"Scratch-WR evaluation acknowledgement for batch {current_batch} "
                f"did not arrive within {timeout_minutes:g} minutes."
            )
        time.sleep(1.0)


def _optimiser_state_dict(optimiser):
    if isinstance(optimiser, dict):
        return {name: value.state_dict() for name, value in optimiser.items()}
    return optimiser.state_dict()


def _load_optimiser_state_dict(optimiser, state) -> None:
    if isinstance(optimiser, dict):
        if not isinstance(state, dict) or set(state) != set(optimiser):
            raise RuntimeError("Resume optimizer state does not match the current optimizer family.")
        for name, value in optimiser.items():
            value.load_state_dict(state[name])
    else:
        optimiser.load_state_dict(state)


def set_optimiser_learning_rate(optimiser, learning_rate: float) -> None:
    learning_rate = float(learning_rate)
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite.")
    optimisers = optimiser.values() if isinstance(optimiser, dict) else (optimiser,)
    for value in optimisers:
        for param_group in value.param_groups:
            param_group["lr"] = learning_rate


def save_scratch_wr_training_state(
    path: Path,
    *,
    policy,
    critic,
    optimiser,
    metadata: dict,
    current_batch: int,
    stage_id: str,
    alpha: float,
) -> None:
    """Save a full same-lineage boundary state, never a pretrained-model source."""
    payload = {
        "format": "scratch_wr_training_state_v1",
        "policy": policy.state_dict(),
        "critic": critic.state_dict(),
        "optimiser": _optimiser_state_dict(optimiser),
        "metadata": metadata,
        "current_batch": int(current_batch),
        "stage_id": str(stage_id),
        "alpha": float(alpha),
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_random_state": random.getstate(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def read_scratch_wr_training_state(path: Path) -> dict:
    """Read and validate a full-lineage state without mutating live objects."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("format") != "scratch_wr_training_state_v1":
        raise RuntimeError("--resume-training-state requires a Scratch-WR full training-state file.")
    saved_metadata = payload.get("metadata", {})
    if not saved_metadata.get("scratch_wr_random_initialization", False):
        raise RuntimeError("Resume state is not part of a verified random-initialized Scratch-WR lineage.")
    return payload


def apply_scratch_wr_training_state(payload: dict, policy, critic, optimiser) -> None:
    """Apply a previously validated lineage state inside the current process."""
    policy.load_state_dict(payload["policy"])
    critic.load_state_dict(payload["critic"])
    _load_optimiser_state_dict(optimiser, payload["optimiser"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    np.random.set_state(payload["numpy_rng_state"])
    if payload.get("python_random_state") is not None:
        random.setstate(payload["python_random_state"])
    if torch.cuda.is_available() and payload.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])


def load_scratch_wr_training_state(path: Path, policy, critic, optimiser) -> dict:
    """Backward-compatible read-then-apply wrapper used by external tests."""
    payload = read_scratch_wr_training_state(path)
    apply_scratch_wr_training_state(payload, policy, critic, optimiser)
    return payload


def _checkpoint_tensor_for_shape(
    saved_tensor: torch.Tensor,
    expected_tensor: torch.Tensor,
    *,
    allow_input_expansion: bool,
    parameter_label: str,
) -> tuple[torch.Tensor, bool]:
    """Adapt legacy sharing and optionally zero-pad appended input columns."""
    candidate = saved_tensor
    expected_shape = expected_tensor.shape
    expanded_input = False
    if candidate.shape == expected_shape:
        return candidate, expanded_input

    # Existing compatibility: replicate a formerly shared tensor over a new
    # leading agent dimension.  Permit a later input-column expansion too.
    if (
        expected_tensor.ndim == candidate.ndim + 1
        and expected_shape[0] >= 1
        and expected_shape[1:-1] == candidate.shape[:-1]
    ):
        candidate = candidate.unsqueeze(0).repeat(
            [expected_shape[0]] + [1] * candidate.ndim
        )

    # Observation extensions append features at the final input dimension of
    # the first linear layer.  Copy every legacy column exactly and initialise
    # only the new feature weights to zero, preserving the old actor output at
    # load time.  Biases and non-input tensors are never silently reshaped.
    if (
        allow_input_expansion
        and expected_tensor.ndim >= 2
        and candidate.ndim == expected_tensor.ndim
        and candidate.shape[:-1] == expected_shape[:-1]
        and candidate.shape[-1] < expected_shape[-1]
    ):
        expanded = torch.zeros_like(expected_tensor)
        expanded[..., : candidate.shape[-1]].copy_(candidate.to(expanded.device))
        candidate = expanded
        expanded_input = True

    if candidate.shape != expected_shape:
        raise ValueError(
            f"Cannot load parameter {parameter_label}: expected shape {expected_shape}, "
            f"found {saved_tensor.shape}."
        )
    return candidate, expanded_input


def load_params(
    obj: dict,
    path: Path,
    *,
    allow_input_expansion: bool = False,
) -> None:
    saved_params = torch.load(path, map_location="cpu", weights_only=False)
    saved_params.pop("metadata", None)
    for module_name, module in obj.items():
        if module_name not in saved_params:
            raise KeyError(f"Checkpoint {path} has no {module_name!r} state.")
        expected_state_dict = module.state_dict()
        saved_state_dict = saved_params[module_name]
        converted_state_dict = {}
        expanded_parameters = []
        for parameter_name, parameter in expected_state_dict.items():
            if parameter_name not in saved_state_dict:
                raise KeyError(f"Checkpoint is missing {module_name}/{parameter_name}.")
            if isinstance(parameter, torch.Tensor):
                converted, expanded_input = _checkpoint_tensor_for_shape(
                    saved_state_dict[parameter_name],
                    parameter,
                    allow_input_expansion=allow_input_expansion,
                    parameter_label=f"{module_name}/{parameter_name}",
                )
                converted_state_dict[parameter_name] = converted
                if expanded_input:
                    expanded_parameters.append(parameter_name)
            else:
                converted_state_dict[parameter_name] = parameter
        module.load_state_dict(converted_state_dict, strict=True)
        print(f"Loaded {module_name} params from {path}")
        if expanded_parameters:
            print(
                f"Zero-padded appended input columns for {module_name}: "
                + ", ".join(expanded_parameters)
            )


def policy_parameter_anchor_loss(
    policy: torch.nn.Module,
    reference: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Mean squared displacement from the actor parameters captured at start."""
    terms = []
    for parameter_name, parameter in policy.named_parameters():
        if parameter.requires_grad and parameter_name in reference:
            terms.append(torch.mean((parameter - reference[parameter_name]) ** 2))
    if not terms:
        return torch.zeros((), device=next(policy.parameters()).device)
    return torch.stack(terms).mean()



def json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    try:
        return str(obj)
    except Exception:
        return None


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=json_default)


def append_training_log(log_path: Path, row: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode",
        "reward_mean",
        "speed_mean",
        "speed_x100",
        "elapsed_sec",
        "frames_per_batch",
        "algorithm",
        "algorithm_family",
        "robot",
        "terrain",
        "channel",
        "actor_update_norm_mean",
        "actor_update_clip_scale_mean",
        "ddpg_loss_actor_mean",
        "ddpg_loss_value_mean",
        "ddpg_actor_grad_norm_mean",
        "ddpg_critic_grad_norm_mean",
        "ddpg_actor_parameter_displacement",
        "ddpg_replay_size",
        "ddpg_replay_transition_bytes",
        "ddpg_exploration_sigma",
        "ddpg_pred_q_mean",
        "ddpg_target_q_mean",
        "ddpg_td_error_mean",
        "ddpg_optimizer_updates",
        "ppo_approx_kl",
        "ppo_early_stop",
        "ppo_updates_completed",
        "policy_anchor_loss",
        "policy_anchor_coeff_effective",
        "closure_score",
        "circularity_score",
        "body_omega",
        "slip_penalty",
        "action_smoothness_penalty",
        "curriculum_progress",
        "rolling_reward",
        "tail_lift_score",
        "tail_forward_score",
        "head_contact_score",
        "curl_prefix_progress",
        "curl_order_penalty",
        "total_signed_curvature",
        "closure_ratio",
        "support_margin",
        "cumulative_rotation",
        "rolling_gate",
        "tail_roll_stage",
        "tail_stage_success",
        "tail_stage_success_rate",
        "tail_roll_reward",
        "fast_roll_phase",
        "fast_roll_phase_steps",
        "fast_roll_phase_progress",
        "fast_roll_flip_event",
        "fast_roll_flip_count",
        "fast_roll_cycle_count",
        "fast_roll_cycle_rotation",
        "fast_roll_cycle_forward",
        "fast_roll_support_migration",
        "fast_roll_direction_fraction",
        "fast_roll_motion_gate",
        "fast_roll_reward",
        "fast_forward_phase",
        "fast_forward_phase_steps",
        "fast_forward_launch_progress",
        "fast_forward_launch_ready_steps",
        "fast_forward_roll_progress",
        "fast_forward_progress_delta",
        "fast_forward_launch_event",
        "fast_forward_event_pulse",
        "fast_forward_event_bonus",
        "fast_forward_event_count",
        "fast_forward_event_rotation",
        "fast_forward_event_forward",
        "fast_forward_support_index",
        "fast_forward_support_migration_nodes",
        "fast_forward_ground_contact_strength",
        "fast_forward_event_direction_fraction",
        "fast_forward_episode_direction_fraction",
        "fast_forward_event_steps",
        "fast_forward_progress_age",
        "fast_forward_stall_penalty",
        "fast_forward_reverse_rotation_penalty",
        "fast_forward_backward_penalty",
        "effort_penalty",
        "scratch_wr_v2_z0_lift_ratio",
        "scratch_wr_v2_z0_forward_ratio",
        "scratch_wr_v2_z0_curl_ratio",
        "scratch_wr_v2_z0_sync_score",
        "scratch_wr_v2_z0_candidate_mask",
        "scratch_wr_v2_z0_active",
        "scratch_wr_v2_z0_penalty_scale",
        "scratch_wr_v2_z0_dense_reward",
        "scratch_wr_v2_progress_delta",
        "fast_forward_reward",
        "wave_amplitude",
        "wave_center",
        "wave_width",
        "wave_hold",
        "wave_kp",
        "wave_kd",
        "scratch_wr_stage_id",
        "scratch_wr_learning_rate",
        "scratch_wr_alpha",
        "scratch_wr_wave_torque_rms",
        "scratch_wr_residual_torque_rms",
        "scratch_wr_applied_residual_torque_rms",
        "scratch_wr_total_torque_rms",
        "scratch_wr_torque_clip_fraction",
        "scratch_wr_residual_saturation_fraction",
        "scratch_wr_v2_wave_ema_beta",
        "scratch_wr_v2_wave_filter_delta_rms",
        "scratch_wr_v2_applied_wave_amplitude",
        "scratch_wr_v2_applied_wave_center",
        "scratch_wr_v2_applied_wave_width",
        "scratch_wr_v2_applied_wave_hold",
        "scratch_wr_v2_applied_wave_kp",
        "scratch_wr_v2_applied_wave_kd",
    ]
    file_exists = log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def coefficient_value_token(value: float | int | None) -> str:
    if value is None:
        return "auto"
    value = float(value)
    if not np.isfinite(value):
        return "neg_inf" if value < 0 else "pos_inf"
    if np.isclose(value, 0.0):
        return "0"
    sign = "neg" if value < 0 else "pos"
    magnitude = abs(value)
    if np.isclose(magnitude, round(magnitude)):
        text = str(int(round(magnitude)))
    else:
        text = f"{magnitude:g}".replace(".", "p")
    return f"{sign}{text}"


def coefficient_range_token(min_value: float | int | None, max_value: float | int | None) -> str:
    return f"{coefficient_value_token(min_value)}_to_{coefficient_value_token(max_value)}"


def effective_k1_range(args: argparse.Namespace) -> tuple[float, float]:
    return (
        float(args.k1_min) if args.k1_min is not None else -np.inf,
        float(args.k1_max) if args.k1_max is not None else np.inf,
    )


def effective_k2_range(args: argparse.Namespace, control_mode: str) -> tuple[float, float]:
    if control_mode == "fixed_k1_k2_positive":
        default_min = float(args.min_k2_magnitude)
        default_max = float(args.max_control_gain)
    elif control_mode == "fixed_k1_k2_negative":
        default_min = -float(args.max_control_gain)
        default_max = -float(args.min_k2_magnitude)
    else:
        default_min = -np.inf
        default_max = np.inf
    return (
        float(args.k2_min) if args.k2_min is not None else default_min,
        float(args.k2_max) if args.k2_max is not None else default_max,
    )


def coefficient_name_suffix(
    args: argparse.Namespace,
    control_mode: str,
    *,
    formula_fix_k1: bool,
    formula_fix_k2: bool,
) -> str:
    if control_mode not in {"formula", "fixed_k1_k2_positive", "fixed_k1_k2_negative"}:
        return ""

    if formula_fix_k1 or control_mode in {"fixed_k1_k2_positive", "fixed_k1_k2_negative"}:
        k1_part = f"k1_fixed_{coefficient_value_token(args.fixed_k1)}"
    else:
        k1_min, k1_max = effective_k1_range(args)
        k1_part = f"k1_range_{coefficient_range_token(k1_min, k1_max)}"

    if formula_fix_k2:
        k2_part = f"k2_fixed_{coefficient_value_token(args.fixed_k2)}"
    else:
        k2_min, k2_max = effective_k2_range(args, control_mode)
        k2_part = f"k2_range_{coefficient_range_token(k2_min, k2_max)}"

    return f"{k1_part}_{k2_part}_seed{args.seed}"


def default_run_name(
    args: argparse.Namespace,
    resolved_channel_slug: str,
    control_mode: str,
    *,
    formula_fix_k1: bool,
    formula_fix_k2: bool,
) -> str:
    name = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.robot}_{args.terrain}_"
        f"{resolved_channel_slug}_{args.algorithm}"
    )
    if args.per_joint_k1_k2:
        name = f"{name}_per_joint_k1_k2"
    suffix = coefficient_name_suffix(
        args,
        control_mode,
        formula_fix_k1=formula_fix_k1,
        formula_fix_k2=formula_fix_k2,
    )
    if suffix:
        name = f"{name}_{suffix}"
    return name


def run_auto_analysis(checkpoint_path: Path, save_dir: Path, args: argparse.Namespace, *, final: bool) -> None:
    """Run the automatic analysis suite without stopping training if analysis fails."""
    if not args.auto_analysis:
        return
    try:
        from analyze_training_results import TerrainArgs, analyze_run

        terrain_args = TerrainArgs(
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
        phase = "final" if final else checkpoint_path.stem
        output_dir = save_dir / "analysis" / phase
        print(f"\nRunning automatic analysis for {checkpoint_path} -> {output_dir}")
        saved = analyze_run(
            run_dir=save_dir,
            checkpoint=checkpoint_path,
            output_dir=output_dir,
            terrains=args.analysis_terrains,
            terrain_args=terrain_args,
            policy_mode="deterministic",
            grid_size=args.analysis_grid_size,
            theta_dot_slices=args.analysis_theta_dot_slices,
            heatmap=True,
            k1_k2_evolution=final,
            equivalent_k_evolution=final,
            motion=True,
            training_curve=True,
            evaluate=True,
            baseline=not args.analysis_no_baseline,
            eval_episodes=args.analysis_eval_episodes,
            eval_steps=args.analysis_eval_steps,
            motion_steps=args.analysis_motion_steps,
            motion_frames=args.analysis_motion_frames,
            dpi=args.analysis_dpi,
        )
        print("Automatic analysis files:")
        for path in saved:
            print(" ", path)
    except Exception as exc:
        print(f"WARNING: automatic analysis failed for {checkpoint_path}: {exc}")


def _deterministic_exploration_type():
    for name in ("DETERMINISTIC", "MEAN", "MODE"):
        if hasattr(ExplorationType, name):
            return getattr(ExplorationType, name)
    raise RuntimeError("This TorchRL version does not expose a deterministic exploration type.")


def _teacher_state_geometry(teacher_env) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return thdot observations plus formula-control state terms."""
    pos = np.ascontiguousarray(teacher_env.pos, dtype=np.complex64)
    thdot = np.ascontiguousarray(teacher_env.thdot, dtype=np.float32)
    dp = np.roll(pos, -1, axis=1) - pos
    dp_norm = dp / np.maximum(np.abs(dp), np.float32(1e-8))
    internode = np.angle(-dp_norm / np.roll(dp_norm, 1, axis=1)) % np.float32(2.0 * np.pi)
    dth = np.asarray(internode - np.float32(teacher_env.angle_eq), dtype=np.float32)
    if teacher_env.material_shape == "crawler":
        dth[:, 0] = 0.0
        dth[:, -1] = 0.0
    dth_next = np.roll(dth, -1, axis=1)
    dth_previous = np.roll(dth, 1, axis=1)
    dth_tot = dth_next - dth_previous
    student_obs = np.stack((dth_next, dth_previous, thdot), axis=2)
    if teacher_env.material_shape == "crawler":
        student_obs = student_obs[:, 1:-1, :]
        dth_tot = dth_tot[:, 1:-1]
        controlled_thdot = thdot[:, 1:-1]
    else:
        controlled_thdot = thdot
    return (
        np.ascontiguousarray(student_obs, dtype=np.float32),
        np.ascontiguousarray(dth_tot, dtype=np.float32),
        np.ascontiguousarray(controlled_thdot, dtype=np.float32),
    )


def _formula_teacher_torque(
    teacher_env,
    teacher_metadata: dict,
    teacher_action: torch.Tensor,
    dth_tot: np.ndarray,
    controlled_thdot: np.ndarray,
) -> np.ndarray:
    action = np.asarray(teacher_action.detach().cpu().numpy(), dtype=np.float32)
    if action.ndim == 2:
        action = action[:, :, None]
    names = list(teacher_metadata.get("formula_action_names", ["k1", "k2"]))
    scale = np.float32(teacher_metadata.get("k_action_scale", teacher_metadata.get("formula_action_scale", 1.0)))
    if bool(teacher_metadata.get("formula_fix_k1", teacher_metadata.get("fix_k1", False))):
        k1 = np.full_like(dth_tot, np.float32(teacher_metadata.get("fixed_k1", -5.0)))
    else:
        k1 = action[:, :, names.index("k1")] * scale
    if bool(teacher_metadata.get("formula_fix_k2", teacher_metadata.get("fix_k2", False))):
        k2 = np.full_like(dth_tot, np.float32(teacher_metadata.get("fixed_k2", 0.0)))
    else:
        k2 = action[:, :, names.index("k2")] * scale
    torque = np.clip(k1 * dth_tot + k2 * controlled_thdot, -teacher_env.max_torque, teacher_env.max_torque)
    return np.ascontiguousarray(torque[:, :, None], dtype=np.float32)


def collect_behavior_cloning_dataset(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Replay a formula teacher and expose its physical torques to a thdot student."""
    from analyze_training_results import TerrainArgs, build_demo_env, load_policy_for_env, metadata_from_checkpoint
    from demo_metamaterial import choose_action

    checkpoint = args.bc_teacher_checkpoint.resolve()
    teacher_metadata = metadata_from_checkpoint(checkpoint)
    if str(teacher_metadata.get("control_mode", "")) != "formula":
        raise ValueError("Behavior-cloning teacher must use formula/K1K2 control.")
    teacher_env, _, _, _ = build_demo_env(
        teacher_metadata,
        "flat",
        TerrainArgs(),
        max_steps=min(1000, max(1, args.bc_steps)),
        render_mode="rgb_array",
        num_envs=1,
    )
    teacher_policy = load_policy_for_env(checkpoint, teacher_env, teacher_metadata)

    observations: list[np.ndarray] = []
    torques: list[np.ndarray] = []
    td = teacher_env.reset()
    episode_step = 0
    for _ in range(args.bc_steps):
        student_obs, dth_tot, controlled_thdot = _teacher_state_geometry(teacher_env)
        teacher_td = choose_action(teacher_policy, td, "deterministic")
        target_torque = _formula_teacher_torque(
            teacher_env,
            teacher_metadata,
            teacher_td[teacher_env.action_key],
            dth_tot,
            controlled_thdot,
        )
        observations.append(student_obs[0])
        torques.append(target_torque[0])
        td = teacher_env.step(teacher_td)["next"]
        episode_step += 1
        if episode_step >= teacher_env.max_steps:
            td = teacher_env.reset()
            episode_step = 0

    if hasattr(teacher_env, "close"):
        teacher_env.close()
    observation_tensor = torch.from_numpy(np.stack(observations, axis=0))
    torque_tensor = torch.from_numpy(np.stack(torques, axis=0))
    dataset_summary = {
        "teacher_checkpoint": str(checkpoint),
        "teacher_control_mode": teacher_metadata.get("control_mode"),
        "teacher_channel": teacher_metadata.get("channel"),
        "samples": int(observation_tensor.shape[0]),
        "num_agents": int(observation_tensor.shape[1]),
        "observation_dim": int(observation_tensor.shape[2]),
        "target_torque_abs_mean": float(torque_tensor.abs().mean().item()),
        "target_torque_abs_max": float(torque_tensor.abs().max().item()),
    }
    return observation_tensor, torque_tensor, dataset_summary


def collect_wave_behavior_cloning_dataset(env, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Roll out the best open-loop wave schedule and clone its six parameters."""
    teacher_path = args.wave_bc_teacher_json.resolve()
    payload = json.loads(teacher_path.read_text(encoding="utf-8"))
    teacher = payload.get("teacher", payload.get("best_parameters", payload))
    required = ("amplitude", "center_start", "center_end", "width", "hold", "kp", "kd")
    missing = [name for name in required if name not in teacher]
    if missing:
        raise ValueError(f"Wave teacher JSON is missing: {missing}")

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    td = env.reset()
    episode_step = 0
    travel_steps = max(1, int(teacher.get("travel_steps", args.episode_steps)))
    num_envs = int(td.batch_size[0])
    for _ in range(args.bc_steps):
        progress = np.float32(np.clip(episode_step / travel_steps, 0.0, 1.0))
        smooth_progress = progress * progress * (np.float32(3.0) - np.float32(2.0) * progress)
        center = np.float32(teacher["center_start"]) + smooth_progress * np.float32(
            float(teacher["center_end"]) - float(teacher["center_start"])
        )
        action_np = np.asarray(
            [[[
                teacher["amplitude"], center, teacher["width"], teacher["hold"],
                teacher["kp"], teacher["kd"],
            ]]],
            dtype=np.float32,
        )
        action_np = np.repeat(action_np, num_envs, axis=0)
        observations.append(td[("agents", "observation")].detach().cpu().numpy())
        actions.append(action_np)
        td.set(env.action_key, torch.as_tensor(action_np, dtype=torch.float32, device=env.device))
        td = env.step(td)["next"]
        episode_step += 1
        if episode_step >= args.episode_steps:
            td = env.reset()
            episode_step = 0

    observation_tensor = torch.from_numpy(np.concatenate(observations, axis=0))
    action_tensor = torch.from_numpy(np.concatenate(actions, axis=0))
    summary = {
        "teacher_json": str(teacher_path),
        "teacher_control_mode": "tail_wave",
        "samples": int(observation_tensor.shape[0]),
        "num_agents": int(observation_tensor.shape[1]),
        "observation_dim": int(observation_tensor.shape[2]),
        "teacher": {name: float(teacher[name]) for name in required},
        "travel_steps": travel_steps,
    }
    return observation_tensor, action_tensor, summary


def run_behavior_cloning(policy, env, args: argparse.Namespace, device: torch.device) -> dict | None:
    if args.bc_teacher_checkpoint is None and args.wave_bc_teacher_json is None:
        return None
    if args.wave_bc_teacher_json is not None:
        observations, target_torques, summary = collect_wave_behavior_cloning_dataset(env, args)
    else:
        observations, target_torques, summary = collect_behavior_cloning_dataset(args)
    bc_optimiser = torch.optim.Adam(policy.parameters(), lr=args.bc_lr)
    deterministic = _deterministic_exploration_type()
    losses: list[float] = []
    sample_count = observations.shape[0]
    action_leaf_spec = env.full_action_spec_unbatched[env.action_key]
    action_low = action_leaf_spec.space.low.to(device)
    action_high = action_leaf_spec.space.high.to(device)

    policy.train()
    for _ in range(args.bc_epochs):
        permutation = torch.randperm(sample_count)
        epoch_losses: list[float] = []
        for start in range(0, sample_count, args.bc_batch_size):
            indices = permutation[start : start + args.bc_batch_size]
            obs = observations[indices].to(device)
            targets = target_torques[indices].to(device)
            td = TensorDict({("agents", "observation"): obs}, batch_size=[obs.shape[0]], device=device)
            normalized_targets = 2.0 * (targets - action_low) / (action_high - action_low) - 1.0
            target_loc = torch.atanh(torch.clamp(normalized_targets, -0.999, 0.999))
            loc_td = policy.module[0](td)
            predicted_loc = loc_td[("agents", "loc")]
            predicted_scale = loc_td[("agents", "scale")]
            target_scale = torch.full_like(predicted_scale, args.bc_target_scale)
            loc_loss = torch.nn.functional.mse_loss(predicted_loc, target_loc)
            scale_loss = torch.nn.functional.mse_loss(torch.log(predicted_scale), torch.log(target_scale))
            loss = loc_loss + scale_loss
            bc_optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            bc_optimiser.step()
            epoch_losses.append(float(loss.detach().cpu().item()))
        losses.append(float(np.mean(epoch_losses)))
    policy.eval()

    with torch.no_grad(), set_exploration_type(deterministic):
        eval_count = min(sample_count, 4096)
        eval_obs = observations[:eval_count].to(device)
        eval_td = TensorDict({("agents", "observation"): eval_obs}, batch_size=[eval_count], device=device)
        predictions = policy(eval_td)[env.action_key].cpu()
        mae = float(torch.mean(torch.abs(predictions - target_torques[:eval_count])).item())

    summary.update(
        {
            "epochs": int(args.bc_epochs),
            "batch_size": int(args.bc_batch_size),
            "learning_rate": float(args.bc_lr),
            "target_scale": float(args.bc_target_scale),
            "first_epoch_mse": losses[0],
            "final_epoch_mse": losses[-1],
            "evaluation_mae": mae,
        }
    )
    print("Behavior cloning complete:", summary)
    return summary


def build_components(env, args: argparse.Namespace, device: torch.device):
    family = algorithm_family(args.algorithm)
    scratch_wr_policy = (
        getattr(getattr(env, "base_env", env), "control_mode", None)
        == "tail_wave_residual"
    )
    policy_net_config = {"depth": args.policy_depth, "num_cells": args.policy_cells}
    critic_net_config = dict(policy_net_config)

    policy_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1],
        n_agent_outputs=env.full_action_spec[env.action_key].shape[-1] * (2 if family == "ppo" else 1),
        n_agents=env.num_agents,
        centralised=False,
        share_params=args.share_policy,
        device=device,
        activation_class=FirstOrderGaussian if args.gaussian_activation else torch.nn.Tanh,
        **policy_net_config,
    )
    if family == "ppo":
        if scratch_wr_policy:
            param_extractor = ScratchWRNormalParamExtractor(
                scale_lb=args.normal_scale_lb,
                wave_action_size=6,
                alpha=args.scratch_wr_alpha,
            )
        else:
            param_extractor = BiasedNormalParamExtractor(scale_lb=args.normal_scale_lb)
        policy_net = torch.nn.Sequential(policy_net, param_extractor)
    elif scratch_wr_policy:
        policy_net = torch.nn.Sequential(
            policy_net,
            ScratchWRDeterministicParamExtractor(
                wave_action_size=6,
                alpha=args.scratch_wr_alpha,
            ),
        )

    temp_keys = [("agents", "loc"), ("agents", "scale")] if family == "ppo" else [("agents", "param")]
    policy_module = TensorDictModule(policy_net, in_keys=[("agents", "observation")], out_keys=temp_keys)

    action_spec = env.action_spec_unbatched.to(device) if hasattr(env.action_spec_unbatched, "to") else env.action_spec_unbatched
    action_leaf_spec = env.full_action_spec_unbatched[env.action_key]
    action_low = action_leaf_spec.space.low.to(device)
    action_high = action_leaf_spec.space.high.to(device)
    bounded_action_spec = type(action_leaf_spec).__name__.startswith("Bounded")
    finite_action_bounds = bounded_action_spec and bool(torch.isfinite(action_low).all().item() and torch.isfinite(action_high).all().item())
    if finite_action_bounds:
        if family == "ppo":
            distribution_class = (
                TanhNormal
                if getattr(args, "ppo_exact_log_prob", False)
                else SafeTanhNormal
            )
        else:
            distribution_class = TanhDelta
        distribution_kwargs = {"low": action_low, "high": action_high}
    else:
        distribution_class = IndependentNormal if family == "ppo" else Delta
        distribution_kwargs = {}

    policy = ProbabilisticActor(
        module=policy_module,
        spec=action_spec,
        in_keys=temp_keys,
        out_keys=[env.action_key],
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=family == "ppo",
    )

    if family == "ddpg":
        if scratch_wr_policy:
            exploration_noise_module = ScratchWRNormalizedAdditiveGaussianModule(
                action_low=action_low,
                action_high=action_high,
                spec=policy.spec,
                annealing_num_steps=args.frames_per_batch * args.episodes // 2,
                action_key=env.action_key,
                sigma_init=args.expl_noise_start,
                sigma_end=args.expl_noise_end,
            )
        else:
            # Preserve the physical-unit noise used by every legacy DDPG run.
            exploration_noise_module = AdditiveGaussianModule(
                spec=policy.spec,
                annealing_num_steps=args.frames_per_batch * args.episodes // 2,
                action_key=("agents", "action"),
                sigma_init=args.expl_noise_start,
                sigma_end=args.expl_noise_end,
            )
        exploration_modules = [
            policy,
            exploration_noise_module,
        ]
        if scratch_wr_policy:
            residual_neutral_action = (
                action_low[..., 6:] + action_high[..., 6:]
            ) * 0.5
            exploration_modules.append(
                TensorDictModule(
                    ScratchWRExplorationActionMask(
                        residual_neutral_action=residual_neutral_action,
                        wave_action_size=6,
                        alpha=args.scratch_wr_alpha,
                    ),
                    in_keys=[env.action_key],
                    out_keys=[env.action_key],
                )
            )
        exploration_policy = TensorDictSequential(*exploration_modules)
    else:
        exploration_policy = policy

    critic_value_type = "state_action_value" if family == "ddpg" else "state_value"
    critic_net = MultiAgentMLP(
        n_agent_inputs=env.observation_spec["agents", "observation"].shape[-1]
        + (env.full_action_spec["agents", "action"].shape[-1] if critic_value_type == "state_action_value" else 0),
        n_agent_outputs=1,
        n_agents=env.num_agents,
        centralised=args.centralised_critic,
        share_params=args.share_critic,
        device=device,
        activation_class=torch.nn.Tanh,
        **critic_net_config,
    )

    if critic_value_type == "state_action_value":
        if scratch_wr_policy:
            critic_input_module = ScratchWRNormalizedCriticInput(
                action_low=action_low,
                action_high=action_high,
            )
        else:
            # Preserve raw physical actions for every legacy DDPG critic.
            critic_input_module = lambda obs, action: torch.cat(
                [obs, action],
                dim=-1,
            )
        critic = TensorDictSequential(
            TensorDictModule(
                critic_input_module,
                in_keys=[("agents", "observation"), ("agents", "action")],
                out_keys=[("agents", "obs_action")],
            ),
            TensorDictModule(critic_net, in_keys=[("agents", "obs_action")], out_keys=[("agents", "state_action_value")]),
        )
    else:
        critic = TensorDictModule(critic_net, in_keys=[("agents", "observation")], out_keys=[("agents", "state_value")])

    if args.pretrained_model_path is not None:
        modules_to_load = {"policy": policy}
        if not args.pretrained_policy_only:
            modules_to_load["critic"] = critic
        load_params(
            modules_to_load,
            args.pretrained_model_path,
            allow_input_expansion=args.compatible_input_expansion,
        )

    behavior_cloning_summary = run_behavior_cloning(policy, env, args, device)
    policy_anchor_reference = (
        {
            parameter_name: parameter.detach().clone()
            for parameter_name, parameter in policy.named_parameters()
            if parameter.requires_grad
        }
        if args.policy_anchor_coeff > 0
        else {}
    )

    collector = SyncDataCollector(
        env,
        exploration_policy,
        device=device,
        storing_device=device,
        frames_per_batch=args.frames_per_batch,
        total_frames=args.frames_per_batch * args.episodes,
    )
    _ACTIVE_COLLECTORS.append(collector)

    buffer_sample_with_replacement = family == "ddpg"
    buffer_sampler_class = RandomSampler if buffer_sample_with_replacement else SamplerWithoutReplacement
    buffer_memory_size = args.memory_size if buffer_sample_with_replacement else args.frames_per_batch

    if args.buffer_storage == "tensor":
        replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(buffer_memory_size, device=device),
            sampler=buffer_sampler_class(),
            batch_size=args.minibatch_size,
        )
    else:
        scratch_dir = tempfile.TemporaryDirectory().name
        replay_buffer = ReplayBuffer(
            storage=LazyMemmapStorage(buffer_memory_size, scratch_dir=scratch_dir),
            sampler=buffer_sampler_class(),
            batch_size=args.minibatch_size,
        )
        if device.type != "cpu":
            replay_buffer.append_transform(lambda x: x.to(device))

    if family == "ppo":
        loss_clip_epsilon = args.clip_epsilon if uses_ppo_update_clip(args.algorithm) else args.ppo_noclip_epsilon
        loss_module = ClipPPOLoss(
            actor_network=policy,
            critic_network=critic,
            clip_epsilon=loss_clip_epsilon,
            entropy_coeff=args.entropy_eps,
            normalize_advantage=args.ppo_normalize_advantage,
        )
        loss_module.set_keys(
            reward=("agents", "reward"),
            action=env.action_key,
            value=("agents", "state_value"),
            done=("agents", "done"),
            terminated=("agents", "terminated"),
        )
        loss_module.make_value_estimator(ValueEstimators.GAE, gamma=args.gamma, lmbda=args.lambda_gae)
        value_estimator = loss_module.value_estimator
        optimiser = torch.optim.Adam(loss_module.parameters(), args.lr, weight_decay=args.weight_decay)
        target_updater = None
    else:
        loss_module = DDPGLoss(
            actor_network=policy,
            value_network=critic,
            delay_actor=scratch_wr_policy,
            delay_value=True,
            loss_function="l2",
        )
        loss_module.set_keys(
            state_action_value=("agents", "state_action_value"),
            reward=("agents", "reward"),
            done=("agents", "done"),
            terminated=("agents", "terminated"),
        )
        loss_module.make_value_estimator(ValueEstimators.TD0, gamma=args.gamma)
        value_estimator = None
        target_updater = SoftUpdate(loss_module, tau=args.polyak_tau)
        optimiser = {
            "loss_actor": torch.optim.Adam(loss_module.actor_network_params.flatten_keys().values(), lr=args.lr, weight_decay=args.weight_decay),
            "loss_value": torch.optim.Adam(loss_module.value_network_params.flatten_keys().values(), lr=args.lr, weight_decay=args.weight_decay),
        }

    return (
        policy,
        exploration_policy,
        critic,
        collector,
        replay_buffer,
        loss_module,
        optimiser,
        value_estimator,
        target_updater,
        behavior_cloning_summary,
        policy_anchor_reference,
    )


def _main_impl() -> None:
    args = parse_args()
    family = algorithm_family(args.algorithm)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
    canonical_channel, observation_func, control_mode = channel_config(
        args.channel,
        observation_func=args.observation_func,
        control_mode=args.control_mode,
    )
    scratch_wr_ddpg = (
        control_mode == "tail_wave_residual" and family == "ddpg"
    )
    scratch_wr_control = None
    scratch_wr_control_cursor = new_scratch_wr_control_cursor()
    if control_mode == "tail_wave_residual":
        scratch_wr_control = read_scratch_wr_control_file(
            args.scratch_wr_control_file,
            default_stage_id=args.scratch_wr_stage_id,
            default_alpha=args.scratch_wr_alpha,
            default_learning_rate=args.lr,
            cursor=scratch_wr_control_cursor,
            retry_seconds=args.scratch_wr_control_read_retry_seconds,
            retry_initial_ms=args.scratch_wr_control_read_retry_initial_ms,
        )
        args.scratch_wr_stage_id = scratch_wr_control["stage_id"]
        args.scratch_wr_alpha = scratch_wr_control["alpha"]
        if scratch_wr_control["learning_rate"] is not None:
            args.lr = scratch_wr_control["learning_rate"]
    if args.bc_teacher_checkpoint is not None and (
        canonical_channel != "thdot" or control_mode != "direct" or observation_func != "dth_neighbours_plus_thdot"
    ):
        raise ValueError("Behavior cloning is supported only for --channel thdot with direct torque control.")
    if args.wave_bc_teacher_json is not None and control_mode != "tail_wave":
        raise ValueError("--wave-bc-teacher-json requires --channel tail_wave or --control-mode tail_wave.")
    formula_fix_k1 = control_mode == "formula" and (
        args.fix_k1 or (args.fixed_k1_was_set and not args.no_fix_k1_was_set)
    )
    formula_fix_k2 = control_mode == "formula" and (
        args.fix_k2 or (args.fixed_k2_was_set and not args.no_fix_k2_was_set)
    )
    per_joint_k1_k2 = control_mode == "formula" and not args.share_policy
    resolved_channel_label = channel_label(canonical_channel, observation_func, control_mode)
    resolved_channel_slug = channel_slug(canonical_channel, observation_func, control_mode)

    run_name = args.run_name or default_run_name(
        args,
        resolved_channel_slug,
        control_mode,
        formula_fix_k1=formula_fix_k1,
        formula_fix_k2=formula_fix_k2,
    )
    save_dir = args.results_dir / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.force_cpu)
    torch.set_default_device("cpu")
    set_composite_lp_aggregate(False).set()

    num_parallel_envs = max(1, args.frames_per_batch // args.episode_steps)
    print("num_parallel_envs:", num_parallel_envs)
    print("robot:", args.robot)
    print("terrain_type:", terrain_type)
    print("terrain_settings:", terrain_settings)
    print("terrain_contact_mode:", args.terrain_contact_mode)
    print("channel:", resolved_channel_label)
    print("observation_func:", observation_func)
    print("control_mode:", control_mode)
    print("algorithm:", args.algorithm)
    print("feedback_gain (fixed):", 1.0)
    print("max_control_gain:", args.max_control_gain)
    print("k1_range:", (args.k1_min, args.k1_max))
    print("k2_range:", (args.k2_min, args.k2_max))
    print("k_action_scale:", args.k_action_scale)
    print("fix_k1:", formula_fix_k1)
    print("fixed_k1:", args.fixed_k1)
    print("fix_k2:", formula_fix_k2)
    print("fixed_k2:", args.fixed_k2)
    print("min_k2_magnitude:", args.min_k2_magnitude)
    print("passive_kappa:", args.passive_kappa)
    print("reward_func:", args.reward_func)
    print("rolling_observation:", args.rolling_observation)
    print("rolling_direction:", args.rolling_direction)
    print("rolling_curriculum:", (args.rolling_curl_episodes, args.rolling_transition_episodes))
    print("rolling_reward_scale:", args.rolling_reward_scale)
    print("action_smoothness_weight:", args.action_smoothness_weight)
    if control_mode == "tail_wave_residual":
        print("scratch_wr_alpha:", args.scratch_wr_alpha)
        print("scratch_wr_stage_id:", args.scratch_wr_stage_id)
        print("scratch_wr_control_file:", args.scratch_wr_control_file)
    print(
        "behavior_cloning:",
        None
        if args.bc_teacher_checkpoint is None and args.wave_bc_teacher_json is None
        else (
            args.bc_teacher_checkpoint or args.wave_bc_teacher_json,
            args.bc_steps, args.bc_epochs, args.bc_batch_size, args.bc_lr,
        ),
    )
    print("initial_state_randomization:", (args.init_pos_randomness, args.init_angle_range_degrees, args.init_height_jitter))
    print("share_policy:", args.share_policy)
    print("policy_parameter_sharing:", "independent_per_joint" if not args.share_policy else "shared")
    print("per_joint_k1_k2:", per_joint_k1_k2)
    print("save_dir:", save_dir)

    base_env = metamaterial.env(
        num_envs=num_parallel_envs,
        material_shape=args.robot,
        num_particles=args.num_particles,
        max_steps=args.episode_steps,
        observation_func=observation_func,
        terrain_type=terrain_type,
        terrain_settings=terrain_settings,
        terrain_contact_mode=args.terrain_contact_mode,
        render=args.render,
        control_mode=control_mode,
        feedback_gain=1.0,
        max_control_gain=args.max_control_gain,
        k1_min=args.k1_min,
        k1_max=args.k1_max,
        k2_min=args.k2_min,
        k2_max=args.k2_max,
        k_action_scale=args.k_action_scale,
        fix_k1=formula_fix_k1,
        fixed_k1=args.fixed_k1,
        fix_k2=formula_fix_k2,
        fixed_k2=args.fixed_k2,
        min_k2_magnitude=args.min_k2_magnitude,
        passive_kappa=args.passive_kappa,
        scratch_wr_alpha=args.scratch_wr_alpha,
        scratch_wr_v2=args.scratch_wr_v2,
        scratch_wr_v2_sync_dense_weight=args.scratch_wr_v2_sync_dense_weight,
        scratch_wr_v2_penalty_start_scale=args.scratch_wr_v2_penalty_start_scale,
        scratch_wr_v2_penalty_anneal_batches=args.scratch_wr_v2_penalty_anneal_batches,
        scratch_wr_v2_wave_ema_beta=args.scratch_wr_v2_wave_ema_beta,
        reward_func=args.reward_func,
        rolling_observation=args.rolling_observation,
        rolling_direction=args.rolling_direction,
        rolling_curl_episodes=args.rolling_curl_episodes,
        rolling_transition_episodes=args.rolling_transition_episodes,
        rolling_speed_ref_x100=args.rolling_speed_ref_x100,
        rolling_omega_ref=args.rolling_omega_ref,
        rolling_reward_scale=args.rolling_reward_scale,
        init_pos_randomness=args.init_pos_randomness,
        init_angle_range_degrees=args.init_angle_range_degrees,
        init_height_jitter=args.init_height_jitter,
        action_smoothness_weight=args.action_smoothness_weight,
        tail_roll_observation=args.tail_roll_observation,
        tail_side=args.tail_side,
        tail_curl_sign=args.tail_curl_sign,
        tail_roll_stage=args.tail_roll_stage,
        tail_roll_reward_scale=args.tail_roll_reward_scale,
        tail_roll_potential_gamma=args.tail_roll_potential_gamma,
        tail_roll_contact_margin=args.tail_roll_contact_margin,
        tail_roll_curl_reference=np.deg2rad(args.tail_roll_curl_reference_degrees),
        tail_roll_init_assist_degrees=args.tail_roll_init_assist_degrees,
        tail_roll_init_assist_segments=args.tail_roll_init_assist_segments,
        tail_roll_init_assist_episodes=args.tail_roll_init_assist_episodes,
        fast_rollover_reward_scale=args.fast_rollover_reward_scale,
        fast_rollover_flip_degrees=args.fast_rollover_flip_degrees,
        fast_rollover_forward_fraction=args.fast_rollover_forward_fraction,
        fast_rollover_support_fraction=args.fast_rollover_support_fraction,
        fast_rollover_reset_open_ratio=args.fast_rollover_reset_open_ratio,
        fast_rollover_cycle_target_steps=args.fast_rollover_cycle_target_steps,
        fast_forward_observation=args.fast_forward_observation,
        fast_forward_reward_scale=args.fast_forward_reward_scale,
        fast_forward_event_degrees=args.fast_forward_event_degrees,
        fast_forward_event_forward_fraction=args.fast_forward_event_forward_fraction,
        fast_forward_event_contact_nodes=args.fast_forward_event_contact_nodes,
        fast_forward_direction_fraction=args.fast_forward_direction_fraction,
        fast_forward_event_target_steps=args.fast_forward_event_target_steps,
        fast_forward_launch_lift=args.fast_forward_launch_lift,
        fast_forward_launch_forward=args.fast_forward_launch_forward,
        fast_forward_launch_curl=args.fast_forward_launch_curl,
        fast_forward_launch_head_contact=args.fast_forward_launch_head_contact,
        fast_forward_launch_hold_steps=args.fast_forward_launch_hold_steps,
        fast_forward_stall_steps=args.fast_forward_stall_steps,
        fast_forward_rotation_step_ref_degrees=args.fast_forward_rotation_step_ref_degrees,
        fast_forward_translation_step_ref=args.fast_forward_translation_step_ref,
    )
    env = TransformedEnv(base_env, RewardSum(in_keys=base_env.reward_keys, reset_keys=["_reset"]))
    check_env_specs(env)
    print("PASSED CHECK!")

    (
        policy,
        exploration_policy,
        critic,
        collector,
        replay_buffer,
        loss_module,
        optimiser,
        value_estimator,
        target_updater,
        behavior_cloning_summary,
        policy_anchor_reference,
    ) = build_components(env, args, device)

    metadata = {
        "scenario": args.robot,
        "robot": args.robot,
        "algorithm": args.algorithm,
        "algorithm_family": family,
        "ppo_clip_enabled": uses_ppo_update_clip(args.algorithm),
        "ppo_loss_clip_epsilon": args.clip_epsilon if uses_ppo_update_clip(args.algorithm) else None,
        "ppo_noclip_epsilon": args.ppo_noclip_epsilon if args.algorithm == "ppo_noclip" else None,
        "ddpg_policy_update_clip_enabled": uses_ddpg_policy_update_clip(args.algorithm),
        "ddpg_policy_update_clip": args.ddpg_policy_update_clip if uses_ddpg_policy_update_clip(args.algorithm) else None,
        "n_particles": args.num_particles,
        "num_particles": args.num_particles,
        "channel": canonical_channel,
        "channel_label": resolved_channel_label,
        "observation_func": observation_func,
        "control_mode": control_mode,
        "control_channel": control_mode,
        "feedback_gain": 1.0,
        "background_friction": float(base_env.background_friction),
        "ground_stiffness": float(base_env.ground_stiffness),
        "ground_damping": float(base_env.ground_damping),
        "max_control_gain": args.max_control_gain,
        "coefficient_limit": args.max_control_gain,
        "k1_min": float(base_env.k1_min),
        "k1_max": float(base_env.k1_max),
        "k2_min": float(base_env.k2_min),
        "k2_max": float(base_env.k2_max),
        "k_action_scale": float(base_env.k_action_scale),
        "formula_action_scale": float(base_env.k_action_scale),
        "fix_k1": formula_fix_k1,
        "formula_fix_k1": formula_fix_k1,
        "fixed_k1": args.fixed_k1,
        "fix_k2": formula_fix_k2,
        "formula_fix_k2": formula_fix_k2,
        "fixed_k2": args.fixed_k2,
        "formula_action_names": list(getattr(base_env, "formula_action_names", [])),
        "tail_wave_action_names": list(getattr(base_env, "tail_wave_action_names", [])),
        "scratch_wr_action_names": list(getattr(base_env, "scratch_wr_action_names", [])),
        "scratch_wr_action_low": getattr(base_env, "scratch_wr_action_low", np.asarray([], dtype=np.float32)).tolist(),
        "scratch_wr_action_high": getattr(base_env, "scratch_wr_action_high", np.asarray([], dtype=np.float32)).tolist(),
        "per_joint_k1_k2": per_joint_k1_k2,
        "policy_parameter_sharing": "independent_per_joint" if not args.share_policy else "shared",
        "num_controlled_joints": int(getattr(base_env, "num_controlled_joints", base_env.num_agents)),
        "min_k2_magnitude": args.min_k2_magnitude,
        "passive_kappa": args.passive_kappa,
        "reward_func": args.reward_func,
        "rolling_observation": args.rolling_observation,
        "rolling_direction": args.rolling_direction,
        "rolling_curl_episodes": args.rolling_curl_episodes,
        "rolling_transition_episodes": args.rolling_transition_episodes,
        "rolling_speed_ref_x100": args.rolling_speed_ref_x100,
        "rolling_omega_ref": args.rolling_omega_ref,
        "rolling_reward_scale": args.rolling_reward_scale,
        "action_smoothness_weight": args.action_smoothness_weight,
        "tail_roll_observation": args.tail_roll_observation,
        "tail_side": args.tail_side,
        "tail_curl_sign": args.tail_curl_sign,
        "tail_roll_stage": args.tail_roll_stage,
        "tail_roll_auto_curriculum": args.tail_roll_auto_curriculum,
        "tail_roll_competence_window": args.tail_roll_competence_window,
        "tail_roll_competence_threshold": args.tail_roll_competence_threshold,
        "tail_roll_min_stage_batches": args.tail_roll_min_stage_batches,
        "tail_roll_reward_scale": args.tail_roll_reward_scale,
        "tail_roll_potential_gamma": args.tail_roll_potential_gamma,
        "tail_roll_contact_margin": args.tail_roll_contact_margin,
        "tail_roll_curl_reference_degrees": args.tail_roll_curl_reference_degrees,
        "tail_roll_init_assist_degrees": args.tail_roll_init_assist_degrees,
        "tail_roll_init_assist_segments": args.tail_roll_init_assist_segments,
        "tail_roll_init_assist_episodes": args.tail_roll_init_assist_episodes,
        "fast_rollover_reward_scale": args.fast_rollover_reward_scale,
        "fast_rollover_flip_degrees": args.fast_rollover_flip_degrees,
        "fast_rollover_forward_fraction": args.fast_rollover_forward_fraction,
        "fast_rollover_support_fraction": args.fast_rollover_support_fraction,
        "fast_rollover_reset_open_ratio": args.fast_rollover_reset_open_ratio,
        "fast_rollover_cycle_target_steps": args.fast_rollover_cycle_target_steps,
        "fast_forward_observation": args.fast_forward_observation,
        "fast_forward_reward_scale": args.fast_forward_reward_scale,
        "fast_forward_event_degrees": args.fast_forward_event_degrees,
        "fast_forward_event_forward_fraction": args.fast_forward_event_forward_fraction,
        "fast_forward_event_contact_nodes": args.fast_forward_event_contact_nodes,
        "fast_forward_direction_fraction": args.fast_forward_direction_fraction,
        "fast_forward_event_target_steps": args.fast_forward_event_target_steps,
        "fast_forward_launch_lift": args.fast_forward_launch_lift,
        "fast_forward_launch_forward": args.fast_forward_launch_forward,
        "fast_forward_launch_curl": args.fast_forward_launch_curl,
        "fast_forward_launch_head_contact": args.fast_forward_launch_head_contact,
        "fast_forward_launch_hold_steps": args.fast_forward_launch_hold_steps,
        "fast_forward_stall_steps": args.fast_forward_stall_steps,
        "fast_forward_rotation_step_ref_degrees": args.fast_forward_rotation_step_ref_degrees,
        "fast_forward_translation_step_ref": args.fast_forward_translation_step_ref,
        "compatible_input_expansion": args.compatible_input_expansion,
        "pretrained_policy_only": args.pretrained_policy_only,
        "ppo_normalize_advantage": args.ppo_normalize_advantage,
        "ppo_target_kl": args.ppo_target_kl,
        "policy_anchor_coeff": args.policy_anchor_coeff,
        "policy_anchor_anneal_batches": args.policy_anchor_anneal_batches,
        "behavior_cloning": behavior_cloning_summary,
        "wave_bc_teacher_json": str(args.wave_bc_teacher_json) if args.wave_bc_teacher_json is not None else None,
        "init_pos_randomness": args.init_pos_randomness,
        "init_angle_range_degrees": args.init_angle_range_degrees,
        "init_height_jitter": args.init_height_jitter,
        "terrain_type": terrain_type,
        "terrain_settings": terrain_settings,
        "terrain_label": args.terrain,
        "terrain_contact_mode": args.terrain_contact_mode,
        "policy_net_config": {"depth": args.policy_depth, "num_cells": args.policy_cells},
        "share_parameters_policy": args.share_policy,
        "share_policy": args.share_policy,
        "gaussian_activation": args.gaussian_activation,
        "normal_scale_lb": args.normal_scale_lb,
        "scratch_wr_random_initialization": control_mode == "tail_wave_residual",
        "source_checkpoint": None if control_mode == "tail_wave_residual" else (
            str(args.pretrained_model_path) if args.pretrained_model_path is not None else None
        ),
        "scratch_wr_lineage_id": str(uuid.uuid4()) if control_mode == "tail_wave_residual" else None,
        "scratch_wr_initial_alpha": float(args.scratch_wr_alpha) if control_mode == "tail_wave_residual" else None,
        "scratch_wr_current_alpha": float(args.scratch_wr_alpha) if control_mode == "tail_wave_residual" else None,
        "scratch_wr_initial_stage_id": str(args.scratch_wr_stage_id) if control_mode == "tail_wave_residual" else None,
        "scratch_wr_current_stage_id": str(args.scratch_wr_stage_id) if control_mode == "tail_wave_residual" else None,
        "scratch_wr_initial_learning_rate": float(args.lr) if control_mode == "tail_wave_residual" else None,
        "scratch_wr_current_learning_rate": float(args.lr) if control_mode == "tail_wave_residual" else None,
        "scratch_wr_current_batch": 0 if control_mode == "tail_wave_residual" else None,
        "scratch_wr_stage_transitions": [],
        "scratch_wr_resumed_from": str(args.resume_training_state) if args.resume_training_state is not None else None,
        "training_args": vars(args),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if control_mode == "tail_wave_residual" and family == "ddpg":
        metadata.update(
            {
                "ppo_log_prob_mode": None,
                "scratch_wr_ddpg_alpha_aware_actor": True,
                "scratch_wr_ddpg_alpha_aware_exploration": True,
                "scratch_wr_ddpg_collector_sync_each_batch": True,
                "scratch_wr_ddpg_delay_actor": True,
                "scratch_wr_ddpg_delay_value": True,
                "scratch_wr_ddpg_soft_update_targets": ["actor", "critic"],
                "scratch_wr_ddpg_exploration_noise_coordinates": "normalized_minus1_plus1",
                "scratch_wr_ddpg_exploration_noise_clamped": True,
                "scratch_wr_ddpg_critic_action_coordinates": "normalized_minus1_plus1",
                "scratch_wr_ddpg_optim_steps_argument": int(args.optim_steps),
                "scratch_wr_ddpg_optimizer_updates_per_batch": int(
                    (args.optim_steps * args.frames_per_batch)
                    // args.minibatch_size
                ),
                "scratch_wr_ddpg_monitor_fields": [
                    "ddpg_loss_actor_mean",
                    "ddpg_loss_value_mean",
                    "ddpg_actor_grad_norm_mean",
                    "ddpg_critic_grad_norm_mean",
                    "ddpg_actor_parameter_displacement",
                    "ddpg_replay_size",
                    "ddpg_replay_transition_bytes",
                    "ddpg_exploration_sigma",
                    "ddpg_pred_q_mean",
                    "ddpg_target_q_mean",
                    "ddpg_td_error_mean",
                    "ddpg_optimizer_updates",
                ],
                "scratch_wr_replay_transition_policy": "clear_on_stage_or_alpha_change",
                "scratch_wr_replay_projection_keys": [
                    list(key) for key in SCRATCH_WR_DDPG_REPLAY_KEYS
                ],
                "scratch_wr_replay_serialized": False,
                "scratch_wr_resume_supported": False,
                "scratch_wr_replay_clear_events": [],
                "scratch_wr_replay_transition_bytes": None,
                "scratch_wr_replay_capacity_bytes_estimate": None,
            }
        )
    if getattr(args, "ppo_exact_log_prob", False):
        metadata["ppo_log_prob_mode"] = "exact_safe_tanh"
    if args.scratch_wr_v2:
        metadata.update(
            {
                "scratch_wr_v2": True,
                "scratch_wr_v2_sync_dense_weight": args.scratch_wr_v2_sync_dense_weight,
                "scratch_wr_v2_penalty_start_scale": args.scratch_wr_v2_penalty_start_scale,
                "scratch_wr_v2_penalty_anneal_batches": args.scratch_wr_v2_penalty_anneal_batches,
                "scratch_wr_v2_wave_ema_beta": args.scratch_wr_v2_wave_ema_beta,
            }
        )
    resume_payload = None
    current_episode_offset = 0
    if args.resume_training_state is not None:
        # Inspect every immutable contract before policy, optimizer, or RNG
        # state is changed. Exact/legacy or v1/v2 mismatches fail atomically.
        resume_payload = read_scratch_wr_training_state(args.resume_training_state)
        saved_metadata = resume_payload["metadata"]
        if saved_metadata.get("control_mode") != "tail_wave_residual":
            raise RuntimeError("Resume state control mode is not tail_wave_residual.")
        if int(saved_metadata.get("num_particles", -1)) != int(args.num_particles):
            raise RuntimeError("Resume state particle count does not match this run.")
        saved_terrain_contact_mode = saved_metadata.get(
            "terrain_contact_mode",
            DEFAULT_TERRAIN_CONTACT_MODE,
        )
        if saved_terrain_contact_mode != args.terrain_contact_mode:
            raise RuntimeError(
                "Resume state terrain contact mode does not match this run: "
                f"saved={saved_terrain_contact_mode!r}, "
                f"requested={args.terrain_contact_mode!r}."
            )
        saved_terrain_label = str(
            saved_metadata.get("terrain_label", "flat")
        )
        saved_terrain_type = str(
            saved_metadata.get("terrain_type", "flat")
        )
        saved_terrain_settings = saved_metadata.get("terrain_settings")
        if saved_terrain_label != str(args.terrain):
            raise RuntimeError(
                "Resume state terrain label does not match this run: "
                f"saved={saved_terrain_label!r}, requested={args.terrain!r}."
            )
        if saved_terrain_type != str(terrain_type):
            raise RuntimeError(
                "Resume state terrain type does not match this run: "
                f"saved={saved_terrain_type!r}, requested={terrain_type!r}."
            )
        if json.dumps(
            saved_terrain_settings,
            sort_keys=True,
            separators=(",", ":"),
        ) != json.dumps(
            terrain_settings,
            sort_keys=True,
            separators=(",", ":"),
        ):
            raise RuntimeError(
                "Resume state terrain settings do not match this run: "
                f"saved={saved_terrain_settings!r}, "
                f"requested={terrain_settings!r}."
            )
        if saved_metadata.get("scratch_wr_action_names") != metadata["scratch_wr_action_names"]:
            raise RuntimeError("Resume state Scratch-WR action layout does not match this run.")
        saved_log_prob_mode = saved_metadata.get("ppo_log_prob_mode", "legacy_epsilon_floor")
        requested_log_prob_mode = (
            "exact_safe_tanh"
            if getattr(args, "ppo_exact_log_prob", False)
            else "legacy_epsilon_floor"
        )
        if saved_log_prob_mode != requested_log_prob_mode:
            raise RuntimeError(
                "Resume state PPO log-probability mode does not match this run: "
                f"saved={saved_log_prob_mode!r}, requested={requested_log_prob_mode!r}."
            )
        if bool(saved_metadata.get("scratch_wr_v2", False)) != bool(args.scratch_wr_v2):
            raise RuntimeError("Resume state Scratch-WR-v2 mode does not match this run.")
        if args.scratch_wr_v2:
            for field_name, requested_value in (
                ("scratch_wr_v2_sync_dense_weight", args.scratch_wr_v2_sync_dense_weight),
                ("scratch_wr_v2_penalty_start_scale", args.scratch_wr_v2_penalty_start_scale),
                ("scratch_wr_v2_wave_ema_beta", args.scratch_wr_v2_wave_ema_beta),
            ):
                saved_value = float(saved_metadata.get(field_name, float("nan")))
                if not np.isclose(saved_value, float(requested_value), rtol=0.0, atol=1e-12):
                    raise RuntimeError(
                        f"Resume state {field_name} does not match this run: "
                        f"saved={saved_value!r}, requested={requested_value!r}."
                    )
            saved_anneal = int(saved_metadata.get("scratch_wr_v2_penalty_anneal_batches", -1))
            if saved_anneal != int(args.scratch_wr_v2_penalty_anneal_batches):
                raise RuntimeError(
                    "Resume state scratch_wr_v2_penalty_anneal_batches does not match this run."
                )
        apply_scratch_wr_training_state(resume_payload, policy, critic, optimiser)
        current_episode_offset = int(resume_payload["current_batch"])
        metadata["scratch_wr_current_batch"] = current_episode_offset
        metadata["scratch_wr_lineage_id"] = saved_metadata["scratch_wr_lineage_id"]
        metadata["scratch_wr_initial_alpha"] = saved_metadata["scratch_wr_initial_alpha"]
        metadata["scratch_wr_initial_stage_id"] = saved_metadata["scratch_wr_initial_stage_id"]
        metadata["scratch_wr_initial_learning_rate"] = saved_metadata.get(
            "scratch_wr_initial_learning_rate", metadata["scratch_wr_initial_learning_rate"]
        )
        metadata["scratch_wr_stage_transitions"] = list(saved_metadata.get("scratch_wr_stage_transitions", []))
        metadata["created_at"] = saved_metadata.get("created_at", metadata["created_at"])
        if args.scratch_wr_control_file is None:
            args.scratch_wr_alpha = float(resume_payload["alpha"])
            args.scratch_wr_stage_id = str(resume_payload["stage_id"])
            args.lr = float(saved_metadata.get("scratch_wr_current_learning_rate", args.lr))
        base_env.set_scratch_wr_alpha(args.scratch_wr_alpha)
        set_scratch_wr_policy_alpha(exploration_policy, args.scratch_wr_alpha)
        # The collector may hold a CUDA policy copy created before the full
        # training state was restored.  Synchronize weights and the
        # non-persistent Scratch-WR alpha buffer before the first resumed batch.
        collector.update_policy_weights_()
        metadata["scratch_wr_current_alpha"] = float(args.scratch_wr_alpha)
        metadata["scratch_wr_current_stage_id"] = str(args.scratch_wr_stage_id)
        set_optimiser_learning_rate(optimiser, args.lr)
        metadata["scratch_wr_current_learning_rate"] = float(args.lr)
        print(
            "Resumed verified Scratch-WR lineage",
            metadata["scratch_wr_lineage_id"],
            "from batch",
            current_episode_offset,
        )
    elif control_mode == "tail_wave_residual":
        # This file is the immutable proof that actor and critic began from the
        # current seed's random initialization, before any collector batch.
        checkpoint_zero_path = save_dir / "checkpoint_0.pt"
        if checkpoint_zero_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite strict-scratch proof checkpoint: {checkpoint_zero_path}"
            )
        save_params({"policy": policy, "critic": critic, "metadata": metadata}, checkpoint_zero_path)
        metadata["scratch_wr_checkpoint_0"] = str(checkpoint_zero_path)
        print("Saved strict-scratch initialization proof:", checkpoint_zero_path)
    bc_checkpoint_path = save_dir / "checkpoint_bc.pt" if behavior_cloning_summary is not None else None
    if bc_checkpoint_path is not None:
        metadata["behavior_cloning_checkpoint"] = str(bc_checkpoint_path)
    write_json(save_dir / "metadata.json", metadata)
    if behavior_cloning_summary is not None:
        write_json(save_dir / "behavior_cloning_summary.json", behavior_cloning_summary)
        save_params({"policy": policy, "critic": critic, "metadata": metadata}, bc_checkpoint_path)
        print("Saved behavior-cloned checkpoint:", bc_checkpoint_path)

    log_path = save_dir / "training_log.csv"
    train_start_time = time.time()

    pbar = tqdm(total=current_episode_offset + args.episodes, initial=current_episode_offset, desc="episode_reward_mean = 0")
    current_episode = current_episode_offset
    last_reward = float("nan")
    last_speed = float("nan")
    tail_stage = int(args.tail_roll_stage)
    tail_stage_batches = 0
    tail_success_history = deque(maxlen=args.tail_roll_competence_window)
    if hasattr(base_env, "set_tail_roll_stage"):
        base_env.set_tail_roll_stage(tail_stage)
    if hasattr(base_env, "set_curriculum_episode"):
        # A resumed v2 lineage must continue its reward anneal before the very
        # first collector batch, rather than briefly reverting to batch zero.
        base_env.set_curriculum_episode(current_episode_offset)
    stop_requested = False

    for iteration, batch in enumerate(collector):
        current_episode = current_episode_offset + iteration + 1
        if control_mode == "tail_wave_residual":
            metadata["scratch_wr_current_batch"] = int(current_episode)
        if hasattr(base_env, "set_curriculum_episode"):
            # The current batch has already been collected; this value is used by the next batch.
            base_env.set_curriculum_episode(current_episode)
        log_info = batch["log_info"]
        last_speed = log_info["speed"].mean().item()
        rolling_log_values = {}
        for metric_name in (
            "closure_score",
            "circularity_score",
            "body_omega",
            "slip_penalty",
            "action_smoothness_penalty",
            "curriculum_progress",
            "rolling_reward",
            "tail_lift_score",
            "tail_forward_score",
            "head_contact_score",
            "curl_prefix_progress",
            "curl_order_penalty",
            "total_signed_curvature",
            "closure_ratio",
            "support_margin",
            "cumulative_rotation",
            "rolling_gate",
            "tail_roll_stage",
            "tail_stage_success",
            "tail_roll_reward",
            "fast_roll_phase",
            "fast_roll_phase_steps",
            "fast_roll_phase_progress",
            "fast_roll_flip_event",
            "fast_roll_flip_count",
            "fast_roll_cycle_count",
            "fast_roll_cycle_rotation",
            "fast_roll_cycle_forward",
            "fast_roll_support_migration",
            "fast_roll_direction_fraction",
            "fast_roll_motion_gate",
            "fast_roll_reward",
            "fast_forward_phase",
            "fast_forward_phase_steps",
            "fast_forward_launch_progress",
            "fast_forward_launch_ready_steps",
            "fast_forward_roll_progress",
            "fast_forward_progress_delta",
            "fast_forward_launch_event",
            "fast_forward_event_pulse",
            "fast_forward_event_bonus",
            "fast_forward_event_count",
            "fast_forward_event_rotation",
            "fast_forward_event_forward",
            "fast_forward_support_index",
            "fast_forward_support_migration_nodes",
            "fast_forward_ground_contact_strength",
            "fast_forward_event_direction_fraction",
            "fast_forward_episode_direction_fraction",
            "fast_forward_event_steps",
            "fast_forward_progress_age",
            "fast_forward_stall_penalty",
            "fast_forward_reverse_rotation_penalty",
            "fast_forward_backward_penalty",
            "effort_penalty",
            "scratch_wr_v2_z0_lift_ratio",
            "scratch_wr_v2_z0_forward_ratio",
            "scratch_wr_v2_z0_curl_ratio",
            "scratch_wr_v2_z0_sync_score",
            "scratch_wr_v2_z0_candidate_mask",
            "scratch_wr_v2_z0_active",
            "scratch_wr_v2_z0_penalty_scale",
            "scratch_wr_v2_z0_dense_reward",
            "scratch_wr_v2_progress_delta",
            "fast_forward_reward",
            "wave_amplitude",
            "wave_center",
            "wave_width",
            "wave_hold",
            "wave_kp",
            "wave_kd",
            "scratch_wr_alpha",
            "scratch_wr_wave_torque_rms",
            "scratch_wr_residual_torque_rms",
            "scratch_wr_applied_residual_torque_rms",
            "scratch_wr_total_torque_rms",
            "scratch_wr_torque_clip_fraction",
            "scratch_wr_residual_saturation_fraction",
            "scratch_wr_v2_wave_ema_beta",
            "scratch_wr_v2_wave_filter_delta_rms",
            "scratch_wr_v2_applied_wave_amplitude",
            "scratch_wr_v2_applied_wave_center",
            "scratch_wr_v2_applied_wave_width",
            "scratch_wr_v2_applied_wave_hold",
            "scratch_wr_v2_applied_wave_kp",
            "scratch_wr_v2_applied_wave_kd",
        ):
            if metric_name in log_info.keys():
                rolling_log_values[metric_name] = log_info[metric_name].mean().item()

        if args.reward_func == "tail_roll_curriculum":
            stage_success_value = float(rolling_log_values.get("tail_stage_success", 0.0))
            tail_success_history.append(stage_success_value)
            tail_stage_batches += 1
            stage_success_rate = float(np.mean(tail_success_history))
            rolling_log_values["tail_stage_success_rate"] = stage_success_rate
            ready_to_promote = (
                args.tail_roll_auto_curriculum
                and tail_stage < 3
                and tail_stage_batches >= args.tail_roll_min_stage_batches
                and len(tail_success_history) == args.tail_roll_competence_window
                and stage_success_rate >= args.tail_roll_competence_threshold
            )
            if ready_to_promote:
                tail_stage += 1
                tail_stage_batches = 0
                tail_success_history.clear()
                base_env.set_tail_roll_stage(tail_stage)
                metadata["tail_roll_stage"] = tail_stage
                metadata.setdefault("tail_roll_stage_transitions", []).append(
                    {
                        "episode": current_episode,
                        "new_stage": tail_stage,
                        "previous_stage_success_rate": stage_success_rate,
                    }
                )
                print(f"\nTail-roll curriculum promoted to stage {tail_stage} at episode {current_episode}.")

        if scratch_wr_ddpg:
            require_finite_tensor(
                batch.get(env.action_key),
                f"collected action at batch {current_episode}",
            )
        batch = process_batch(batch)
        if value_estimator is not None:
            with torch.no_grad():
                value_estimator(
                    batch,
                    params=loss_module.critic_network_params,
                    target_params=loss_module.target_critic_network_params,
                )

        replay_batch = (
            scratch_wr_ddpg_replay_batch(batch)
            if scratch_wr_ddpg
            else batch
        )
        current_replay_transition_bytes = (
            replay_transition_bytes(replay_batch) if scratch_wr_ddpg else 0
        )
        if (
            scratch_wr_ddpg
            and metadata["scratch_wr_replay_transition_bytes"] is None
        ):
            metadata["scratch_wr_replay_transition_bytes"] = int(
                current_replay_transition_bytes
            )
            metadata["scratch_wr_replay_capacity_bytes_estimate"] = int(
                current_replay_transition_bytes * args.memory_size
            )
            write_json(save_dir / "metadata.json", metadata)
            print(
                "Scratch-WR DDPG minimal replay layout:",
                {
                    "keys": metadata["scratch_wr_replay_projection_keys"],
                    "bytes_per_transition": current_replay_transition_bytes,
                    "capacity": int(args.memory_size),
                    "capacity_bytes_estimate": metadata[
                        "scratch_wr_replay_capacity_bytes_estimate"
                    ],
                },
            )
        replay_buffer.extend(replay_batch.reshape(-1))
        actor_update_norms: list[float] = []
        actor_update_clip_scales: list[float] = []
        ddpg_loss_actor_values: list[float] = []
        ddpg_loss_value_values: list[float] = []
        ddpg_actor_grad_norms: list[float] = []
        ddpg_critic_grad_norms: list[float] = []
        ddpg_pred_q_values: list[float] = []
        ddpg_target_q_values: list[float] = []
        ddpg_td_error_values: list[float] = []
        ddpg_optimizer_updates = 0
        ddpg_actor_batch_snapshot = (
            parameter_update_snapshot(
                optimiser["loss_actor"].param_groups[0]["params"]
            )
            if scratch_wr_ddpg
            else []
        )
        ddpg_exploration_sigma = (
            float(exploration_policy[1].sigma.detach().cpu().item())
            if scratch_wr_ddpg
            else float("nan")
        )
        ppo_approx_kl_values: list[float] = []
        ppo_early_stop = False
        ppo_updates_completed = 0
        policy_anchor_losses: list[float] = []
        if args.policy_anchor_anneal_batches > 0:
            anchor_fraction = max(
                0.0,
                1.0 - (current_episode - 1) / args.policy_anchor_anneal_batches,
            )
        else:
            anchor_fraction = 1.0
        effective_anchor_coeff = args.policy_anchor_coeff * anchor_fraction

        for _ in range((args.optim_steps * args.frames_per_batch) // args.minibatch_size):
            subdata = replay_buffer.sample()
            loss_vals = loss_module(subdata)

            if family == "ppo":
                approx_kl_tensor = None
                for kl_key in ("kl_approx", "approx_kl", "kl"):
                    if kl_key in loss_vals.keys():
                        approx_kl_tensor = loss_vals[kl_key]
                        break
                if approx_kl_tensor is not None:
                    approx_kl_value = float(approx_kl_tensor.detach().mean().item())
                    if np.isfinite(approx_kl_value):
                        ppo_approx_kl_values.append(approx_kl_value)
                        if args.ppo_target_kl > 0 and approx_kl_value > args.ppo_target_kl:
                            ppo_early_stop = True
                            break

                total_ppo_loss = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                if effective_anchor_coeff > 0 and policy_anchor_reference:
                    anchor_loss = policy_parameter_anchor_loss(policy, policy_anchor_reference)
                    policy_anchor_losses.append(float(anchor_loss.detach().item()))
                    total_ppo_loss = total_ppo_loss + effective_anchor_coeff * anchor_loss
                losses = [total_ppo_loss]
                params = [loss_module.parameters()]
                optims = [optimiser]
                loss_names = ["loss_ppo"]
            else:
                losses = [loss_vals[name] for name in ["loss_actor", "loss_value"]]
                params = [optimiser[name].param_groups[0]["params"] for name in ["loss_actor", "loss_value"]]
                optims = [optimiser[name] for name in ["loss_actor", "loss_value"]]
                loss_names = ["loss_actor", "loss_value"]
                if scratch_wr_ddpg:
                    ddpg_loss_actor_values.append(
                        float(loss_vals["loss_actor"].detach().mean().cpu().item())
                    )
                    ddpg_loss_value_values.append(
                        float(loss_vals["loss_value"].detach().mean().cpu().item())
                    )
                    for ddpg_loss_name in ("loss_actor", "loss_value"):
                        require_finite_tensor(
                            loss_vals[ddpg_loss_name],
                            f"{ddpg_loss_name} at batch {current_episode}",
                        )
                    for q_key, q_values in (
                        ("pred_value", ddpg_pred_q_values),
                        ("target_value", ddpg_target_q_values),
                        ("td_error", ddpg_td_error_values),
                    ):
                        if q_key not in loss_vals.keys():
                            raise RuntimeError(
                                "Scratch-WR DDPG loss output is missing "
                                f"required diagnostic key {q_key!r}."
                            )
                        require_finite_tensor(
                            loss_vals[q_key],
                            f"{q_key} at batch {current_episode}",
                        )
                        q_values.append(
                            float(
                                loss_vals[q_key]
                                .detach()
                                .mean()
                                .cpu()
                                .item()
                            )
                        )

            for loss_name, loss, param, optim in zip(loss_names, losses, params, optims):
                loss.backward()
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(
                    param,
                    args.max_grad_norm,
                )
                if scratch_wr_ddpg:
                    grad_norm = float(grad_norm_tensor.detach().cpu().item())
                    if loss_name == "loss_actor":
                        ddpg_actor_grad_norms.append(grad_norm)
                    elif loss_name == "loss_value":
                        ddpg_critic_grad_norms.append(grad_norm)
                    require_finite_scalar(
                        grad_norm,
                        "gradient norm "
                        f"for {loss_name} at batch {current_episode}",
                    )
                update_snapshot = None
                if uses_ddpg_policy_update_clip(args.algorithm) and loss_name == "loss_actor":
                    update_snapshot = parameter_update_snapshot(param)
                optim.step()
                if update_snapshot is not None:
                    update_norm, clip_scale = clip_parameter_update(update_snapshot, args.ddpg_policy_update_clip)
                    actor_update_norms.append(update_norm)
                    actor_update_clip_scales.append(clip_scale)
                optim.zero_grad()

            if family == "ppo":
                ppo_updates_completed += 1

            if target_updater is not None:
                target_updater.step()
            if scratch_wr_ddpg:
                ddpg_optimizer_updates += 1

        ddpg_actor_parameter_displacement = (
            parameter_displacement_norm(ddpg_actor_batch_snapshot)
            if scratch_wr_ddpg
            else float("nan")
        )
        if scratch_wr_ddpg:
            if ddpg_optimizer_updates <= 0:
                raise RuntimeError(
                    "Scratch-WR DDPG completed no optimizer updates in "
                    f"batch {current_episode}; check optim_steps, "
                    "frames_per_batch, and minibatch_size."
                )
            require_finite_scalar(
                ddpg_actor_parameter_displacement,
                f"actor parameter displacement at batch {current_episode}",
            )
            require_finite_scalar(
                ddpg_exploration_sigma,
                f"exploration sigma at batch {current_episode}",
            )
        if family == "ddpg":
            exploration_policy[1].step(batch.numel())
        collector.update_policy_weights_()

        done = batch.get(("next", "done"))
        last_reward = batch.get(("next", "episode_reward"))[done].mean().item()
        append_training_log(
            log_path,
            {
                "episode": current_episode,
                "reward_mean": float(last_reward),
                "speed_mean": float(last_speed),
                "speed_x100": float(last_speed) * 100.0,
                "elapsed_sec": time.time() - train_start_time,
                "frames_per_batch": args.frames_per_batch,
                "algorithm": args.algorithm,
                "algorithm_family": family,
                "robot": args.robot,
                "terrain": args.terrain,
                "channel": resolved_channel_slug,
                "actor_update_norm_mean": float(np.mean(actor_update_norms)) if actor_update_norms else None,
                "actor_update_clip_scale_mean": float(np.mean(actor_update_clip_scales)) if actor_update_clip_scales else None,
                "ddpg_loss_actor_mean": float(np.mean(ddpg_loss_actor_values)) if ddpg_loss_actor_values else None,
                "ddpg_loss_value_mean": float(np.mean(ddpg_loss_value_values)) if ddpg_loss_value_values else None,
                "ddpg_actor_grad_norm_mean": float(np.mean(ddpg_actor_grad_norms)) if ddpg_actor_grad_norms else None,
                "ddpg_critic_grad_norm_mean": float(np.mean(ddpg_critic_grad_norms)) if ddpg_critic_grad_norms else None,
                "ddpg_actor_parameter_displacement": ddpg_actor_parameter_displacement if scratch_wr_ddpg else None,
                "ddpg_replay_size": int(len(replay_buffer)) if scratch_wr_ddpg else None,
                "ddpg_replay_transition_bytes": int(current_replay_transition_bytes) if scratch_wr_ddpg else None,
                "ddpg_exploration_sigma": ddpg_exploration_sigma if scratch_wr_ddpg else None,
                "ddpg_pred_q_mean": float(np.mean(ddpg_pred_q_values)) if ddpg_pred_q_values else None,
                "ddpg_target_q_mean": float(np.mean(ddpg_target_q_values)) if ddpg_target_q_values else None,
                "ddpg_td_error_mean": float(np.mean(ddpg_td_error_values)) if ddpg_td_error_values else None,
                "ddpg_optimizer_updates": int(ddpg_optimizer_updates) if scratch_wr_ddpg else None,
                "ppo_approx_kl": float(np.mean(ppo_approx_kl_values)) if ppo_approx_kl_values else None,
                "ppo_early_stop": int(ppo_early_stop) if family == "ppo" else None,
                "ppo_updates_completed": int(ppo_updates_completed) if family == "ppo" else None,
                "policy_anchor_loss": (
                    float(np.mean(policy_anchor_losses))
                    if policy_anchor_losses
                    else (0.0 if family == "ppo" else None)
                ),
                "policy_anchor_coeff_effective": float(effective_anchor_coeff) if family == "ppo" else None,
                "scratch_wr_stage_id": str(args.scratch_wr_stage_id) if control_mode == "tail_wave_residual" else "",
                "scratch_wr_learning_rate": float(args.lr) if control_mode == "tail_wave_residual" else float("nan"),
                **rolling_log_values,
            },
        )
        pbar.set_description(f"episode_reward_mean = {last_reward:.4f}, speed = {last_speed:.4f}", refresh=False)
        pbar.update()

        scratch_sync_due = (
            control_mode == "tail_wave_residual"
            and args.scratch_wr_eval_sync_every > 0
            and current_episode % args.scratch_wr_eval_sync_every == 0
        )
        if current_episode % args.save_every == 0 or scratch_sync_due:
            checkpoint_path = save_dir / f"checkpoint_{current_episode}.pt"
            save_params({"policy": policy, "critic": critic, "metadata": metadata}, checkpoint_path)
            print("\nSaved:", checkpoint_path)
            if control_mode == "tail_wave_residual":
                save_scratch_wr_training_state(
                    save_dir / f"training_state_{current_episode}.pt",
                    policy=policy,
                    critic=critic,
                    optimiser=optimiser,
                    metadata=metadata,
                    current_batch=current_episode,
                    stage_id=args.scratch_wr_stage_id,
                    alpha=args.scratch_wr_alpha,
                )
            if args.analysis_every > 0 and current_episode % args.analysis_every == 0:
                run_auto_analysis(checkpoint_path, save_dir, args, final=False)

        if control_mode == "tail_wave_residual" and args.scratch_wr_control_file is not None:
            if scratch_sync_due:
                print(
                    f"\nScratch-WR batch {current_episode} paused for deterministic evaluation",
                    f"(stage={args.scratch_wr_stage_id}, alpha={args.scratch_wr_alpha:g}).",
                )
                next_control = wait_for_scratch_wr_evaluation(
                    args.scratch_wr_control_file,
                    current_batch=current_episode,
                    stage_id=args.scratch_wr_stage_id,
                    alpha=args.scratch_wr_alpha,
                    timeout_minutes=args.scratch_wr_eval_sync_timeout_minutes,
                    learning_rate=args.lr,
                    control_cursor=scratch_wr_control_cursor,
                    control_read_retry_seconds=args.scratch_wr_control_read_retry_seconds,
                    control_read_retry_initial_ms=args.scratch_wr_control_read_retry_initial_ms,
                )
            else:
                next_control = read_scratch_wr_control_file(
                    args.scratch_wr_control_file,
                    default_stage_id=args.scratch_wr_stage_id,
                    default_alpha=args.scratch_wr_alpha,
                    default_learning_rate=args.lr,
                    cursor=scratch_wr_control_cursor,
                    retry_seconds=args.scratch_wr_control_read_retry_seconds,
                    retry_initial_ms=args.scratch_wr_control_read_retry_initial_ms,
                )
            new_stage_id = next_control["stage_id"]
            new_alpha = float(next_control["alpha"])
            new_learning_rate = float(next_control["learning_rate"] or args.lr)
            stage_or_alpha_changed = (
                new_stage_id != args.scratch_wr_stage_id
                or not np.isclose(new_alpha, args.scratch_wr_alpha)
            )
            if (
                stage_or_alpha_changed
                or not np.isclose(new_learning_rate, args.lr)
            ):
                metadata["scratch_wr_stage_transitions"].append(
                    {
                        "batch": current_episode,
                        "previous_stage_id": str(args.scratch_wr_stage_id),
                        "new_stage_id": str(new_stage_id),
                        "previous_alpha": float(args.scratch_wr_alpha),
                        "new_alpha": new_alpha,
                        "previous_learning_rate": float(args.lr),
                        "new_learning_rate": new_learning_rate,
                    }
                )
                if family == "ddpg" and stage_or_alpha_changed:
                    stale_transition_count = len(replay_buffer)
                    replay_buffer.empty()
                    replay_clear_event = {
                        "batch": int(current_episode),
                        "previous_stage_id": str(args.scratch_wr_stage_id),
                        "new_stage_id": str(new_stage_id),
                        "previous_alpha": float(args.scratch_wr_alpha),
                        "new_alpha": float(new_alpha),
                        "discarded_transitions": int(stale_transition_count),
                        "reason": "stage_or_alpha_change",
                    }
                    metadata["scratch_wr_replay_clear_events"].append(
                        replay_clear_event
                    )
                    print(
                        "Scratch-WR DDPG replay cleared at curriculum boundary:",
                        replay_clear_event,
                    )
                args.scratch_wr_stage_id = new_stage_id
                args.scratch_wr_alpha = new_alpha
                args.lr = new_learning_rate
                base_env.set_scratch_wr_alpha(new_alpha)
                set_scratch_wr_policy_alpha(exploration_policy, new_alpha)
                # SyncDataCollector owns a policy copy on CUDA.  Keep the
                # collector's action distribution aligned with the policy
                # used to compute the first loss after a stage change.
                collector.update_policy_weights_()
                set_optimiser_learning_rate(optimiser, new_learning_rate)
                metadata["scratch_wr_current_stage_id"] = str(new_stage_id)
                metadata["scratch_wr_current_alpha"] = new_alpha
                metadata["scratch_wr_current_learning_rate"] = new_learning_rate
                write_json(save_dir / "metadata.json", metadata)
                save_scratch_wr_training_state(
                    save_dir / f"training_state_{current_episode}.pt",
                    policy=policy,
                    critic=critic,
                    optimiser=optimiser,
                    metadata=metadata,
                    current_batch=current_episode,
                    stage_id=args.scratch_wr_stage_id,
                    alpha=args.scratch_wr_alpha,
                )
                print(
                    f"Scratch-WR transition -> {new_stage_id}, alpha={new_alpha:g}, lr={new_learning_rate:g}."
                )
            stop_requested = bool(next_control["stop_requested"])
            if stop_requested:
                checkpoint_path = save_dir / f"checkpoint_{current_episode}.pt"
                save_params({"policy": policy, "critic": critic, "metadata": metadata}, checkpoint_path)
                print("Scratch-WR clean stop requested after batch", current_episode)
                break

    shutdown_collector(collector)

    final_checkpoint_path = save_dir / f"checkpoint_{current_episode}.pt"
    if current_episode % args.save_every != 0:
        save_params({"policy": policy, "critic": critic, "metadata": metadata}, final_checkpoint_path)
        print("\nSaved:", final_checkpoint_path)
    elif not final_checkpoint_path.exists():
        # Defensive fallback for unusual collector behaviour.
        save_params({"policy": policy, "critic": critic, "metadata": metadata}, final_checkpoint_path)
        print("\nSaved:", final_checkpoint_path)
    if control_mode == "tail_wave_residual":
        save_scratch_wr_training_state(
            save_dir / f"training_state_{current_episode}.pt",
            policy=policy,
            critic=critic,
            optimiser=optimiser,
            metadata=metadata,
            current_batch=current_episode,
            stage_id=args.scratch_wr_stage_id,
            alpha=args.scratch_wr_alpha,
        )

    simulation_command_path = write_simulation_command(
        save_dir,
        final_checkpoint_path,
        PROJECT_ROOT,
    ).path
    print("Simulation command:", simulation_command_path)

    write_json(
        save_dir / "training_summary.json",
        {
            "final_checkpoint": str(final_checkpoint_path),
            "final_reward_mean": float(last_reward),
            "final_speed_mean": float(last_speed),
            "final_speed_x100": float(last_speed) * 100.0,
            "episodes": current_episode,
            "status": "stopped" if stop_requested else "complete",
            "save_dir": str(save_dir),
            "simulation_command": str(simulation_command_path),
            "terrain_contact_mode": args.terrain_contact_mode,
            "metadata": metadata,
            "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    run_auto_analysis(final_checkpoint_path, save_dir, args, final=True)

    print("Training complete.")
    print("Final reward mean:", last_reward)
    print("Final speed mean:", last_speed)
    print("Result directory:", save_dir)


def main() -> None:
    try:
        _main_impl()
    finally:
        # If collection, evaluation synchronization, optimization, or logging
        # raises, do not leave a worker alive and make controllers misread it
        # as a still-running training process.
        for collector in list(_ACTIVE_COLLECTORS):
            try:
                shutdown_collector(collector)
            except Exception as exc:
                print(f"Collector cleanup warning: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
