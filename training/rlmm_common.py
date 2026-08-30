"""Shared helpers for RL metamaterial training and demo scripts."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules import TanhNormal


DEFAULT_TERRAIN_CONTACT_MODE = "legacy_flat"
TERRAIN_CONTACT_MODE_CHOICES = ("legacy_flat", "mesh_v1")


def terrain_contact_mode_from_metadata(
    metadata: dict[str, Any],
    override: str | None = None,
) -> str:
    """Resolve terrain contact mode while preserving old-checkpoint behavior."""
    value = override
    if value is None:
        value = metadata.get("terrain_contact_mode")
    if value is None:
        training_args = metadata.get("training_args")
        if isinstance(training_args, dict):
            value = training_args.get("terrain_contact_mode")
    if value is None:
        value = DEFAULT_TERRAIN_CONTACT_MODE

    mode = str(value)
    if mode not in TERRAIN_CONTACT_MODE_CHOICES:
        raise ValueError(
            f"Unsupported terrain_contact_mode {mode!r}; "
            f"expected one of {TERRAIN_CONTACT_MODE_CHOICES}."
        )
    return mode


def find_project_root(start: Path) -> Path:
    """Find the project root from a file path inside the project."""
    start = start.resolve()
    candidates = [start] + list(start.parents)
    for p in candidates:
        has_env_package = (
            (p / "metamaterial_envs").exists()
            or (p / "packages" / "metamaterial_envs").exists()
        )
        if has_env_package and (p / "training").exists():
            return p
    raise RuntimeError(
        "Cannot find project root. Put this file inside the project folder, "
        "preferably in the training/ directory."
    )


def add_env_package_to_path(project_root: Path) -> None:
    """Make `from metamaterial_envs.env import metamaterial` resolve to the local package."""
    import sys

    packaged_layout = project_root / "packages" / "metamaterial_envs"
    package_root = str(
        packaged_layout if packaged_layout.exists() else project_root / "metamaterial_envs"
    )
    if package_root not in sys.path:
        sys.path.insert(0, package_root)


def stairs_settings(
    start_stairs: float = 5,
    step_width: float = 5,
    step_height: float = 0.2,
    steps: int = 10,
) -> dict[str, Any]:
    return {
        "type": "stairs",
        "start_stairs": start_stairs,
        "step_width": step_width,
        "step_height": step_height,
        "steps": steps,
    }


def tunnel_settings(
    start: float = 10,
    slope: float = 5,
    slope_height: float = 1,
    tunnel_length: float = 10,
    tunnel_height: float = 5,
) -> dict[str, Any]:
    """Return MeshTerrain settings for the built-in tunnel preset."""
    return {
        "type": "tunnel",
        "start": start,
        "slope": slope,
        "slope_height": slope_height,
        "tunnel": tunnel_length,
        "tunnel_height": tunnel_height,
    }


def terrain_config(
    kind: str,
    *,
    start_stairs: float = 5,
    step_width: float = 5,
    step_height: float = 0.2,
    steps: int = 10,
    tunnel_start: float = 10,
    tunnel_slope: float = 5,
    tunnel_slope_height: float = 1,
    tunnel_length: float = 10,
    tunnel_height: float = 5,
):
    """Return `(terrain_type, terrain_settings)` for the environment constructor."""
    kind = kind.lower()
    if kind == "flat":
        return "flat", None
    if kind == "stairs":
        return "mesh", stairs_settings(start_stairs, step_width, step_height, steps)
    if kind == "tunnel":
        return "mesh", tunnel_settings(
            start=tunnel_start,
            slope=tunnel_slope,
            slope_height=tunnel_slope_height,
            tunnel_length=tunnel_length,
            tunnel_height=tunnel_height,
        )
    raise ValueError(f"Unsupported terrain: {kind!r}. Use 'flat', 'stairs', or 'tunnel'.")


def terrain_label(terrain_type: str, terrain_settings: Any) -> str:
    if terrain_type == "flat":
        return "flat"
    if terrain_type == "mesh" and isinstance(terrain_settings, dict):
        preset_type = terrain_settings.get("type")
        if preset_type in {"stairs", "tunnel"}:
            return preset_type
    if terrain_type == "mesh" and terrain_settings in {"stairs", "tunnel"}:
        return str(terrain_settings)
    return str(terrain_type)


# User-facing channel/control presets used by train_metamaterial.py and demo_metamaterial.py.
#
# Exact trainable channels used in the thesis/paper:
#   --channel dth
#       obs_i = [delta_theta_(i-1), delta_theta_(i+1)]
#       policy action is the active torque directly.
#   --channel thdot
#       obs_i = [delta_theta_(i-1), delta_theta_(i+1), theta_dot_i]
#       policy action is the active torque directly.
#
# Project extensions retained for ablation/comparison:
#   --channel obs
#       obs_i = delta_theta_(i+1) - delta_theta_(i-1)
#       policy action is the physical torque directly.
#   --channel action
#       policy action is [k1_i, k2_i], and the simulator applies
#       tau_i = k1_i * [delta_theta_(i+1)-delta_theta_(i-1)]
#             + k2_i * theta_dot_i.
#   --channel paper
#       policy action is kappa_alpha_i in the constrained non-reciprocity
#       formula. This is a project extension, not the paper's direct-torque RL
#       channel.
#   --channel k2_positive / k2_negative
#       k1 is fixed and the policy learns only signed k2.
#
# F is fixed to 1.0 everywhere.
CHANNEL_CHOICES = (
    "obs",
    "action",
    "dth",
    "thdot",
    "paper",
    "k2_positive",
    "k2_negative",
    "tail_wave",
    "tail_wave_residual",
    "theta",
    "formula",
    "theta_feedback",
    "wave",
)
CONTROL_MODE_CHOICES = (
    "direct",
    "formula",
    "nonreciprocity",
    "fixed_k1_k2_positive",
    "fixed_k1_k2_negative",
    "tail_wave",
    "tail_wave_residual",
)

_CHANNEL_ALIASES = {
    # Existing single-difference direct-torque channel.
    "obs": "obs",
    "theta": "obs",
    "theta_diff": "obs",
    "theta_difference": "obs",
    "single": "obs",
    "single_channel": "obs",
    "one_channel": "obs",
    "1ch": "obs",
    "direct": "obs",
    "torque": "obs",

    # Exact two-neighbour observation used in Eq. (2.3) and the dth columns.
    "dth": "dth",
    "paper_dth": "dth",
    "dth_direct": "dth",
    "dth_neighbours": "dth",
    "dth_neighbors": "dth",
    "neighbours": "dth",
    "neighbors": "dth",

    # Exact alternate observation used in Sec. 3.2.2 and challenging terrain.
    "thdot": "thdot",
    "paper_thdot": "thdot",
    "thdot_direct": "thdot",
    "dth_neighbours_plus_thdot": "thdot",
    "dth_neighbors_plus_thdot": "thdot",
    "neighbours_thdot": "thdot",
    "neighbors_thdot": "thdot",

    # k1/k2 formula channel from this project.
    "action": "action",
    "formula": "action",
    "gain": "action",
    "feedback": "action",
    "k1k2": "action",
    "k1_k2": "action",
    "action_formula": "action",
    "two_channel_action": "action",
    "formula_action": "action",

    # Constrained learned non-reciprocity coefficient.
    "paper": "paper",
    "paper_formula": "paper",
    "learned_kappa_alpha": "paper",
    "nonreciprocity": "paper",
    "non_reciprocity": "paper",
    "nonreciprocal": "paper",
    "odd": "paper",
    "odd_elasticity": "paper",
    "kappa_alpha": "paper",

    # Fixed-k1 signed-k2 ablations.
    "k2_positive": "k2_positive",
    "k2_pos": "k2_positive",
    "positive_k2": "k2_positive",
    "fixed_k1_positive": "k2_positive",
    "fixed_k1_k2_positive": "k2_positive",
    "k2_negative": "k2_negative",
    "k2_neg": "k2_negative",
    "negative_k2": "k2_negative",
    "fixed_k1_negative": "k2_negative",
    "fixed_k1_k2_negative": "k2_negative",

    # Single global six-parameter tail-to-head curl-wave controller.
    "tail_wave": "tail_wave",
    "tail_curl_wave": "tail_wave",
    "curl_wave": "tail_wave",
    "tail_wave_residual": "tail_wave_residual",
    "scratch_wr": "tail_wave_residual",
    "wave_residual": "tail_wave_residual",

    # Optional observation ablations retained for old checkpoints.
    "theta_feedback": "theta_feedback",
    "feedback_obs": "theta_feedback",
    "theta_dot": "theta_feedback",
    "theta_friction": "theta_feedback",
    "friction": "theta_feedback",
    "two_channel": "theta_feedback",
    "two_channels": "theta_feedback",
    "two_channel_obs": "theta_feedback",
    "2ch": "theta_feedback",
    "wave": "wave",
    "wave_feedback": "wave",
    "wave_sum": "wave",
    "combined": "wave",
    "formula_signal": "wave",
}

_CHANNEL_TO_OBSERVATION = {
    "obs": "dth_tot",
    "action": "dth_tot",
    "dth": "dth_neighbours",
    "thdot": "dth_neighbours_plus_thdot",
    "paper": "dth_tot",
    "k2_positive": "dth_tot",
    "k2_negative": "dth_tot",
    "tail_wave": "dth_tot",
    "tail_wave_residual": "dth_tot",
    "theta_feedback": "dth_tot_plus_friction_thdot",
    "wave": "dth_wave_feedback",
}

_CHANNEL_TO_DEFAULT_CONTROL = {
    "obs": "direct",
    "action": "formula",
    "dth": "direct",
    "thdot": "direct",
    "paper": "nonreciprocity",
    "k2_positive": "fixed_k1_k2_positive",
    "k2_negative": "fixed_k1_k2_negative",
    "tail_wave": "tail_wave",
    "tail_wave_residual": "tail_wave_residual",
    "theta_feedback": "direct",
    "wave": "direct",
}

_OBSERVATION_TO_CHANNEL = {
    "dth_tot": "obs",
    "dth_neighbours": "dth",
    "dth_neighbours_plus_thdot": "thdot",
    "dth_tot_plus_friction_thdot": "theta_feedback",
    "dth_tot_plus_feedback_thdot": "theta_feedback",
    "dth_wave_feedback": "wave",
}

_CONTROL_MODE_ALIASES = {
    "direct": "direct",
    "torque": "direct",
    "raw": "direct",
    "single": "direct",
    "single_channel": "direct",
    "formula": "formula",
    "action": "formula",
    "gain": "formula",
    "feedback": "formula",
    "k1k2": "formula",
    "k1_k2": "formula",
    "two": "formula",
    "two_channel": "formula",
    "action_formula": "formula",
    "wave_formula": "formula",
    "wave_feedback": "formula",
    "tail_wave": "tail_wave",
    "tail_curl_wave": "tail_wave",
    "curl_wave": "tail_wave",
    "tail_wave_residual": "tail_wave_residual",
    "scratch_wr": "tail_wave_residual",
    "wave_residual": "tail_wave_residual",
    "paper": "nonreciprocity",
    "paper_formula": "nonreciprocity",
    "nonreciprocity": "nonreciprocity",
    "non_reciprocity": "nonreciprocity",
    "nonreciprocal": "nonreciprocity",
    "odd": "nonreciprocity",
    "odd_elasticity": "nonreciprocity",
    "kappa_alpha": "nonreciprocity",
    "k2_positive": "fixed_k1_k2_positive",
    "k2_pos": "fixed_k1_k2_positive",
    "positive_k2": "fixed_k1_k2_positive",
    "fixed_k1_positive": "fixed_k1_k2_positive",
    "fixed_k1_k2_positive": "fixed_k1_k2_positive",
    "k2_negative": "fixed_k1_k2_negative",
    "k2_neg": "fixed_k1_k2_negative",
    "negative_k2": "fixed_k1_k2_negative",
    "fixed_k1_negative": "fixed_k1_k2_negative",
    "fixed_k1_k2_negative": "fixed_k1_k2_negative",
}


def _normalise_key(value: str | None, default: str = "") -> str:
    if value is None:
        value = default
    return str(value).strip().lower().replace("-", "_")


def _is_auto(value: str | None) -> bool:
    return value is None or _normalise_key(value) in {"", "auto", "checkpoint"}


def normalise_channel(channel: str | None = "obs") -> str:
    """Normalize user-facing channel aliases to one canonical channel name."""
    key = _normalise_key(channel, "obs")
    if key in {"", "auto", "checkpoint"}:
        key = "obs"
    if key not in _CHANNEL_ALIASES:
        raise ValueError(
            f"Unsupported channel {channel!r}. Main choices: 'dth', 'thdot', "
            "'obs', 'action', 'paper', 'k2_positive', 'k2_negative', "
            "'tail_wave', and 'tail_wave_residual'. "
            "Optional observation ablations: 'theta_feedback', 'wave'."
        )
    return _CHANNEL_ALIASES[key]


# Backward-compatible US spelling used by some generated scripts.
normalize_channel = normalise_channel


def normalise_control_mode(mode: str | None = "direct") -> str:
    """Normalize an explicit environment control mode.

    `auto`/`checkpoint` are accepted for compatibility and resolve to `direct`.
    Scripts that need channel-specific defaults should call `channel_config`, which
    maps `--channel action --control-mode auto` to formula mode.
    """
    key = _normalise_key(mode, "direct")
    if key in {"", "auto", "checkpoint"}:
        return "direct"
    if key not in _CONTROL_MODE_ALIASES:
        raise ValueError(
            f"Unsupported control mode {mode!r}. Use one of: "
            f"{', '.join(CONTROL_MODE_CHOICES)}."
        )
    return _CONTROL_MODE_ALIASES[key]


def resolve_control_mode(mode: str | None = "direct") -> str:
    """Backward-compatible helper for older scripts."""
    return normalise_control_mode(mode)


def resolve_observation_func(channel: str | None = "obs", observation_func: str | None = None) -> tuple[str, str]:
    """Return `(observation_func, canonical_channel)` for a channel preset."""
    canonical = normalise_channel(channel)
    if not _is_auto(observation_func):
        key = _normalise_key(observation_func)
        # Allow accidental use of a channel alias in --observation-func.
        if key in _CHANNEL_ALIASES:
            canonical = normalise_channel(key)
            return _CHANNEL_TO_OBSERVATION[canonical], canonical
        # Keep the requested channel/control semantics and only override the raw
        # observation function. This lets users test formula control with a custom obs.
        return str(observation_func), canonical
    return _CHANNEL_TO_OBSERVATION[canonical], canonical


def channel_config(
    channel: str = "obs",
    *,
    observation_func: str | None = None,
    control_mode: str | None = None,
) -> tuple[str, str, str]:
    """Resolve a channel preset into `(canonical_channel, observation_func, control_mode)`.

    Main presets:
      dth          -> paper observation [dtheta_(i-1), dtheta_(i+1)],
                      actor outputs active torque directly.
      thdot        -> paper observation [dtheta_(i-1), dtheta_(i+1), theta_dot_i],
                      actor outputs active torque directly.
      obs          -> observation=dth_tot, actor outputs one direct torque value.
      action       -> observation=dth_tot, actor outputs [k1, k2] and the
                      simulator applies tau=k1*theta_diff+k2*theta_dot (F=1).
      paper        -> observation=dth_tot, actor outputs kappa_alpha and the
                      simulator adds the fixed passive -kappa*dtheta_i term.
      k2_positive  -> observation=dth_tot, fixed k1, actor output k2 > 0.
      k2_negative  -> observation=dth_tot, fixed k1, actor output k2 < 0.
    """
    observation_func_resolved, canonical = resolve_observation_func(channel, observation_func)
    if _is_auto(control_mode):
        control_mode_resolved = _CHANNEL_TO_DEFAULT_CONTROL[canonical]
    else:
        control_mode_resolved = resolve_control_mode(control_mode)
    return canonical, observation_func_resolved, control_mode_resolved


def default_observation_for_channel(channel: str | None) -> str:
    return channel_config(channel)[1]


def control_channel_for_channel(channel: str | None) -> str:
    return channel_config(channel)[2]


def channel_description(channel: str | None) -> str:
    canonical = normalise_channel(channel)
    if canonical == "dth":
        return "dth: paper observation [dtheta_prev,dtheta_next], policy outputs active torque directly"
    if canonical == "thdot":
        return "thdot: paper observation [dtheta_prev,dtheta_next,theta_dot_i], policy outputs active torque directly"
    if canonical == "obs":
        return "obs: obs = dtheta_next-dtheta_prev, policy outputs physical torque directly"
    if canonical == "action":
        return "action: obs = dtheta_next-dtheta_prev, policy outputs [k1,k2], env applies tau=k1*theta_diff+k2*theta_dot with F=1"
    if canonical == "paper":
        return "paper: policy outputs kappa_alpha; env applies tau=-kappa*dtheta_i+kappa_alpha*(dtheta_next-dtheta_prev)"
    if canonical == "k2_positive":
        return "k2_positive: k1 is fixed, policy outputs strictly positive k2, env applies tau=k1*theta_diff+k2*theta_dot with F=1"
    if canonical == "k2_negative":
        return "k2_negative: k1 is fixed, policy outputs strictly negative k2, env applies tau=k1*theta_diff+k2*theta_dot with F=1"
    if canonical == "tail_wave":
        return "tail_wave: one global policy outputs six tail-to-head target-curvature PD parameters"
    if canonical == "tail_wave_residual":
        return "tail_wave_residual: one global scratch policy outputs six wave parameters plus per-joint residual K1/K2 pairs"
    if canonical == "theta_feedback":
        return "theta_feedback: obs = [dtheta_next-dtheta_prev, theta_dot], policy outputs torque directly; F=1"
    return "wave: obs = dtheta_next-dtheta_prev+theta_dot, policy outputs torque directly; F=1"


def channel_label(channel: str | None, observation_func: str, control_mode: str) -> str:
    try:
        c = normalise_channel(channel)
    except Exception:
        c = str(channel or "custom")
    return f"{c}/{observation_func}/{control_mode}"


def channel_slug(channel: str | None, observation_func: str, control_mode: str) -> str:
    """Filesystem-safe channel label for result folder names."""
    return channel_label(channel, observation_func, control_mode).replace("/", "_").replace("-", "_").replace(" ", "_")


def infer_channel_from_metadata(metadata: dict[str, Any]) -> str:
    """Infer a user-facing channel from old/new checkpoint metadata."""
    if not metadata:
        return "obs"
    observation_func = str(metadata.get("observation_func", "dth_tot"))
    saved = metadata.get("channel")
    if saved:
        try:
            saved_key = _normalise_key(str(saved))
            # Backward compatibility: an older README used `wave` for the two-field
            # observation. Keep old checkpoints readable.
            if saved_key == "wave" and observation_func == "dth_tot_plus_friction_thdot":
                return "theta_feedback"
            return normalise_channel(str(saved))
        except Exception:
            pass
    saved_control = metadata.get("control_mode", metadata.get("control_channel", "direct"))
    control_key = _normalise_key(str(saved_control))
    if control_key in {"formula", "action", "gain", "feedback", "k1k2", "k1_k2"}:
        return "action"
    if control_key in {"paper", "paper_formula", "nonreciprocity", "non_reciprocity", "nonreciprocal", "odd", "odd_elasticity", "kappa_alpha"}:
        return "paper"
    if control_key in {"k2_positive", "k2_pos", "positive_k2", "fixed_k1_positive", "fixed_k1_k2_positive"}:
        return "k2_positive"
    if control_key in {"k2_negative", "k2_neg", "negative_k2", "fixed_k1_negative", "fixed_k1_k2_negative"}:
        return "k2_negative"
    return _OBSERVATION_TO_CHANNEL.get(observation_func, "obs")


def infer_control_mode_from_metadata(metadata: dict[str, Any]) -> str:
    if not metadata:
        return "direct"
    return resolve_control_mode(metadata.get("control_mode", metadata.get("control_channel", "direct")))


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


def find_latest_checkpoint(project_root: Path) -> Path:
    """Find newest checkpoint_*.pt under common result folders."""
    candidates: list[Path] = []
    for root in [project_root / "results", project_root / "training" / "results", project_root]:
        if root.exists():
            candidates.extend(root.rglob("checkpoint_*.pt"))
    candidates = sorted(set(candidates), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No checkpoint_*.pt found. Train first or pass --checkpoint PATH.")
    return candidates[0]


def load_checkpoint(path: Path):
    """Load a local checkpoint, allowing the metadata object saved by Sacred-era scripts."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class FirstOrderGaussian(torch.nn.Module):
    """Optional activation kept for compatibility with older experiments."""

    def __init__(self, std: float = 1):
        super().__init__()
        self.std = std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.pi * x * torch.exp((-x**2) / (2 * self.std**2)) / self.std


class BiasedNormalParamExtractor(NormalParamExtractor):
    """Matches the PPO actor head used by this project."""

    def forward(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        tensor, *others = tensors
        loc, scale = tensor.chunk(2, -1)
        scale = self.scale_mapping(scale) + self.scale_lb
        return (loc, scale, *others)


class SafeTanhNormal(TanhNormal):
    """Numerically safer log_prob for PPO training."""

    def log_prob(self, value):
        log_prob = super().log_prob(value)
        epsilon = 1e-6
        return torch.log(torch.exp(log_prob) + epsilon)


def choose_device(force_cpu: bool = False) -> torch.device:
    from torch import multiprocessing

    is_fork = multiprocessing.get_start_method() == "fork"
    cuda_available = torch.cuda.is_available()

    if force_cpu:
        device = torch.device("cpu")
        print("force_cpu=True, using CPU.")
    elif cuda_available and not is_fork:
        device = torch.device("cuda:0")
        print("CUDA is available, using GPU.")
        print("CUDA device:", torch.cuda.get_device_name(device))
        print("CUDA version used by PyTorch:", torch.version.cuda)
    else:
        device = torch.device("cpu")
        if not cuda_available:
            print("CUDA is not available in this Python environment; falling back to CPU.")
        elif is_fork:
            print("Multiprocessing start method is 'fork'; falling back to CPU for safety.")

    print("Using device:", device)
    return device
