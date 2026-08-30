"""Actor-only O1-sham extension over the immutable parent trainer.

The environment and critic retain the full two-channel observation.  In
``spatial_only_sham`` mode, a parameter-free forward pre-hook replaces only the
second input channel seen by the actor backbone with exact zeros.  Installing
the hook after component construction preserves the actor/critic/optimiser and
RNG initialisation hashes of the parent formal run.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import torch


DEFAULT_PARENT_TRAINER = Path(
    "C:\\Users\\PUBLIC_USER\\CloudStorage\\Desktop\\finalproject\\job\\roll_learning\\"
    "obs2_roll_repro_v2_1_formal_20260803_r2\\_control\\code_snapshot\\"
    "training\\train_metamaterial.py"
)
ALLOWED_ACTOR_OBSERVATION_MODES = ("full_o2", "spatial_only_sham")
ACTOR_OBSERVATION_FLAG = "--actor-observation-mode"


def _load_parent_trainer() -> Any:
    path = Path(os.environ.get("FORMAL_PARENT_TRAINER", str(DEFAULT_PARENT_TRAINER))).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Immutable parent trainer not found: {path}")
    # The parent trainer imports sibling modules (rlmm_common and
    # simulation_command) exactly as it did when executed as a script.
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("formal_parent_train_metamaterial", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_BASE = _load_parent_trainer()
_BASE_PARSE_ARGS = _BASE.parse_args
_BASE_BUILD_COMPONENTS = _BASE.build_components
_BASE_MAIN = _BASE.main
_BASE_SAVE_PARAMS = _BASE.save_params
_BASE_WRITE_JSON = _BASE.write_json
_ACTIVE_ACTOR_OBSERVATION_MODE = "full_o2"

# The formal audit wrapper patches this exact attribute.
metamaterial = _BASE.metamaterial


def split_actor_observation_mode(argv: Sequence[str]) -> tuple[str, list[str]]:
    """Extract exactly zero or one actor-observation flag without reordering args."""

    mode = "full_o2"
    cleaned: list[str] = []
    occurrences = 0
    index = 0
    while index < len(argv):
        token = str(argv[index])
        if token == ACTOR_OBSERVATION_FLAG:
            occurrences += 1
            if index + 1 >= len(argv):
                raise ValueError(f"{ACTOR_OBSERVATION_FLAG} requires a value")
            mode = str(argv[index + 1])
            index += 2
            continue
        prefix = f"{ACTOR_OBSERVATION_FLAG}="
        if token.startswith(prefix):
            occurrences += 1
            mode = token[len(prefix) :]
            index += 1
            continue
        cleaned.append(token)
        index += 1
    if occurrences > 1:
        raise ValueError(f"{ACTOR_OBSERVATION_FLAG} may appear at most once")
    if mode not in ALLOWED_ACTOR_OBSERVATION_MODES:
        raise ValueError(
            f"Unsupported actor observation mode {mode!r}; "
            f"expected one of {ALLOWED_ACTOR_OBSERVATION_MODES}"
        )
    return mode, cleaned


def spatial_only_sham_tensor(observation: torch.Tensor) -> torch.Tensor:
    """Return ``[s_i, 0]`` while preserving shape, dtype, device and gradients."""

    if not torch.is_tensor(observation):
        raise TypeError(f"Actor observation must be a tensor, got {type(observation).__name__}")
    if observation.ndim < 1 or int(observation.shape[-1]) != 2:
        raise RuntimeError(
            "O1-sham requires the locked two-channel O2 actor input; "
            f"received shape {tuple(observation.shape)}"
        )
    masked = observation.clone()
    masked[..., 1] = 0
    return masked


def _spatial_only_pre_hook(module: torch.nn.Module, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
    if len(inputs) != 1:
        raise RuntimeError(
            f"Locked actor backbone expected one positional tensor, received {len(inputs)}"
        )
    return (spatial_only_sham_tensor(inputs[0]),)


def locate_actor_backbone(policy: torch.nn.Module) -> torch.nn.Module:
    candidates = [
        module
        for module in policy.modules()
        if isinstance(module, _BASE.MultiAgentMLP)
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "Expected exactly one MultiAgentMLP actor backbone inside policy; "
            f"found {len(candidates)}"
        )
    return candidates[0]


def install_actor_observation_mode(policy: torch.nn.Module, mode: str) -> torch.nn.Module:
    """Install the locked actor-only mode without adding state-dict entries."""

    if mode not in ALLOWED_ACTOR_OBSERVATION_MODES:
        raise ValueError(mode)
    backbone = locate_actor_backbone(policy)
    if hasattr(backbone, "_formal_actor_observation_mode"):
        raise RuntimeError("Actor observation mode has already been installed")
    if mode == "spatial_only_sham":
        handle = backbone.register_forward_pre_hook(_spatial_only_pre_hook)
        # Keep the handle alive for the lifetime of the backbone.  Neither
        # attribute participates in state_dict serialisation.
        backbone._formal_actor_observation_hook_handle = handle
    backbone._formal_actor_observation_mode = mode
    policy._formal_actor_observation_mode = mode
    return backbone


def actor_observation_mode_from_metadata(metadata: dict[str, Any]) -> str:
    mode = metadata.get("actor_observation_mode")
    if mode is None:
        mode = metadata.get("training_args", {}).get("actor_observation_mode", "full_o2")
    mode = str(mode)
    if mode not in ALLOWED_ACTOR_OBSERVATION_MODES:
        raise RuntimeError(f"Checkpoint contains unsupported actor observation mode: {mode!r}")
    return mode


def install_actor_observation_from_metadata(
    policy: torch.nn.Module, metadata: dict[str, Any]
) -> torch.nn.Module:
    """Required by replay/evaluation loaders after reconstructing a policy."""

    return install_actor_observation_mode(policy, actor_observation_mode_from_metadata(metadata))


def _annotated_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    value = dict(metadata)
    value.update(
        {
            "actor_observation_mode": _ACTIVE_ACTOR_OBSERVATION_MODE,
            "critic_observation_mode": "full_o2",
            "actor_observation_intervention": (
                "none"
                if _ACTIVE_ACTOR_OBSERVATION_MODE == "full_o2"
                else "second_channel_replaced_by_exact_zero_at_actor_backbone"
            ),
            "physical_k2_theta_dot_enabled": True,
        }
    )
    return value


def _save_params_with_observation_metadata(obj: dict[str, Any], path: Path) -> None:
    value = dict(obj)
    if isinstance(value.get("metadata"), dict):
        value["metadata"] = _annotated_metadata(value["metadata"])
    _BASE_SAVE_PARAMS(value, path)


def _write_json_with_observation_metadata(path: Path, value: Any) -> None:
    output = value
    if isinstance(value, dict) and Path(path).name == "metadata.json":
        output = _annotated_metadata(value)
    elif isinstance(value, dict) and isinstance(value.get("metadata"), dict):
        output = dict(value)
        output["metadata"] = _annotated_metadata(value["metadata"])
    _BASE_WRITE_JSON(path, output)


def parse_args() -> Any:
    global _ACTIVE_ACTOR_OBSERVATION_MODE
    mode, cleaned = split_actor_observation_mode(sys.argv[1:])
    original = sys.argv[:]
    try:
        sys.argv = [original[0], *cleaned]
        args = _BASE_PARSE_ARGS()
    finally:
        sys.argv = original
    args.actor_observation_mode = mode
    args.critic_observation_mode = "full_o2"
    _ACTIVE_ACTOR_OBSERVATION_MODE = mode
    return args


def build_components(env: Any, args: Any, device: torch.device) -> tuple[Any, ...]:
    result = _BASE_BUILD_COMPONENTS(env, args, device)
    policy = result[0]
    install_actor_observation_mode(policy, str(args.actor_observation_mode))
    return result


def main() -> Any:
    # An outer audit wrapper may replace this module's parse/build functions.
    # Synchronise those replacements into the immutable parent's globals before
    # entering its main function.
    _BASE.parse_args = globals()["parse_args"]
    _BASE.build_components = globals()["build_components"]
    _BASE.metamaterial = globals()["metamaterial"]
    _BASE.save_params = _save_params_with_observation_metadata
    _BASE.write_json = _write_json_with_observation_metadata
    return _BASE_MAIN()


def __getattr__(name: str) -> Any:
    return getattr(_BASE, name)


if __name__ == "__main__":
    main()
