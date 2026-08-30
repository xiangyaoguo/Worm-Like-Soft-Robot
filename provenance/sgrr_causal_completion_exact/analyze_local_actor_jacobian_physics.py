from __future__ import annotations

"""Read-only local-policy and control-boundary physics analysis.

This script deliberately does not construct an environment, run an episode, train a
policy, mutate a checkpoint, or alter a trace.  It reads the five frozen formal
Rroll checkpoint_1500 policies and genuine recorded trajectories, then writes a
separate analysis bundle under this study directory.

The formal actor consists of eight independent local MLPs.  Joint j receives only
[delta_theta_j, theta_dot_j] and emits only [K1_j, K2_j].  Consequently the direct
16 x 16 policy Jacobian is block diagonal by construction: eight 2 x 2 local
blocks and 224 structurally zero cross-joint entries.  Physical cross-joint effects
belong to the closed-loop intervention analysis, not this direct actor Jacobian.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
FORMAL_ROOT_DEFAULT = Path(
    r"C:\Users\PUBLIC_USER\CloudStorage\Desktop\finalproject\job\roll_learning"
    r"\obs2_roll_repro_v2_1_formal_20260803_r2"
)
OLD_MECHANISM_ROOT = HERE.parent / "mechanism_runtime_exact"
DEFAULT_OUTPUT = HERE / "analysis_local_actor"

POLICY_KEYS = {
    "w0": "module.0.module.0.params.0.weight",
    "b0": "module.0.module.0.params.0.bias",
    "w2": "module.0.module.0.params.2.weight",
    "b2": "module.0.module.0.params.2.bias",
    "w4": "module.0.module.0.params.4.weight",
    "b4": "module.0.module.0.params.4.bias",
}
EXPECTED_SHAPES = {
    "w0": (8, 256, 2),
    "b0": (8, 256),
    "w2": (8, 256, 256),
    "b2": (8, 256),
    "w4": (8, 4, 256),
    "b4": (8, 4),
}
TRAINING_SEEDS = (9201, 9202, 9203, 9204, 9205)
JOINT_COUNT = 8
K_SCALE = 100.0
MAX_TORQUE = 9.0

EVENT_BINS = (
    "official_prelaunch",
    "official_rolling_outside_pulse",
    "official_pulse_q1",
    "official_pulse_q2",
    "official_pulse_q3",
    "official_pulse_q4",
    "official_pulse_q5",
)
EVENT_BIN_TO_CODE = {name: index for index, name in enumerate(EVENT_BINS)}
DEFAULT_BASELINE_REGEX = (
    r"^(C11|BASELINE_C11|C11_BASELINE|RROLL_BASELINE|BASELINE_RROLL)$"
)


@dataclass(frozen=True)
class PolicyBundle:
    seed: int
    checkpoint_path: Path
    checkpoint_sha256: str
    tensors: dict[str, torch.Tensor]
    metadata: dict[str, Any]
    feedback_gain: float


@dataclass(frozen=True)
class TraceRecord:
    path: Path
    source_root: Path
    sha256: str
    content_sha256: str
    training_seed: int
    episode_seed: int
    condition_id: str
    observation: np.ndarray
    positions: np.ndarray
    support_index: np.ndarray
    contact_strength: np.ndarray
    recorded_roll_action: np.ndarray | None


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def scalar_from_array(value: np.ndarray | Any, default: Any = None) -> Any:
    array = np.asarray(value)
    if array.size == 0:
        return default
    item = array.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    if isinstance(item, np.generic):
        return item.item()
    return item


def metadata_value(metadata: Mapping[str, Any], key: str, default: Any = None) -> Any:
    candidates: list[Mapping[str, Any]] = [metadata]
    for container_key in (
        "training_args",
        "environment_config",
        "env_config",
        "config",
    ):
        value = metadata.get(container_key)
        if isinstance(value, Mapping):
            candidates.append(value)
    for candidate in candidates:
        if key in candidate:
            return candidate[key]
    return default


def checkpoint_path(formal_root: Path, seed: int) -> Path:
    return (
        formal_root
        / "formal"
        / "runs"
        / f"formal__seed{seed}__Rroll"
        / "checkpoint_1500.pt"
    )


def load_policy_bundle(formal_root: Path, seed: int) -> PolicyBundle:
    path = checkpoint_path(formal_root, seed)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    policy = payload.get("policy")
    metadata = dict(payload.get("metadata", {}))
    if not isinstance(policy, Mapping):
        raise RuntimeError(f"Missing policy state_dict: {path}")
    if set(policy) != set(POLICY_KEYS.values()):
        raise RuntimeError(
            f"Unexpected policy signature for seed {seed}: {sorted(policy)}"
        )
    tensors: dict[str, torch.Tensor] = {}
    for alias, key in POLICY_KEYS.items():
        value = policy[key].detach().to(device="cpu", dtype=torch.float32).contiguous()
        if tuple(value.shape) != EXPECTED_SHAPES[alias]:
            raise RuntimeError(
                f"Shape mismatch seed {seed} {key}: {tuple(value.shape)} != "
                f"{EXPECTED_SHAPES[alias]}"
            )
        if not bool(torch.isfinite(value).all().item()):
            raise RuntimeError(f"Non-finite policy tensor: seed {seed} {key}")
        tensors[alias] = value

    required_metadata = {
        "algorithm": "ppo",
        "observation_func": "dth_tot_plus_friction_thdot",
        "control_mode": "formula",
        "per_joint_k1_k2": True,
        "policy_parameter_sharing": "independent_per_joint",
        "reward_func": "obs2_roll_repro_v2_1",
    }
    for key, expected in required_metadata.items():
        actual = metadata_value(metadata, key)
        if actual != expected:
            raise RuntimeError(
                f"Frozen checkpoint contract drift seed {seed}: {key}={actual!r}, "
                f"expected {expected!r}"
            )
    if float(metadata_value(metadata, "k_action_scale", math.nan)) != K_SCALE:
        raise RuntimeError(f"K scale is not {K_SCALE:g} in seed {seed}")
    source_checkpoint = metadata_value(metadata, "source_checkpoint")
    if source_checkpoint not in (None, ""):
        raise RuntimeError(f"Formal seed {seed} unexpectedly cites a source checkpoint")

    feedback_value = metadata_value(metadata, "feedback_gain", 1.0)
    feedback_gain = float(feedback_value)
    if not math.isfinite(feedback_gain):
        raise RuntimeError(f"Non-finite feedback_gain seed {seed}: {feedback_value!r}")
    return PolicyBundle(
        seed=seed,
        checkpoint_path=path.resolve(),
        checkpoint_sha256=sha256_file(path),
        tensors=tensors,
        metadata=metadata,
        feedback_gain=feedback_gain,
    )


def actor_forward_physical(
    tensors: Mapping[str, torch.Tensor], observations: np.ndarray, batch_size: int = 8192
) -> np.ndarray:
    """Return physical K in [N, 8, 2] for actual observations."""
    observations = np.asarray(observations, dtype=np.float32)
    if observations.ndim != 3 or observations.shape[1:] != (8, 2):
        raise ValueError(f"Observation shape must be [N,8,2], got {observations.shape}")
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            obs = torch.as_tensor(observations[start : start + batch_size]).permute(1, 0, 2)
            h1 = torch.tanh(
                torch.einsum("aoi,ani->ano", tensors["w0"], obs)
                + tensors["b0"].unsqueeze(1)
            )
            h2 = torch.tanh(
                torch.einsum("aoi,ani->ano", tensors["w2"], h1)
                + tensors["b2"].unsqueeze(1)
            )
            loc = (
                torch.einsum("aoi,ani->ano", tensors["w4"], h2)
                + tensors["b4"].unsqueeze(1)
            )[..., :2]
            chunks.append((K_SCALE * loc).permute(1, 0, 2).cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float64, copy=False)


def analytic_local_jacobian(
    tensors: Mapping[str, torch.Tensor], observations: np.ndarray, batch_size: int = 4096
) -> np.ndarray:
    """Compute d[K1,K2]/d[delta_theta,theta_dot] in physical K units.

    Output shape is [N, 8, 2, 2].  Output channel is the third axis and input
    channel the fourth.  Cross-joint entries are absent here because the actor has
    eight independent local subnetworks.
    """
    observations = np.asarray(observations, dtype=np.float32)
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(observations), batch_size):
            obs = torch.as_tensor(observations[start : start + batch_size]).permute(1, 0, 2)
            h1 = torch.tanh(
                torch.einsum("aoi,ani->ano", tensors["w0"], obs)
                + tensors["b0"].unsqueeze(1)
            )
            h2 = torch.tanh(
                torch.einsum("aoi,ani->ano", tensors["w2"], h1)
                + tensors["b2"].unsqueeze(1)
            )
            # Chain rule: W4 * diag(1-h2^2) * W2 * diag(1-h1^2) * W0.
            propagated = tensors["w0"].unsqueeze(1) * (1.0 - h1.square()).unsqueeze(-1)
            propagated = torch.einsum(
                "aoh,anhi->anoi", tensors["w2"], propagated
            )
            propagated = propagated * (1.0 - h2.square()).unsqueeze(-1)
            local = K_SCALE * torch.einsum(
                "akh,anhi->anki", tensors["w4"][:, :2, :], propagated
            )
            chunks.append(local.permute(1, 0, 2, 3).cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float64, copy=False)


def full_jacobian_from_local(local: np.ndarray) -> np.ndarray:
    local = np.asarray(local)
    if local.shape[-3:] != (8, 2, 2):
        raise ValueError(f"Local Jacobian shape must end in [8,2,2], got {local.shape}")
    prefix = local.shape[:-3]
    full = np.zeros(prefix + (16, 16), dtype=local.dtype)
    for joint in range(8):
        row = slice(2 * joint, 2 * joint + 2)
        col = slice(2 * joint, 2 * joint + 2)
        full[..., row, col] = local[..., joint, :, :]
    return full


def _infer_seed(path: Path, data: Mapping[str, np.ndarray]) -> int:
    for key in ("training_seed", "seed"):
        if key in data:
            return int(scalar_from_array(data[key]))
    match = re.search(r"seed[_-]?(\d+)", str(path), flags=re.IGNORECASE)
    if not match:
        raise RuntimeError(f"Cannot infer training seed from trace: {path}")
    return int(match.group(1))


def _infer_episode_seed(data: Mapping[str, np.ndarray]) -> int:
    for key in ("evaluation_seed", "episode_seed", "initial_state_seed"):
        if key in data:
            return int(scalar_from_array(data[key]))
    return -1


def _infer_condition(path: Path, data: Mapping[str, np.ndarray]) -> str:
    for key in ("condition_id", "condition", "case_id"):
        if key in data:
            return str(scalar_from_array(data[key]))
    return path.stem


def _first_present(data: Mapping[str, np.ndarray], names: Sequence[str]) -> np.ndarray | None:
    for name in names:
        if name in data:
            return np.asarray(data[name])
    return None


def read_trace(path: Path, source_root: Path) -> TraceRecord:
    with np.load(path, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}
    observation = _first_present(
        data, ("observation", "rroll_observation_input", "policy_observation")
    )
    positions = _first_present(data, ("positions", "trajectory_positions"))
    if observation is None or positions is None:
        raise RuntimeError(f"Trace lacks observation/positions arrays: {path}")
    observation = np.asarray(observation, dtype=np.float32)
    positions = np.asarray(positions, dtype=np.complex128)
    if observation.ndim != 3 or observation.shape[1:] != (8, 2):
        raise RuntimeError(f"Invalid observation shape {observation.shape}: {path}")
    if positions.ndim != 2 or positions.shape[1] != 10:
        raise RuntimeError(f"Invalid positions shape {positions.shape}: {path}")
    if len(positions) != len(observation) + 1:
        raise RuntimeError(
            f"Trace positions/observation length mismatch {positions.shape}/{observation.shape}: {path}"
        )
    if not np.isfinite(observation).all() or not np.isfinite(positions).all():
        raise RuntimeError(f"Trace contains NaN/Inf: {path}")

    support = _first_present(data, ("support_index", "fast_forward_support_index"))
    contact = _first_present(
        data, ("ground_contact_strength", "fast_forward_ground_contact_strength")
    )
    if support is None:
        support = np.full(len(positions), np.nan, dtype=np.float64)
    if contact is None:
        contact = np.full(len(positions), np.nan, dtype=np.float64)
    support = np.asarray(support, dtype=np.float64).reshape(-1)
    contact = np.asarray(contact, dtype=np.float64).reshape(-1)
    if len(support) != len(positions) or len(contact) != len(positions):
        raise RuntimeError(f"Support/contact length mismatch: {path}")

    action = _first_present(data, ("roll_action", "rroll_action", "policy_action"))
    recorded_roll_action: np.ndarray | None = None
    if action is not None:
        action = np.asarray(action, dtype=np.float32)
        if action.shape == observation.shape:
            recorded_roll_action = action

    content_digest = hashlib.sha256()
    content_digest.update(np.ascontiguousarray(observation).view(np.uint8))
    content_digest.update(np.ascontiguousarray(positions).view(np.uint8))
    return TraceRecord(
        path=path.resolve(),
        source_root=source_root.resolve(),
        sha256=sha256_file(path),
        content_sha256=content_digest.hexdigest(),
        training_seed=_infer_seed(path, data),
        episode_seed=_infer_episode_seed(data),
        condition_id=_infer_condition(path, data),
        observation=observation,
        positions=positions,
        support_index=support,
        contact_strength=contact,
        recorded_roll_action=recorded_roll_action,
    )


def default_trace_roots() -> list[Path]:
    candidates = [
        HERE / "traces",
        HERE / "trace_archive",
        HERE / "main_traces",
        HERE / "main" / "traces",
        HERE / "results" / "traces",
        OLD_MECHANISM_ROOT / "independent_trace_audit",
    ]
    return [path for path in candidates if path.is_dir()]


def discover_traces(
    roots: Sequence[Path], condition_pattern: re.Pattern[str]
) -> tuple[list[TraceRecord], list[dict[str, str]]]:
    records: list[TraceRecord] = []
    skipped: list[dict[str, str]] = []
    seen_content: set[str] = set()
    for root in roots:
        if not root.is_dir():
            skipped.append({"path": str(root), "reason": "trace_root_missing"})
            continue
        for path in sorted(root.rglob("*.npz")):
            # Never recursively ingest this script's own outputs.
            if DEFAULT_OUTPUT in path.parents or "analysis_local_actor" in path.parts:
                continue
            try:
                record = read_trace(path, root)
            except (KeyError, OSError, RuntimeError, ValueError) as exc:
                skipped.append({"path": str(path.resolve()), "reason": str(exc)})
                continue
            if record.training_seed not in TRAINING_SEEDS:
                skipped.append({"path": str(path.resolve()), "reason": "nonformal_training_seed"})
                continue
            if condition_pattern.search(record.condition_id) is None:
                skipped.append({"path": str(path.resolve()), "reason": "nonbaseline_condition"})
                continue
            if record.content_sha256 in seen_content:
                skipped.append({"path": str(path.resolve()), "reason": "duplicate_trace_content"})
                continue
            seen_content.add(record.content_sha256)
            records.append(record)
    records.sort(key=lambda item: (item.training_seed, item.episode_seed, str(item.path)))
    return records, skipped


def load_official_evaluator(formal_root: Path) -> Any:
    path = formal_root / "_control" / "code_snapshot" / "training" / "evaluate_fast_forward_roll.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("frozen_roll_evaluator_for_analysis", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import frozen evaluator: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite_or_none(values: np.ndarray) -> list[float | None]:
    return [float(value) if math.isfinite(float(value)) else None for value in values]


def official_metrics(module: Any, trace: TraceRecord) -> dict[str, Any]:
    args = module._parser().parse_args([])
    return module._episode_metrics(
        [row.copy() for row in trace.positions],
        "right",
        "left",
        args,
        finite_or_none(trace.support_index),
        finite_or_none(trace.contact_strength),
    )


def event_bins_from_metrics(steps: int, metrics: Mapping[str, Any]) -> np.ndarray:
    """Construct explicitly derived event-aligned bins.

    These are analysis bins, not hidden environment states and not reward phases.
    Valid pulse intervals come from the frozen official evaluator.  A pulse is split
    into five equal normalized-time quintiles.  Outside valid pulses, frames before
    the first pulse-matched launch (or first valid pulse if no matched launch exists)
    are `official_prelaunch`; later frames are `official_rolling_outside_pulse`.
    """
    pulses = list(metrics.get("roll_pulses", []))
    matched_launches = [int(value) for value in metrics.get("tail_launch_steps", [])]
    if matched_launches:
        transition = min(matched_launches)
    elif pulses:
        transition = min(int(pulse["start_step"]) for pulse in pulses)
    else:
        transition = steps
    codes = np.full(steps, EVENT_BIN_TO_CODE["official_rolling_outside_pulse"], dtype=np.int8)
    codes[: max(0, min(steps, transition))] = EVENT_BIN_TO_CODE["official_prelaunch"]
    for pulse in pulses:
        start = max(0, min(steps - 1, int(pulse["start_step"])))
        end = max(start, min(steps - 1, int(pulse["end_step"])))
        denominator = max(end - start + 1, 1)
        for step in range(start, end + 1):
            quintile = min(4, int(5 * (step - start) / denominator))
            codes[step] = EVENT_BIN_TO_CODE[f"official_pulse_q{quintile + 1}"]
    return codes


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    mask = np.isfinite(left) & np.isfinite(right)
    if int(np.sum(mask)) < 3:
        return math.nan
    left_valid = left[mask]
    right_valid = right[mask]
    if float(np.std(left_valid)) <= 1e-15 or float(np.std(right_valid)) <= 1e-15:
        return math.nan
    return float(np.corrcoef(left_valid, right_valid)[0, 1])


def distribution(values: np.ndarray, prefix: str) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": math.nan,
            f"{prefix}_median": math.nan,
            f"{prefix}_q25": math.nan,
            f"{prefix}_q75": math.nan,
            f"{prefix}_std": math.nan,
            f"{prefix}_abs_mean": math.nan,
        }
    return {
        f"{prefix}_n": int(len(finite)),
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_q25": float(np.quantile(finite, 0.25)),
        f"{prefix}_q75": float(np.quantile(finite, 0.75)),
        f"{prefix}_std": float(np.std(finite, ddof=0)),
        f"{prefix}_abs_mean": float(np.mean(np.abs(finite))),
    }


def summarize_jacobian(
    seed: int, jacobian: np.ndarray, event_codes: np.ndarray, zero_tolerance: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    output_names = ("K1", "K2")
    input_names = ("delta_theta", "theta_dot")
    for phase_code, event_bin in enumerate(EVENT_BINS):
        mask = event_codes == phase_code
        if not np.any(mask):
            continue
        phase_local_median = np.median(jacobian[mask], axis=0)
        full = full_jacobian_from_local(phase_local_median)
        for joint in range(8):
            for output_index, output_name in enumerate(output_names):
                for input_index, input_name in enumerate(input_names):
                    values = jacobian[mask, joint, output_index, input_index]
                    positive = float(np.mean(values > zero_tolerance))
                    negative = float(np.mean(values < -zero_tolerance))
                    near_zero = float(np.mean(np.abs(values) <= zero_tolerance))
                    row: dict[str, Any] = {
                        "training_seed": seed,
                        "event_bin": event_bin,
                        "joint": f"J{joint + 1:02d}",
                        "K_channel": output_name,
                        "observation_channel": input_name,
                        "positive_fraction": positive,
                        "negative_fraction": negative,
                        "near_zero_fraction": near_zero,
                        "sign_consistency": max(positive, negative),
                    }
                    row.update(distribution(values, "derivative"))
                    rows.append(row)
        for output_index in range(16):
            out_joint, out_channel = divmod(output_index, 2)
            for input_index in range(16):
                in_joint, in_channel = divmod(input_index, 2)
                full_rows.append(
                    {
                        "training_seed": seed,
                        "event_bin": event_bin,
                        "output_index": output_index,
                        "output_label": f"J{out_joint + 1:02d}_{output_names[out_channel]}",
                        "input_index": input_index,
                        "input_label": f"J{in_joint + 1:02d}_{input_names[in_channel]}",
                        "median_derivative": float(full[output_index, input_index]),
                        "structural_zero": bool(out_joint != in_joint),
                    }
                )
    return rows, full_rows


def clip_shapley(
    u1: np.ndarray, u2: np.ndarray, bound: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    f1 = np.clip(u1, -bound, bound)
    f2 = np.clip(u2, -bound, bound)
    f12 = np.clip(u1 + u2, -bound, bound)
    phi1 = 0.5 * (f1 + (f12 - f2))
    phi2 = 0.5 * (f2 + (f12 - f1))
    reconstruction = float(np.max(np.abs((phi1 + phi2) - f12)))
    return phi1, phi2, f12, reconstruction


def summarize_physics(
    seed: int,
    observations: np.ndarray,
    physical_k: np.ndarray,
    event_codes: np.ndarray,
    support: np.ndarray,
    contact: np.ndarray,
    feedback_gain: float,
    max_torque: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    delta = observations[..., 0]
    theta_dot = observations[..., 1]
    u1 = physical_k[..., 0] * delta
    u2 = physical_k[..., 1] * feedback_gain * theta_dot
    phi1, phi2, clipped_tau, reconstruction = clip_shapley(u1, u2, max_torque)
    power1 = phi1 * theta_dot
    power2 = phi2 * theta_dot
    saturated = np.abs(u1 + u2) >= max_torque
    channel_values = (
        ("K1", physical_k[..., 0], u1, phi1, power1),
        ("K2", physical_k[..., 1], u2, phi2, power2),
    )
    rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for phase_code, event_bin in enumerate(EVENT_BINS):
        mask = event_codes == phase_code
        if not np.any(mask):
            continue
        event_rows.append(
            {
                "training_seed": seed,
                "event_bin": event_bin,
                "step_count": int(np.sum(mask)),
                "support_index_mean": float(np.nanmean(support[mask]))
                if np.any(np.isfinite(support[mask]))
                else math.nan,
                "support_index_std": float(np.nanstd(support[mask]))
                if np.any(np.isfinite(support[mask]))
                else math.nan,
                "contact_strength_mean": float(np.nanmean(contact[mask]))
                if np.any(np.isfinite(contact[mask]))
                else math.nan,
                "contact_strength_std": float(np.nanstd(contact[mask]))
                if np.any(np.isfinite(contact[mask]))
                else math.nan,
                "saturation_rate_all_joints": float(np.mean(saturated[mask])),
                "clipped_active_torque_abs_mean": float(np.mean(np.abs(clipped_tau[mask]))),
            }
        )
        for joint in range(8):
            for channel, k_values, u_values, phi_values, power_values in channel_values:
                selected_k = k_values[mask, joint]
                selected_u = u_values[mask, joint]
                selected_phi = phi_values[mask, joint]
                selected_power = power_values[mask, joint]
                row: dict[str, Any] = {
                    "training_seed": seed,
                    "event_bin": event_bin,
                    "joint": f"J{joint + 1:02d}",
                    "K_channel": channel,
                    "feedback_gain": feedback_gain,
                    "max_torque": max_torque,
                    "saturation_rate": float(np.mean(saturated[mask, joint])),
                    "positive_power_proxy_fraction": float(np.mean(selected_power > 0.0)),
                    "negative_power_proxy_fraction": float(np.mean(selected_power < 0.0)),
                    "phi_vs_contact_pearson": safe_correlation(
                        selected_phi, contact[mask]
                    ),
                    "power_proxy_vs_contact_pearson": safe_correlation(
                        selected_power, contact[mask]
                    ),
                    "abs_phi_vs_support_pearson": safe_correlation(
                        np.abs(selected_phi), support[mask]
                    ),
                }
                row.update(distribution(selected_k, "physical_K"))
                row.update(distribution(selected_u, "unclipped_channel_torque"))
                row.update(distribution(selected_phi, "shapley_clipped_torque"))
                row.update(distribution(selected_power, "active_power_proxy"))
                rows.append(row)
    return rows, event_rows, reconstruction


def local_forward_double(tensors: Mapping[str, torch.Tensor], joint: int, x: torch.Tensor) -> torch.Tensor:
    w0 = tensors["w0"][joint].to(dtype=torch.float64)
    b0 = tensors["b0"][joint].to(dtype=torch.float64)
    w2 = tensors["w2"][joint].to(dtype=torch.float64)
    b2 = tensors["b2"][joint].to(dtype=torch.float64)
    w4 = tensors["w4"][joint, :2].to(dtype=torch.float64)
    b4 = tensors["b4"][joint, :2].to(dtype=torch.float64)
    h1 = torch.tanh(w0 @ x + b0)
    h2 = torch.tanh(w2 @ h1 + b2)
    return K_SCALE * (w4 @ h2 + b4)


def analytic_one_double(
    tensors: Mapping[str, torch.Tensor], joint: int, observation: np.ndarray
) -> np.ndarray:
    x = torch.as_tensor(observation, dtype=torch.float64)
    w0 = tensors["w0"][joint].to(dtype=torch.float64)
    b0 = tensors["b0"][joint].to(dtype=torch.float64)
    w2 = tensors["w2"][joint].to(dtype=torch.float64)
    b2 = tensors["b2"][joint].to(dtype=torch.float64)
    w4 = tensors["w4"][joint, :2].to(dtype=torch.float64)
    h1 = torch.tanh(w0 @ x + b0)
    h2 = torch.tanh(w2 @ h1 + b2)
    result = K_SCALE * w4 @ torch.diag(1.0 - h2.square()) @ w2 @ torch.diag(
        1.0 - h1.square()
    ) @ w0
    return result.detach().cpu().numpy()


def validate_jacobian_samples(
    bundle: PolicyBundle,
    observations: np.ndarray,
    sample_count: int,
    finite_difference_epsilon: float,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    if len(observations) == 0:
        return []
    count = min(sample_count, len(observations))
    indices = rng.choice(len(observations), size=count, replace=False)
    rows: list[dict[str, Any]] = []
    for sample_ordinal, state_index in enumerate(indices):
        for joint in range(8):
            observation = observations[state_index, joint].astype(np.float64)
            analytic = analytic_one_double(bundle.tensors, joint, observation)
            x = torch.as_tensor(observation, dtype=torch.float64).requires_grad_(True)
            autograd = torch.autograd.functional.jacobian(
                lambda value: local_forward_double(bundle.tensors, joint, value),
                x,
                create_graph=False,
                strict=True,
            ).detach().cpu().numpy()
            finite_difference = np.zeros((2, 2), dtype=np.float64)
            for input_index in range(2):
                plus = observation.copy()
                minus = observation.copy()
                plus[input_index] += finite_difference_epsilon
                minus[input_index] -= finite_difference_epsilon
                with torch.no_grad():
                    y_plus = local_forward_double(
                        bundle.tensors,
                        joint,
                        torch.as_tensor(plus, dtype=torch.float64),
                    ).cpu().numpy()
                    y_minus = local_forward_double(
                        bundle.tensors,
                        joint,
                        torch.as_tensor(minus, dtype=torch.float64),
                    ).cpu().numpy()
                finite_difference[:, input_index] = (
                    y_plus - y_minus
                ) / (2.0 * finite_difference_epsilon)
            full = full_jacobian_from_local(
                np.stack(
                    [
                        analytic_one_double(bundle.tensors, other_joint, observations[state_index, other_joint])
                        for other_joint in range(8)
                    ],
                    axis=0,
                )
            )
            off_block_values = []
            for out_joint in range(8):
                for in_joint in range(8):
                    if out_joint != in_joint:
                        off_block_values.append(
                            full[
                                2 * out_joint : 2 * out_joint + 2,
                                2 * in_joint : 2 * in_joint + 2,
                            ]
                        )
            rows.append(
                {
                    "training_seed": bundle.seed,
                    "sample_ordinal": sample_ordinal,
                    "state_index": int(state_index),
                    "joint": f"J{joint + 1:02d}",
                    "delta_theta": float(observation[0]),
                    "theta_dot": float(observation[1]),
                    "analytic_vs_autograd_max_abs": float(np.max(np.abs(analytic - autograd))),
                    "analytic_vs_finite_difference_max_abs": float(
                        np.max(np.abs(analytic - finite_difference))
                    ),
                    "full_16x16_cross_joint_max_abs": float(
                        np.max(np.abs(np.asarray(off_block_values)))
                    ),
                }
            )
    return rows


def structural_mask_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for output_index in range(16):
        output_joint, output_channel = divmod(output_index, 2)
        for input_index in range(16):
            input_joint, input_channel = divmod(input_index, 2)
            rows.append(
                {
                    "output_index": output_index,
                    "output_label": f"J{output_joint + 1:02d}_{('K1', 'K2')[output_channel]}",
                    "input_index": input_index,
                    "input_label": f"J{input_joint + 1:02d}_{('delta_theta', 'theta_dot')[input_channel]}",
                    "structural_zero": bool(output_joint != input_joint),
                    "local_block_entry": bool(output_joint == input_joint),
                }
            )
    return rows


def trace_success(metrics: Mapping[str, Any]) -> bool:
    interval = metrics.get("mean_roll_pulse_interval_steps")
    return bool(
        int(metrics.get("roll_pulse_count", 0)) >= 4
        and float(metrics.get("desired_net_rotation_degrees", -math.inf)) >= 360.0
        and float(metrics.get("desired_active_rotation_fraction", -math.inf)) >= 0.70
        and float(metrics.get("forward_body_lengths", -math.inf)) >= 1.0
        and interval is not None
        and float(interval) <= 250.0
    )


def write_dataframe(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    frame = pd.DataFrame(list(rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", quoting=csv.QUOTE_MINIMAL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, default=FORMAL_ROOT_DEFAULT)
    parser.add_argument(
        "--trace-root",
        type=Path,
        action="append",
        default=None,
        help="Repeat for multiple immutable trace roots. Defaults include new-study trace folders and the old independent audit.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-condition-regex", default=DEFAULT_BASELINE_REGEX)
    parser.add_argument("--validation-samples-per-seed", type=int, default=12)
    parser.add_argument("--finite-difference-epsilon", type=float, default=1e-5)
    parser.add_argument("--autograd-tolerance", type=float, default=1e-8)
    parser.add_argument("--finite-difference-tolerance", type=float, default=1e-5)
    parser.add_argument("--derivative-zero-tolerance", type=float, default=1e-10)
    parser.add_argument("--max-torque", type=float, default=MAX_TORQUE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formal_root = args.formal_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    trace_roots = (
        [path.resolve() for path in args.trace_root]
        if args.trace_root
        else [path.resolve() for path in default_trace_roots()]
    )
    if not trace_roots:
        raise RuntimeError(
            "No trace roots found. Pass one or more --trace-root paths containing genuine .npz trajectories."
        )
    condition_pattern = re.compile(args.baseline_condition_regex, flags=re.IGNORECASE)
    traces, skipped = discover_traces(trace_roots, condition_pattern)
    if not traces:
        raise RuntimeError(
            "No baseline Rroll traces matched. Inspect trace_discovery_skipped.json or adjust --baseline-condition-regex."
        )
    seeds_present = {trace.training_seed for trace in traces}
    missing_seeds = sorted(set(TRAINING_SEEDS) - seeds_present)
    if missing_seeds:
        raise RuntimeError(f"Baseline trajectories missing formal seeds: {missing_seeds}")

    evaluator = load_official_evaluator(formal_root)
    bundles = {seed: load_policy_bundle(formal_root, seed) for seed in TRAINING_SEEDS}
    checkpoint_hashes_before = {
        str(seed): bundle.checkpoint_sha256 for seed, bundle in bundles.items()
    }
    rng = np.random.default_rng(2026080407)

    inventory_rows: list[dict[str, Any]] = []
    jacobian_rows: list[dict[str, Any]] = []
    full_jacobian_rows: list[dict[str, Any]] = []
    physics_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    reconstruction_errors: list[float] = []
    action_replay_errors: list[float] = []

    for seed in TRAINING_SEEDS:
        seed_traces = [trace for trace in traces if trace.training_seed == seed]
        observation_parts: list[np.ndarray] = []
        event_code_parts: list[np.ndarray] = []
        support_parts: list[np.ndarray] = []
        contact_parts: list[np.ndarray] = []
        for trace in seed_traces:
            metrics = official_metrics(evaluator, trace)
            codes = event_bins_from_metrics(len(trace.observation), metrics)
            observation_parts.append(trace.observation)
            event_code_parts.append(codes)
            support_parts.append(trace.support_index[:-1])
            contact_parts.append(trace.contact_strength[:-1])
            inventory_rows.append(
                {
                    "training_seed": seed,
                    "episode_seed": trace.episode_seed,
                    "condition_id": trace.condition_id,
                    "path": str(trace.path),
                    "source_root": str(trace.source_root),
                    "file_sha256": trace.sha256,
                    "content_sha256": trace.content_sha256,
                    "steps": len(trace.observation),
                    "roll_pulse_count": int(metrics.get("roll_pulse_count", 0)),
                    "tail_launch_count": int(metrics.get("tail_launch_count", 0)),
                    "desired_net_rotation_degrees": float(
                        metrics.get("desired_net_rotation_degrees", math.nan)
                    ),
                    "desired_active_rotation_fraction": float(
                        metrics.get("desired_active_rotation_fraction", math.nan)
                    ),
                    "forward_body_lengths": float(metrics.get("forward_body_lengths", math.nan)),
                    "mean_roll_pulse_interval_steps": metrics.get(
                        "mean_roll_pulse_interval_steps"
                    ),
                    "frozen_joint_success": trace_success(metrics),
                }
            )

        observations = np.concatenate(observation_parts, axis=0)
        event_codes = np.concatenate(event_code_parts, axis=0)
        support = np.concatenate(support_parts, axis=0)
        contact = np.concatenate(contact_parts, axis=0)
        bundle = bundles[seed]
        physical_k = actor_forward_physical(bundle.tensors, observations)
        jacobian = analytic_local_jacobian(bundle.tensors, observations)
        if not np.isfinite(physical_k).all() or not np.isfinite(jacobian).all():
            raise RuntimeError(f"Non-finite actor analysis output for seed {seed}")

        offset = 0
        for trace in seed_traces:
            length = len(trace.observation)
            if trace.recorded_roll_action is not None:
                replay_error = float(
                    np.max(
                        np.abs(
                            physical_k[offset : offset + length]
                            - K_SCALE * trace.recorded_roll_action.astype(np.float64)
                        )
                    )
                )
                action_replay_errors.append(replay_error)
                for row in reversed(inventory_rows):
                    if row["path"] == str(trace.path):
                        row["policy_vs_recorded_physical_K_max_abs"] = replay_error
                        break
            offset += length

        seed_jacobian_rows, seed_full_rows = summarize_jacobian(
            seed, jacobian, event_codes, float(args.derivative_zero_tolerance)
        )
        jacobian_rows.extend(seed_jacobian_rows)
        full_jacobian_rows.extend(seed_full_rows)
        seed_physics_rows, seed_event_rows, reconstruction = summarize_physics(
            seed,
            observations,
            physical_k,
            event_codes,
            support,
            contact,
            bundle.feedback_gain,
            float(args.max_torque),
        )
        physics_rows.extend(seed_physics_rows)
        event_rows.extend(seed_event_rows)
        reconstruction_errors.append(reconstruction)
        validation_rows.extend(
            validate_jacobian_samples(
                bundle,
                observations,
                int(args.validation_samples_per_seed),
                float(args.finite_difference_epsilon),
                rng,
            )
        )

    write_dataframe(output / "trace_inventory.csv", inventory_rows)
    write_dataframe(output / "jacobian_local_summary_by_seed_event.csv", jacobian_rows)
    write_dataframe(output / "jacobian_full_16x16_by_seed_event.csv", full_jacobian_rows)
    write_dataframe(output / "jacobian_structural_mask_16x16.csv", structural_mask_rows())
    write_dataframe(output / "shapley_torque_power_by_seed_event.csv", physics_rows)
    write_dataframe(output / "event_bin_contact_summary.csv", event_rows)
    write_dataframe(output / "jacobian_validation.csv", validation_rows)
    atomic_json(output / "trace_discovery_skipped.json", skipped)

    validation_frame = pd.DataFrame(validation_rows)
    autograd_error = float(validation_frame["analytic_vs_autograd_max_abs"].max())
    finite_difference_error = float(
        validation_frame["analytic_vs_finite_difference_max_abs"].max()
    )
    cross_error = float(validation_frame["full_16x16_cross_joint_max_abs"].max())
    structural_rows = structural_mask_rows()
    structural_zero_count = sum(bool(row["structural_zero"]) for row in structural_rows)
    local_entry_count = sum(bool(row["local_block_entry"]) for row in structural_rows)
    if structural_zero_count != 224 or local_entry_count != 32:
        raise RuntimeError("16x16 structural mask count mismatch")
    passed = bool(
        autograd_error <= float(args.autograd_tolerance)
        and finite_difference_error <= float(args.finite_difference_tolerance)
        and cross_error == 0.0
        and max(reconstruction_errors, default=0.0) <= 1e-12
    )

    checkpoint_hashes_after = {
        str(seed): sha256_file(bundle.checkpoint_path) for seed, bundle in bundles.items()
    }
    if checkpoint_hashes_after != checkpoint_hashes_before:
        raise RuntimeError("A protected formal checkpoint changed during read-only analysis")
    manifest = {
        "schema": "obs2_v2_1_local_actor_jacobian_physics/v1",
        "study_root": str(HERE),
        "formal_root": str(formal_root),
        "training_seeds": list(TRAINING_SEEDS),
        "checkpoint_batch": 1500,
        "checkpoint_sha256": checkpoint_hashes_after,
        "trace_roots": [str(path) for path in trace_roots],
        "trace_count": len(traces),
        "trace_steps": int(sum(len(trace.observation) for trace in traces)),
        "event_bins": list(EVENT_BINS),
        "event_bin_semantics": "derived from frozen official roll pulses and pulse-matched tail launches; pulse intervals split into equal normalized-time quintiles",
        "observation_per_joint": ["delta_theta", "theta_dot"],
        "action_per_joint": ["K1", "K2"],
        "actor_architecture": "eight independent local 2-input/2-output MLPs",
        "physical_K_scale": K_SCALE,
        "direct_full_jacobian_shape": [16, 16],
        "local_2x2_entry_count": 32,
        "structural_cross_joint_zero_count": 224,
        "max_torque": float(args.max_torque),
        "shapley_reconstruction_max_abs": max(reconstruction_errors, default=0.0),
        "policy_vs_recorded_physical_K_max_abs": max(action_replay_errors, default=None),
        "validation": {
            "passed": passed,
            "analytic_vs_autograd_max_abs": autograd_error,
            "analytic_vs_finite_difference_max_abs": finite_difference_error,
            "cross_joint_max_abs": cross_error,
            "autograd_tolerance": float(args.autograd_tolerance),
            "finite_difference_tolerance": float(args.finite_difference_tolerance),
            "finite_difference_epsilon": float(args.finite_difference_epsilon),
        },
        "interpretation_limits": [
            "Event bins are analysis alignment bins, not latent reward phases.",
            "The direct actor Jacobian cannot measure physical cross-joint propagation.",
            "Shapley phi1/phi2 exactly allocate the clipped control-boundary active torque for the two channels.",
            "phi*theta_dot is an active-control power proxy, not exact ten-physics-substep mechanical energy.",
            "No environment rollout or training is performed by this script.",
        ],
        "outputs": {
            "trace_inventory": "trace_inventory.csv",
            "local_jacobian": "jacobian_local_summary_by_seed_event.csv",
            "full_jacobian": "jacobian_full_16x16_by_seed_event.csv",
            "structural_mask": "jacobian_structural_mask_16x16.csv",
            "torque_power": "shapley_torque_power_by_seed_event.csv",
            "event_contact": "event_bin_contact_summary.csv",
            "validation": "jacobian_validation.csv",
        },
    }
    atomic_json(output / "ANALYSIS_MANIFEST.json", manifest)
    if not passed:
        raise RuntimeError(
            "Jacobian validation failed; inspect jacobian_validation.csv and ANALYSIS_MANIFEST.json"
        )
    atomic_json(output / "JACOBIAN_VALIDATION_PASS.json", manifest["validation"])


if __name__ == "__main__":
    main()
