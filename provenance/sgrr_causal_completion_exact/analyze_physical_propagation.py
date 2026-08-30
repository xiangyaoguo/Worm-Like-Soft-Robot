"""Paired, read-only analysis of closed-loop cross-joint propagation.

This program consumes the trace files produced by ``run_causal_completion.py``
and writes *only* to ``analysis_physical_propagation`` below this study root.
It never imports the simulator, loads a checkpoint, evaluates a policy, trains,
or modifies an existing result/trace.

The intervention is active from action step 0.  Controller-side arrays
(``observation``, ``physical_k``, and ``tau_clipped``) are indexed at action
step t.  State-side arrays (position/contact/support) are shifted so lag t is
the state after action t.  Rotation at lag t is the cumulative desired
rotation through transition t.

The formal actor is eight independent local two-input/two-output policies.
Consequently its 16x16 direct actor Jacobian is block diagonal and all direct
cross-joint derivatives are exactly zero by architecture.  Any paired
cross-joint difference that appears after lag 0 is therefore interpreted as
closed-loop physical propagation, not direct cross-joint actor input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "study_config.json"
TRACE_ROOT = ROOT / "traces"
RESULT_ROOT = ROOT / "results"
DEFAULT_OUTPUT = ROOT / "analysis_physical_propagation"

SD_THRESHOLD = 0.5
SUSTAINED_STEPS = 3
MAX_LAG = 50
NUM_JOINTS = 8
NUM_PARTICLES = 10
TRANSFORMS = ("ZERO", "SIGN_FLIP")
CONDITION_RE = re.compile(
    r"^C11_J(?P<joint>\d{2})_(?P<channel>K[12])_"
    r"(?P<transform>ZERO|SIGN_FLIP)$"
)


@dataclass(frozen=True)
class Feature:
    family: str
    component: str
    targets: tuple[str, ...]
    values: np.ndarray  # [time, target]

    @property
    def key(self) -> str:
        return f"{self.family}.{self.component}"


@dataclass(frozen=True)
class Scale:
    raw_sd: np.ndarray  # [time, target]
    effective_sd: np.ndarray  # [time, target]
    floor: np.ndarray  # [target]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def quantile(values: Sequence[float], q: float) -> float:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if finite.size == 0:
        return math.nan
    return float(np.quantile(finite, q))


def finite_or_none(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def target_names() -> tuple[str, ...]:
    return tuple(f"J{index:02d}" for index in range(1, NUM_JOINTS + 1))


def intervention_condition_ids() -> tuple[str, ...]:
    values: list[str] = []
    for joint in range(1, NUM_JOINTS + 1):
        for channel in ("K1", "K2"):
            for transform in TRANSFORMS:
                values.append(f"C11_J{joint:02d}_{channel}_{transform}")
    return tuple(values)


def expected_trace_path(seed: int, condition_id: str, eval_seed: int) -> Path:
    return TRACE_ROOT / f"seed{seed}" / f"{condition_id}__evalseed{eval_seed}.npz"


def result_trace_receipts(seed: int, condition_id: str) -> dict[Path, str]:
    path = RESULT_ROOT / f"seed{seed}" / f"{condition_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing completed condition result: {path}")
    payload = load_json(path)
    if payload.get("study_id") != load_json(CONFIG_PATH).get("study_id"):
        raise RuntimeError(f"Study identity mismatch: {path}")
    if int(payload.get("training_seed", -1)) != seed:
        raise RuntimeError(f"Training-seed mismatch: {path}")
    condition = payload.get("condition", {})
    if condition.get("id") != condition_id:
        raise RuntimeError(f"Condition identity mismatch: {path}")
    expected_episodes = int(load_json(CONFIG_PATH)["main_evaluation"]["episodes"])
    if len(payload.get("episodes", [])) != expected_episodes:
        raise RuntimeError(f"Condition is not endpoint-complete: {path}")
    records: dict[Path, str] = {}
    for record in payload.get("trace_files", []):
        trace_path = (ROOT / str(record["path"])).resolve()
        try:
            trace_path.relative_to(TRACE_ROOT.resolve())
        except ValueError as error:
            raise RuntimeError(f"Trace escapes frozen trace root: {trace_path}") from error
        records[trace_path] = str(record["sha256"]).lower()
    return records


def preflight(config: Mapping[str, Any]) -> tuple[list[Path], dict[str, str]]:
    seeds = [int(value) for value in config["training_seeds"]]
    base_seed = int(config["main_evaluation"]["base_seed"])
    episodes = int(config["main_evaluation"]["episodes"])
    if seeds != [9201, 9202, 9203, 9204, 9205]:
        raise RuntimeError(f"Unexpected frozen training-seed set: {seeds}")
    if episodes != 20:
        raise RuntimeError(f"This method requires 20 C11 scale traces, got {episodes}")
    if int(config["locked_contract"]["joint_count"]) != NUM_JOINTS:
        raise RuntimeError("Joint-count drift from the frozen contract")
    required: list[Path] = []
    receipt_hashes: dict[str, str] = {}
    for seed in seeds:
        baseline_receipts = result_trace_receipts(seed, "C11")
        for eval_seed in range(base_seed, base_seed + episodes):
            path = expected_trace_path(seed, "C11", eval_seed).resolve()
            if path not in baseline_receipts:
                raise RuntimeError(f"Missing C11 trace receipt for {path}")
            required.append(path)
            receipt_hashes[str(path)] = baseline_receipts[path]
        for condition_id in intervention_condition_ids():
            receipts = result_trace_receipts(seed, condition_id)
            path = expected_trace_path(seed, condition_id, base_seed).resolve()
            if path not in receipts:
                raise RuntimeError(f"Missing first-episode trace receipt for {path}")
            required.append(path)
            receipt_hashes[str(path)] = receipts[path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required traces are missing:\n" + "\n".join(missing))
    actual_hashes: dict[str, str] = {}
    for path in required:
        actual = sha256_file(path)
        expected = receipt_hashes[str(path)]
        if actual != expected:
            raise RuntimeError(f"Frozen trace hash mismatch: {path}: {actual} != {expected}")
        actual_hashes[str(path)] = actual
    return required, actual_hashes


def load_trace(path: Path, seed: int, eval_seed: int, condition_id: str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    required_shapes: dict[str, tuple[int, ...]] = {
        "observation": (1000, 8, 2),
        "physical_k": (1000, 8, 2),
        "tau1_unclipped": (1000, 8),
        "tau2_unclipped": (1000, 8),
        "tau_unclipped": (1000, 8),
        "tau_clipped": (1000, 8),
        "positions": (1001, 10),
        "support_index": (1001,),
        "ground_contact_strength": (1001,),
        "desired_cumulative_rotation_rad": (1000,),
    }
    for key, shape in required_shapes.items():
        if key not in arrays or arrays[key].shape != shape:
            raise RuntimeError(f"Trace field/shape mismatch in {path}: {key} != {shape}")
    scalar_checks = {
        "training_seed": seed,
        "evaluation_seed": eval_seed,
    }
    for key, expected in scalar_checks.items():
        if int(np.asarray(arrays[key]).item()) != expected:
            raise RuntimeError(f"Trace {key} mismatch in {path}")
    if str(np.asarray(arrays["condition_id"]).item()) != condition_id:
        raise RuntimeError(f"Trace condition mismatch in {path}")
    return arrays


def trace_features(trace: Mapping[str, np.ndarray]) -> dict[str, Feature]:
    joints = target_names()
    observation = np.asarray(trace["observation"], dtype=float)
    physical_k = np.asarray(trace["physical_k"], dtype=float)
    tau1 = np.asarray(trace["tau1_unclipped"], dtype=float)
    tau2 = np.asarray(trace["tau2_unclipped"], dtype=float)
    tau_unclipped = np.asarray(trace["tau_unclipped"], dtype=float)
    tau_clipped = np.asarray(trace["tau_clipped"], dtype=float)
    positions = np.asarray(trace["positions"])
    if not np.iscomplexobj(positions):
        raise RuntimeError("Expected complex-valued [time,particle] positions")
    # J01--J08 are the eight interior particle/joint sites.  Position lag t is
    # displacement after action t relative to the same episode's initial state.
    joint_displacement = positions[1:, 1:9] - positions[0:1, 1:9]
    com = np.mean(positions, axis=1)
    com_displacement = com[1:] - com[0]
    features = (
        Feature("observation", "delta_theta", joints, observation[:, :, 0]),
        Feature("observation", "theta_dot", joints, observation[:, :, 1]),
        Feature("K", "K1", joints, physical_k[:, :, 0]),
        Feature("K", "K2", joints, physical_k[:, :, 1]),
        Feature("tau", "K1_term_unclipped", joints, tau1),
        Feature("tau", "K2_term_unclipped", joints, tau2),
        Feature("tau", "total_unclipped", joints, tau_unclipped),
        Feature("tau", "total_clipped", joints, tau_clipped),
        Feature("position", "x", joints, np.real(joint_displacement)),
        Feature("position", "y", joints, np.imag(joint_displacement)),
        Feature(
            "contact",
            "ground_strength",
            ("GLOBAL",),
            np.asarray(trace["ground_contact_strength"], dtype=float)[1:, None],
        ),
        Feature(
            "support",
            "material_index",
            ("GLOBAL",),
            np.asarray(trace["support_index"], dtype=float)[1:, None],
        ),
        Feature(
            "rotation",
            "desired_cumulative_rad",
            ("GLOBAL",),
            np.asarray(trace["desired_cumulative_rotation_rad"], dtype=float)[:, None],
        ),
        Feature("position", "com_x", ("GLOBAL",), np.real(com_displacement)[:, None]),
        Feature("position", "com_y", ("GLOBAL",), np.imag(com_displacement)[:, None]),
    )
    output = {feature.key: feature for feature in features}
    for feature in output.values():
        if feature.values.ndim != 2 or feature.values.shape[0] != 1000:
            raise RuntimeError(f"Derived feature shape failure: {feature.key}")
        if feature.values.shape[1] != len(feature.targets):
            raise RuntimeError(f"Derived target shape failure: {feature.key}")
    return output


def baseline_scales(
    traces: Sequence[Mapping[str, np.ndarray]],
) -> tuple[dict[str, Feature], dict[str, Scale]]:
    extracted = [trace_features(trace) for trace in traces]
    keys = tuple(extracted[0])
    paired = extracted[0]
    scales: dict[str, Scale] = {}
    for key in keys:
        stack = np.stack([values[key].values for values in extracted], axis=0)
        if stack.shape[0] != 20:
            raise RuntimeError(f"Expected 20 C11 scale trajectories for {key}")
        with np.errstate(invalid="ignore", divide="ignore"):
            raw_sd = np.nanstd(stack, axis=0, ddof=1)
            pooled_sd = np.nanstd(stack.reshape(-1, stack.shape[-1]), axis=0, ddof=1)
            pooled_level = np.nanmedian(np.abs(stack), axis=(0, 1))
        # The registered threshold remains exactly 0.5 SD.  This floor only
        # resolves an undefined numerical threshold when a fixed-time C11 SD
        # is zero/non-finite; both raw and effective SD are reported.
        floor = np.maximum(1e-12, np.maximum(pooled_sd * 1e-6, pooled_level * 1e-12))
        floor = np.where(np.isfinite(floor), floor, 1e-12)
        effective = np.where(
            np.isfinite(raw_sd) & (raw_sd > floor[None, :]),
            raw_sd,
            floor[None, :],
        )
        scales[key] = Scale(raw_sd=raw_sd, effective_sd=effective, floor=floor)
    return paired, scales


def first_true(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    return int(indices[0]) if indices.size else None


def first_sustained(mask: np.ndarray, length: int = SUSTAINED_STEPS) -> int | None:
    clean = np.asarray(mask, dtype=bool)
    if clean.size < length:
        return None
    run = np.convolve(clean.astype(np.int8), np.ones(length, dtype=np.int8), mode="valid")
    indices = np.flatnonzero(run == length)
    return int(indices[0]) if indices.size else None


def target_role(source_index: int, target: str) -> tuple[int | None, str]:
    if target == "GLOBAL":
        return None, "global"
    target_index = int(target[1:]) - 1
    distance = abs(target_index - source_index)
    if distance == 0:
        role = "source"
    elif distance == 1:
        role = "neighbor"
    else:
        role = "far"
    return distance, role


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        values = numerator / denominator
    return np.where(np.isfinite(values), values, np.nan)


def write_csv_header(path: Path, fields: Sequence[str]) -> tuple[Any, csv.DictWriter]:
    handle = path.open("w", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
    writer.writeheader()
    return handle, writer


def min_optional(values: Iterable[int | None]) -> int | None:
    finite = [int(value) for value in values if value is not None]
    return min(finite) if finite else None


def aggregate_stage(
    first_by_feature: Mapping[tuple[str, str, str], int | None],
    family: str,
    role: str,
) -> int | None:
    return min_optional(
        step
        for (candidate_family, _target, candidate_role), step in first_by_feature.items()
        if candidate_family == family and candidate_role == role
    )


def build_stage_rows(
    seed: int,
    eval_seed: int,
    condition_id: str,
    source_joint: str,
    source_channel: str,
    transform: str,
    first_by_feature: Mapping[tuple[str, str, str], int | None],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_k = first_by_feature.get(("K", source_joint, "source"))
    source_tau = first_by_feature.get(("tau", source_joint, "source"))
    source_observation = first_by_feature.get(("observation", source_joint, "source"))
    stages: dict[str, int | None] = {
        "source_K_direct": source_k,
        "source_tau": source_tau,
        "source_observation_feedback": source_observation,
        "neighbor_observation": aggregate_stage(first_by_feature, "observation", "neighbor"),
        "far_observation": aggregate_stage(first_by_feature, "observation", "far"),
        "neighbor_K_feedback": aggregate_stage(first_by_feature, "K", "neighbor"),
        "far_K_feedback": aggregate_stage(first_by_feature, "K", "far"),
        "source_position": aggregate_stage(first_by_feature, "position", "source"),
        "neighbor_position": aggregate_stage(first_by_feature, "position", "neighbor"),
        "far_position": aggregate_stage(first_by_feature, "position", "far"),
        "support": aggregate_stage(first_by_feature, "support", "global"),
        "contact": aggregate_stage(first_by_feature, "contact", "global"),
        "rotation": aggregate_stage(first_by_feature, "rotation", "global"),
        "center_of_mass_position": aggregate_stage(first_by_feature, "position", "global"),
    }
    rows: list[dict[str, Any]] = []
    for stage, step in stages.items():
        rows.append(
            {
                "training_seed": seed,
                "evaluation_seed": eval_seed,
                "condition_id": condition_id,
                "source_joint": source_joint,
                "source_channel": source_channel,
                "transform": transform.lower(),
                "stage": stage,
                "first_sustained_separation_step": step,
            }
        )
    detected = sorted(
        ((step, stage) for stage, step in stages.items() if step is not None),
        key=lambda item: (item[0], item[1]),
    )
    earliest_stage = detected[0][1] if detected else None
    earliest_step = detected[0][0] if detected else None

    def last_precursor(global_stage: str) -> tuple[str | None, int | None]:
        global_step = stages[global_stage]
        if global_step is None:
            return None, None
        local = [
            (step, stage)
            for stage, step in stages.items()
            if stage not in {"contact", "support", "rotation"}
            and step is not None
            and step <= global_step
        ]
        if not local:
            return None, None
        step, stage = max(local, key=lambda item: (item[0], item[1]))
        return stage, step

    pre_contact_stage, pre_contact_step = last_precursor("contact")
    pre_rotation_stage, pre_rotation_step = last_precursor("rotation")
    summary = {
        "training_seed": seed,
        "evaluation_seed": eval_seed,
        "condition_id": condition_id,
        "source_joint": source_joint,
        "source_channel": source_channel,
        "transform": transform.lower(),
        "intervention_start_step": 0,
        "earliest_detected_stage": earliest_stage,
        "earliest_detected_step": earliest_step,
        "source_K_step": stages["source_K_direct"],
        "source_tau_step": stages["source_tau"],
        "source_observation_step": stages["source_observation_feedback"],
        "neighbor_observation_step": stages["neighbor_observation"],
        "far_observation_step": stages["far_observation"],
        "neighbor_K_step": stages["neighbor_K_feedback"],
        "far_K_step": stages["far_K_feedback"],
        "source_position_step": stages["source_position"],
        "neighbor_position_step": stages["neighbor_position"],
        "far_position_step": stages["far_position"],
        "support_step": stages["support"],
        "contact_step": stages["contact"],
        "rotation_step": stages["rotation"],
        "center_of_mass_position_step": stages["center_of_mass_position"],
        "last_local_precursor_before_contact": pre_contact_stage,
        "last_local_precursor_before_contact_step": pre_contact_step,
        "last_local_precursor_before_rotation": pre_rotation_stage,
        "last_local_precursor_before_rotation_step": pre_rotation_step,
        "detected_stage_order": " > ".join(f"{stage}@{step}" for step, stage in detected),
    }
    return rows, summary


def plot_heatmaps(
    output: Path,
    condition_labels: Sequence[str],
    heat_values: Mapping[tuple[str, str, int], Sequence[float]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = ("observation", "K", "tau", "position")
    fig, axes = plt.subplots(2, 2, figsize=(15, 18), constrained_layout=True)
    for axis, family in zip(axes.flat, families):
        matrix = np.full((len(condition_labels), NUM_JOINTS), np.nan, dtype=float)
        for row, condition in enumerate(condition_labels):
            for target in range(NUM_JOINTS):
                values = [
                    value
                    for value in heat_values.get((condition, family, target), [])
                    if math.isfinite(value)
                ]
                if values:
                    matrix[row, target] = float(np.median(values))
        image = axis.imshow(
            np.ma.masked_invalid(np.clip(matrix, 0, 100)),
            aspect="auto",
            cmap="viridis_r",
            vmin=0,
            vmax=100,
        )
        axis.set_title(f"{family}: median first sustained separation (clipped at 100)")
        axis.set_xticks(np.arange(NUM_JOINTS), target_names())
        axis.set_yticks(np.arange(len(condition_labels)), condition_labels, fontsize=7)
        axis.set_xlabel("target joint")
        axis.set_ylabel("source channel intervention")
        fig.colorbar(image, ax=axis, label="action-step lag")
    fig.savefig(output / "first_separation_heatmaps.png", dpi=180)
    plt.close(fig)


def plot_lag_response(
    output: Path,
    lag_values: Mapping[tuple[str, str, int], Sequence[float]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = ("observation", "K", "tau", "position")
    roles = ("source", "neighbor", "far")
    colors = {"source": "#1f77b4", "neighbor": "#ff7f0e", "far": "#2ca02c"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for axis, family in zip(axes.flat, families):
        for role in roles:
            median: list[float] = []
            lower: list[float] = []
            upper: list[float] = []
            for lag in range(MAX_LAG + 1):
                values = [
                    value
                    for value in lag_values.get((family, role, lag), [])
                    if math.isfinite(value)
                ]
                median.append(float(np.median(values)) if values else math.nan)
                lower.append(float(np.quantile(values, 0.25)) if values else math.nan)
                upper.append(float(np.quantile(values, 0.75)) if values else math.nan)
            x = np.arange(MAX_LAG + 1)
            axis.plot(x, median, label=role, color=colors[role])
            axis.fill_between(x, lower, upper, color=colors[role], alpha=0.18)
        axis.axhline(SD_THRESHOLD, color="black", linestyle="--", linewidth=1)
        axis.set_title(f"{family}: |paired difference| / C11 SD")
        axis.set_xlabel("action-step lag")
        axis.set_ylabel("standardized absolute response")
        axis.set_ylim(bottom=0)
        axis.legend()
    fig.savefig(output / "lag_response_by_joint_distance.png", dpi=180)
    plt.close(fig)


def plot_global_order(output: Path, chain_summaries: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = (
        "source_K_step",
        "source_tau_step",
        "neighbor_observation_step",
        "far_observation_step",
        "source_position_step",
        "neighbor_position_step",
        "contact_step",
        "support_step",
        "rotation_step",
    )
    labels = (
        "source K",
        "source tau",
        "neighbor obs",
        "far obs",
        "source position",
        "neighbor position",
        "contact",
        "support",
        "rotation",
    )
    data = [
        [float(row[column]) for row in chain_summaries if row.get(column) is not None]
        for column in columns
    ]
    fig, axis = plt.subplots(figsize=(13, 6), constrained_layout=True)
    axis.boxplot(data, labels=labels, showfliers=False)
    axis.set_ylabel("first sustained separation step")
    axis.set_title("Closed-loop causal-chain timing across single-K interventions")
    axis.tick_params(axis="x", rotation=35)
    fig.savefig(output / "contact_rotation_change_order.png", dpi=180)
    plt.close(fig)


def write_method(output: Path, config: Mapping[str, Any]) -> None:
    text = f"""# Cross-Joint Closed-Loop Physical-Propagation Analysis

## Analysis Boundary

- Inputs are only NPZ traces generated by the frozen main experiment; this analysis does not run the environment, load a policy, train, or modify checkpoints, results, or traces.
- Analysis set: `ZERO` and `SIGN_FLIP` for 16 individual K channels across five training seeds ({', '.join(map(str, config['training_seeds']))}). Each condition uses only its preregistered first saved episode and is paired with `C11` from the same training seed and evaluation seed.
- Interventions take effect at action step 0. Lag t for `observation/K/tau` denotes action step t; lag t for `position/contact/support` denotes state t+1 after action t; lag t for `rotation` includes transition t.
- Position for J01--J08 is the two-dimensional displacement of internal particles 1--8 relative to the start of that episode; global center-of-mass displacement is reported separately.

## Frozen Definition of Material Separation

- The 20 `C11` trajectories for a given training seed estimate sample SD at every time point for every target quantity.
- The threshold is exactly `0.5 x C11 SD`; unstandardized continuous differences, standardized magnitude, raw SD, and effective SD are all retained.
- When the SD at a fixed time is zero or nonfinite, only a tiny numeric floor prevents division by zero. `sd_floor_used` explicitly identifies such cases in the CSV; this numeric floor must not be interpreted as empirical variability.
- `first_threshold_crossing_step` is the first threshold crossing; the primary `first_sustained_separation_step` requires the threshold to be exceeded for {SUSTAINED_STEPS} consecutive steps.

## Direct Policy Effects and Physical Propagation

The formal actor consists of eight local 2-to-2 subpolicies with no shared inputs. Therefore, its 16-by-16 direct actor Jacobian is block diagonal with eight 2-by-2 blocks, and direct cross-joint partial derivatives are exactly zero. Consequently:

1. The source-channel K difference at lag 0 is the imposed intervention itself.
2. Direct actor differences at other joints should theoretically be zero at lag 0 and are audited numerically by the script.
3. Observation, K, tau, and position differences appearing at other joints for lag > 0 can only be interpreted as closed-loop propagation through intervention -> dynamics/contact -> changed local observation -> changed local policy output.

This is not population-level inference about success rate. Each nonbaseline condition has only one mechanism trace; success/failure and cross-seed robustness must be supported by the frozen endpoint evaluation of 5 x 20 episodes per condition.

## Primary Outputs

- `per_channel_propagation.csv`: first separation, peak, mean, and AUC from each source condition to each target quantity/joint.
- `lag_response.csv`: signed difference, absolute difference, C11 SD, and standardized magnitude for lags 0--{MAX_LAG}.
- `causal_chain_order.csv`: long-format staged first-separation table for every condition.
- `contact_rotation_first_change.csv`: first contact, support, and rotation changes and their preceding local stages.
- `stage_timing_summary.csv`: stage detection rates and lag distributions across conditions/seeds.
- Three PNG files: first-separation heat map, adjacent/distant lag response, and contact/rotation-chain timing.
"""
    atomic_text(output / "METHOD.md", text)


def run(output_dir: Path) -> None:
    config = load_json(CONFIG_PATH)
    required_inputs, hashes_before = preflight(config)
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing analysis directory: {output_dir}"
        )
    temporary = Path(
        tempfile.mkdtemp(prefix=".analysis_physical_propagation.", dir=str(ROOT))
    )
    per_fields = (
        "training_seed",
        "evaluation_seed",
        "condition_id",
        "source_joint",
        "source_channel",
        "transform",
        "intervention_start_step",
        "target_family",
        "target_component",
        "target_joint",
        "graph_distance",
        "target_role",
        "direct_actor_cross_joint_jacobian",
        "first_threshold_crossing_step",
        "first_sustained_separation_step",
        "first_sustained_raw_difference",
        "first_sustained_standardized_abs_difference",
        "peak_abs_difference_full_episode",
        "peak_standardized_abs_difference_full_episode",
        "mean_abs_difference_lag0_50",
        "mean_standardized_abs_difference_lag0_50",
        "standardized_abs_auc_lag0_50",
        "mean_abs_difference_full_episode",
        "mean_standardized_abs_difference_full_episode",
        "median_raw_baseline_sd",
        "median_effective_baseline_sd",
        "sd_floor",
        "sd_floor_used_fraction",
    )
    lag_fields = (
        "training_seed",
        "evaluation_seed",
        "condition_id",
        "source_joint",
        "source_channel",
        "transform",
        "target_family",
        "target_component",
        "target_joint",
        "graph_distance",
        "target_role",
        "lag",
        "raw_difference",
        "absolute_difference",
        "raw_baseline_sd",
        "effective_baseline_sd",
        "threshold_0p5_sd",
        "standardized_abs_difference",
        "threshold_exceeded",
        "sd_floor_used",
    )
    stage_fields = (
        "training_seed",
        "evaluation_seed",
        "condition_id",
        "source_joint",
        "source_channel",
        "transform",
        "stage",
        "first_sustained_separation_step",
    )
    chain_fields = (
        "training_seed",
        "evaluation_seed",
        "condition_id",
        "source_joint",
        "source_channel",
        "transform",
        "intervention_start_step",
        "earliest_detected_stage",
        "earliest_detected_step",
        "source_K_step",
        "source_tau_step",
        "source_observation_step",
        "neighbor_observation_step",
        "far_observation_step",
        "neighbor_K_step",
        "far_K_step",
        "source_position_step",
        "neighbor_position_step",
        "far_position_step",
        "support_step",
        "contact_step",
        "rotation_step",
        "center_of_mass_position_step",
        "last_local_precursor_before_contact",
        "last_local_precursor_before_contact_step",
        "last_local_precursor_before_rotation",
        "last_local_precursor_before_rotation_step",
        "detected_stage_order",
    )
    per_handle, per_writer = write_csv_header(
        temporary / "per_channel_propagation.csv", per_fields
    )
    lag_handle, lag_writer = write_csv_header(temporary / "lag_response.csv", lag_fields)
    stage_rows: list[dict[str, Any]] = []
    chain_summaries: list[dict[str, Any]] = []
    heat_values: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    plot_lag_values: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    cross_joint_k_lag0_max = 0.0
    condition_labels = [
        condition_id.replace("C11_", "").replace("_SIGN_FLIP", " flip").replace("_ZERO", " zero")
        for condition_id in intervention_condition_ids()
    ]
    base_seed = int(config["main_evaluation"]["base_seed"])
    episodes = int(config["main_evaluation"]["episodes"])
    try:
        for seed in [int(value) for value in config["training_seeds"]]:
            baseline_traces = [
                load_trace(
                    expected_trace_path(seed, "C11", eval_seed),
                    seed,
                    eval_seed,
                    "C11",
                )
                for eval_seed in range(base_seed, base_seed + episodes)
            ]
            paired_baseline, scales = baseline_scales(baseline_traces)
            for condition_id in intervention_condition_ids():
                match = CONDITION_RE.fullmatch(condition_id)
                if match is None:
                    raise RuntimeError(f"Unparseable frozen condition: {condition_id}")
                source_joint = match.group("joint")
                source_label = f"J{source_joint}"
                source_index = int(source_joint) - 1
                source_channel = match.group("channel")
                transform = match.group("transform")
                trace = load_trace(
                    expected_trace_path(seed, condition_id, base_seed),
                    seed,
                    base_seed,
                    condition_id,
                )
                if not np.allclose(
                    trace["observation"][0],
                    baseline_traces[0]["observation"][0],
                    rtol=0.0,
                    atol=1e-7,
                    equal_nan=True,
                ):
                    raise RuntimeError(f"Paired initial observation mismatch: seed {seed} {condition_id}")
                features = trace_features(trace)
                first_by_component: dict[tuple[str, str, str, str], int | None] = {}
                for key, feature in features.items():
                    baseline_feature = paired_baseline[key]
                    scale = scales[key]
                    if feature.targets != baseline_feature.targets:
                        raise RuntimeError(f"Target drift for feature {key}")
                    difference = feature.values - baseline_feature.values
                    absolute = np.abs(difference)
                    standardized = safe_divide(absolute, scale.effective_sd)
                    threshold = SD_THRESHOLD * scale.effective_sd
                    crossed = np.isfinite(absolute) & (absolute >= threshold)
                    for target_index, target in enumerate(feature.targets):
                        distance, role = target_role(source_index, target)
                        series = difference[:, target_index]
                        abs_series = absolute[:, target_index]
                        standardized_series = standardized[:, target_index]
                        raw_sd = scale.raw_sd[:, target_index]
                        effective_sd = scale.effective_sd[:, target_index]
                        mask = crossed[:, target_index]
                        first_cross = first_true(mask)
                        first_separation = first_sustained(mask)
                        first_by_component[(feature.family, feature.component, target, role)] = first_separation
                        if (
                            feature.family == "K"
                            and target != source_label
                            and target != "GLOBAL"
                        ):
                            cross_joint_k_lag0_max = max(
                                cross_joint_k_lag0_max,
                                float(abs_series[0]) if math.isfinite(float(abs_series[0])) else 0.0,
                            )
                        first_raw = (
                            float(series[first_separation])
                            if first_separation is not None and math.isfinite(float(series[first_separation]))
                            else math.nan
                        )
                        first_standardized = (
                            float(standardized_series[first_separation])
                            if first_separation is not None
                            and math.isfinite(float(standardized_series[first_separation]))
                            else math.nan
                        )
                        window = slice(0, MAX_LAG + 1)
                        per_writer.writerow(
                            {
                                "training_seed": seed,
                                "evaluation_seed": base_seed,
                                "condition_id": condition_id,
                                "source_joint": source_label,
                                "source_channel": source_channel,
                                "transform": transform.lower(),
                                "intervention_start_step": 0,
                                "target_family": feature.family,
                                "target_component": feature.component,
                                "target_joint": target,
                                "graph_distance": distance,
                                "target_role": role,
                                "direct_actor_cross_joint_jacobian": (
                                    0.0 if target not in {source_label, "GLOBAL"} else "not_cross_joint"
                                ),
                                "first_threshold_crossing_step": first_cross,
                                "first_sustained_separation_step": first_separation,
                                "first_sustained_raw_difference": first_raw,
                                "first_sustained_standardized_abs_difference": first_standardized,
                                "peak_abs_difference_full_episode": float(np.nanmax(abs_series)),
                                "peak_standardized_abs_difference_full_episode": float(
                                    np.nanmax(standardized_series)
                                ),
                                "mean_abs_difference_lag0_50": float(np.nanmean(abs_series[window])),
                                "mean_standardized_abs_difference_lag0_50": float(
                                    np.nanmean(standardized_series[window])
                                ),
                                "standardized_abs_auc_lag0_50": float(
                                    np.nansum(standardized_series[window])
                                ),
                                "mean_abs_difference_full_episode": float(np.nanmean(abs_series)),
                                "mean_standardized_abs_difference_full_episode": float(
                                    np.nanmean(standardized_series)
                                ),
                                "median_raw_baseline_sd": float(np.nanmedian(raw_sd)),
                                "median_effective_baseline_sd": float(np.nanmedian(effective_sd)),
                                "sd_floor": float(scale.floor[target_index]),
                                "sd_floor_used_fraction": float(
                                    np.mean(~np.isfinite(raw_sd) | (raw_sd <= scale.floor[target_index]))
                                ),
                            }
                        )
                        for lag in range(MAX_LAG + 1):
                            value = float(series[lag])
                            absolute_value = float(abs_series[lag])
                            standardized_value = float(standardized_series[lag])
                            raw_sd_value = float(raw_sd[lag])
                            effective_sd_value = float(effective_sd[lag])
                            floor_used = not math.isfinite(raw_sd_value) or raw_sd_value <= float(
                                scale.floor[target_index]
                            )
                            lag_writer.writerow(
                                {
                                    "training_seed": seed,
                                    "evaluation_seed": base_seed,
                                    "condition_id": condition_id,
                                    "source_joint": source_label,
                                    "source_channel": source_channel,
                                    "transform": transform.lower(),
                                    "target_family": feature.family,
                                    "target_component": feature.component,
                                    "target_joint": target,
                                    "graph_distance": distance,
                                    "target_role": role,
                                    "lag": lag,
                                    "raw_difference": value,
                                    "absolute_difference": absolute_value,
                                    "raw_baseline_sd": raw_sd_value,
                                    "effective_baseline_sd": effective_sd_value,
                                    "threshold_0p5_sd": SD_THRESHOLD * effective_sd_value,
                                    "standardized_abs_difference": standardized_value,
                                    "threshold_exceeded": bool(mask[lag]),
                                    "sd_floor_used": floor_used,
                                }
                            )
                            if feature.family in {"observation", "K", "tau", "position"} and role in {
                                "source",
                                "neighbor",
                                "far",
                            }:
                                plot_lag_values[(feature.family, role, lag)].append(
                                    standardized_value
                                )
                # Collapse x/y or K1/K2 component crossings to family/target
                # crossings for propagation distance and causal-chain order.
                first_by_feature: dict[tuple[str, str, str], int | None] = {}
                for family in {item[0] for item in first_by_component}:
                    targets = {item[2] for item in first_by_component if item[0] == family}
                    for target in targets:
                        roles = {item[3] for item in first_by_component if item[0] == family and item[2] == target}
                        for role in roles:
                            first_by_feature[(family, target, role)] = min_optional(
                                step
                                for (candidate_family, _component, candidate_target, candidate_role), step
                                in first_by_component.items()
                                if candidate_family == family
                                and candidate_target == target
                                and candidate_role == role
                            )
                condition_label = condition_id.replace("C11_", "").replace(
                    "_SIGN_FLIP", " flip"
                ).replace("_ZERO", " zero")
                for family in ("observation", "K", "tau", "position"):
                    for target_index, target in enumerate(target_names()):
                        _distance, role = target_role(source_index, target)
                        step = first_by_feature.get((family, target, role))
                        if step is not None:
                            heat_values[(condition_label, family, target_index)].append(float(step))
                rows, summary = build_stage_rows(
                    seed,
                    base_seed,
                    condition_id,
                    source_label,
                    source_channel,
                    transform,
                    first_by_feature,
                )
                stage_rows.extend(rows)
                chain_summaries.append(summary)
    finally:
        per_handle.close()
        lag_handle.close()

    # A direct actor cross-joint response at lag 0 would contradict the frozen
    # local-policy architecture or the paired-initial-state contract.
    if cross_joint_k_lag0_max > 1e-6:
        raise RuntimeError(
            "Cross-joint K differed at lag 0 despite block-diagonal local actor: "
            f"max |delta K|={cross_joint_k_lag0_max}"
        )

    stage_path = temporary / "causal_chain_order.csv"
    with stage_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=stage_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(stage_rows)
    chain_path = temporary / "contact_rotation_first_change.csv"
    with chain_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=chain_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(chain_summaries)

    grouped_stages: dict[tuple[str, str, str], list[int | None]] = defaultdict(list)
    for row in stage_rows:
        grouped_stages[(row["source_channel"], row["transform"], row["stage"])].append(
            row["first_sustained_separation_step"]
        )
    stage_summary_rows: list[dict[str, Any]] = []
    for (channel, transform, stage), values in sorted(grouped_stages.items()):
        detected = [float(value) for value in values if value is not None]
        stage_summary_rows.append(
            {
                "source_channel": channel,
                "transform": transform,
                "stage": stage,
                "comparison_count": len(values),
                "detected_count": len(detected),
                "detection_fraction": len(detected) / len(values),
                "first_step_median": statistics.median(detected) if detected else "",
                "first_step_q25": quantile(detected, 0.25),
                "first_step_q75": quantile(detected, 0.75),
                "first_step_min": min(detected) if detected else "",
                "first_step_max": max(detected) if detected else "",
            }
        )
    summary_fields = (
        "source_channel",
        "transform",
        "stage",
        "comparison_count",
        "detected_count",
        "detection_fraction",
        "first_step_median",
        "first_step_q25",
        "first_step_q75",
        "first_step_min",
        "first_step_max",
    )
    with (temporary / "stage_timing_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(stage_summary_rows)

    plot_heatmaps(temporary, condition_labels, heat_values)
    plot_lag_response(temporary, plot_lag_values)
    plot_global_order(temporary, chain_summaries)
    write_method(temporary, config)

    hashes_after = {str(path): sha256_file(path) for path in required_inputs}
    changed = [
        path for path, before in hashes_before.items() if hashes_after.get(path) != before
    ]
    if changed:
        raise RuntimeError(
            "Frozen inputs changed during analysis; refusing final output:\n"
            + "\n".join(changed)
        )
    output_files = sorted(path for path in temporary.iterdir() if path.is_file())
    audit = {
        "schema": "obs2_v2_1_k_causal_completion/physical_propagation_audit/v1",
        "study_id": config["study_id"],
        "analysis": "paired_single_trace_closed_loop_physical_propagation",
        "status": "complete",
        "training_seeds": [int(value) for value in config["training_seeds"]],
        "evaluation_seed": base_seed,
        "conditions": list(intervention_condition_ids()),
        "condition_count_per_training_seed": len(intervention_condition_ids()),
        "paired_comparison_count": len(chain_summaries),
        "baseline_scale_trajectory_count_per_training_seed": episodes,
        "sd_threshold": SD_THRESHOLD,
        "first_separation_consecutive_steps": SUSTAINED_STEPS,
        "lag_window": [0, MAX_LAG],
        "intervention_start_step": 0,
        "direct_actor_architecture": "8 independent local 2-input/2-output actors",
        "direct_16x16_actor_jacobian": "block diagonal; all cross-joint blocks exactly zero",
        "cross_joint_K_lag0_numeric_audit_max_abs_difference": cross_joint_k_lag0_max,
        "input_trace_count": len(required_inputs),
        "inputs_unchanged": True,
        "input_sha256": hashes_after,
        "caveat": (
            "Each nonbaseline condition contributes one paired trace. These outputs are "
            "mechanism diagnostics and cannot replace 5x20 endpoint success inference."
        ),
        "output_sha256_before_audit_file": {
            path.name: sha256_file(path) for path in output_files
        },
    }
    atomic_json(temporary / "ANALYSIS_AUDIT.json", audit)
    os.replace(temporary, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="New output directory. Existing directories are never overwritten.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output_dir.resolve()
    try:
        output.relative_to(ROOT.resolve())
    except ValueError as error:
        raise RuntimeError("Output must remain inside this study root") from error
    run(output)
    print(json.dumps({"status": "complete", "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
