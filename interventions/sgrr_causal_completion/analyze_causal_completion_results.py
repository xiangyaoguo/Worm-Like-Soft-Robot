"""Contract-bound analysis for the 113-condition K1/K2 completion matrix.

The program is intentionally outcome-agnostic.  It refuses to emit a final
analysis unless both evidence inventories are complete:

* causal completion: 113 conditions x 5 training seeds x 20 repeated states;
* matched C00: 1 condition x the same 5 seeds x the same 20 main states;
* sealed legacy study: 59 conditions x 5 training seeds x 20 repeated states.

The five training seeds are the only independent inference units.  The twenty
evaluation seeds are paired repeated initial states and are never treated as
independent policies.  Decision labels are computed only from the thresholds
already frozen in ``study_contract.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


SOURCE_ROOT = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("THESIS_SGRR_OUTPUT", str(SOURCE_ROOT))).resolve()
ANALYSIS = ROOT / "analysis_causal_completion"
FIGURES = ANALYSIS / "figures"
CONTRACT_PATH = SOURCE_ROOT / "study_contract.json"
CONFIG_PATH = SOURCE_ROOT / "study_config.json"
LEGACY_ROOT = Path(
    os.environ.get(
        "THESIS_SGRR_LEGACY_ROOT",
        str(SOURCE_ROOT.parent / "mechanism_runtime"),
    )
).resolve()
LEGACY_ANALYSIS = LEGACY_ROOT / "analysis"
MATCHED_C00_ROOT = ROOT / "matched_c00"
MATCHED_C00_RESULTS = MATCHED_C00_ROOT / "results"
MATCHED_C00_GATE = MATCHED_C00_ROOT / "MATCHED_C00_COMPLETE.json"

TRAINING_SEEDS = (9201, 9202, 9203, 9204, 9205)
JOINTS = tuple(f"J{number:02d}" for number in range(1, 9))
CHANNELS = ("K1", "K2")
TRANSFORMS = (
    ("ZERO", "zero"),
    ("SCALE_0P5", "scale_0p5"),
    ("SCALE_1P5", "scale_1p5"),
    ("SIGN_FLIP", "sign_flip"),
    ("STATIC_MEAN", "static_mean"),
    ("TIME_PERMUTED", "time_permuted"),
)
FAILURES = (
    "pulse_fail",
    "rotation_fail",
    "direction_fail",
    "forward_fail",
    "interval_fail",
)
FAILURE_LABELS = {
    "pulse_fail": "pulse < 4",
    "rotation_fail": "rotation < 360°",
    "direction_fail": "direction < 0.70",
    "forward_fail": "forward < 1 body",
    "interval_fail": "interval > 250",
}
PRIMARY_METRICS = (
    "forward_body_lengths",
    "desired_net_rotation_degrees",
    "direction_fraction",
    "roll_pulses",
    "mean_pulse_interval",
)
HIGHER_IS_BETTER = {
    "forward_body_lengths": True,
    "desired_net_rotation_degrees": True,
    "direction_fraction": True,
    "roll_pulses": True,
    "mean_pulse_interval": False,
}
DESCRIPTIVE_METRICS = PRIMARY_METRICS + ("tail_launch_count",)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(json_ready(value), ensure_ascii=False, indent=2, allow_nan=False),
    )


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, path)


def atomic_figure(path: Path, figure: plt.Figure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    figure.savefig(temporary, format="png", dpi=180, bbox_inches="tight")
    plt.close(figure)
    os.replace(temporary, path)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def expected_new_condition_ids() -> tuple[str, ...]:
    ids = ["C11"]
    for joint in JOINTS:
        for channel in CHANNELS:
            for suffix, _ in TRANSFORMS:
                ids.append(f"C11_{joint}_{channel}_{suffix}")
    ids.extend(f"C11_PAIR_NEC_{joint}" for joint in JOINTS)
    ids.extend(f"C00_PAIR_SUFF_{joint}" for joint in JOINTS)
    if len(ids) != 113 or len(set(ids)) != 113:
        raise AssertionError("Internal 113-condition inventory is not unique")
    return tuple(ids)


EXPECTED_NEW_IDS = expected_new_condition_ids()


def expected_new_condition_identity(condition_id: str) -> tuple[str, dict[str, Any]]:
    """Return the frozen family/spec expected for one canonical condition ID."""
    if condition_id == "C11":
        return "identity_baseline", {"op": "baseline_c11"}
    pair_nec = re.fullmatch(r"C11_PAIR_NEC_J(0[1-8])", condition_id)
    if pair_nec:
        return "c11_joint_pair_necessity", {
            "op": "c11_joint_pair_necessity",
            "joint": int(pair_nec.group(1)) - 1,
        }
    pair_suff = re.fullmatch(r"C00_PAIR_SUFF_J(0[1-8])", condition_id)
    if pair_suff:
        return "c00_joint_pair_sufficiency", {
            "op": "c00_joint_pair_sufficiency",
            "joint": int(pair_suff.group(1)) - 1,
        }
    match = re.fullmatch(
        r"C11_J(0[1-8])_(K[12])_(ZERO|SCALE_0P5|SCALE_1P5|SIGN_FLIP|STATIC_MEAN|TIME_PERMUTED)",
        condition_id,
    )
    if not match:
        raise ValueError(f"Not a canonical completion condition: {condition_id}")
    joint = int(match.group(1)) - 1
    channel_label = match.group(2)
    channel = 0 if channel_label == "K1" else 1
    suffix = match.group(3)
    transform_by_suffix = {
        "ZERO": ("zero", {"transform": "zero"}),
        "SCALE_0P5": ("scale_0p5", {"transform": "scale", "alpha": 0.5}),
        "SCALE_1P5": ("scale_1p5", {"transform": "scale", "alpha": 1.5}),
        "SIGN_FLIP": ("sign_flip", {"transform": "sign_flip"}),
        "STATIC_MEAN": ("static_mean", {"transform": "static_mean"}),
        "TIME_PERMUTED": ("time_permuted", {"transform": "time_permuted"}),
    }
    family_suffix, transform = transform_by_suffix[suffix]
    return f"full_c11_channel_{family_suffix}", {
        "op": "c11_channel_intervention",
        "joint": joint,
        "channel": channel,
        "channel_label": channel_label,
        **transform,
    }


def finite_float(value: Any, name: str, *, allow_none: bool = False) -> float:
    if value is None and allow_none:
        return math.nan
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} is non-finite: {value!r}")
    return number


def failure_flags(row: dict[str, Any], criteria: dict[str, Any]) -> dict[str, int]:
    interval = row.get("mean_pulse_interval")
    return {
        "pulse_fail": int(
            int(row["roll_pulses"]) < int(criteria["minimum_roll_pulses"])
        ),
        "rotation_fail": int(
            float(row["desired_net_rotation_degrees"])
            < float(criteria["minimum_desired_net_rotation_degrees"])
        ),
        "direction_fail": int(
            float(row["direction_fraction"])
            < float(criteria["minimum_direction_fraction"])
        ),
        "forward_fail": int(
            float(row["forward_body_lengths"])
            < float(criteria["minimum_forward_body_lengths"])
        ),
        "interval_fail": int(
            interval is None
            or not math.isfinite(float(interval))
            or float(interval)
            > float(criteria["maximum_mean_inter_pulse_interval_steps"])
        ),
    }


def parse_new_identity(condition: dict[str, Any]) -> dict[str, Any]:
    condition_id = str(condition["id"])
    spec = condition["spec"]
    op = str(spec["op"])
    result: dict[str, Any] = {
        "joint": None,
        "channel": None,
        "transform": None,
        "intervention_scope": op,
    }
    if op == "c11_channel_intervention":
        result["joint"] = f"J{int(spec['joint']) + 1:02d}"
        result["channel"] = str(spec["channel_label"])
        transform = str(spec["transform"])
        if transform == "scale":
            transform = "scale_0p5" if float(spec["alpha"]) == 0.5 else "scale_1p5"
        result["transform"] = transform
    elif op in {"c11_joint_pair_necessity", "c00_joint_pair_sufficiency"}:
        result["joint"] = f"J{int(spec['joint']) + 1:02d}"
        result["channel"] = "K1+K2"
        result["transform"] = "pair_zero" if "necessity" in op else "pair_transplant"
    elif condition_id != "C11":
        raise ValueError(f"Unknown condition operation: {condition_id}: {op}")
    return result


def normalize_new_episode(
    payload: dict[str, Any], metrics: dict[str, Any], episode_index: int, criteria: dict[str, Any]
) -> dict[str, Any]:
    condition = payload["condition"]
    identity = parse_new_identity(condition)
    interval = metrics.get("mean_roll_pulse_interval_steps")
    row: dict[str, Any] = {
        "source_study": "causal_completion_113",
        "condition_uid": f"new::{condition['id']}",
        "condition_id": condition["id"],
        "family": condition["family"],
        "module": "causal_completion",
        "description": condition.get("description", ""),
        **identity,
        "training_seed": int(payload["training_seed"]),
        "evaluation_seed": int(metrics["seed"]),
        "paired_repeat_index": episode_index + 1,
        "pairing_group": "new_main_20264401_20264420",
        "success": int(bool(metrics["success"])),
        "roll_pulses": int(metrics["roll_pulse_count"]),
        "desired_net_rotation_degrees": finite_float(
            metrics["desired_net_rotation_degrees"], "desired_net_rotation_degrees"
        ),
        "net_best_fit_rotation_degrees": finite_float(
            metrics["net_best_fit_rotation_degrees"], "net_best_fit_rotation_degrees"
        ),
        "direction_fraction": finite_float(
            metrics["desired_active_rotation_fraction"], "direction_fraction"
        ),
        "forward_body_lengths": finite_float(
            metrics["forward_body_lengths"], "forward_body_lengths"
        ),
        "forward_displacement": finite_float(
            metrics["forward_displacement"], "forward_displacement"
        ),
        "mean_pulse_interval": (
            math.nan
            if interval is None
            else finite_float(interval, "mean_roll_pulse_interval_steps")
        ),
        "tail_launch_detected": int(bool(metrics["tail_launch_detected"])),
        "tail_launch_count": int(metrics["tail_launch_count"]),
        "contact_metric_source": metrics["contact_metric_source"],
        "initial_body_length": finite_float(metrics["initial_body_length"], "initial_body_length"),
        "source_file": "",
        "joint_summary_json": json.dumps(
            metrics.get("joint_summary", {}), ensure_ascii=False, separators=(",", ":")
        ),
    }
    row.update(failure_flags(row, criteria))
    recomputed = int(not any(row[name] for name in FAILURES))
    if recomputed != row["success"]:
        raise ValueError(
            f"Stored/recomputed success mismatch: {condition['id']} seed "
            f"{payload['training_seed']} evaluation {metrics['seed']}"
        )
    return row


def load_new_evidence(
    contract: dict[str, Any], config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, Any]]:
    criteria = config["episode_success"]
    expected_hash = str(contract["conditions"]["canonical_sha256"])
    records: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}
    source_hashes: dict[str, str] = {}
    for seed in TRAINING_SEEDS:
        directory = ROOT / "results" / f"seed{seed}"
        if not directory.is_dir():
            raise FileNotFoundError(f"Incomplete matrix: missing {directory}")
        actual_ids = {path.stem for path in directory.glob("*.json")}
        if actual_ids != set(EXPECTED_NEW_IDS):
            missing = sorted(set(EXPECTED_NEW_IDS) - actual_ids)
            extra = sorted(actual_ids - set(EXPECTED_NEW_IDS))
            raise RuntimeError(
                f"Incomplete/extra result inventory for seed {seed}: "
                f"missing={missing}, extra={extra}"
            )
        for condition_id in EXPECTED_NEW_IDS:
            path = directory / f"{condition_id}.json"
            payload = load_json(path)
            if payload.get("study_id") != contract["study_id"]:
                raise ValueError(f"Study identity mismatch: {path}")
            if int(payload.get("training_seed")) != seed:
                raise ValueError(f"Training seed mismatch: {path}")
            if payload.get("canonical_condition_sha256") != expected_hash:
                raise ValueError(f"Condition hash mismatch: {path}")
            if payload.get("condition", {}).get("id") != condition_id:
                raise ValueError(f"Condition identity mismatch: {path}")
            expected_family, expected_spec = expected_new_condition_identity(condition_id)
            if payload["condition"].get("family") != expected_family:
                raise ValueError(f"Condition family drift: {path}")
            if payload["condition"].get("spec") != expected_spec:
                raise ValueError(f"Condition intervention-spec drift: {path}")
            if int(payload.get("evaluation_base_seed")) != 20264401:
                raise ValueError(f"Evaluation base drift: {path}")
            if int(payload.get("evaluation_steps")) != 1000:
                raise ValueError(f"Evaluation step-count drift: {path}")
            if payload.get("checkpoint_sha256") != contract["checkpoint_sha256"][str(seed)]:
                raise ValueError(f"Frozen checkpoint hash receipt drift: {path}")
            episodes = payload.get("episodes", [])
            if len(episodes) != 20:
                raise ValueError(f"Expected 20 episodes: {path}")
            expected_evaluation = list(range(20264401, 20264421))
            if [int(item["seed"]) for item in episodes] != expected_evaluation:
                raise ValueError(f"Evaluation initial-state drift: {path}")
            metadata[condition_id] = payload["condition"]
            if int(payload.get("success_episodes")) != sum(
                int(bool(item["success"])) for item in episodes
            ):
                raise ValueError(f"Condition-level success count drift: {path}")
            source_hashes[str(path.relative_to(ROOT))] = sha256_file(path)
            for index, metrics in enumerate(episodes):
                row = normalize_new_episode(payload, metrics, index, criteria)
                row["source_file"] = str(path.relative_to(ROOT))
                records.append(row)
    frame = pd.DataFrame.from_records(records)
    if len(frame) != 11300:
        raise RuntimeError(f"New episode inventory is {len(frame)}, expected 11300")
    counts = frame.groupby(["condition_id", "training_seed"]).size()
    if len(counts) != 565 or not bool((counts == 20).all()):
        raise RuntimeError("New condition x training-seed cells are not exactly 113 x 5 x 20")
    return frame, metadata, {
        "result_file_count": len(source_hashes),
        "episode_count": len(frame),
        "condition_count": frame["condition_id"].nunique(),
        "sha256": source_hashes,
    }


def normalize_matched_c00_episode(
    payload: dict[str, Any], metrics: dict[str, Any], episode_index: int, criteria: dict[str, Any]
) -> dict[str, Any]:
    interval = metrics.get("mean_roll_pulse_interval_steps")
    row: dict[str, Any] = {
        "source_study": "matched_c00_main",
        "condition_uid": "matched::C00",
        "condition_id": "C00",
        "family": "matched_c00_identity",
        "module": "causal_completion_matched_reference",
        "description": (
            "Complete R0/C00 evaluated on the same 20264401..20264420 initial states "
            "as the 113-condition causal-completion matrix."
        ),
        "joint": None,
        "channel": None,
        "transform": None,
        "intervention_scope": "matched_c00_identity_reference",
        "training_seed": int(payload["training_seed"]),
        "evaluation_seed": int(metrics["seed"]),
        "paired_repeat_index": episode_index + 1,
        "pairing_group": "new_main_20264401_20264420",
        "success": int(bool(metrics["success"])),
        "roll_pulses": int(metrics["roll_pulse_count"]),
        "desired_net_rotation_degrees": finite_float(
            metrics["desired_net_rotation_degrees"], "matched C00 desired rotation"
        ),
        "net_best_fit_rotation_degrees": finite_float(
            metrics["net_best_fit_rotation_degrees"], "matched C00 net rotation"
        ),
        "direction_fraction": finite_float(
            metrics["desired_active_rotation_fraction"], "matched C00 direction"
        ),
        "forward_body_lengths": finite_float(
            metrics["forward_body_lengths"], "matched C00 forward"
        ),
        "forward_displacement": finite_float(
            metrics["forward_displacement"], "matched C00 displacement"
        ),
        "mean_pulse_interval": (
            math.nan
            if interval is None
            else finite_float(interval, "matched C00 pulse interval")
        ),
        "tail_launch_detected": int(bool(metrics["tail_launch_detected"])),
        "tail_launch_count": int(metrics["tail_launch_count"]),
        "contact_metric_source": str(metrics["contact_metric_source"]),
        "initial_body_length": finite_float(
            metrics["initial_body_length"], "matched C00 initial body length"
        ),
        "source_file": "",
        "joint_summary_json": json.dumps(
            metrics.get("joint_summary", {}), ensure_ascii=False, separators=(",", ":")
        ),
    }
    row.update(failure_flags(row, criteria))
    recomputed = int(not any(row[name] for name in FAILURES))
    if recomputed != row["success"]:
        raise ValueError(
            "Matched C00 stored/recomputed success mismatch: training seed "
            f"{payload['training_seed']}, evaluation seed {metrics['seed']}"
        )
    return row


def load_matched_c00_evidence(
    contract: dict[str, Any], config: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the mandatory same-seed/same-initial-state C00 reference gate."""
    if not MATCHED_C00_GATE.is_file():
        raise FileNotFoundError(
            f"Matched C00 gate is mandatory for joint-pair sufficiency: {MATCHED_C00_GATE}"
        )
    gate = load_json(MATCHED_C00_GATE)
    if gate.get("schema") != "obs2_v2_1_k_causal_completion/matched_c00_complete/v1":
        raise RuntimeError("Matched C00 gate schema drift")
    if gate.get("study_id") != contract["study_id"]:
        raise RuntimeError("Matched C00 gate study_id drift")
    if gate.get("condition_id") != "C00":
        raise RuntimeError("Matched C00 gate condition identity drift")
    gate_complete = bool(gate.get("passed")) or str(gate.get("status", "")).lower() in {
        "complete",
        "completed",
        "pass",
        "passed",
    }
    if not gate_complete:
        raise RuntimeError("Matched C00 completion gate is not PASS/complete")
    if not bool(gate.get("all_checkpoint_and_policy_immutability_gates_passed")):
        raise RuntimeError("Matched C00 checkpoint/policy immutability gate is not PASS")
    if "training_seeds" in gate and [int(item) for item in gate["training_seeds"]] != list(TRAINING_SEEDS):
        raise RuntimeError("Matched C00 gate training-seed inventory drift")
    if "training_seed_count" in gate and int(gate["training_seed_count"]) != 5:
        raise RuntimeError("Matched C00 gate must bind five training seeds")
    if "evaluation_base_seed" in gate and int(gate["evaluation_base_seed"]) != 20264401:
        raise RuntimeError("Matched C00 gate evaluation base drift")
    gate_evaluation_seeds = gate.get(
        "evaluation_seeds", gate.get("evaluation_episode_seeds")
    )
    if gate_evaluation_seeds is None or [int(item) for item in gate_evaluation_seeds] != list(
        range(20264401, 20264421)
    ):
        raise RuntimeError("Matched C00 gate must bind evaluation seeds 20264401..20264420")
    gate_steps = gate.get("evaluation_steps", gate.get("steps_per_episode"))
    if gate_steps is None or int(gate_steps) != 1000:
        raise RuntimeError("Matched C00 gate must bind 1000 evaluation steps")
    gate_episodes_per_seed = gate.get(
        "episodes_per_seed", gate.get("episodes_per_training_seed")
    )
    if gate_episodes_per_seed is None or int(gate_episodes_per_seed) != 20:
        raise RuntimeError("Matched C00 gate episodes-per-seed drift")
    declared_total = gate.get("total_episodes", gate.get("episode_count"))
    if declared_total is not None and int(declared_total) != 100:
        raise RuntimeError("Matched C00 gate total episode count drift")

    seed_results = gate.get("seed_results")
    if not isinstance(seed_results, list) or len(seed_results) != 5:
        raise RuntimeError("Matched C00 gate seed_results must contain exactly five records")
    gate_seed_records: dict[int, dict[str, Any]] = {}
    for item in seed_results:
        if not isinstance(item, dict):
            raise RuntimeError("Matched C00 gate seed result is not an object")
        training_seed = int(item.get("training_seed", -1))
        if training_seed in gate_seed_records or training_seed not in TRAINING_SEEDS:
            raise RuntimeError("Matched C00 gate seed_results identity/uniqueness drift")
        if int(item.get("evaluation_episodes", -1)) != 20:
            raise RuntimeError("Matched C00 gate per-seed episode count drift")
        immutability = item.get("immutability")
        if not isinstance(immutability, dict):
            raise RuntimeError("Matched C00 gate lacks per-seed immutability receipt")
        if not bool(immutability.get("checkpoints_before_after_equal")) or not bool(
            immutability.get("policies_before_after_equal")
        ):
            raise RuntimeError("Matched C00 per-seed immutability receipt is not PASS")
        gate_seed_records[training_seed] = item
    if set(gate_seed_records) != set(TRAINING_SEEDS):
        raise RuntimeError("Matched C00 gate does not cover seeds 9201..9205 exactly")
    records: list[dict[str, Any]] = []
    source_hashes: dict[str, str] = {}
    expected_evaluation = list(range(20264401, 20264421))
    for seed in TRAINING_SEEDS:
        path = MATCHED_C00_RESULTS / f"seed{seed}" / "C00.json"
        if not path.is_file():
            raise FileNotFoundError(f"Matched C00 5x20 gate is incomplete: {path}")
        relative_to_matched = str(path.relative_to(MATCHED_C00_ROOT)).replace("\\", "/")
        actual_hash = sha256_file(path)
        gate_seed = gate_seed_records[seed]
        declared_result_path = str(gate_seed.get("result_path", "")).replace("\\", "/")
        allowed_result_paths = {
            relative_to_matched,
            str(path.relative_to(ROOT)).replace("\\", "/"),
            str(path).replace("\\", "/"),
        }
        if declared_result_path not in allowed_result_paths:
            raise RuntimeError(f"Matched C00 result path is not bound by gate: {path}")
        if str(gate_seed.get("result_sha256", "")).lower() != actual_hash:
            raise RuntimeError(f"Matched C00 result hash is not bound by gate: {path}")
        trace_path_value = str(gate_seed.get("trace_path", ""))
        trace_path = Path(trace_path_value)
        if not trace_path.is_absolute():
            candidate_under_root = ROOT / trace_path
            candidate_under_matched = MATCHED_C00_ROOT / trace_path
            trace_path = (
                candidate_under_root
                if candidate_under_root.is_file()
                else candidate_under_matched
            )
        if not trace_path.is_file() or sha256_file(trace_path) != str(
            gate_seed.get("trace_sha256", "")
        ).lower():
            raise RuntimeError(f"Matched C00 trace is missing or hash-drifted: {trace_path}")
        payload = load_json(path)
        if payload.get("schema") != "obs2_v2_1_k_causal_completion/matched_c00_seed/v1":
            raise RuntimeError(f"Matched C00 result schema drift: {path}")
        payload_parent_study = payload.get("parent_study_id", payload.get("study_id"))
        if payload_parent_study != contract["study_id"]:
            raise RuntimeError(f"Matched C00 result parent study_id drift: {path}")
        gate_supplemental_study = gate.get("supplemental_study_id")
        payload_supplemental_study = payload.get("supplemental_study_id")
        if (
            gate_supplemental_study is None
            or payload_supplemental_study != gate_supplemental_study
        ):
            raise RuntimeError(f"Matched C00 supplemental study_id drift: {path}")
        if int(payload.get("training_seed")) != seed:
            raise RuntimeError(f"Matched C00 training seed drift: {path}")
        condition_object = payload.get("condition")
        condition_id = (
            condition_object.get("id")
            if isinstance(condition_object, dict)
            else payload.get("condition_id")
        )
        if condition_id not in {"C00", "MATCHED_C00"}:
            raise RuntimeError(f"Matched C00 condition identity drift: {path}")
        if int(payload.get("evaluation_base_seed")) != 20264401:
            raise RuntimeError(f"Matched C00 evaluation base drift: {path}")
        if int(payload.get("evaluation_steps")) != 1000:
            raise RuntimeError(f"Matched C00 evaluation step-count drift: {path}")
        episodes = payload.get("episodes", [])
        if len(episodes) != 20:
            raise RuntimeError(f"Matched C00 requires exactly 20 episodes: {path}")
        if [int(item["seed"]) for item in episodes] != expected_evaluation:
            raise RuntimeError(f"Matched C00 evaluation initial states are not exactly paired: {path}")
        if any(item.get("condition_id") != "C00" for item in episodes):
            raise RuntimeError(f"Matched C00 episode condition identity drift: {path}")
        if int(payload.get("success_episodes")) != sum(
            int(bool(item["success"])) for item in episodes
        ):
            raise RuntimeError(f"Matched C00 condition success count drift: {path}")
        if int(gate_seed.get("success_episodes", -1)) != int(payload["success_episodes"]):
            raise RuntimeError(f"Matched C00 result/gate success counts differ: {path}")
        trace_files = payload.get("trace_files")
        if not isinstance(trace_files, list) or len(trace_files) != 1:
            raise RuntimeError(f"Matched C00 result must bind exactly one trace: {path}")
        result_trace = trace_files[0]
        if not isinstance(result_trace, dict):
            raise RuntimeError(f"Matched C00 trace receipt is malformed: {path}")
        if str(result_trace.get("path", "")).replace("\\", "/") != str(
            gate_seed.get("trace_path", "")
        ).replace("\\", "/"):
            raise RuntimeError(f"Matched C00 result/gate trace paths differ: {path}")
        if str(result_trace.get("sha256", "")).lower() != str(
            gate_seed.get("trace_sha256", "")
        ).lower():
            raise RuntimeError(f"Matched C00 result/gate trace hashes differ: {path}")
        checkpoint_receipt = payload.get("checkpoint_sha256")
        if checkpoint_receipt != contract["checkpoint_sha256"][str(seed)]:
            raise RuntimeError(f"Matched C00 frozen checkpoint bundle drift: {path}")
        expected_r0_checkpoint = contract["checkpoint_sha256"][str(seed)]["R0"]
        if str(gate_seed.get("R0_checkpoint_sha256", "")).lower() != str(
            expected_r0_checkpoint
        ).lower():
            raise RuntimeError(f"Matched C00 did not use the frozen same-seed R0 checkpoint: {path}")
        policy_receipt = payload.get("policy_state_sha256")
        if not isinstance(policy_receipt, dict) or not policy_receipt.get("R0"):
            raise RuntimeError(f"Matched C00 policy-state receipt missing: {path}")
        if str(gate_seed.get("R0_policy_state_sha256", "")).lower() != str(
            policy_receipt["R0"]
        ).lower():
            raise RuntimeError(f"Matched C00 R0 policy hash differs between result and gate: {path}")
        payload_immutability = payload.get("immutability")
        if not isinstance(payload_immutability, dict):
            raise RuntimeError(f"Matched C00 result lacks immutability receipt: {path}")
        if not bool(payload_immutability.get("checkpoints_before_after_equal")) or not bool(
            payload_immutability.get("policies_before_after_equal")
        ):
            raise RuntimeError(f"Matched C00 result immutability gate is not PASS: {path}")
        if payload_immutability != gate_seed["immutability"]:
            raise RuntimeError(f"Matched C00 result/gate immutability receipts differ: {path}")
        if payload_immutability.get("checkpoint_sha256") != checkpoint_receipt:
            raise RuntimeError(f"Matched C00 checkpoint immutability hash bundle drift: {path}")
        if payload_immutability.get("policy_state_sha256") != policy_receipt:
            raise RuntimeError(f"Matched C00 policy immutability hash bundle drift: {path}")
        if not bool(payload.get("source_audit", {}).get("passed")):
            raise RuntimeError(f"Matched C00 source audit is not PASS: {path}")
        expected_evaluator = contract["known_source_sha256"][
            "formal_evaluate_fast_forward_roll.py"
        ]
        if str(payload.get("frozen_evaluator_sha256", "")).lower() != str(
            expected_evaluator
        ).lower():
            raise RuntimeError(f"Matched C00 frozen evaluator hash drift: {path}")
        for index, metrics in enumerate(episodes):
            row = normalize_matched_c00_episode(
                payload, metrics, index, config["episode_success"]
            )
            row["source_file"] = str(path.relative_to(ROOT)).replace("\\", "/")
            records.append(row)
        source_hashes[str(path.relative_to(ROOT)).replace("\\", "/")] = actual_hash
    frame = pd.DataFrame.from_records(records)
    if len(frame) != 100:
        raise RuntimeError(f"Matched C00 episode inventory is {len(frame)}, expected 100")
    counts = frame.groupby("training_seed").size()
    if set(counts.index.astype(int)) != set(TRAINING_SEEDS) or not bool((counts == 20).all()):
        raise RuntimeError("Matched C00 gate is not exactly five training seeds x 20 episodes")
    return frame, {
        "gate_path": str(MATCHED_C00_GATE.relative_to(ROOT)).replace("\\", "/"),
        "gate_sha256": sha256_file(MATCHED_C00_GATE),
        "result_file_count": len(source_hashes),
        "episode_count": len(frame),
        "condition_count": 1,
        "training_seed_count": frame["training_seed"].nunique(),
        "episodes_per_training_seed": 20,
        "evaluation_seeds": expected_evaluation,
        "same_training_seed_and_initial_state_pairing": True,
        "sha256": source_hashes,
    }


def validate_matched_initial_state_pairing(
    new: pd.DataFrame, matched_c00: pd.DataFrame
) -> dict[str, Any]:
    """Confirm identical reset geometry for every paired main evaluation state."""
    relevant = pd.concat([new, matched_c00], ignore_index=True, sort=False)
    spread = relevant.groupby(["training_seed", "evaluation_seed"])[
        "initial_body_length"
    ].agg(lambda values: float(values.max() - values.min()))
    if len(spread) != 100:
        raise RuntimeError("Matched C00 pairing does not cover exactly 5 x 20 reset states")
    maximum = float(spread.max())
    if not math.isfinite(maximum) or maximum > 1e-12:
        raise RuntimeError(
            "Matched C00/new matrix initial reset geometry differs; maximum body-length "
            f"error={maximum}"
        )
    return {
        "paired_training_seed_evaluation_state_cells": len(spread),
        "maximum_initial_body_length_absolute_error": maximum,
        "absolute_tolerance": 1e-12,
        "passed": True,
    }


def normalize_legacy_episode(
    source: pd.Series,
    descriptions: dict[str, dict[str, Any]],
    criteria: dict[str, Any],
) -> dict[str, Any]:
    condition_id = str(source["condition_id"])
    meta = descriptions[condition_id]
    interval = source.get("mean_pulse_interval")
    interval_value = math.nan if pd.isna(interval) else finite_float(interval, "legacy interval")
    row: dict[str, Any] = {
        "source_study": "legacy_sealed_59",
        "condition_uid": f"legacy::{condition_id}",
        "condition_id": condition_id,
        "family": str(meta["family"]),
        "module": str(meta.get("module", "")),
        "description": str(meta.get("description", "")),
        "joint": None,
        "channel": None,
        "transform": None,
        "intervention_scope": "legacy_frozen_condition",
        "training_seed": int(source["training_seed"]),
        "evaluation_seed": int(source["evaluation_seed"]),
        "paired_repeat_index": int(source["evaluation_seed"]) - 20264200,
        "pairing_group": "legacy_main_20264201_20264220",
        "success": int(source["success"]),
        "roll_pulses": int(source["roll_pulses"]),
        "desired_net_rotation_degrees": finite_float(
            source["desired_net_rotation_degrees"], "legacy rotation"
        ),
        "net_best_fit_rotation_degrees": finite_float(
            source["net_best_fit_rotation_degrees"], "legacy net rotation"
        ),
        "direction_fraction": finite_float(source["direction_fraction"], "legacy direction"),
        "forward_body_lengths": finite_float(
            source["forward_body_lengths"], "legacy forward"
        ),
        "forward_displacement": finite_float(
            source["forward_displacement"], "legacy displacement"
        ),
        "mean_pulse_interval": interval_value,
        "tail_launch_detected": int(source["tail_launch"]),
        "tail_launch_count": math.nan,
        "contact_metric_source": str(source["contact_metric_source"]),
        "initial_body_length": math.nan,
        "source_file": str((LEGACY_ANALYSIS / "episode_results.csv").relative_to(ROOT.parent)),
        "joint_summary_json": "",
    }
    joint_match = re.search(r"_(J\d{2})$", condition_id)
    if joint_match:
        row["joint"] = joint_match.group(1)
    if condition_id.startswith("K1_"):
        row["channel"] = "K1"
    elif condition_id.startswith("K2_"):
        row["channel"] = "K2"
    if "_SUFF_" in condition_id:
        row["transform"] = "legacy_transplant"
    elif "_NEC_" in condition_id:
        row["transform"] = "legacy_recipient_replacement"
    row.update(failure_flags(row, criteria))
    recomputed = int(not any(row[name] for name in FAILURES))
    if recomputed != row["success"]:
        raise ValueError(
            f"Legacy stored/recomputed success mismatch: {condition_id}, "
            f"training seed {row['training_seed']}, evaluation seed {row['evaluation_seed']}"
        )
    return row


def normalize_legacy_raw_episode(
    payload: dict[str, Any], metrics: dict[str, Any], episode_index: int, criteria: dict[str, Any]
) -> dict[str, Any]:
    """Normalize one hash-validated legacy raw-result episode."""
    condition = payload["condition"]
    condition_id = str(condition["id"])
    interval = metrics.get("mean_roll_pulse_interval_steps")
    row: dict[str, Any] = {
        "source_study": "legacy_sealed_59",
        "condition_uid": f"legacy::{condition_id}",
        "condition_id": condition_id,
        "family": str(condition["family"]),
        "module": str(condition.get("module", "")),
        "description": str(condition.get("description", "")),
        "joint": None,
        "channel": None,
        "transform": None,
        "intervention_scope": "legacy_frozen_condition",
        "training_seed": int(payload["training_seed"]),
        "evaluation_seed": int(metrics["seed"]),
        "paired_repeat_index": episode_index + 1,
        "pairing_group": "legacy_main_20264201_20264220",
        "success": int(bool(metrics["success"])),
        "roll_pulses": int(metrics["roll_pulse_count"]),
        "desired_net_rotation_degrees": finite_float(
            metrics["desired_net_rotation_degrees"], "legacy desired rotation"
        ),
        "net_best_fit_rotation_degrees": finite_float(
            metrics["net_best_fit_rotation_degrees"], "legacy net rotation"
        ),
        "direction_fraction": finite_float(
            metrics["desired_active_rotation_fraction"], "legacy direction fraction"
        ),
        "forward_body_lengths": finite_float(
            metrics["forward_body_lengths"], "legacy forward body lengths"
        ),
        "forward_displacement": finite_float(
            metrics["forward_displacement"], "legacy displacement"
        ),
        "mean_pulse_interval": (
            math.nan
            if interval is None
            else finite_float(interval, "legacy mean pulse interval")
        ),
        "tail_launch_detected": int(bool(metrics["tail_launch_detected"])),
        "tail_launch_count": int(metrics["tail_launch_count"]),
        "contact_metric_source": str(metrics["contact_metric_source"]),
        "initial_body_length": finite_float(
            metrics["initial_body_length"], "legacy initial body length"
        ),
        "source_file": "",
        "joint_summary_json": json.dumps(
            metrics.get("joint_summary", {}), ensure_ascii=False, separators=(",", ":")
        ),
    }
    joint_match = re.search(r"_(J\d{2})$", condition_id)
    if joint_match:
        row["joint"] = joint_match.group(1)
    if condition_id.startswith("K1_"):
        row["channel"] = "K1"
    elif condition_id.startswith("K2_"):
        row["channel"] = "K2"
    if "_SUFF_" in condition_id:
        row["transform"] = "legacy_transplant"
    elif "_NEC_" in condition_id:
        row["transform"] = "legacy_recipient_replacement"
    row.update(failure_flags(row, criteria))
    recomputed = int(not any(row[name] for name in FAILURES))
    if recomputed != row["success"]:
        raise ValueError(
            f"Legacy stored/recomputed success mismatch: {condition_id}, "
            f"training seed {row['training_seed']}, evaluation seed {row['evaluation_seed']}"
        )
    return row


def load_legacy_evidence(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, dict[str, Any]], dict[str, Any]]:
    validation_path = LEGACY_ROOT / "VALIDATION_PASS.json"
    seal_path = LEGACY_ROOT / "FINAL_EVIDENCE_SEAL.json"
    if not validation_path.is_file() or not seal_path.is_file():
        raise FileNotFoundError("Sealed legacy 59-condition evidence is missing")
    validation = load_json(validation_path)
    seal = load_json(seal_path)
    if not bool(validation.get("passed")):
        raise RuntimeError("Legacy validation is not PASS")
    sealed_files = seal.get("files", {})
    if sha256_file(validation_path) != sealed_files.get("VALIDATION_PASS.json"):
        raise RuntimeError("Legacy VALIDATION_PASS hash no longer matches the final evidence seal")
    legacy_manifest_path = LEGACY_ANALYSIS / "analysis_manifest.json"
    if sha256_file(legacy_manifest_path) != sealed_files.get("analysis/analysis_manifest.json"):
        raise RuntimeError("Legacy analysis manifest hash no longer matches the final evidence seal")
    inventory = seal.get("inventory", {})
    expected_inventory = {
        "training_seed_count": 5,
        "condition_count": 59,
        "main_episode_count": 5900,
    }
    for key, expected in expected_inventory.items():
        if int(inventory.get(key, -1)) != expected:
            raise RuntimeError(f"Legacy sealed inventory drift: {key}")
    old_summary = pd.read_csv(LEGACY_ANALYSIS / "condition_summary.csv", encoding="utf-8-sig")
    if len(old_summary) != 59 or old_summary["condition_id"].nunique() != 59:
        raise RuntimeError("Legacy condition summary is not 59 unique rows")
    descriptions = {
        str(row["condition_id"]): row.to_dict() for _, row in old_summary.iterrows()
    }
    result_hashes = validation.get("result_file_sha256", {})
    if not isinstance(result_hashes, dict) or len(result_hashes) != 295:
        raise RuntimeError("Legacy validator does not bind exactly 295 raw result files")
    rows: list[dict[str, Any]] = []
    for seed in TRAINING_SEEDS:
        for condition_id in old_summary["condition_id"].astype(str).tolist():
            relative = f"results/seed{seed}/{condition_id}.json"
            path = LEGACY_ROOT / relative
            if not path.is_file() or sha256_file(path) != result_hashes.get(relative):
                raise RuntimeError(f"Legacy raw result missing or hash-drifted: {path}")
            payload = load_json(path)
            if payload.get("study_id") != "obs2_v2_1_k_mechanism_20260804":
                raise RuntimeError(f"Legacy study identity drift: {path}")
            if int(payload.get("training_seed")) != seed:
                raise RuntimeError(f"Legacy training seed drift: {path}")
            condition = payload.get("condition", {})
            if condition.get("id") != condition_id:
                raise RuntimeError(f"Legacy condition identity drift: {path}")
            summary_meta = descriptions[condition_id]
            if condition.get("family") != summary_meta["family"] or condition.get("module") != summary_meta["module"]:
                raise RuntimeError(f"Legacy raw/summary metadata drift: {path}")
            if int(payload.get("evaluation_base_seed")) != 20264201:
                raise RuntimeError(f"Legacy evaluation base drift: {path}")
            if int(payload.get("evaluation_steps")) != 1000:
                raise RuntimeError(f"Legacy evaluation step-count drift: {path}")
            episodes = payload.get("episodes", [])
            if len(episodes) != 20:
                raise RuntimeError(f"Legacy episode count drift: {path}")
            if [int(item["seed"]) for item in episodes] != list(range(20264201, 20264221)):
                raise RuntimeError(f"Legacy evaluation-state inventory drift: {path}")
            if int(payload.get("success_episodes")) != sum(
                int(bool(item["success"])) for item in episodes
            ):
                raise RuntimeError(f"Legacy condition success count drift: {path}")
            for index, metrics in enumerate(episodes):
                row = normalize_legacy_raw_episode(
                    payload, metrics, index, config["episode_success"]
                )
                row["source_file"] = relative
                rows.append(row)
    frame = pd.DataFrame.from_records(rows)
    if len(frame) != 5900:
        raise RuntimeError(f"Legacy episode inventory is {len(frame)}, expected 5900")
    counts = frame.groupby(["condition_id", "training_seed"]).size()
    if len(counts) != 295 or not bool((counts == 20).all()):
        raise RuntimeError("Legacy condition x seed cells are not exactly 59 x 5 x 20")
    for _, group in frame.groupby(["condition_id", "training_seed"]):
        if sorted(group["evaluation_seed"].tolist()) != list(range(20264201, 20264221)):
            raise RuntimeError("Legacy paired evaluation-seed inventory drift")
    return frame, descriptions, {
        "validation_sha256": sha256_file(validation_path),
        "evidence_seal_sha256": sha256_file(seal_path),
        "episode_csv_sha256": sha256_file(LEGACY_ANALYSIS / "episode_results.csv"),
        "condition_summary_sha256": sha256_file(LEGACY_ANALYSIS / "condition_summary.csv"),
        "hash_validated_raw_result_file_count": len(result_hashes),
        "episode_count": len(frame),
        "condition_count": frame["condition_id"].nunique(),
    }


def safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if len(numeric) else math.nan


def safe_std(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.std(ddof=1)) if len(numeric) > 1 else math.nan


def condition_seed_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_keys = ["source_study", "condition_uid", "condition_id", "training_seed"]
    for keys, group in episodes.groupby(group_keys, sort=False, dropna=False):
        first = group.iloc[0]
        row: dict[str, Any] = {
            "source_study": keys[0],
            "condition_uid": keys[1],
            "condition_id": keys[2],
            "training_seed": int(keys[3]),
            "family": first["family"],
            "module": first["module"],
            "joint": first["joint"],
            "channel": first["channel"],
            "transform": first["transform"],
            "episode_repetitions": len(group),
            "success_episodes": int(group["success"].sum()),
            "success_rate": float(group["success"].mean()),
            "training_seed_reaches_10_of_20": int(group["success"].sum() >= 10),
            "independent_inference_unit": "one frozen training seed/policy",
        }
        for metric in DESCRIPTIVE_METRICS:
            row[f"{metric}_mean"] = safe_mean(group[metric])
            row[f"{metric}_episode_sd_descriptive"] = safe_std(group[metric])
        for failure in FAILURES:
            row[failure] = int(group[failure].sum())
            row[f"{failure}_rate"] = float(group[failure].mean())
        rows.append(row)
    result = pd.DataFrame(rows)
    expected = (113 + 1 + 59) * 5
    if len(result) != expected:
        raise RuntimeError(f"Condition-seed summary has {len(result)} rows, expected {expected}")
    return result.sort_values(["source_study", "condition_id", "training_seed"]).reset_index(drop=True)


def condition_summary(episodes: pd.DataFrame, seed_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (source_study, condition_uid, condition_id), group in episodes.groupby(
        ["source_study", "condition_uid", "condition_id"], sort=False
    ):
        first = group.iloc[0]
        seed_rows = seed_summary[seed_summary["condition_uid"] == condition_uid]
        seed_success = {
            str(int(row["training_seed"])): int(row["success_episodes"])
            for _, row in seed_rows.iterrows()
        }
        row: dict[str, Any] = {
            "source_study": source_study,
            "condition_uid": condition_uid,
            "condition_id": condition_id,
            "family": first["family"],
            "module": first["module"],
            "description": first["description"],
            "joint": first["joint"],
            "channel": first["channel"],
            "transform": first["transform"],
            "training_seed_count_inferential_n": seed_rows["training_seed"].nunique(),
            "paired_repeated_initial_states_per_training_seed": 20,
            "episode_count_descriptive": len(group),
            "success_episodes_descriptive": int(group["success"].sum()),
            "success_rate_equal_weight_training_seed_mean": float(seed_rows["success_rate"].mean()),
            "success_rate_training_seed_sd": safe_std(seed_rows["success_rate"]),
            "minimum_training_seed_success_rate": float(seed_rows["success_rate"].min()),
            "maximum_training_seed_success_rate": float(seed_rows["success_rate"].max()),
            "training_seeds_reaching_10_of_20": int(seed_rows["training_seed_reaches_10_of_20"].sum()),
            "success_episodes_by_training_seed": json.dumps(
                seed_success, sort_keys=True, separators=(",", ":")
            ),
            "inference_warning": "n=5 training seeds; 20 initial states are paired repetitions",
        }
        for metric in DESCRIPTIVE_METRICS:
            row[f"{metric}_mean"] = safe_mean(group[metric])
            row[f"{metric}_episode_sd_reference"] = safe_std(group[metric])
            row[f"{metric}_training_seed_mean_sd"] = safe_std(seed_rows[f"{metric}_mean"])
        for failure in FAILURES:
            row[failure] = int(group[failure].sum())
            row[f"{failure}_rate"] = float(group[failure].mean())
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != 173:
        raise RuntimeError(f"Combined condition summary has {len(result)} rows, expected 173")
    return result.sort_values(["source_study", "condition_id"]).reset_index(drop=True)


def lookups(
    summary: pd.DataFrame, seed_summary: pd.DataFrame, episodes: pd.DataFrame
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[int, dict[str, Any]]], dict[str, pd.DataFrame]]:
    summary_lookup = {str(row["condition_uid"]): row.to_dict() for _, row in summary.iterrows()}
    seed_lookup: dict[str, dict[int, dict[str, Any]]] = {}
    for _, row in seed_summary.iterrows():
        seed_lookup.setdefault(str(row["condition_uid"]), {})[int(row["training_seed"])] = row.to_dict()
    episode_lookup = {
        str(uid): group.copy() for uid, group in episodes.groupby("condition_uid", sort=False)
    }
    return summary_lookup, seed_lookup, episode_lookup


def standardized_metric_difference(
    condition: dict[str, Any], reference: dict[str, Any], metric: str
) -> float:
    difference = abs(float(condition[f"{metric}_mean"]) - float(reference[f"{metric}_mean"]))
    standard_deviation = float(reference[f"{metric}_episode_sd_reference"])
    if not math.isfinite(standard_deviation) or standard_deviation <= 0.0:
        return 0.0 if difference == 0.0 else math.inf
    return difference / standard_deviation


def degraded_seed_count(
    condition_uid: str,
    reference_uid: str,
    seed_lookup: dict[str, dict[int, dict[str, Any]]],
    metric: str = "success_rate",
) -> int:
    count = 0
    for seed in TRAINING_SEEDS:
        target = float(seed_lookup[condition_uid][seed][metric])
        reference = float(seed_lookup[reference_uid][seed][metric])
        metric_name = metric.removesuffix("_mean")
        higher = HIGHER_IS_BETTER.get(metric_name, True)
        if (higher and target < reference) or (not higher and target > reference):
            count += 1
    return count


def condition_effect(
    condition_uid: str,
    reference_uid: str,
    summary_lookup: dict[str, dict[str, Any]],
    seed_lookup: dict[str, dict[int, dict[str, Any]]],
    equivalent_abs_rate: float,
    equivalent_sd: float,
) -> dict[str, Any]:
    condition = summary_lookup[condition_uid]
    reference = summary_lookup[reference_uid]
    rate = float(condition["success_rate_equal_weight_training_seed_mean"])
    reference_rate = float(reference["success_rate_equal_weight_training_seed_mean"])
    metric_z = {
        metric: standardized_metric_difference(condition, reference, metric)
        for metric in PRIMARY_METRICS
    }
    success_degraded = degraded_seed_count(condition_uid, reference_uid, seed_lookup)
    metric_degraded = {
        metric: degraded_seed_count(
            condition_uid, reference_uid, seed_lookup, f"{metric}_mean"
        )
        for metric in PRIMARY_METRICS
    }
    equivalent = (
        abs(rate - reference_rate) <= equivalent_abs_rate + 1e-12
        and all(value <= equivalent_sd + 1e-12 for value in metric_z.values())
        and success_degraded < 4
        and all(value < 4 for value in metric_degraded.values())
    )
    return {
        "condition_uid": condition_uid,
        "reference_uid": reference_uid,
        "success_rate": rate,
        "reference_success_rate": reference_rate,
        "success_rate_difference": rate - reference_rate,
        "success_rate_difference_percentage_points": 100.0 * (rate - reference_rate),
        "success_rate_drop": reference_rate - rate,
        "success_rate_drop_percentage_points": 100.0 * (reference_rate - rate),
        "training_seeds_degraded_success": success_degraded,
        "training_seeds_degraded_primary_metrics": metric_degraded,
        "primary_continuous_metric_abs_difference_reference_sd": metric_z,
        "equivalent_or_redundant": equivalent,
    }


def thresholds(contract: dict[str, Any]) -> dict[str, Any]:
    raw = contract["causal_decision_rules_frozen_before_outcomes"]
    return {
        "necessary_drop": float(
            raw["necessary_contribution"][
                "success_rate_drop_from_C11_minimum_percentage_points"
            ]
        )
        / 100.0,
        "necessary_seeds": int(raw["necessary_contribution"]["minimum_training_seeds_degraded"]),
        "strong_drop": float(
            raw["strong_necessity"][
                "success_rate_drop_from_C11_minimum_percentage_points"
            ]
        )
        / 100.0,
        "strong_seeds": int(raw["strong_necessity"]["minimum_training_seeds_degraded"]),
        "timing_drop": float(
            raw["timing_critical"][
                "static_or_permuted_success_rate_drop_from_C11_minimum_percentage_points"
            ]
        )
        / 100.0,
        "timing_seeds": int(raw["timing_critical"]["minimum_training_seeds_degraded"]),
        "suff_gain": float(
            raw["single_channel_or_joint_pair_sufficiency"][
                "success_rate_gain_from_C00_minimum_percentage_points"
            ]
        )
        / 100.0,
        "suff_seeds_reach": int(
            raw["single_channel_or_joint_pair_sufficiency"][
                "minimum_training_seeds_reaching_10_of_20_successes"
            ]
        ),
        "equivalent_rate": float(
            raw["equivalent_or_redundant"][
                "success_rate_absolute_difference_maximum_percentage_points"
            ]
        )
        / 100.0,
        "equivalent_sd": float(
            raw["equivalent_or_redundant"][
                "primary_continuous_metric_absolute_difference_maximum_reference_sd"
            ]
        ),
    }


def is_necessary(effect: dict[str, Any], rule: dict[str, Any]) -> bool:
    return (
        float(effect["success_rate_drop"]) >= rule["necessary_drop"] - 1e-12
        and int(effect["training_seeds_degraded_success"]) >= rule["necessary_seeds"]
    )


def is_strong_necessity(effect: dict[str, Any], rule: dict[str, Any]) -> bool:
    return (
        float(effect["success_rate_drop"]) >= rule["strong_drop"] - 1e-12
        and int(effect["training_seeds_degraded_success"]) >= rule["strong_seeds"]
    )


def is_timing_critical(effect: dict[str, Any], rule: dict[str, Any]) -> bool:
    return (
        float(effect["success_rate_drop"]) >= rule["timing_drop"] - 1e-12
        and int(effect["training_seeds_degraded_success"]) >= rule["timing_seeds"]
    )


def is_sufficient(
    effect: dict[str, Any],
    condition_uid: str,
    summary_lookup: dict[str, dict[str, Any]],
    rule: dict[str, Any],
) -> bool:
    condition = summary_lookup[condition_uid]
    return (
        float(effect["success_rate_difference"]) >= rule["suff_gain"] - 1e-12
        and int(condition["training_seeds_reaching_10_of_20"]) >= rule["suff_seeds_reach"]
    )


def transform_effect_fields(
    prefix: str,
    effect: dict[str, Any],
    summary_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    condition = summary_lookup[str(effect["condition_uid"])]
    row: dict[str, Any] = {
        f"{prefix}_condition_uid": effect["condition_uid"],
        f"{prefix}_success_rate": effect["success_rate"],
        f"{prefix}_success_rate_difference_percentage_points": effect[
            "success_rate_difference_percentage_points"
        ],
        f"{prefix}_success_rate_drop_percentage_points": effect[
            "success_rate_drop_percentage_points"
        ],
        f"{prefix}_training_seeds_degraded_success": effect[
            "training_seeds_degraded_success"
        ],
        f"{prefix}_equivalent_or_redundant": int(effect["equivalent_or_redundant"]),
        f"{prefix}_primary_metric_max_abs_difference_reference_sd": max(
            effect["primary_continuous_metric_abs_difference_reference_sd"].values()
        ),
    }
    for failure in FAILURES:
        row[f"{prefix}_{failure}_rate"] = condition[f"{failure}_rate"]
    for metric in PRIMARY_METRICS:
        row[f"{prefix}_{metric}_mean"] = condition[f"{metric}_mean"]
        row[f"{prefix}_{metric}_abs_difference_reference_sd"] = effect[
            "primary_continuous_metric_abs_difference_reference_sd"
        ][metric]
    return row


def build_per_k_cards(
    summary_lookup: dict[str, dict[str, Any]],
    seed_lookup: dict[str, dict[int, dict[str, Any]]],
    rule: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    reference_uid = "new::C11"
    legacy_c00 = "legacy::C00"
    for joint in JOINTS:
        for channel in CHANNELS:
            effects: dict[str, dict[str, Any]] = {}
            for suffix, transform in TRANSFORMS:
                uid = f"new::C11_{joint}_{channel}_{suffix}"
                effects[transform] = condition_effect(
                    uid,
                    reference_uid,
                    summary_lookup,
                    seed_lookup,
                    rule["equivalent_rate"],
                    rule["equivalent_sd"],
                )
            zero = effects["zero"]
            static = effects["static_mean"]
            permuted = effects["time_permuted"]
            strong = is_strong_necessity(zero, rule)
            necessary = is_necessary(zero, rule)
            timing = is_timing_critical(static, rule) or is_timing_critical(permuted, rule)

            legacy_suff_uid = f"legacy::{channel}_SUFF_{joint}"
            legacy_suff_effect = condition_effect(
                legacy_suff_uid,
                legacy_c00,
                summary_lookup,
                seed_lookup,
                rule["equivalent_rate"],
                rule["equivalent_sd"],
            )
            sufficient = is_sufficient(
                legacy_suff_effect, legacy_suff_uid, summary_lookup, rule
            )
            legacy_nec_reference = "legacy::C10" if channel == "K1" else "legacy::C11"
            legacy_nec_uid = f"legacy::{channel}_NEC_{joint}"
            legacy_nec_effect = condition_effect(
                legacy_nec_uid,
                legacy_nec_reference,
                summary_lookup,
                seed_lookup,
                rule["equivalent_rate"],
                rule["equivalent_sd"],
            )
            classifications: list[str] = []
            if strong:
                classifications.append("strong_necessity")
            elif necessary:
                classifications.append("necessary_contribution")
            if timing:
                classifications.append("timing_critical")
            if sufficient:
                classifications.append("single_channel_sufficient_in_legacy_C00_background")
            if bool(zero["equivalent_or_redundant"]):
                classifications.append("equivalent_or_redundant_when_zeroed_in_C11")
            if not classifications:
                classifications.append("no_frozen_threshold_label")
            row: dict[str, Any] = {
                "joint": joint,
                "channel": channel,
                "channel_id": f"{joint}-{channel}",
                "C11_reference_success_rate": summary_lookup[reference_uid][
                    "success_rate_equal_weight_training_seed_mean"
                ],
                "strong_necessity": int(strong),
                "necessary_contribution": int(necessary),
                "timing_critical": int(timing),
                "legacy_single_channel_sufficient": int(sufficient),
                "zero_equivalent_or_redundant": int(zero["equivalent_or_redundant"]),
                "formal_classifications": ";".join(classifications),
                "inference_unit": "5 independent training seeds",
                "episode_role": "20 paired repeated initial states per seed",
                "single_channel_sufficiency_reference": "legacy C00; same legacy evaluation states",
                "legacy_necessity_background": (
                    "C10 (Rroll K1/R0 K2)" if channel == "K1" else "legacy C11"
                ),
                "legacy_necessity_success_rate_drop_percentage_points": legacy_nec_effect[
                    "success_rate_drop_percentage_points"
                ],
                "legacy_sufficiency_success_rate_gain_percentage_points": legacy_suff_effect[
                    "success_rate_difference_percentage_points"
                ],
                "legacy_sufficiency_training_seeds_reaching_10_of_20": summary_lookup[
                    legacy_suff_uid
                ]["training_seeds_reaching_10_of_20"],
            }
            for _, transform in TRANSFORMS:
                row.update(transform_effect_fields(transform, effects[transform], summary_lookup))
            rows.append(row)
            details.append(
                {
                    "joint": joint,
                    "channel": channel,
                    "formal_classifications": classifications,
                    "strong_necessity": strong,
                    "necessary_contribution": necessary,
                    "timing_critical": timing,
                    "single_channel_sufficient_legacy_C00_background": sufficient,
                    "zero_equivalent_or_redundant": bool(zero["equivalent_or_redundant"]),
                    "transform_effects": effects,
                    "legacy_single_channel_sufficiency": legacy_suff_effect,
                    "legacy_recipient_replacement_necessity_descriptive": legacy_nec_effect,
                }
            )
    return pd.DataFrame(rows), details


def build_joint_pair_effects(
    summary_lookup: dict[str, dict[str, Any]],
    seed_lookup: dict[str, dict[int, dict[str, Any]]],
    rule: dict[str, Any],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for joint in JOINTS:
        nec_uid = f"new::C11_PAIR_NEC_{joint}"
        suff_uid = f"new::C00_PAIR_SUFF_{joint}"
        nec = condition_effect(
            nec_uid,
            "new::C11",
            summary_lookup,
            seed_lookup,
            rule["equivalent_rate"],
            rule["equivalent_sd"],
        )
        suff = condition_effect(
            suff_uid,
            "matched::C00",
            summary_lookup,
            seed_lookup,
            rule["equivalent_rate"],
            rule["equivalent_sd"],
        )
        historical_suff = condition_effect(
            suff_uid,
            "legacy::C00",
            summary_lookup,
            seed_lookup,
            rule["equivalent_rate"],
            rule["equivalent_sd"],
        )
        strong = is_strong_necessity(nec, rule)
        necessary = is_necessary(nec, rule)
        sufficient = is_sufficient(suff, suff_uid, summary_lookup, rule)
        row: dict[str, Any] = {
            "joint": joint,
            "necessity_condition_uid": nec_uid,
            "necessity_success_rate": nec["success_rate"],
            "necessity_success_rate_drop_percentage_points": nec[
                "success_rate_drop_percentage_points"
            ],
            "necessity_training_seeds_degraded": nec["training_seeds_degraded_success"],
            "strong_joint_pair_necessity": int(strong),
            "joint_pair_necessary_contribution": int(necessary),
            "joint_pair_zero_equivalent_or_redundant": int(nec["equivalent_or_redundant"]),
            "sufficiency_condition_uid": suff_uid,
            "sufficiency_reference_uid": "matched::C00",
            "sufficiency_reference_pairing": "same training seed and same 20264401..20264420 initial states",
            "matched_C00_reference_success_rate": summary_lookup["matched::C00"][
                "success_rate_equal_weight_training_seed_mean"
            ],
            "legacy_C00_historical_success_rate": summary_lookup["legacy::C00"][
                "success_rate_equal_weight_training_seed_mean"
            ],
            "matched_minus_legacy_C00_success_rate_percentage_points": 100.0
            * (
                float(
                    summary_lookup["matched::C00"][
                        "success_rate_equal_weight_training_seed_mean"
                    ]
                )
                - float(
                    summary_lookup["legacy::C00"][
                        "success_rate_equal_weight_training_seed_mean"
                    ]
                )
            ),
            "sufficiency_success_rate": suff["success_rate"],
            "sufficiency_success_rate_gain_percentage_points": suff[
                "success_rate_difference_percentage_points"
            ],
            "sufficiency_training_seeds_reaching_10_of_20": summary_lookup[suff_uid][
                "training_seeds_reaching_10_of_20"
            ],
            "joint_pair_sufficient": int(sufficient),
            "historical_unpaired_gain_vs_legacy_C00_percentage_points": historical_suff[
                "success_rate_difference_percentage_points"
            ],
            "inference_unit": "5 independent training seeds",
        }
        for failure in FAILURES:
            row[f"necessity_{failure}_rate"] = summary_lookup[nec_uid][f"{failure}_rate"]
            row[f"sufficiency_{failure}_rate"] = summary_lookup[suff_uid][f"{failure}_rate"]
        rows.append(row)
        details.append(
            {
                "joint": joint,
                "strong_joint_pair_necessity": strong,
                "joint_pair_necessary_contribution": necessary,
                "joint_pair_sufficient": sufficient,
                "necessity_effect": nec,
                "sufficiency_effect": suff,
                "historical_unpaired_effect_vs_legacy_C00": historical_suff,
                "sufficiency_reference_pairing": (
                    "Matched C00 and C00_PAIR_SUFF use the same learned training seed "
                    "and the same ordered evaluation states 20264401..20264420."
                ),
                "legacy_C00_role": "historical replication comparison only",
            }
        )
    return pd.DataFrame(rows), details


def failure_decomposition(episodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for condition_uid, group in episodes.groupby("condition_uid", sort=False):
        first = group.iloc[0]
        pattern_counts: dict[str, int] = {}
        for _, episode in group.iterrows():
            failed = [name.removesuffix("_fail") for name in FAILURES if int(episode[name])]
            pattern = "+".join(failed) if failed else "success_all_five"
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        row: dict[str, Any] = {
            "source_study": first["source_study"],
            "condition_uid": condition_uid,
            "condition_id": first["condition_id"],
            "family": first["family"],
            "joint": first["joint"],
            "channel": first["channel"],
            "transform": first["transform"],
            "episode_count_descriptive": len(group),
            "training_seed_count_inferential_n": group["training_seed"].nunique(),
            "success_all_five_count": int(group["success"].sum()),
            "success_all_five_rate": float(group["success"].mean()),
            "joint_failure_pattern_counts_json": json.dumps(
                pattern_counts, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        }
        for failure in FAILURES:
            row[failure] = int(group[failure].sum())
            row[f"{failure}_rate"] = float(group[failure].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["source_study", "condition_id"]).reset_index(drop=True)


def annotate_condition_references(
    summary: pd.DataFrame,
    summary_lookup: dict[str, dict[str, Any]],
    seed_lookup: dict[str, dict[int, dict[str, Any]]],
    rule: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, source in summary.iterrows():
        row = source.to_dict()
        uid = str(row["condition_uid"])
        reference: str | None = None
        if uid.startswith("new::C11_"):
            reference = "new::C11"
        if uid.startswith("new::C00_PAIR_SUFF_"):
            reference = "matched::C00"
        if uid.startswith("legacy::K1_SUFF_") or uid.startswith("legacy::K2_SUFF_"):
            reference = "legacy::C00"
        if uid.startswith("legacy::K1_NEC_"):
            reference = "legacy::C10"
        if uid.startswith("legacy::K2_NEC_"):
            reference = "legacy::C11"
        if reference is not None:
            effect = condition_effect(
                uid,
                reference,
                summary_lookup,
                seed_lookup,
                rule["equivalent_rate"],
                rule["equivalent_sd"],
            )
            row.update(
                {
                    "reference_condition_uid": reference,
                    "success_rate_difference_percentage_points": effect[
                        "success_rate_difference_percentage_points"
                    ],
                    "success_rate_drop_percentage_points": effect[
                        "success_rate_drop_percentage_points"
                    ],
                    "training_seeds_degraded_success": effect[
                        "training_seeds_degraded_success"
                    ],
                    "equivalent_or_redundant": int(effect["equivalent_or_redundant"]),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def plot_heatmap(cards: pd.DataFrame) -> str:
    transforms = [item[1] for item in TRANSFORMS]
    labels = cards["channel_id"].tolist()
    data = np.asarray(
        [
            [float(row[f"{transform}_success_rate_difference_percentage_points"]) for transform in transforms]
            for _, row in cards.iterrows()
        ]
    )
    limit = max(10.0, float(np.nanmax(np.abs(data))))
    figure, axis = plt.subplots(figsize=(10.5, 8.5))
    image = axis.imshow(data, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(range(len(transforms)), transforms, rotation=25, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_title("Per-channel success-rate effect vs complete C11 (percentage points)")
    for row in range(data.shape[0]):
        for column in range(data.shape[1]):
            axis.text(column, row, f"{data[row, column]:.0f}", ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axis, label="condition - C11 (pp)")
    path = FIGURES / "fig_01_16_channel_effect_heatmap.png"
    atomic_figure(path, figure)
    return str(path.relative_to(ROOT))


def plot_dose(cards: pd.DataFrame) -> str:
    figure, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True)
    alphas = [0.0, 0.5, 1.0, 1.5]
    for channel, axis in zip(CHANNELS, axes):
        subset = cards[cards["channel"] == channel]
        for _, row in subset.iterrows():
            values = [
                float(row["zero_success_rate"]),
                float(row["scale_0p5_success_rate"]),
                float(row["C11_reference_success_rate"]),
                float(row["scale_1p5_success_rate"]),
            ]
            axis.plot(alphas, values, marker="o", linewidth=1.4, label=row["joint"])
        axis.set_title(f"{channel} dose response in complete C11")
        axis.set_xlabel("target channel scale")
        axis.set_xticks(alphas)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("success rate (seed-equal mean)")
    axes[1].legend(ncol=2, fontsize=8, title="joint")
    path = FIGURES / "fig_02_channel_dose_response.png"
    atomic_figure(path, figure)
    return str(path.relative_to(ROOT))


def plot_timing(cards: pd.DataFrame) -> str:
    labels = cards["channel_id"].tolist()
    static = cards["static_mean_success_rate_difference_percentage_points"].to_numpy(float)
    permuted = cards["time_permuted_success_rate_difference_percentage_points"].to_numpy(float)
    x = np.arange(len(labels))
    width = 0.38
    figure, axis = plt.subplots(figsize=(12, 5.8))
    axis.bar(x - width / 2, static, width, label="static mean")
    axis.bar(x + width / 2, permuted, width, label="time permuted")
    axis.axhline(-30, color="black", linestyle="--", linewidth=1, label="-30 pp rule")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, labels, rotation=55, ha="right")
    axis.set_ylabel("condition - C11 success rate (pp)")
    axis.set_title("Per-channel timing interventions")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    path = FIGURES / "fig_03_channel_timing_effects.png"
    atomic_figure(path, figure)
    return str(path.relative_to(ROOT))


def plot_joint_pairs(pairs: pd.DataFrame) -> str:
    x = np.arange(len(pairs))
    width = 0.38
    nec = -pairs["necessity_success_rate_drop_percentage_points"].to_numpy(float)
    suff = pairs["sufficiency_success_rate_gain_percentage_points"].to_numpy(float)
    figure, axis = plt.subplots(figsize=(10, 5.8))
    axis.bar(x - width / 2, nec, width, label="pair zero in C11 (negative = loss)")
    axis.bar(x + width / 2, suff, width, label="pair transplant into C00 (gain)")
    axis.axhline(-30, color="black", linestyle="--", linewidth=1)
    axis.axhline(30, color="black", linestyle=":", linewidth=1)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(x, pairs["joint"].tolist())
    axis.set_ylabel("success-rate effect (pp)")
    axis.set_title("Joint K1+K2 necessity and sufficiency effects")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.2)
    path = FIGURES / "fig_04_joint_pair_effects.png"
    atomic_figure(path, figure)
    return str(path.relative_to(ROOT))


def plot_failures(cards: pd.DataFrame) -> str:
    labels = cards["channel_id"].tolist()
    data = np.asarray(
        [[float(row[f"zero_{failure}_rate"]) for failure in FAILURES] for _, row in cards.iterrows()]
    )
    figure, axis = plt.subplots(figsize=(9, 8.5))
    image = axis.imshow(data, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_xticks(range(len(FAILURES)), [FAILURE_LABELS[item] for item in FAILURES], rotation=25, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_title("Five-criterion failure decomposition after channel zeroing")
    for row in range(data.shape[0]):
        for column in range(data.shape[1]):
            axis.text(column, row, f"{100 * data[row, column]:.0f}%", ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axis, label="episode failure rate (descriptive)")
    path = FIGURES / "fig_05_zero_failure_decomposition.png"
    atomic_figure(path, figure)
    return str(path.relative_to(ROOT))


def english_bool(value: Any) -> str:
    return "yes" if bool(value) else "no"


def classification_markdown(
    cards: pd.DataFrame,
    pairs: pd.DataFrame,
    rule: dict[str, Any],
    validation: dict[str, Any],
    figure_paths: list[str],
) -> str:
    lines = [
        "# K1/K2 causal completion: contract decisions",
        "",
        "> This file is generated automatically by `analyze_causal_completion_results.py` under the frozen contract. The independent inference units are the five training seeds; the twenty initial states per seed are paired repetitions, not 100 independent policy samples.",
        "",
        "## Evidence integrity",
        "",
        f"- New matrix: {validation['new']['condition_count']} conditions, {validation['new']['result_file_count']} condition-seed files, and {validation['new']['episode_count']} episodes.",
        f"- Strictly matched C00: {validation['matched_c00']['training_seed_count']} training seeds × {validation['matched_c00']['episodes_per_training_seed']} episodes, {validation['matched_c00']['episode_count']} episodes in total; initial states 20264401–20264420 are shared with the new matrix.",
        f"- Archived legacy evidence: {validation['legacy']['condition_count']} conditions and {validation['legacy']['episode_count']} episodes.",
        "- Every success label was recomputed and checked episode by episode against the joint pulse, rotation, direction, progress, and pulse-interval thresholds.",
        "",
        "## Frozen decision rules",
        "",
        f"- Strong necessity: zeroing under C11 reduces success by at least {100*rule['strong_drop']:.0f} percentage points, with a decrease in at least {rule['strong_seeds']}/5 training seeds.",
        f"- Necessary contribution: the decrease is at least {100*rule['necessary_drop']:.0f} percentage points, with a decrease in at least {rule['necessary_seeds']}/5 training seeds.",
        f"- Timing critical: static mean or time permuted reduces success by at least {100*rule['timing_drop']:.0f} percentage points, with a decrease in at least {rule['timing_seeds']}/5 seeds.",
        f"- Sufficient: the gain relative to C00 is at least {100*rule['suff_gain']:.0f} percentage points, with at least {rule['suff_seeds_reach']}/5 seeds reaching 10/20.",
        f"- Equivalent/redundant: the absolute success-rate difference is no more than {100*rule['equivalent_rate']:.0f} percentage points, every primary continuous-metric difference is within {rule['equivalent_sd']:.1f} reference SD, and there is no consistent degradation in at least 4/5 seeds.",
        "",
        "## Per-joint, per-K decisions",
        "",
        "|Channel|Zeroing difference (pp)|Strong necessity|Necessary contribution|Static difference (pp)|Permutation difference (pp)|Timing critical|Legacy C00 single-channel sufficient|Zeroing equivalent/redundant|",
        "|---|---:|:---:|:---:|---:|---:|:---:|:---:|:---:|",
    ]
    for _, row in cards.iterrows():
        lines.append(
            "|{channel}|{zero:.1f}|{strong}|{necessary}|{static:.1f}|{permuted:.1f}|{timing}|{suff}|{equiv}|".format(
                channel=row["channel_id"],
                zero=row["zero_success_rate_difference_percentage_points"],
                strong=english_bool(row["strong_necessity"]),
                necessary=english_bool(row["necessary_contribution"]),
                static=row["static_mean_success_rate_difference_percentage_points"],
                permuted=row["time_permuted_success_rate_difference_percentage_points"],
                timing=english_bool(row["timing_critical"]),
                suff=english_bool(row["legacy_single_channel_sufficient"]),
                equiv=english_bool(row["zero_equivalent_or_redundant"]),
            )
        )
    lines.extend(
        [
            "",
            "## Joint K1+K2: per-joint necessity and sufficiency",
            "",
            "|Joint|Whole-pair zeroing difference (pp)|Strong necessity|Necessary contribution|Whole-pair transplant gain (pp)|Seeds reaching 10/20|Sufficient|",
            "|---|---:|:---:|:---:|---:|---:|:---:|",
        ]
    )
    for _, row in pairs.iterrows():
        lines.append(
            "|{joint}|{nec:.1f}|{strong}|{necessary}|{suff:.1f}|{seeds}|{sufficient}|".format(
                joint=row["joint"],
                nec=-float(row["necessity_success_rate_drop_percentage_points"]),
                strong=english_bool(row["strong_joint_pair_necessity"]),
                necessary=english_bool(row["joint_pair_necessary_contribution"]),
                suff=row["sufficiency_success_rate_gain_percentage_points"],
                seeds=int(row["sufficiency_training_seeds_reaching_10_of_20"]),
                sufficient=english_bool(row["joint_pair_sufficient"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- Single-channel sufficiency comes from the archived legacy 59-condition matrix, in which the candidate and C00 used the same legacy evaluation initial states.",
            "- Both the new whole-pair sufficiency conditions and matched C00 use the same training seeds and the same ordered initial states 20264401–20264420. Joint sufficiency is therefore a strictly paired comparison. Legacy C00 is retained only as a historical reproduction control and does not enter the primary new joint-sufficiency decision.",
            "- The primary continuous metrics for equivalence were fixed in advance as forward body lengths, desired net rotation, direction ratio, rolling-pulse count, and mean pulse interval. Each is standardised by the episode-level SD of the reference condition.",
            "- `success rate` is the equally weighted mean of the five seed-level success rates. Episode-level failure rates in the CSV are descriptive mechanism measures only; the 100 episodes must not be treated as an independent n=100 for policy-level inference.",
            "",
            "## Figures",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in figure_paths)
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the complete inventories but do not write derived tables/figures.",
    )
    args = parser.parse_args()
    contract = load_json(CONTRACT_PATH)
    config = load_json(CONFIG_PATH)
    if contract.get("study_id") != config.get("study_id"):
        raise RuntimeError("study_config/study_contract identity mismatch")
    if int(contract["conditions"]["count"]) != 113:
        raise RuntimeError("Frozen contract is not the 113-condition matrix")
    if tuple(int(item) for item in config["training_seeds"]) != TRAINING_SEEDS:
        raise RuntimeError("Training-seed contract drift")

    new, new_meta, new_validation = load_new_evidence(contract, config)
    matched_c00, matched_c00_validation = load_matched_c00_evidence(contract, config)
    matched_pairing_validation = validate_matched_initial_state_pairing(new, matched_c00)
    legacy, legacy_meta, legacy_validation = load_legacy_evidence(config)
    validation = {
        "study_id": contract["study_id"],
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "config_sha256": sha256_file(CONFIG_PATH),
        "condition_canonical_sha256": contract["conditions"]["canonical_sha256"],
        "new": new_validation,
        "matched_c00": matched_c00_validation,
        "matched_c00_initial_state_pairing": matched_pairing_validation,
        "legacy": legacy_validation,
        "training_seeds_are_independent_inference_units": True,
        "evaluation_initial_states_are_paired_repetitions": True,
        "new_metadata_condition_count": len(new_meta),
        "legacy_metadata_condition_count": len(legacy_meta),
        "passed": True,
    }
    if args.validate_only:
        print(json.dumps(json_ready(validation), ensure_ascii=False, indent=2, allow_nan=False))
        return

    episodes = pd.concat([new, matched_c00, legacy], ignore_index=True, sort=False)
    episodes = episodes.sort_values(
        ["source_study", "condition_id", "training_seed", "evaluation_seed"]
    ).reset_index(drop=True)
    seed_summary = condition_seed_summary(episodes)
    summary = condition_summary(episodes, seed_summary)
    summary_lookup, seed_lookup, _ = lookups(summary, seed_summary, episodes)
    rule = thresholds(contract)
    cards, card_details = build_per_k_cards(summary_lookup, seed_lookup, rule)
    pairs, pair_details = build_joint_pair_effects(summary_lookup, seed_lookup, rule)
    failures = failure_decomposition(episodes)
    summary = annotate_condition_references(summary, summary_lookup, seed_lookup, rule)

    atomic_csv(ANALYSIS / "episode_long.csv", episodes)
    atomic_csv(ANALYSIS / "condition_seed_summary.csv", seed_summary)
    atomic_csv(ANALYSIS / "condition_summary.csv", summary)
    atomic_csv(ANALYSIS / "per_k_mechanism_cards.csv", cards)
    atomic_csv(ANALYSIS / "failure_decomposition.csv", failures)
    atomic_csv(ANALYSIS / "joint_pair_effects.csv", pairs)

    figure_paths = [
        plot_heatmap(cards),
        plot_dose(cards),
        plot_timing(cards),
        plot_joint_pairs(pairs),
        plot_failures(cards),
    ]
    classification = {
        "schema": "obs2_v2_1_k_causal_completion/classification/v1",
        "study_id": contract["study_id"],
        "analysis_source_sha256": sha256_file(Path(__file__)),
        "validation": validation,
        "inference_contract": {
            "independent_unit": "training seed / frozen learned policy",
            "independent_n": 5,
            "paired_repetitions_per_training_seed": 20,
            "prohibited_interpretation": "Do not treat 100 episodes as 100 independent learned policies.",
        },
        "decision_thresholds_from_study_contract": rule,
        "equivalence_primary_continuous_metrics": list(PRIMARY_METRICS),
        "equivalence_no_consistent_degradation_interpretation": (
            "No success or primary metric may degrade in at least 4 of 5 training seeds."
        ),
        "per_k": card_details,
        "joint_pairs": pair_details,
        "figures": figure_paths,
        "output_tables": [
            "analysis_causal_completion/episode_long.csv",
            "analysis_causal_completion/condition_seed_summary.csv",
            "analysis_causal_completion/condition_summary.csv",
            "analysis_causal_completion/per_k_mechanism_cards.csv",
            "analysis_causal_completion/failure_decomposition.csv",
            "analysis_causal_completion/joint_pair_effects.csv",
        ],
    }
    atomic_json(ANALYSIS / "classification.json", classification)
    markdown = classification_markdown(cards, pairs, rule, validation, figure_paths)
    atomic_text(ANALYSIS / "classification.md", markdown)
    manifest = {
        "schema": "obs2_v2_1_k_causal_completion/analysis_manifest/v1",
        "study_id": contract["study_id"],
        "analysis_source": str(Path(__file__).name),
        "analysis_source_sha256": sha256_file(Path(__file__)),
        "contract_sha256": sha256_file(CONTRACT_PATH),
        "inputs": validation,
        "files": {
            str(path.relative_to(ROOT)): sha256_file(path)
            for path in sorted(ANALYSIS.rglob("*"))
            if path.is_file() and path.name != "analysis_manifest.json"
        },
    }
    atomic_json(ANALYSIS / "analysis_manifest.json", manifest)
    print(json.dumps(json_ready(manifest), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
