"""Fail-closed checkpoint-1500 evaluator for the six formal configurations.

The default action is a read-only contract scan.  ``--execute`` always means
the complete frozen 6 x 5 x 20 matrix.  Scientific settings cannot be changed
from the command line.  Interrupted runs may resume only from task artifacts
whose signatures and hashes pass validation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import os
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "formal_six_config_checkpoint1500_evaluator/v1"
TASK_SCHEMA = "formal_six_config_checkpoint1500_evaluator/task/v1"
VALIDATION_SCHEMA = "formal_six_config_checkpoint1500_evaluator/validation/v1"
CONFIG_SCHEMA = "formal_six_config_checkpoint1500_evaluator_config/v1"
REQUIRED_CHECKPOINT_KEYS = {"policy", "critic", "metadata"}
EXPECTED_ARMS = (
    "HPR_DTH_PS",
    "HPR_THDOT_PS",
    "HPR_OBS_PS",
    "HPR_O2_PS",
    "HPR_O2_JS",
    "SGRR_O2_JS",
)
FINITE_EPISODE_FIELDS = (
    "initial_body_length",
    "forward_displacement",
    "forward_body_lengths",
    "scaled_signed_horizontal_progress_x100",
    "net_best_fit_rotation_degrees",
    "desired_net_rotation_degrees",
    "desired_active_rotation_fraction",
    "best_fit_rotation_range_degrees",
    "max_rotation_excursion_degrees",
    "max_abs_step_rotation_degrees",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite JSON float: {value}")
        return value
    if hasattr(value, "item"):
        return json_safe(value.item())
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows([{key: json_safe(value) for key, value in row.items()} for row in rows])
    os.replace(temporary, path)


def atomic_npz(path: Path, np: Any, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def metadata_value(metadata: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in metadata:
        return metadata[key]
    training_args = metadata.get("training_args")
    if isinstance(training_args, dict) and key in training_args:
        return training_args[key]
    return default


def exact_number(actual: Any, expected: Any, name: str) -> None:
    if actual is None or not math.isclose(
        float(actual), float(expected), rel_tol=0.0, abs_tol=1e-12
    ):
        raise RuntimeError(f"Locked contract drift: {name}={actual!r}, expected {expected!r}")


def sample_stats(values: Iterable[float], prefix: str) -> dict[str, float]:
    numbers = [float(value) for value in values]
    if not numbers or not all(math.isfinite(value) for value in numbers):
        raise RuntimeError(f"Invalid values for {prefix}: {numbers}")
    return {
        f"{prefix}_mean": statistics.fmean(numbers),
        f"{prefix}_sd": statistics.stdev(numbers) if len(numbers) > 1 else 0.0,
        f"{prefix}_median": statistics.median(numbers),
        f"{prefix}_min": min(numbers),
        f"{prefix}_max": max(numbers),
    }


def exact_two_sided_sign_flip_p(differences: Iterable[float]) -> float:
    values = [float(value) for value in differences]
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError("Sign-flip input is empty or non-finite")
    observed = abs(statistics.fmean(values))
    total = 2 ** len(values)
    extreme = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        permuted = abs(statistics.fmean(sign * value for sign, value in zip(signs, values)))
        if permuted >= observed - 1e-15:
            extreme += 1
    return extreme / total


def validate_config(config: dict[str, Any]) -> None:
    if config.get("schema") != CONFIG_SCHEMA:
        raise RuntimeError("Evaluator config schema drift")
    if tuple(config.get("arm_order", ())) != EXPECTED_ARMS:
        raise RuntimeError("Six-arm order drift")
    if tuple(config.get("arms", {}).keys()) != EXPECTED_ARMS:
        raise RuntimeError("Six-arm definition drift")
    if list(config["training_seeds"]) != list(range(9201, 9206)):
        raise RuntimeError("Training-seed matrix drift")
    if config["seed_to_formal_run"] != {
        str(seed): seed - 9201 for seed in range(9201, 9206)
    }:
        raise RuntimeError("Formal-run mapping drift")
    if int(config["checkpoint_batch"]) != 1500:
        raise RuntimeError("Endpoint checkpoint drift")
    evaluation = config["evaluation"]
    if list(evaluation["reset_seeds"]) != list(range(20264101, 20264121)):
        raise RuntimeError("Reset-seed panel drift")
    if int(evaluation["steps"]) != 1000:
        raise RuntimeError("Rollout duration drift")
    if evaluation["policy_mode"] != "deterministic_distribution_location":
        raise RuntimeError("Policy mode drift")
    if evaluation["terrain"] != "flat" or evaluation["terrain_contact_mode"] != "legacy_flat":
        raise RuntimeError("Terrain contract drift")
    if evaluation["desired_direction"] != "right" or evaluation["tail_side"] != "left":
        raise RuntimeError("Direction/tail contract drift")
    exact_number(evaluation["active_rotation_increment_degrees"], 0.05, "active rotation")
    if evaluation["primary_lenient_rotation_span"] != {
        "minimum_rotation_span_degrees": 360.0,
        "direction_gate_used": False,
        "displacement_gate_used": False,
        "pulse_or_contact_gate_used": False,
    }:
        raise RuntimeError("Primary lenient endpoint drift")
    if evaluation["secondary_strict_common_kinematic"] != {
        "minimum_desired_net_rotation_degrees": 360.0,
        "minimum_desired_active_rotation_fraction": 0.7,
        "minimum_forward_body_lengths": 1.0,
        "pulse_or_contact_gate_used": False,
    }:
        raise RuntimeError("Secondary strict endpoint drift")
    if int(evaluation["run_discovery_threshold_successes"]) != 10:
        raise RuntimeError("Run-level discovery threshold drift")
    if len(config["pre_registered_local_comparisons"]) != 5:
        raise RuntimeError("Local-comparison matrix drift")


@dataclass(frozen=True)
class Task:
    arm_id: str
    seed: int
    checkpoint_batch: int = 1500

    @property
    def formal_run(self) -> int:
        return self.seed - 9201

    @property
    def task_id(self) -> str:
        return f"{self.arm_id}__run{self.formal_run}__checkpoint{self.checkpoint_batch:04d}"


def all_tasks(config: dict[str, Any]) -> list[Task]:
    return [
        Task(arm_id, int(seed), int(config["checkpoint_batch"]))
        for arm_id in config["arm_order"]
        for seed in config["training_seeds"]
    ]


def run_name(config: dict[str, Any], task: Task) -> str:
    tag = config["arms"][task.arm_id]["archive_tag"]
    return f"formal__seed{task.seed}__{tag}"


def run_dir(config: dict[str, Any], task: Task) -> Path:
    return (Path(config["formal_runs_root"]) / run_name(config, task)).resolve()


def checkpoint_path(config: dict[str, Any], task: Task) -> Path:
    return run_dir(config, task) / f"checkpoint_{task.checkpoint_batch}.pt"


def task_paths(output_root: Path, task: Task) -> tuple[Path, Path]:
    directory = output_root / "tasks" / task.arm_id / f"run{task.formal_run}"
    base = directory / f"checkpoint_{task.checkpoint_batch:04d}"
    return base.with_suffix(".json"), base.with_suffix(".npz")


class Dependencies:
    def __init__(self, config: dict[str, Any]) -> None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        site_packages = str(Path(config["runtime"]["site_packages"]).resolve())
        snapshot = Path(config["source_snapshot_root"]).resolve()
        training = str((snapshot / "training").resolve())
        # ProcessPool workers inherit the parent's sys.path on Windows.  Move,
        # rather than merely add, the formal paths to the front so an analysis
        # helper from another frozen study can never shadow the formal runtime.
        for path in (training, site_packages):
            while path in sys.path:
                sys.path.remove(path)
        sys.path.insert(0, site_packages)
        sys.path.insert(0, training)
        import numpy as np  # type: ignore
        import torch  # type: ignore
        from analyze_training_results import (  # type: ignore
            TerrainArgs,
            build_demo_env,
            load_policy_for_env,
            metadata_from_checkpoint,
        )
        from demo_metamaterial import choose_action  # type: ignore
        from metamaterial_envs.env import metamaterial as metamaterial_module  # type: ignore

        torch.set_num_threads(int(config["runtime"]["torch_num_threads_per_worker"]))
        self.np = np
        self.torch = torch
        self.TerrainArgs = TerrainArgs
        self.build_demo_env = build_demo_env
        self.load_policy_for_env = load_policy_for_env
        self.metadata_from_checkpoint = metadata_from_checkpoint
        self.choose_action = choose_action
        actual_environment_path = Path(metamaterial_module.__file__).resolve()
        expected_environment_path = (
            snapshot
            / config["immutable_runtime"]["environment_source"]["relative"]
        ).resolve()
        if actual_environment_path != expected_environment_path:
            raise RuntimeError(
                "Wrong environment module imported: "
                f"{actual_environment_path} != {expected_environment_path}"
            )
        actual_environment_hash = sha256_file(actual_environment_path)
        if actual_environment_hash != config["immutable_runtime"]["environment_source"]["sha256"]:
            raise RuntimeError("Imported environment module hash drift")
        self.environment_module_path = actual_environment_path
        self.environment_module_sha256 = actual_environment_hash
        metric_path = snapshot / config["immutable_runtime"]["metric_helper"]["relative"]
        self.metric = load_module(metric_path, f"formal_common_metric_{os.getpid()}")
        self.metric_args = self.metric._parser().parse_args([])
        self.metric_args.active_rotation_degrees = float(
            config["evaluation"]["active_rotation_increment_degrees"]
        )
        span_path = Path(
            config["immutable_analysis_extension"]["rotation_span_helper"]["absolute"]
        ).resolve()
        span_training = str(span_path.parent)
        span_path_was_present = span_training in sys.path
        if not span_path_was_present:
            sys.path.insert(0, span_training)
        try:
            self.span_metric = load_module(span_path, f"formal_span_metric_{os.getpid()}")
        finally:
            if not span_path_was_present and span_training in sys.path:
                sys.path.remove(span_training)


def source_identity(config: dict[str, Any]) -> dict[str, Any]:
    snapshot = Path(config["source_snapshot_root"]).resolve()
    records: dict[str, Any] = {}
    for name, item in config["immutable_runtime"].items():
        path = (snapshot / item["relative"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != item["sha256"]:
            raise RuntimeError(f"Frozen source hash drift for {name}: {actual}")
        records[name] = {"path": str(path), "sha256": actual}
    span_item = config["immutable_analysis_extension"]["rotation_span_helper"]
    span_path = Path(span_item["absolute"]).resolve()
    if not span_path.is_file():
        raise FileNotFoundError(span_path)
    span_hash = sha256_file(span_path)
    if span_hash != span_item["sha256"]:
        raise RuntimeError(f"Frozen span helper hash drift: {span_hash}")
    records["rotation_span_helper"] = {
        "path": str(span_path),
        "sha256": span_hash,
        "permitted_symbols": span_item["permitted_symbols"],
    }
    python = Path(config["runtime"]["python"]).resolve()
    if not python.is_file() or sha256_file(python) != config["runtime"]["python_sha256"]:
        raise RuntimeError("Frozen Python executable identity drift")
    records["python"] = {"path": str(python), "sha256": sha256_file(python)}
    evaluator_script = Path(__file__).resolve()
    records["evaluator_script"] = {
        "path": str(evaluator_script),
        "sha256": sha256_file(evaluator_script),
    }
    return records


def validate_metadata(
    metadata: dict[str, Any], config: dict[str, Any], task: Task
) -> dict[str, Any]:
    common = config["locked_common_controller"]
    arm = config["arms"][task.arm_id]
    expected_run = run_name(config, task)
    exact_fields = {
        "scenario": common["scenario"],
        "num_particles": common["num_particles"],
        "algorithm": common["algorithm"],
        "channel": arm["channel"],
        "observation_func": arm["observation_func"],
        "control_mode": arm["control_mode"],
        "reward_func": arm["reward_func"],
        "share_policy": arm["share_policy"],
        "per_joint_k1_k2": arm["per_joint_k1_k2"],
        "share_critic": common["share_critic"],
        "centralised_critic": common["centralised_critic"],
        "terrain_type": common["terrain_type"],
        "terrain_contact_mode": common["terrain_contact_mode"],
        "init_pos_randomness": common["init_pos_randomness"],
        "init_angle_range_degrees": common["init_angle_range_degrees"],
        "init_height_jitter": common["init_height_jitter"],
        "episodes": common["training_batches"],
        "seed": task.seed,
        "run_name": expected_run,
    }
    aliases = {"num_particles": ("num_particles", "n_particles")}
    for key, expected in exact_fields.items():
        actual = None
        for candidate in aliases.get(key, (key,)):
            actual = metadata_value(metadata, candidate, None)
            if actual is not None:
                break
        if actual != expected:
            raise RuntimeError(
                f"Metadata drift for {task.task_id}: {key}={actual!r}, expected {expected!r}"
            )
    for key, expected in (
        ("feedback_gain", common["feedback_gain"]),
        ("k_action_scale", common["k_action_scale"]),
        ("max_control_gain", common["max_active_torque"]),
        ("passive_kappa", common["passive_kappa"]),
    ):
        exact_number(metadata_value(metadata, key), expected, f"{task.task_id}.{key}")
    if bool(metadata_value(metadata, "fix_k1", True)) or bool(
        metadata_value(metadata, "fix_k2", True)
    ):
        raise RuntimeError(f"Fixed K output flag in {task.task_id}")
    if metadata_value(metadata, "pretrained_model_path", None) is not None:
        raise RuntimeError(f"Non-scratch lineage in {task.task_id}")
    if bool(metadata_value(metadata, "rolling_observation", True)):
        raise RuntimeError(f"Rolling observation unexpectedly enabled in {task.task_id}")
    if bool(metadata_value(metadata, "tail_roll_observation", True)):
        raise RuntimeError(f"Tail-roll observation unexpectedly enabled in {task.task_id}")
    if bool(metadata_value(metadata, "fast_forward_observation", True)):
        raise RuntimeError(f"Fast-forward observation unexpectedly enabled in {task.task_id}")
    if arm["control_mode"] == "formula" and tuple(
        metadata_value(metadata, "formula_action_names", ())
    ) != tuple(common["formula_action_names"]):
        raise RuntimeError(f"Formula action order drift in {task.task_id}")
    return {
        "metadata_contract_valid": True,
        "expected_run_name": expected_run,
        "paper_label": arm["paper_label"],
        "action_semantics": arm["action_semantics"],
        "observation_dim": arm["observation_dim"],
        "action_dim": arm["action_dim"],
    }


def tensor_tree_is_finite(torch: Any, value: Any, label: str) -> int:
    checked = 0
    if isinstance(value, dict):
        for key, item in value.items():
            checked += tensor_tree_is_finite(torch, item, f"{label}.{key}")
    elif torch.is_tensor(value):
        checked += 1
        if value.is_floating_point() or value.is_complex():
            if not bool(torch.isfinite(value).all().item()):
                raise RuntimeError(f"Non-finite checkpoint tensor: {label}")
    return checked


def validate_training_artifacts(
    config: dict[str, Any], task: Task, deps: Dependencies
) -> dict[str, Any]:
    directory = run_dir(config, task)
    metadata_path = directory / "metadata.json"
    summary_path = directory / "training_summary.json"
    log_path = directory / "training_log.csv"
    checkpoint = checkpoint_path(config, task)
    for path in (metadata_path, summary_path, log_path, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    file_metadata = read_json(metadata_path)
    summary = read_json(summary_path)
    if summary.get("status") != "complete" or int(summary.get("episodes", -1)) != 1500:
        raise RuntimeError(f"Incomplete training summary: {summary_path}")
    with log_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    episodes = [int(row["episode"]) for row in rows]
    if len(rows) != 1500 or episodes != list(range(1, 1501)):
        raise RuntimeError(f"Non-contiguous 1500-batch training log: {log_path}")
    for row_index, row in enumerate(rows, start=1):
        for field in ("reward_mean", "speed_x100", "approx_kl"):
            if field in row and row[field] not in ("", None):
                if not math.isfinite(float(row[field])):
                    raise RuntimeError(f"Non-finite {field} at {log_path}:{row_index}")
    raw_checkpoint = deps.torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(raw_checkpoint, dict) or not REQUIRED_CHECKPOINT_KEYS.issubset(raw_checkpoint):
        raise RuntimeError(f"Checkpoint payload key drift: {checkpoint}")
    tensor_count = tensor_tree_is_finite(deps.torch, raw_checkpoint["policy"], "policy")
    tensor_count += tensor_tree_is_finite(deps.torch, raw_checkpoint["critic"], "critic")
    if tensor_count <= 0:
        raise RuntimeError(f"Checkpoint contains no policy/critic tensors: {checkpoint}")
    checkpoint_metadata = dict(deps.metadata_from_checkpoint(checkpoint))
    file_contract = validate_metadata(file_metadata, config, task)
    checkpoint_contract = validate_metadata(checkpoint_metadata, config, task)
    identity_keys = (
        "run_name",
        "seed",
        "channel",
        "observation_func",
        "control_mode",
        "reward_func",
        "share_policy",
        "per_joint_k1_k2",
    )
    for key in identity_keys:
        if metadata_value(file_metadata, key) != metadata_value(checkpoint_metadata, key):
            raise RuntimeError(f"File/checkpoint metadata mismatch in {task.task_id}: {key}")
    return {
        "status": "passed",
        "run_dir": str(directory),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "metadata": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "training_summary": str(summary_path),
        "training_summary_sha256": sha256_file(summary_path),
        "training_log": str(log_path),
        "training_log_sha256": sha256_file(log_path),
        "training_log_rows": len(rows),
        "checkpoint_finite_tensor_count": tensor_count,
        "file_contract": file_contract,
        "checkpoint_contract": checkpoint_contract,
    }


def close_env(env: Any) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def validate_environment(env: Any, config: dict[str, Any], task: Task) -> dict[str, Any]:
    common = config["locked_common_controller"]
    arm = config["arms"][task.arm_id]
    if int(getattr(env, "num_particles", -1)) != int(common["num_particles"]):
        raise RuntimeError("Environment particle-count drift")
    if str(getattr(env, "observation_func", "")) != arm["observation_func"]:
        raise RuntimeError("Environment observation-function drift")
    if str(getattr(env, "control_mode", "")) != arm["control_mode"]:
        raise RuntimeError("Environment control-mode drift")
    if str(getattr(env, "terrain_contact_mode", "")) != common["terrain_contact_mode"]:
        raise RuntimeError("Environment contact-mode drift")
    for key, expected in (
        ("feedback_gain", common["feedback_gain"]),
        ("k_action_scale", common["k_action_scale"]),
        ("max_torque", common["max_active_torque"]),
    ):
        exact_number(getattr(env, key, None), expected, f"environment.{key}")
    if arm["control_mode"] == "formula" and tuple(
        getattr(env, "formula_action_names", ())
    ) != tuple(common["formula_action_names"]):
        raise RuntimeError("Environment formula-action order drift")
    obs_shape = tuple(env.observation_spec[("agents", "observation")].shape)
    action_shape = tuple(env.action_spec[env.action_key].shape)
    expected_obs = (1, 8, int(arm["observation_dim"]))
    expected_action = (1, 8, int(arm["action_dim"]))
    if obs_shape != expected_obs or action_shape != expected_action:
        raise RuntimeError(
            f"Environment shape drift for {task.task_id}: {obs_shape}/{action_shape}, "
            f"expected {expected_obs}/{expected_action}"
        )
    return {
        "validated": True,
        "observation_shape": list(obs_shape),
        "action_shape": list(action_shape),
        "observation_func": arm["observation_func"],
        "control_mode": arm["control_mode"],
        "action_semantics": arm["action_semantics"],
        "max_active_torque": float(env.max_torque),
    }


def policy_sha256(policy: Any) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(policy.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def strict_common_success(metrics: dict[str, Any], criterion: dict[str, Any]) -> bool:
    if criterion.get("pulse_or_contact_gate_used") is not False:
        raise RuntimeError("Strict common criterion unexpectedly includes pulse/contact")
    return bool(
        float(metrics["desired_net_rotation_degrees"])
        >= float(criterion["minimum_desired_net_rotation_degrees"])
        and float(metrics["desired_active_rotation_fraction"])
        >= float(criterion["minimum_desired_active_rotation_fraction"])
        and float(metrics["forward_body_lengths"])
        >= float(criterion["minimum_forward_body_lengths"])
    )


def run_span_self_tests(deps: Dependencies) -> dict[str, Any]:
    np = deps.np
    helper = deps.span_metric
    cases = {
        "positive_370": (np.full(37, math.radians(10.0)), True),
        "negative_370": (np.full(37, math.radians(-10.0)), True),
        "return_to_start": (
            np.r_[np.full(37, math.radians(10.0)), np.full(37, math.radians(-10.0))],
            True,
        ),
        "rocking_below_360_span": (
            np.tile(np.r_[np.full(18, math.radians(10.0)), np.full(18, math.radians(-10.0))], 5),
            False,
        ),
        "boundary_359_9": (np.asarray([math.radians(359.9)]), False),
        "boundary_360": (np.asarray([math.radians(360.0)]), True),
    }
    results: dict[str, bool] = {}
    for name, (increments, expected) in cases.items():
        actual = bool(helper._rotation_excursion_metrics(increments)["loose_360_detected"])
        if actual != expected:
            raise RuntimeError(f"Rotation-span self-test failed: {name}")
        results[name] = actual
    return {"status": "passed", "cases": results}


def paired_reset_and_deterministic_action_gate(
    config: dict[str, Any], deps: Dependencies
) -> dict[str, Any]:
    """Verify paired initial positions and repeatable deterministic actions."""
    np, torch = deps.np, deps.torch
    action_checks: dict[str, Any] = {}
    reference_positions = None
    maximum_initial_position_error = 0.0
    for arm_id in config["arm_order"]:
        task = Task(arm_id, 9201, int(config["checkpoint_batch"]))
        checkpoint = checkpoint_path(config, task)
        metadata = dict(deps.metadata_from_checkpoint(checkpoint))
        env, _, _, _ = deps.build_demo_env(
            metadata,
            config["evaluation"]["terrain"],
            deps.TerrainArgs(),
            max_steps=int(config["evaluation"]["steps"]),
            render_mode="rgb_array",
            num_envs=1,
        )
        try:
            validate_environment(env, config, task)
            policy = deps.load_policy_for_env(checkpoint, env, metadata)
            arm_positions = []
            for reset_seed in config["evaluation"]["reset_seeds"]:
                random.seed(int(reset_seed))
                np.random.seed(int(reset_seed))
                torch.manual_seed(int(reset_seed))
                td = env.reset()
                position = np.asarray(deps.metric._positions(env), dtype=np.complex128)
                arm_positions.append(position)
                if int(reset_seed) == int(config["evaluation"]["reset_seeds"][0]):
                    first = deps.choose_action(policy, td.clone(recurse=True), "deterministic")[
                        "agents", "action"
                    ].detach()
                    second = deps.choose_action(policy, td.clone(recurse=True), "deterministic")[
                        "agents", "action"
                    ].detach()
                    if not bool(torch.equal(first, second)) or not bool(
                        torch.isfinite(first).all().item()
                    ):
                        raise RuntimeError(
                            f"Deterministic action repeatability failed: {task.task_id}"
                        )
                    action_checks[task.arm_id] = {
                        "shape": list(first.shape),
                        "exact_repeat": True,
                        "finite": True,
                    }
        finally:
            close_env(env)
        arm_positions_array = np.stack(arm_positions, axis=0)
        if reference_positions is None:
            reference_positions = arm_positions_array
        else:
            error = float(np.max(np.abs(arm_positions_array - reference_positions)))
            maximum_initial_position_error = max(maximum_initial_position_error, error)
            if error > 1e-12:
                raise RuntimeError(f"Paired initial-state mismatch for {task.arm_id}: {error}")
    return {
        "status": "passed",
        "reset_seed_count": len(config["evaluation"]["reset_seeds"]),
        "configuration_count": len(config["arm_order"]),
        "maximum_initial_position_absolute_error": maximum_initial_position_error,
        "deterministic_action_checks": action_checks,
    }


def run_episode(
    deps: Dependencies,
    env: Any,
    policy: Any,
    config: dict[str, Any],
    task: Task,
    reset_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    np, torch = deps.np, deps.torch
    arm = config["arms"][task.arm_id]
    steps = int(config["evaluation"]["steps"])
    obs_dim = int(arm["observation_dim"])
    action_dim = int(arm["action_dim"])
    torch.manual_seed(int(reset_seed))
    np.random.seed(int(reset_seed))
    random.seed(int(reset_seed))
    td = env.reset()
    positions = [deps.metric._positions(env)]
    support = [deps.metric._log_info_scalar(td, "fast_forward_support_index")]
    contact = [deps.metric._log_info_scalar(td, "fast_forward_ground_contact_strength")]
    raw_obs = np.empty((steps, 8, obs_dim), dtype=np.float32)
    policy_action = np.empty((steps, 8, action_dim), dtype=np.float32)
    tau_pre = np.empty((steps, 8), dtype=np.float32)
    tau_executed = np.empty_like(tau_pre)
    saturated = np.empty((steps, 8), dtype=np.uint8)
    if arm["control_mode"] == "formula":
        physical_gain = np.empty((steps, 8, 2), dtype=np.float32)
        tau_k1 = np.empty((steps, 8), dtype=np.float32)
        tau_k2 = np.empty((steps, 8), dtype=np.float32)
    max_torque = float(env.max_torque)
    for step in range(steps):
        observation_t = td["agents", "observation"].detach().clone()
        expected_obs_shape = (1, 8, obs_dim)
        if tuple(observation_t.shape) != expected_obs_shape or not bool(
            torch.isfinite(observation_t).all().item()
        ):
            raise RuntimeError(f"Observation contract failed at {task.task_id}, step {step}")
        chosen = deps.choose_action(policy, td, "deterministic")
        action_t = chosen["agents", "action"].detach().clone()
        expected_action_shape = (1, 8, action_dim)
        if tuple(action_t.shape) != expected_action_shape or not bool(
            torch.isfinite(action_t).all().item()
        ):
            raise RuntimeError(f"Action contract failed at {task.task_id}, step {step}")
        obs_np = observation_t[0].cpu().numpy().astype(np.float32, copy=True)
        action_np = action_t[0].cpu().numpy().astype(np.float32, copy=True)
        raw_obs[step] = obs_np
        policy_action[step] = action_np
        if arm["control_mode"] == "direct":
            command = action_np[:, 0]
            if float(np.max(np.abs(command))) > max_torque + 1e-5:
                raise RuntimeError(f"Direct action exceeds active-torque bound in {task.task_id}")
            tau_pre[step] = command
            tau_executed[step] = np.clip(command, -max_torque, max_torque)
            saturated[step] = (np.abs(command) >= max_torque - 1e-6).astype(np.uint8)
        else:
            gain_np = float(env.k_action_scale) * action_np
            tau1_np = gain_np[:, 0] * obs_np[:, 0]
            tau2_np = gain_np[:, 1] * obs_np[:, 1]
            total_np = tau1_np + tau2_np
            physical_gain[step] = gain_np
            tau_k1[step] = tau1_np
            tau_k2[step] = tau2_np
            tau_pre[step] = total_np
            tau_executed[step] = np.clip(total_np, -max_torque, max_torque)
            saturated[step] = (np.abs(total_np) >= max_torque).astype(np.uint8)
        env_td = td.clone(recurse=True)
        env_td["agents", "action"] = action_t
        td = env.step(env_td)["next"]
        position = deps.metric._positions(env)
        if not np.isfinite(np.real(position)).all() or not np.isfinite(np.imag(position)).all():
            raise RuntimeError(f"Non-finite position at {task.task_id}, step {step + 1}")
        positions.append(position)
        support.append(deps.metric._log_info_scalar(td, "fast_forward_support_index"))
        contact.append(
            deps.metric._log_info_scalar(td, "fast_forward_ground_contact_strength")
        )
    metrics = deps.metric._episode_metrics(
        positions,
        config["evaluation"]["desired_direction"],
        config["evaluation"]["tail_side"],
        deps.metric_args,
        support,
        contact,
    )
    position_complex = np.asarray(positions, dtype=np.complex128)
    increments = np.asarray(
        [
            deps.span_metric._best_fit_rotation(position_complex[index], position_complex[index + 1])
            for index in range(steps)
        ],
        dtype=np.float64,
    )
    old_increments = np.asarray(
        [
            deps.metric._best_fit_rotation(position_complex[index], position_complex[index + 1])
            for index in range(steps)
        ],
        dtype=np.float64,
    )
    if not np.array_equal(increments, old_increments):
        maximum_error = float(np.max(np.abs(increments - old_increments)))
        if maximum_error > 1e-15:
            raise RuntimeError(f"Rotation helper mismatch: {maximum_error}")
    excursion = deps.span_metric._rotation_excursion_metrics(
        increments,
        threshold_radians=math.radians(
            float(
                config["evaluation"]["primary_lenient_rotation_span"][
                    "minimum_rotation_span_degrees"
                ]
            )
        ),
    )
    metrics.update(excursion)
    metrics["reset_seed"] = int(reset_seed)
    metrics["success_lenient_rotation_span"] = bool(excursion["loose_360_detected"])
    metrics["success_secondary_strict_common_kinematic"] = strict_common_success(
        metrics, config["evaluation"]["secondary_strict_common_kinematic"]
    )
    metrics["scaled_signed_horizontal_progress_x100"] = float(
        float(config["evaluation"]["scaled_signed_horizontal_progress_multiplier"])
        * float(metrics["forward_displacement"])
        / steps
    )
    metrics["torque_saturation_fraction_pre_step"] = float(np.mean(saturated))
    metrics["mean_abs_active_torque_pre_step"] = float(np.mean(np.abs(tau_pre)))
    if arm["control_mode"] == "formula":
        metrics["mean_abs_k1"] = float(np.mean(np.abs(physical_gain[..., 0])))
        metrics["mean_abs_k2"] = float(np.mean(np.abs(physical_gain[..., 1])))
    for field in FINITE_EPISODE_FIELDS:
        if not math.isfinite(float(metrics[field])):
            raise RuntimeError(f"Non-finite endpoint {field} in {task.task_id}")
    cumulative_degrees = np.degrees(np.r_[0.0, np.cumsum(increments)]).astype(np.float32)
    arrays: dict[str, Any] = {
        "positions_xy": np.stack(
            (np.real(position_complex), np.imag(position_complex)), axis=-1
        ).astype(np.float32),
        "raw_environment_observation": raw_obs,
        "deterministic_policy_action": policy_action,
        "best_fit_rotation_increment_degrees": np.degrees(increments).astype(np.float32),
        "unwrapped_best_fit_rotation_degrees": cumulative_degrees,
        "tau_active_pre_step_unclipped": tau_pre,
        "tau_active_pre_step_executed": tau_executed,
        "torque_saturated_pre_step": saturated,
        "support_index": np.asarray(
            [np.nan if value is None else value for value in support], dtype=np.float32
        ),
        "ground_contact_strength": np.asarray(
            [np.nan if value is None else value for value in contact], dtype=np.float32
        ),
    }
    if arm["control_mode"] == "formula":
        arrays.update(
            {
                "physical_gain": physical_gain,
                "tau_k1_pre_step_unclipped": tau_k1,
                "tau_k2_pre_step_unclipped": tau_k2,
            }
        )
    return metrics, arrays


def official_metric_projection(row: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(row[key])
        for key in (
            "initial_body_length",
            "forward_displacement",
            "forward_body_lengths",
            "net_best_fit_rotation_degrees",
            "desired_net_rotation_degrees",
            "desired_active_rotation_fraction",
        )
    }


def official_identity_gate(
    config: dict[str, Any], task: Task, episodes: list[dict[str, Any]]
) -> dict[str, Any] | None:
    template = config["arms"][task.arm_id]["official_endpoint_relative_template"]
    if template is None:
        return None
    official_path = (
        Path(config["official_parent_formal_root"]) / template.format(seed=task.seed)
    ).resolve()
    if not official_path.is_file():
        raise FileNotFoundError(official_path)
    payload = read_json(official_path)
    official = payload["results"][0]["episodes"]
    expected = {int(row["seed"]): row for row in official}
    tolerance = float(config["evaluation"]["official_identity_absolute_tolerance"])
    maximum = {key: 0.0 for key in official_metric_projection(episodes[0])}
    if len(episodes) != 20 or len(expected) != 20:
        raise RuntimeError("Official endpoint identity episode-count mismatch")
    for row in episodes:
        reset_seed = int(row["reset_seed"])
        if reset_seed not in expected:
            raise RuntimeError(f"Official endpoint lacks reset seed {reset_seed}")
        actual_fields = official_metric_projection(row)
        expected_fields = official_metric_projection(expected[reset_seed])
        for key, actual in actual_fields.items():
            error = abs(actual - expected_fields[key])
            maximum[key] = max(maximum[key], error)
            if error > tolerance:
                raise RuntimeError(
                    f"Official identity mismatch {task.task_id}, reset {reset_seed}, "
                    f"{key}: {error:.3g} > {tolerance:.3g}"
                )
    return {
        "passed": True,
        "official_path": str(official_path),
        "official_sha256": sha256_file(official_path),
        "absolute_tolerance": tolerance,
        "maximum_absolute_error": maximum,
    }


def task_signature(
    config_path: Path, config: dict[str, Any], task: Task, checkpoint_sha256: str
) -> dict[str, Any]:
    return {
        "config_sha256": sha256_file(config_path),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "task": asdict(task),
        "checkpoint_path": str(checkpoint_path(config, task)),
        "checkpoint_sha256": checkpoint_sha256,
        "steps": config["evaluation"]["steps"],
        "reset_seeds": config["evaluation"]["reset_seeds"],
        "policy_mode": config["evaluation"]["policy_mode"],
        "terrain": config["evaluation"]["terrain"],
        "primary_endpoint": config["evaluation"]["primary_lenient_rotation_span"],
        "secondary_endpoint": config["evaluation"]["secondary_strict_common_kinematic"],
    }


def validate_existing_task(
    config_path: Path, config: dict[str, Any], output_root: Path, task: Task
) -> dict[str, Any] | None:
    manifest_path, trace_path = task_paths(output_root, task)
    if not manifest_path.exists() and not trace_path.exists():
        return None
    if not manifest_path.is_file() or not trace_path.is_file():
        raise RuntimeError(f"Partial existing task output: {manifest_path} / {trace_path}")
    payload = read_json(manifest_path)
    if payload.get("schema") != TASK_SCHEMA or payload.get("status") != "complete":
        raise RuntimeError(f"Invalid existing task manifest: {manifest_path}")
    current_checkpoint_hash = sha256_file(checkpoint_path(config, task))
    expected_signature = task_signature(
        config_path, config, task, current_checkpoint_hash
    )
    if payload.get("task_signature") != expected_signature:
        raise RuntimeError(f"Existing task signature drift: {manifest_path}")
    if payload.get("task_signature_sha256") != sha256_json(expected_signature):
        raise RuntimeError(f"Existing task signature hash drift: {manifest_path}")
    if sha256_file(trace_path) != payload["trace_archive"]["sha256"]:
        raise RuntimeError(f"Existing trace hash drift: {trace_path}")
    if len(payload.get("episodes", [])) != 20:
        raise RuntimeError(f"Existing task episode-count drift: {manifest_path}")
    for row in payload["episodes"]:
        for field in FINITE_EPISODE_FIELDS:
            if not math.isfinite(float(row[field])):
                raise RuntimeError(f"Non-finite existing task field: {manifest_path}:{field}")
    identity = payload.get("official_identity_gate")
    if config["arms"][task.arm_id]["official_endpoint_relative_template"] is not None:
        if not isinstance(identity, dict) or identity.get("passed") is not True:
            raise RuntimeError(f"Existing official identity gate absent: {manifest_path}")
        official_path = Path(identity["official_path"]).resolve()
        if not official_path.is_file() or sha256_file(official_path) != identity["official_sha256"]:
            raise RuntimeError(f"Archived official endpoint identity source drift: {official_path}")
    return {
        "task_id": task.task_id,
        "status": "reused_verified",
        "manifest": str(manifest_path),
        "lenient_success_count": payload["evaluation"]["lenient_success_count"],
        "strict_success_count": payload["evaluation"]["secondary_strict_success_count"],
        "wall_seconds": 0.0,
    }


def evaluate_task(
    config_path_string: str, output_root_string: str, task_dict: dict[str, Any]
) -> dict[str, Any]:
    config_path = Path(config_path_string).resolve()
    config = read_json(config_path)
    validate_config(config)
    task = Task(**task_dict)
    output_root = Path(output_root_string).resolve()
    existing = validate_existing_task(config_path, config, output_root, task)
    if existing is not None:
        return existing
    manifest_path, trace_path = task_paths(output_root, task)
    checkpoint = checkpoint_path(config, task)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha_before = sha256_file(checkpoint)
    deps = Dependencies(config)
    training_contract = validate_training_artifacts(config, task, deps)
    metadata = dict(deps.metadata_from_checkpoint(checkpoint))
    metadata_contract = validate_metadata(metadata, config, task)
    env, resolved_name, resolved_type, resolved_settings = deps.build_demo_env(
        metadata,
        config["evaluation"]["terrain"],
        deps.TerrainArgs(),
        max_steps=int(config["evaluation"]["steps"]),
        render_mode="rgb_array",
        num_envs=1,
    )
    environment_contract = validate_environment(env, config, task)
    policy = deps.load_policy_for_env(checkpoint, env, metadata)
    policy_hash_before = policy_sha256(policy)
    episodes: list[dict[str, Any]] = []
    episode_arrays: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        for reset_seed in config["evaluation"]["reset_seeds"]:
            metrics, arrays = run_episode(
                deps, env, policy, config, task, int(reset_seed)
            )
            episodes.append(metrics)
            episode_arrays.append(arrays)
        policy_hash_after = policy_sha256(policy)
    finally:
        close_env(env)
    checkpoint_sha_after = sha256_file(checkpoint)
    if checkpoint_sha_after != checkpoint_sha_before:
        raise RuntimeError(f"Checkpoint changed during evaluation: {checkpoint}")
    if policy_hash_after != policy_hash_before:
        raise RuntimeError(f"Policy state changed during evaluation: {task.task_id}")
    identity = official_identity_gate(config, task, episodes)
    np = deps.np
    trace_arrays: dict[str, Any] = {
        "reset_seeds": np.asarray(config["evaluation"]["reset_seeds"], dtype=np.int64)
    }
    keys = tuple(episode_arrays[0])
    if any(tuple(arrays) != keys for arrays in episode_arrays):
        raise RuntimeError(f"Trace-array key drift within task: {task.task_id}")
    for key in keys:
        trace_arrays[key] = np.stack([arrays[key] for arrays in episode_arrays], axis=0)
    atomic_npz(trace_path, np, trace_arrays)
    lenient_success_count = int(
        sum(bool(row["success_lenient_rotation_span"]) for row in episodes)
    )
    strict_success_count = int(
        sum(bool(row["success_secondary_strict_common_kinematic"]) for row in episodes)
    )
    threshold = int(config["evaluation"]["run_discovery_threshold_successes"])
    signature = task_signature(config_path, config, task, checkpoint_sha_before)
    arm = config["arms"][task.arm_id]
    payload = {
        "schema": TASK_SCHEMA,
        "study_id": config["study_id"],
        "status": "complete",
        "completed_at_utc": utc_now(),
        "task": {
            **asdict(task),
            "task_id": task.task_id,
            "formal_run": task.formal_run,
            "paper_label": arm["paper_label"],
            "archive_run_name": run_name(config, task),
        },
        "task_signature": signature,
        "task_signature_sha256": sha256_json(signature),
        "training_contract": training_contract,
        "metadata_contract": metadata_contract,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256_before_after": checkpoint_sha_before,
            "unchanged": True,
        },
        "policy": {
            "state_sha256_before_after": policy_hash_before,
            "unchanged": True,
            "mode": config["evaluation"]["policy_mode"],
        },
        "environment": {
            **environment_contract,
            "imported_environment_module": str(deps.environment_module_path),
            "imported_environment_module_sha256": deps.environment_module_sha256,
            "resolved_terrain_name": resolved_name,
            "resolved_terrain_type": resolved_type,
            "resolved_terrain_settings": resolved_settings,
            "device_policy": "CPU; CUDA_VISIBLE_DEVICES is empty",
            "formula_torque_trace_note": (
                "For formula controllers, saved K/tau reconstructions use the pre-control-step "
                "observation. The frozen environment recomputes formula torque during each physics substep."
                if arm["control_mode"] == "formula"
                else None
            ),
        },
        "evaluation": {
            "steps": config["evaluation"]["steps"],
            "reset_seeds": config["evaluation"]["reset_seeds"],
            "primary_lenient_rotation_span": config["evaluation"][
                "primary_lenient_rotation_span"
            ],
            "secondary_strict_common_kinematic": config["evaluation"][
                "secondary_strict_common_kinematic"
            ],
            "lenient_success_count": lenient_success_count,
            "lenient_success_rate": lenient_success_count / len(episodes),
            "run_discovered_lenient_rolling": lenient_success_count >= threshold,
            "secondary_strict_success_count": strict_success_count,
            "secondary_strict_success_rate": strict_success_count / len(episodes),
            "run_met_secondary_strict_threshold": strict_success_count >= threshold,
            "wall_seconds": time.perf_counter() - started,
        },
        "official_identity_gate": identity,
        "trace_archive": {
            "path": str(trace_path),
            "sha256": sha256_file(trace_path),
            "array_shapes": {key: list(value.shape) for key, value in trace_arrays.items()},
            "diagnostic_nan_policy": (
                "NaN is permitted only in support_index/ground_contact_strength when the legacy "
                "environment does not export these optional diagnostics."
            ),
        },
        "episodes": episodes,
    }
    atomic_json(manifest_path, payload)
    return {
        "task_id": task.task_id,
        "status": "evaluated",
        "manifest": str(manifest_path),
        "lenient_success_count": lenient_success_count,
        "strict_success_count": strict_success_count,
        "wall_seconds": payload["evaluation"]["wall_seconds"],
    }


def process_isolation_smoke_worker(config_path_string: str, arm_id: str) -> dict[str, Any]:
    """One reset/step in a spawned worker; writes no evaluation artifacts."""
    config_path = Path(config_path_string).resolve()
    config = read_json(config_path)
    validate_config(config)
    deps = Dependencies(config)
    task = Task(arm_id, 9201, int(config["checkpoint_batch"]))
    checkpoint = checkpoint_path(config, task)
    metadata = dict(deps.metadata_from_checkpoint(checkpoint))
    validate_metadata(metadata, config, task)
    env, _, _, _ = deps.build_demo_env(
        metadata,
        config["evaluation"]["terrain"],
        deps.TerrainArgs(),
        max_steps=int(config["evaluation"]["steps"]),
        render_mode="rgb_array",
        num_envs=1,
    )
    try:
        contract = validate_environment(env, config, task)
        policy = deps.load_policy_for_env(checkpoint, env, metadata)
        reset_seed = int(config["evaluation"]["reset_seeds"][0])
        random.seed(reset_seed)
        deps.np.random.seed(reset_seed)
        deps.torch.manual_seed(reset_seed)
        td = env.reset()
        action_td = deps.choose_action(policy, td.clone(recurse=True), "deterministic")
        action = action_td["agents", "action"].detach()
        if not bool(deps.torch.isfinite(action).all().item()):
            raise RuntimeError(f"Non-finite process smoke action: {task.task_id}")
        env_td = td.clone(recurse=True)
        env_td["agents", "action"] = action
        next_td = env.step(env_td)["next"]
        if not bool(deps.torch.isfinite(next_td["agents", "observation"]).all().item()):
            raise RuntimeError(f"Non-finite process smoke next observation: {task.task_id}")
        return {
            "status": "passed",
            "task_id": task.task_id,
            "environment_module": str(deps.environment_module_path),
            "environment_sha256": deps.environment_module_sha256,
            "observation_shape": contract["observation_shape"],
            "action_shape": list(action.shape),
        }
    finally:
        close_env(env)


def contract_scan(config_path: Path) -> dict[str, Any]:
    config = read_json(config_path)
    validate_config(config)
    sources = source_identity(config)
    deps = Dependencies(config)
    span_tests = run_span_self_tests(deps)
    paired_reset_gate = paired_reset_and_deterministic_action_gate(config, deps)
    records: list[dict[str, Any]] = []
    for task in all_tasks(config):
        records.append(
            {
                "task": {
                    **asdict(task),
                    "task_id": task.task_id,
                    "formal_run": task.formal_run,
                    "paper_label": config["arms"][task.arm_id]["paper_label"],
                },
                "training_artifacts": validate_training_artifacts(config, task, deps),
            }
        )
    checkpoint_hashes = [
        row["training_artifacts"]["checkpoint_sha256"] for row in records
    ]
    result = {
        "schema": SCHEMA,
        "mode": "contract_scan",
        "status": "passed",
        "checked_at_utc": utc_now(),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "source_identity": sources,
        "rotation_span_self_tests": span_tests,
        "paired_reset_and_deterministic_action_gate": paired_reset_gate,
        "task_count": len(records),
        "unique_checkpoint_paths": len(
            {row["training_artifacts"]["checkpoint"] for row in records}
        ),
        "unique_checkpoint_hashes": len(set(checkpoint_hashes)),
        "training_log_rows_total": sum(
            int(row["training_artifacts"]["training_log_rows"]) for row in records
        ),
        "records": records,
        "ready_for_complete_evaluation": len(records) == 30,
    }
    if not result["ready_for_complete_evaluation"]:
        raise RuntimeError("Contract scan did not cover 30 tasks")
    return result


def episode_csv_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    task = payload["task"]
    fields = (
        "reset_seed",
        "success_lenient_rotation_span",
        "success_secondary_strict_common_kinematic",
        "scaled_signed_horizontal_progress_x100",
        "forward_displacement",
        "forward_body_lengths",
        "best_fit_rotation_range_degrees",
        "max_rotation_excursion_degrees",
        "net_best_fit_rotation_degrees",
        "desired_net_rotation_degrees",
        "desired_active_rotation_fraction",
        "torque_saturation_fraction_pre_step",
        "mean_abs_active_torque_pre_step",
    )
    rows: list[dict[str, Any]] = []
    for episode in payload["episodes"]:
        row = {
            "configuration_id": task["arm_id"],
            "paper_label": task["paper_label"],
            "formal_run": task["formal_run"],
            "internal_training_seed": task["seed"],
            "checkpoint_batch": task["checkpoint_batch"],
            **{field: episode[field] for field in fields},
            "mean_abs_k1": episode.get("mean_abs_k1"),
            "mean_abs_k2": episode.get("mean_abs_k2"),
        }
        rows.append(row)
    return rows


def make_run_result(payload: dict[str, Any]) -> dict[str, Any]:
    task = payload["task"]
    episodes = payload["episodes"]
    row: dict[str, Any] = {
        "configuration_id": task["arm_id"],
        "paper_label": task["paper_label"],
        "formal_run": task["formal_run"],
        "internal_training_seed": task["seed"],
        "checkpoint_batch": task["checkpoint_batch"],
        "checkpoint_sha256": payload["checkpoint"]["sha256_before_after"],
        "evaluation_episode_count": len(episodes),
        "lenient_success_count": payload["evaluation"]["lenient_success_count"],
        "lenient_success_rate": payload["evaluation"]["lenient_success_rate"],
        "run_discovered_lenient_rolling": payload["evaluation"][
            "run_discovered_lenient_rolling"
        ],
        "secondary_strict_success_count": payload["evaluation"][
            "secondary_strict_success_count"
        ],
        "secondary_strict_success_rate": payload["evaluation"][
            "secondary_strict_success_rate"
        ],
        "run_met_secondary_strict_threshold": payload["evaluation"][
            "run_met_secondary_strict_threshold"
        ],
        "official_identity_gate_passed": (
            payload["official_identity_gate"]["passed"]
            if payload["official_identity_gate"] is not None
            else None
        ),
    }
    for field, prefix in (
        ("scaled_signed_horizontal_progress_x100", "scaled_progress_x100"),
        ("forward_displacement", "forward_displacement"),
        ("forward_body_lengths", "forward_body_lengths"),
        ("best_fit_rotation_range_degrees", "rotation_span_degrees"),
        ("desired_net_rotation_degrees", "desired_net_rotation_degrees"),
        ("desired_active_rotation_fraction", "desired_active_rotation_fraction"),
        ("torque_saturation_fraction_pre_step", "torque_saturation_fraction_pre_step"),
    ):
        row.update(sample_stats([episode[field] for episode in episodes], prefix))
    return row


def configuration_summary_rows(
    config: dict[str, Any], run_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for arm_id in config["arm_order"]:
        rows = sorted(
            [row for row in run_rows if row["configuration_id"] == arm_id],
            key=lambda row: int(row["formal_run"]),
        )
        if len(rows) != 5 or [int(row["formal_run"]) for row in rows] != list(range(5)):
            raise RuntimeError(f"Incomplete five-run configuration: {arm_id}")
        arm = config["arms"][arm_id]
        summary: dict[str, Any] = {
            "configuration_id": arm_id,
            "paper_label": arm["paper_label"],
            "independent_run_count": 5,
            "nested_rollout_count": 100,
            "runs_discovered_lenient_rolling": int(
                sum(bool(row["run_discovered_lenient_rolling"]) for row in rows)
            ),
            "lenient_episode_successes_nested": int(
                sum(int(row["lenient_success_count"]) for row in rows)
            ),
            "runs_meeting_secondary_strict_threshold": int(
                sum(bool(row["run_met_secondary_strict_threshold"]) for row in rows)
            ),
            "secondary_strict_episode_successes_nested": int(
                sum(int(row["secondary_strict_success_count"]) for row in rows)
            ),
        }
        for run_field, prefix in (
            ("scaled_progress_x100_mean", "run_mean_scaled_progress_x100"),
            ("forward_body_lengths_mean", "run_mean_forward_body_lengths"),
            ("rotation_span_degrees_mean", "run_mean_rotation_span_degrees"),
            ("desired_net_rotation_degrees_mean", "run_mean_desired_rotation_degrees"),
            (
                "desired_active_rotation_fraction_mean",
                "run_mean_desired_active_rotation_fraction",
            ),
            ("lenient_success_rate", "run_lenient_success_rate"),
            ("secondary_strict_success_rate", "run_secondary_strict_success_rate"),
        ):
            summary.update(sample_stats([row[run_field] for row in rows], prefix))
        result.append(summary)
    return result


def paired_comparison_rows(
    config: dict[str, Any], run_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key = {
        (str(row["configuration_id"]), int(row["formal_run"])): row for row in run_rows
    }
    outcomes = (
        ("scaled_progress_x100_mean", "run_mean_scaled_progress_x100"),
        ("forward_body_lengths_mean", "run_mean_forward_body_lengths"),
        ("rotation_span_degrees_mean", "run_mean_rotation_span_degrees"),
        ("lenient_success_rate", "run_lenient_success_rate"),
        ("secondary_strict_success_rate", "run_secondary_strict_success_rate"),
    )
    difference_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for comparison in config["pre_registered_local_comparisons"]:
        arm_a, arm_b = comparison["a"], comparison["b"]
        per_run: list[dict[str, Any]] = []
        for formal_run in range(5):
            row_a = by_key[(arm_a, formal_run)]
            row_b = by_key[(arm_b, formal_run)]
            row: dict[str, Any] = {
                "comparison_id": comparison["comparison_id"],
                "configuration_a": arm_a,
                "configuration_b": arm_b,
                "paper_label_a": config["arms"][arm_a]["paper_label"],
                "paper_label_b": config["arms"][arm_b]["paper_label"],
                "difference_definition": "B_minus_A",
                "formal_run": formal_run,
                "internal_training_seed": formal_run + 9201,
            }
            for field, name in outcomes:
                row[f"difference_{name}"] = float(row_b[field]) - float(row_a[field])
            per_run.append(row)
            difference_rows.append(row)
        for _, name in outcomes:
            differences = [float(row[f"difference_{name}"]) for row in per_run]
            summary = {
                "comparison_id": comparison["comparison_id"],
                "configuration_a": arm_a,
                "configuration_b": arm_b,
                "paper_label_a": config["arms"][arm_a]["paper_label"],
                "paper_label_b": config["arms"][arm_b]["paper_label"],
                "outcome": name,
                "difference_definition": "B_minus_A",
                "paired_independent_run_count": 5,
                **sample_stats(differences, "paired_difference"),
                "exact_two_sided_sign_flip_p": exact_two_sided_sign_flip_p(differences),
                **{
                    f"difference_formal_run_{index}": difference
                    for index, difference in enumerate(differences)
                },
            }
            summary_rows.append(summary)
    return difference_rows, summary_rows


def aggregate_and_validate(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    validate_config(config)
    source_records = source_identity(config)
    deps = Dependencies(config)
    run_span_self_tests(deps)
    paired_reset_gate = paired_reset_and_deterministic_action_gate(config, deps)
    np = deps.np
    task_records: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    identity_gate_count = 0
    for task in all_tasks(config):
        manifest_path, trace_path = task_paths(output_root, task)
        if not manifest_path.is_file() or not trace_path.is_file():
            raise RuntimeError(f"Incomplete evaluation task: {task.task_id}")
        payload = read_json(manifest_path)
        if payload.get("schema") != TASK_SCHEMA or payload.get("status") != "complete":
            raise RuntimeError(f"Invalid task manifest: {manifest_path}")
        verified = validate_existing_task(config_path, config, output_root, task)
        if verified is None:
            raise RuntimeError(f"Task unexpectedly absent during validation: {task.task_id}")
        expected_seeds = list(config["evaluation"]["reset_seeds"])
        actual_seeds = [int(row["reset_seed"]) for row in payload["episodes"]]
        if actual_seeds != expected_seeds:
            raise RuntimeError(f"Reset-seed order drift: {task.task_id}")
        if any(int(row["steps"]) != 1000 for row in payload["episodes"]):
            raise RuntimeError(f"Rollout-length drift: {task.task_id}")
        if payload["checkpoint"].get("unchanged") is not True:
            raise RuntimeError(f"Checkpoint unchanged flag failed: {task.task_id}")
        if payload["policy"].get("unchanged") is not True:
            raise RuntimeError(f"Policy unchanged flag failed: {task.task_id}")
        current_checkpoint_hash = sha256_file(checkpoint_path(config, task))
        if current_checkpoint_hash != payload["checkpoint"]["sha256_before_after"]:
            raise RuntimeError(f"Post-evaluation checkpoint hash drift: {task.task_id}")
        for row in payload["episodes"]:
            for field in FINITE_EPISODE_FIELDS:
                if not math.isfinite(float(row[field])):
                    raise RuntimeError(f"Non-finite final result {task.task_id}:{field}")
        if payload["official_identity_gate"] is not None:
            if payload["official_identity_gate"].get("passed") is not True:
                raise RuntimeError(f"Failed official identity gate: {task.task_id}")
            identity_gate_count += 1
        with np.load(trace_path, allow_pickle=False) as archive:
            if not np.array_equal(
                archive["reset_seeds"],
                np.asarray(config["evaluation"]["reset_seeds"], dtype=np.int64),
            ):
                raise RuntimeError(f"Trace reset-seed drift: {task.task_id}")
            expected_shapes = payload["trace_archive"]["array_shapes"]
            if set(archive.files) != set(expected_shapes):
                raise RuntimeError(f"Trace key drift: {task.task_id}")
            for key in archive.files:
                value = archive[key]
                if list(value.shape) != list(expected_shapes[key]):
                    raise RuntimeError(f"Trace shape drift: {task.task_id}:{key}")
                if key not in ("support_index", "ground_contact_strength"):
                    if not bool(np.isfinite(value).all()):
                        raise RuntimeError(f"Non-finite trace array: {task.task_id}:{key}")
        task_records.append(
            {
                **payload["task"],
                "manifest": str(manifest_path),
                "manifest_sha256": sha256_file(manifest_path),
                "trace": str(trace_path),
                "trace_sha256": sha256_file(trace_path),
                "lenient_success_count": payload["evaluation"]["lenient_success_count"],
                "secondary_strict_success_count": payload["evaluation"][
                    "secondary_strict_success_count"
                ],
            }
        )
        episode_rows.extend(episode_csv_rows(payload))
        run_rows.append(make_run_result(payload))
    if len(task_records) != 30 or len(episode_rows) != 600 or len(run_rows) != 30:
        raise RuntimeError("Aggregate is not the complete 6 x 5 x 20 matrix")
    episode_keys = {
        (
            str(row["configuration_id"]),
            int(row["formal_run"]),
            int(row["reset_seed"]),
        )
        for row in episode_rows
    }
    if len(episode_keys) != 600:
        raise RuntimeError("Duplicate configuration/run/reset episode identity")
    temporary_files = list(output_root.rglob("*.tmp"))
    if temporary_files:
        raise RuntimeError(f"Temporary evaluation artifacts remain: {temporary_files[0]}")
    episode_csv = output_root / "episode_results.csv"
    run_csv = output_root / "run_results.csv"
    config_csv = output_root / "configuration_summary.csv"
    differences_csv = output_root / "paired_run_differences.csv"
    comparisons_csv = output_root / "pairwise_local_comparisons.csv"
    atomic_csv(episode_csv, episode_rows)
    atomic_csv(run_csv, run_rows)
    config_rows = configuration_summary_rows(config, run_rows)
    atomic_csv(config_csv, config_rows)
    difference_rows, comparison_rows = paired_comparison_rows(config, run_rows)
    atomic_csv(differences_csv, difference_rows)
    atomic_csv(comparisons_csv, comparison_rows)
    result = {
        "schema": SCHEMA,
        "mode": "complete_evaluation",
        "status": "complete",
        "study_id": config["study_id"],
        "validated_at_utc": utc_now(),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "source_identity": source_records,
        "paired_reset_and_deterministic_action_gate": paired_reset_gate,
        "task_count": len(task_records),
        "independent_training_run_count": len(run_rows),
        "nested_rollout_count": len(episode_rows),
        "official_identity_gate_count": identity_gate_count,
        "complete_locked_matrix": True,
        "endpoint_checkpoint": 1500,
        "primary_lenient_rotation_span": config["evaluation"][
            "primary_lenient_rotation_span"
        ],
        "secondary_strict_common_kinematic": config["evaluation"][
            "secondary_strict_common_kinematic"
        ],
        "inference_unit": "independent training run; n=5 per configuration",
        "nested_repeat_unit": "20 paired reset rollouts within each frozen policy",
        "configuration_summary": config_rows,
        "pairwise_local_comparisons": comparison_rows,
        "task_records": task_records,
        "artifacts": {
            "episode_results_csv": str(episode_csv),
            "episode_results_csv_sha256": sha256_file(episode_csv),
            "run_results_csv": str(run_csv),
            "run_results_csv_sha256": sha256_file(run_csv),
            "configuration_summary_csv": str(config_csv),
            "configuration_summary_csv_sha256": sha256_file(config_csv),
            "paired_run_differences_csv": str(differences_csv),
            "paired_run_differences_csv_sha256": sha256_file(differences_csv),
            "pairwise_local_comparisons_csv": str(comparisons_csv),
            "pairwise_local_comparisons_csv_sha256": sha256_file(comparisons_csv),
        },
    }
    study_manifest = output_root / "STUDY_MANIFEST.json"
    atomic_json(study_manifest, result)
    validation = {
        "schema": VALIDATION_SCHEMA,
        "status": "PASS",
        "validated_at_utc": utc_now(),
        "checks": {
            "six_configurations": len(config_rows) == 6,
            "thirty_independent_runs": len(run_rows) == 30,
            "six_hundred_nested_rollouts": len(episode_rows) == 600,
            "all_rollouts_1000_steps": True,
            "exact_paired_reset_seed_panel": True,
            "all_endpoint_fields_finite": True,
            "all_checkpoint_hashes_unchanged": True,
            "all_policy_hashes_unchanged": True,
            "all_task_trace_hashes_valid": True,
            "all_trace_arrays_parse_and_match_declared_shapes": True,
            "no_duplicate_episode_identity": len(episode_keys) == 600,
            "no_temporary_artifacts": not temporary_files,
            "ten_archived_r0_rroll_identity_gates_passed": identity_gate_count == 10,
            "source_hashes_match_frozen_config": True,
            "paired_initial_states_identical_across_six_configurations": (
                paired_reset_gate["maximum_initial_position_absolute_error"] <= 1e-12
            ),
            "deterministic_actions_repeat_exactly": all(
                item["exact_repeat"]
                for item in paired_reset_gate["deterministic_action_checks"].values()
            ),
            "run_level_inference_preserved": True,
        },
        "failed_checks": [],
        "study_manifest": str(study_manifest),
        "study_manifest_sha256": sha256_file(study_manifest),
    }
    if not all(validation["checks"].values()):
        validation["status"] = "FAIL"
        validation["failed_checks"] = [
            key for key, passed in validation["checks"].items() if not passed
        ]
        atomic_json(output_root / "FINAL_VALIDATION.json", validation)
        raise RuntimeError(f"Final validation failed: {validation['failed_checks']}")
    atomic_json(output_root / "FINAL_VALIDATION.json", validation)
    return result


def update_progress(
    output_root: Path,
    config_path: Path,
    completed: int,
    total: int,
    last_result: dict[str, Any] | None,
    started_perf: float,
) -> None:
    atomic_json(
        output_root / "EVALUATION_PROGRESS.json",
        {
            "schema": SCHEMA,
            "status": "running" if completed < total else "tasks_complete",
            "updated_at_utc": utc_now(),
            "config_sha256": sha256_file(config_path),
            "completed_tasks": completed,
            "total_tasks": total,
            "remaining_tasks": total - completed,
            "last_result": last_result,
            "wall_seconds": time.perf_counter() - started_perf,
        },
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("evaluator_config.json")
    )
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--contract-only", action="store_true")
    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--process-smoke", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--validate-only", action="store_true")
    result.add_argument("--workers", type=int, default=1)
    return result


def main() -> None:
    args = parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    config_path = args.config.resolve()
    config = read_json(config_path)
    validate_config(config)
    output_root = Path(config["output_root"]).resolve()
    if args.self_test:
        deps = Dependencies(config)
        print(json.dumps(run_span_self_tests(deps), ensure_ascii=False, indent=2))
        return
    if args.process_smoke:
        smoke_arms = ("HPR_DTH_PS", "HPR_O2_JS")
        with ProcessPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(
                    process_isolation_smoke_worker,
                    [str(config_path)] * len(smoke_arms),
                    smoke_arms,
                )
            )
        print(json.dumps({"status": "passed", "workers": results}, ensure_ascii=False, indent=2))
        return
    if args.validate_only:
        study = aggregate_and_validate(config_path, output_root)
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "task_count": study["task_count"],
                    "nested_rollout_count": study["nested_rollout_count"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if not args.execute:
        result = contract_scan(config_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    workers = int(args.workers)
    if workers < 1 or workers > int(config["runtime"]["maximum_workers"]):
        raise ValueError(f"workers must be 1..{config['runtime']['maximum_workers']}")
    gate = contract_scan(config_path)
    if not gate["ready_for_complete_evaluation"]:
        raise RuntimeError("Formal matrix did not pass the complete contract scan")
    output_root.mkdir(parents=True, exist_ok=True)
    atomic_json(output_root / "CONTRACT_SCAN.json", gate)
    start_marker = output_root / "EVALUATION_STARTED.json"
    start_payload = {
        "schema": SCHEMA,
        "status": "started",
        "started_at_utc": utc_now(),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "evaluator": str(Path(__file__).resolve()),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "task_count": 30,
        "nested_rollout_count": 600,
        "workers": workers,
        "resume_policy": "only verified identical complete tasks",
    }
    if start_marker.exists():
        existing_start = read_json(start_marker)
        for key in (
            "schema",
            "config",
            "config_sha256",
            "evaluator",
            "evaluator_sha256",
            "task_count",
            "nested_rollout_count",
        ):
            if existing_start.get(key) != start_payload.get(key):
                raise RuntimeError(f"Incompatible existing evaluation start marker: {key}")
    else:
        atomic_json(start_marker, start_payload)
    tasks = all_tasks(config)
    started = time.perf_counter()
    update_progress(output_root, config_path, 0, len(tasks), None, started)
    if workers == 1:
        for index, task in enumerate(tasks, start=1):
            result = evaluate_task(str(config_path), str(output_root), asdict(task))
            update_progress(output_root, config_path, index, len(tasks), result, started)
            print(
                f"[{index:02d}/30] {result['task_id']}: "
                f"lenient={result['lenient_success_count']}/20, "
                f"strict={result['strict_success_count']}/20, {result['status']}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    evaluate_task, str(config_path), str(output_root), asdict(task)
                ): task
                for task in tasks
            }
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                update_progress(
                    output_root, config_path, completed, len(tasks), result, started
                )
                print(
                    f"[{completed:02d}/30] {result['task_id']}: "
                    f"lenient={result['lenient_success_count']}/20, "
                    f"strict={result['strict_success_count']}/20, {result['status']}",
                    flush=True,
                )
    study = aggregate_and_validate(config_path, output_root)
    complete_payload = {
        "schema": SCHEMA,
        "status": "complete",
        "completed_at_utc": utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "study_manifest": str(output_root / "STUDY_MANIFEST.json"),
        "study_manifest_sha256": sha256_file(output_root / "STUDY_MANIFEST.json"),
        "final_validation": str(output_root / "FINAL_VALIDATION.json"),
        "final_validation_sha256": sha256_file(output_root / "FINAL_VALIDATION.json"),
        "complete_locked_matrix": study["complete_locked_matrix"],
        "task_count": study["task_count"],
        "nested_rollout_count": study["nested_rollout_count"],
    }
    atomic_json(output_root / "EVALUATION_COMPLETE.json", complete_payload)
    update_progress(output_root, config_path, len(tasks), len(tasks), complete_payload, started)


if __name__ == "__main__":
    main()
