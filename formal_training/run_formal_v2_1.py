from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATE_SCHEMA = "roll_learning_v2_1_formal_state/v1"
PAIR_HASH_KEYS = (
    "actor_sha256",
    "critic_sha256",
    "optimizer_sha256",
    "torch_cpu_rng_sha256",
    "torch_cuda_rng_sha256",
    "numpy_rng_sha256",
    "python_rng_sha256",
)


class ContractFailure(RuntimeError):
    """A scientific-contract failure that must never be auto-retried."""


class TechnicalFailure(RuntimeError):
    """A technical failure eligible for the pre-registered whole-pair retry."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    # CloudStorage/Defender can briefly hold the destination immediately after a
    # sync notification on Windows.  Retrying the *same* completed temp file
    # preserves atomicity and never changes scientific content.
    for attempt in range(1, 41):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if os.name != "nt" or attempt == 40:
                raise
            time.sleep(min(0.05 * attempt, 0.5))


def append_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"time_utc": utc_now(), **event}
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_object_hash(value: Any) -> str:
    """Reproduce the frozen audit wrapper's typed object hash for JSON values."""
    digest = hashlib.sha256()

    def length(payload: bytes) -> None:
        digest.update(struct.pack(">Q", len(payload)))
        digest.update(payload)

    def update(item: Any) -> None:
        if isinstance(item, dict):
            digest.update(b"D")
            items = sorted(item.items(), key=lambda pair: repr(pair[0]))
            digest.update(struct.pack(">Q", len(items)))
            for key, child in items:
                update(key)
                update(child)
        elif isinstance(item, list):
            digest.update(b"L")
            digest.update(struct.pack(">Q", len(item)))
            for child in item:
                update(child)
        elif item is None:
            digest.update(b"N")
        elif isinstance(item, bool):
            digest.update(b"B1" if item else b"B0")
        elif isinstance(item, int):
            digest.update(b"I")
            length(str(item).encode("ascii"))
        elif isinstance(item, float):
            digest.update(b"F")
            digest.update(struct.pack(">d", item))
        elif isinstance(item, str):
            digest.update(b"U")
            length(item.encode("utf-8"))
        else:
            raise TypeError(f"Unsupported stable hash value: {type(item).__name__}")

    update(value)
    return digest.hexdigest()


def csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
        return max(0, sum(1 for _ in handle) - 1)


def latest_checkpoint(run_dir: Path) -> int:
    latest = 0
    if run_dir.is_dir():
        for checkpoint in run_dir.glob("checkpoint_*.pt"):
            try:
                latest = max(latest, int(checkpoint.stem.rsplit("_", 1)[-1]))
            except ValueError:
                continue
    return latest


def tail_text(path: Path, maximum_bytes: int = 16000) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        size = path.stat().st_size
        handle.seek(max(0, size - maximum_bytes))
        return handle.read().decode("utf-8", errors="replace")


def artifact_inventory(root: Path) -> list[dict[str, Any]]:
    try:
        if not root.exists():
            return []
        files = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
    except OSError as error:
        return [{"path": str(root), "inventory_error": f"{type(error).__name__}: {error}"}]
    result: list[dict[str, Any]] = []
    for path in files:
        try:
            result.append(
                {
                    "path": str(path),
                    "relative_path": path.name if root.is_file() else path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
        except OSError as error:
            result.append({"path": str(path), "inventory_error": f"{type(error).__name__}: {error}"})
    return result


class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def available_memory_gb() -> float | None:
    if os.name != "nt":
        return None
    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    try:
        success = ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    except (AttributeError, OSError):
        return None
    if not success:
        return None
    return float(status.ullAvailPhys / (1024**3))


class FormalSupervisor:
    def __init__(self, root: Path, config_path: Path):
        self.root = root.resolve()
        self.config_path = config_path.resolve()
        self.config = read_json(self.config_path)
        expected_root = Path(self.config["output_root"]).resolve()
        if self.root != expected_root:
            raise ContractFailure(f"Output root mismatch: {self.root} != {expected_root}")

        runtime = self.config["runtime"]
        self.python = Path(runtime["python"]).resolve()
        self.site_packages = Path(runtime["site_packages"]).resolve()
        self.code = (self.root / runtime["code_snapshot_relative"]).resolve()
        self.trainer = self.code / runtime["trainer_relative"]
        self.audit_wrapper = self.code / runtime["audit_wrapper_relative"]
        self.contract_test = self.code / runtime["contract_test_relative"]
        self.evaluator = self.code / runtime["evaluation_wrapper_relative"]
        self.state_path = self.root / "orchestrator_state.json"
        self.events_path = self.root / "events.jsonl"
        self.manifest_path = self.root / "_control" / "source_manifest.json"
        self.approval_path = self.root / self.config["formal_approval"]["required_marker_relative"]
        self.gate_path = self.root / self.config["prerequisite"]["regression_gate_relative"]
        self.regression_config_path = self.root / self.config["prerequisite"]["regression_config_relative"]
        self.regression_manifest_path = self.root / self.config["prerequisite"]["regression_source_manifest_relative"]
        self.protocol_path = self.root / "_control" / "FORMAL_PROTOCOL.md"
        self.results_by_case: dict[str, dict[str, Any]] = {}
        self.evaluation_results: list[dict[str, Any]] = []
        self.state: dict[str, Any] = {
            "schema": STATE_SCHEMA,
            "study_id": self.config["study_id"],
            "status": "initializing",
            "phase": "PRECHECK",
            "started_at_utc": utc_now(),
            "updated_at_utc": utc_now(),
            "orchestrator_pid": os.getpid(),
            "execution_mode": "paired_concurrent_training",
            "maximum_concurrent_training_processes": 2,
            "current_pair": None,
            "active_cases": {},
            "completed_cases": [],
            "evaluated_cases": [],
            "technical_retries": [],
            "error": None,
            "formal_result": None,
        }

    def write_state(self) -> None:
        self.state["updated_at_utc"] = utc_now()
        atomic_json(self.state_path, self.state)

    def event(self, name: str, **fields: Any) -> None:
        append_event(self.events_path, {"event": name, **fields})

    def environment(self, *, evaluation: bool = False) -> dict[str, str]:
        value = os.environ.copy()
        threads = str(int(self.config["resource_policy"]["thread_limit_per_process"]))
        value.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
                "PYTHONUTF8": "1",
                "PYTHONIOENCODING": "utf-8:strict",
                "PYGAME_HIDE_SUPPORT_PROMPT": "1",
                "MPLBACKEND": "Agg",
                "PYTHONPATH": str(self.site_packages),
                "OMP_NUM_THREADS": threads,
                "MKL_NUM_THREADS": threads,
                "OPENBLAS_NUM_THREADS": threads,
                "NUMEXPR_NUM_THREADS": threads,
                "CUDA_VISIBLE_DEVICES": "" if evaluation else str(self.config["resource_policy"]["cuda_visible_devices"]),
            }
        )
        return value

    def required_sources(self) -> dict[str, Path]:
        sources: dict[str, Path] = {
            "formal_config": self.config_path,
            "formal_runner": Path(__file__).resolve(),
            "formal_protocol": self.protocol_path,
            "formal_approval": self.approval_path,
            "regression_gate": self.gate_path,
            "regression_config": self.regression_config_path,
            "regression_source_manifest": self.regression_manifest_path,
            "trainer": self.trainer,
            "audit_wrapper": self.audit_wrapper,
            "reward_contract_test": self.contract_test,
            "frozen_evaluator": self.evaluator,
            "environment": self.code / "metamaterial_envs" / "metamaterial_envs" / "env" / "metamaterial.py",
        }
        for path in sorted(self.code.rglob("*")):
            if not path.is_file() or "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            sources[f"code_snapshot::{path.relative_to(self.code).as_posix()}"] = path
        return sources

    def freeze_sources(self) -> None:
        sources = self.required_sources()
        missing = [str(path) for path in sources.values() if not path.is_file()]
        if missing:
            raise ContractFailure(f"Missing frozen sources: {missing}")
        manifest = {
            "schema": "roll_learning_formal_source_manifest/v1",
            "study_id": self.config["study_id"],
            "created_at_utc": utc_now(),
            "files": {
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in sources.items()
            },
        }
        atomic_json(self.manifest_path, manifest)

    def verify_sources(self) -> None:
        manifest = read_json(self.manifest_path)
        for name, item in manifest["files"].items():
            path = Path(item["path"])
            actual = sha256_file(path)
            if actual != item["sha256"]:
                raise ContractFailure(f"Frozen source drift: {name}: {actual} != {item['sha256']}")

    def verify_static_contract(self) -> None:
        approval = read_json(self.approval_path)
        expected_statement = self.config["formal_approval"]["required_statement"]
        if approval.get("approved_statement") != expected_statement:
            raise ContractFailure("Formal approval marker is missing the required statement")
        if approval.get("study_id") != self.config["study_id"]:
            raise ContractFailure("Formal approval study_id mismatch")

        if sha256_file(self.gate_path) != self.config["prerequisite"]["regression_gate_sha256"]:
            raise ContractFailure("Approved regression gate hash mismatch")
        gate = read_json(self.gate_path)
        expected_counts = {
            str(key): int(value)
            for key, value in self.config["prerequisite"]["required_success_episodes_by_seed"].items()
        }
        actual_counts = {str(key): int(value) for key, value in gate.get("success_episodes_by_seed", {}).items()}
        if gate.get("passed") is not True or actual_counts != expected_counts:
            raise ContractFailure(f"Regression prerequisite changed: {actual_counts} != {expected_counts}")
        if gate.get("formal_started") is not False:
            raise ContractFailure("Regression gate does not certify formal_started=false")

        prerequisite = self.config["prerequisite"]
        if sha256_file(self.regression_config_path) != prerequisite["regression_config_sha256"]:
            raise ContractFailure("Frozen v2.1 regression config hash mismatch")
        if sha256_file(self.regression_manifest_path) != prerequisite["regression_source_manifest_sha256"]:
            raise ContractFailure("Frozen v2.1 source manifest hash mismatch")
        regression = read_json(self.regression_config_path)
        if self.config["training"] != regression["training"]:
            raise ContractFailure("Training/PPO contract differs from the passed v2.1 regression")
        if self.config["reward_schedule"] != regression["reward_schedule"]:
            raise ContractFailure("Reward schedule differs from the passed v2.1 regression")
        if self.config["from_scratch_contract"] != regression["from_scratch_contract"]:
            raise ContractFailure("From-scratch contract differs from the passed v2.1 regression")
        if self.config["episode_success"] != regression["episode_success"]:
            raise ContractFailure("Frozen endpoint criteria differ from the passed v2.1 regression")
        if regression["rewards"].get("V2_1") != self.config["rewards"].get("Rroll"):
            raise ContractFailure("Rroll is not the passed v2.1 reward")
        revision = regression.get("reward_revision", {})
        if revision.get("only_semantic_change") != "rolling-phase motion_quality coefficient 0.08 -> 0.16":
            raise ContractFailure("The approved v2.1 reward revision is not frozen")
        if float(revision.get("rolling_motion_quality_coefficient_v2_1", math.nan)) != 0.16:
            raise ContractFailure("The approved v2.1 motion-quality coefficient is not 0.16")

        reference_manifest = read_json(self.regression_manifest_path)
        reference_snapshot = {
            name.split("::", 1)[1]: item["sha256"]
            for name, item in reference_manifest["files"].items()
            if name.startswith("code_snapshot::")
        }
        current_snapshot = {
            path.relative_to(self.code).as_posix(): sha256_file(path)
            for path in self.code.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        }
        if current_snapshot != reference_snapshot:
            missing = sorted(set(reference_snapshot) - set(current_snapshot))
            extra = sorted(set(current_snapshot) - set(reference_snapshot))
            changed = sorted(
                key for key in set(current_snapshot) & set(reference_snapshot)
                if current_snapshot[key] != reference_snapshot[key]
            )
            raise ContractFailure(
                f"Formal code snapshot differs from passed v2.1: missing={missing}, extra={extra}, changed={changed}"
            )

        formal = self.config["formal"]
        seeds = [int(seed) for seed in formal["training_seeds"]]
        if seeds != [9201, 9202, 9203, 9204, 9205] or formal["execution_order"] != seeds:
            raise ContractFailure("Formal seed contract changed")
        if formal["arms"] != ["R0", "Rroll"] or int(formal["batches"]) != 1500:
            raise ContractFailure("Formal arm or endpoint contract changed")
        if int(formal["max_concurrent_processes"]) != 2 or formal["pair_seed_serial"] is not True:
            raise ContractFailure("Formal concurrency contract changed")
        if self.config["rewards"] != {"R0": "horizontal_speed", "Rroll": "obs2_roll_repro_v2_1"}:
            raise ContractFailure("Formal reward contract changed")

        controller = self.config["controller_contract"]
        expected_controller = {
            "robot": "crawler",
            "terrain": "flat",
            "terrain_contact_mode": "legacy_flat",
            "num_particles": 10,
            "observation_func": "dth_tot_plus_friction_thdot",
            "observation_shape_per_joint": 2,
            "action_names": ["k1", "k2"],
            "action_shape_per_joint": 2,
            "control_mode": "formula",
            "per_joint_k1_k2": True,
            "share_policy": False,
            "share_critic": True,
            "centralised_critic": True,
            "k_action_scale": 100.0,
            "max_torque": 9.0,
            "torque_formula": "clip(K1*dth_tot + K2*theta_dot, -9, 9)",
            "rolling_observation": False,
            "tail_roll_observation": False,
            "fast_forward_observation": False,
        }
        for key, expected in expected_controller.items():
            if controller.get(key) != expected:
                raise ContractFailure(f"Controller contract drift: {key}")

        scratch = self.config["from_scratch_contract"]
        forbidden_values = {
            "pretrained_model_path": None,
            "pretrained_policy_only": False,
            "resume_training_state": None,
            "bc_teacher_checkpoint": None,
            "wave_bc_teacher_json": None,
            "bc_steps": 0,
            "bc_epochs": 0,
            "policy_anchor_coeff": 0.0,
            "policy_anchor_anneal_batches": 0,
            "compatible_input_expansion": False,
            "pilot_or_historical_checkpoint_as_initialization": False,
        }
        for key, expected in forbidden_values.items():
            if scratch.get(key) != expected:
                raise ContractFailure(f"From-scratch contract drift: {key}")
        if list(self.code.rglob("*.pt")):
            raise ContractFailure("Checkpoint file found inside the formal code snapshot")

        expected_formal = {
            "evaluation_device": "cpu",
            "evaluation_concurrency": 1,
            "evaluation_base_seed": 20264101,
            "evaluation_episodes": 20,
            "evaluation_steps": 1000,
            "single_training_seed_success_min_episodes": 10,
            "condition_reproducible_min_training_seeds": 3,
            "condition_robust_min_training_seeds": 4,
        }
        for key, expected in expected_formal.items():
            if formal.get(key) != expected:
                raise ContractFailure(f"Formal endpoint contract drift: {key}")
        expected_retry = {
            "max_pair_attempts": 2,
            "retry_scope": "whole_pair",
            "restart_from_batch0": True,
            "preserve_failed_attempts": True,
            "retryable_abrupt_exit_after_valid_batch0": True,
            "retryable_trainer_error_prefixes": [
                "OutOfMemoryError:",
                "RuntimeError: CUDA error:",
                "OSError: [WinError 1455]",
            ],
            "evaluation_max_attempts": 2,
            "batch0_audit_timeout_seconds": 180,
            "training_log_stall_timeout_seconds": 900,
        }
        if self.config["technical_retry"] != expected_retry:
            raise ContractFailure("Technical retry allowlist or limits drifted")
        expected_resources = {
            "cuda_visible_devices": "0",
            "thread_limit_per_process": 8,
            "minimum_available_memory_gb_before_pair": 5.0,
            "parallel_training_processes": 2,
            "parallel_evaluation_processes": 1,
        }
        if self.config["resource_policy"] != expected_resources:
            raise ContractFailure("Formal resource/concurrency policy drifted")
        expected_failure_policy = {
            "fail_closed": True,
            "automatic_resume": False,
            "seed_replacement": False,
            "intermediate_checkpoint_selection": False,
            "extend_beyond_1500": False,
            "automatic_reward_change": False,
            "automatic_hyperparameter_change": False,
            "automatic_concurrency_change": False,
            "curve_only_early_stop": False,
            "historical_checkpoint_as_initialization": False,
        }
        if self.config["failure_policy"] != expected_failure_policy:
            raise ContractFailure("Formal fail-closed policy drifted")

    def write_launcher_receipt(self) -> None:
        atomic_json(
            self.root / "_control" / "launcher_receipt.json",
            {
                "schema": "roll_learning_v2_1_formal_launcher_receipt/v1",
                "study_id": self.config["study_id"],
                "created_at_utc": utc_now(),
                "root": str(self.root),
                "config": str(self.config_path),
                "orchestrator": str(Path(__file__).resolve()),
                "orchestrator_pid": os.getpid(),
                "stdout": str(self.root / "_control" / "orchestrator.stdout.log"),
                "stderr": str(self.root / "_control" / "orchestrator.stderr.log"),
                "formal_training_allowed": True,
                "approved_from_scratch": True,
                "maximum_concurrent_training_processes": 2,
                "whole_pair_technical_retry_limit": 1,
                "automatic_resume": False,
                "historical_checkpoint_loaded": False,
            },
        )

    def run_sync(
        self,
        command: list[str],
        stdout_path: Path,
        stderr_path: Path,
        label: str,
        *,
        evaluation: bool = False,
    ) -> int:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        self.event("process_started", label=label, command=command)
        creationflags = 0x08000000 if os.name == "nt" else 0
        process: subprocess.Popen[Any] | None = None
        try:
            with stdout_path.open("w", encoding="utf-8", newline="\n") as out, stderr_path.open(
                "w", encoding="utf-8", newline="\n"
            ) as err:
                process = subprocess.Popen(
                    command,
                    cwd=str(self.code),
                    env=self.environment(evaluation=evaluation),
                    stdout=out,
                    stderr=err,
                    text=True,
                    creationflags=creationflags,
                )
                self.state["auxiliary_process"] = {"label": label, "pid": process.pid}
                self.write_state()
                code = int(process.wait())
        except BaseException as error:
            if process is not None and process.poll() is None:
                self.terminate_process(process)
            self.state["auxiliary_process"] = None
            self.write_state()
            self.event("process_wait_failed", label=label, error=f"{type(error).__name__}: {error}")
            raise
        self.state["auxiliary_process"] = None
        self.write_state()
        self.event("process_finished", label=label, exit_code=code)
        return code

    def run_prechecks(self) -> None:
        logs = self.root / "_control" / "precheck_logs"
        code = self.run_sync(
            [
                str(self.python),
                "-c",
                (
                    "import torch; "
                    "assert torch.cuda.is_available(), 'CUDA unavailable'; "
                    "assert torch.cuda.device_count() >= 1, 'no CUDA device'; "
                    "x=torch.ones(1, device='cuda:0'); "
                    "assert float(x.item()) == 1.0; "
                    "print(torch.cuda.get_device_name(0))"
                ),
            ],
            logs / "cuda_preflight.stdout.log",
            logs / "cuda_preflight.stderr.log",
            "cuda_training_preflight",
        )
        if code != 0:
            raise ContractFailure("CUDA training preflight failed")
        code = self.run_sync(
            [str(self.python), str(self.evaluator), "--self-test"],
            logs / "evaluator_self_test.stdout.log",
            logs / "evaluator_self_test.stderr.log",
            "frozen_evaluator_self_test",
            evaluation=True,
        )
        if code != 0:
            raise ContractFailure("Frozen evaluator self-test failed")
        code = self.run_sync(
            [str(self.python), str(self.contract_test)],
            logs / "reward_contract_test.stdout.log",
            logs / "reward_contract_test.stderr.log",
            "reward_contract_test",
        )
        if code != 0:
            raise ContractFailure("Reward contract test failed")

    def base_trainer_args(self, reward: str, seed: int, run_name: str, results_dir: Path) -> list[str]:
        training = self.config["training"]
        schedule = self.config["reward_schedule"]
        return [
            "--robot", "crawler",
            "--terrain", "flat",
            "--terrain-contact-mode", "legacy_flat",
            "--num-particles", "10",
            "--channel", "action",
            "--observation-func", "dth_tot_plus_friction_thdot",
            "--control-mode", "formula",
            "--feedback-gain", "1.0",
            "--max-control-gain", "9.0",
            "--no-fix-k1",
            "--no-fix-k2",
            "--k-action-scale", "100.0",
            "--passive-kappa", "4.0",
            "--per-joint-k1-k2",
            "--no-share-policy",
            "--share-critic",
            "--centralised-critic",
            "--algorithm", "ppo",
            "--policy-depth", str(training["policy_depth"]),
            "--policy-cells", str(training["policy_cells"]),
            "--normal-scale-lb", "0.0001",
            "--episode-steps", str(training["episode_steps"]),
            "--frames-per-batch", str(training["frames_per_batch"]),
            "--memory-size", str(training["memory_size"]),
            "--minibatch-size", str(training["minibatch_size"]),
            "--optim-steps", str(training["optim_steps"]),
            "--lr", str(training["learning_rate"]),
            "--weight-decay", str(training["weight_decay"]),
            "--max-grad-norm", str(training["max_grad_norm"]),
            "--gamma", str(training["gamma"]),
            "--clip-epsilon", str(training["clip_epsilon"]),
            "--lambda-gae", str(training["lambda_gae"]),
            "--entropy-eps", str(training["entropy_epsilon"]),
            "--no-ppo-normalize-advantage",
            "--ppo-target-kl", "0.0",
            "--init-pos-randomness", str(training["init_pos_randomness"]),
            "--init-angle-range-degrees", str(training["init_angle_range_degrees"]),
            "--init-height-jitter", str(training["init_height_jitter"]),
            "--action-smoothness-weight", "0.0",
            "--policy-anchor-coeff", "0.0",
            "--policy-anchor-anneal-batches", "0",
            "--bc-steps", "0",
            "--bc-epochs", "0",
            "--rolling-direction", "right",
            "--rolling-curl-episodes", "300",
            "--rolling-transition-episodes", "300",
            "--rolling-reward-scale", "3.0",
            "--tail-side", "left",
            "--tail-roll-init-assist-degrees", "0.0",
            "--tail-roll-init-assist-episodes", "0",
            "--no-rolling-observation",
            "--no-tail-roll-observation",
            "--no-fast-forward-observation",
            "--fast-forward-event-degrees", str(schedule["pulse_rotation_degrees"]),
            "--fast-forward-event-forward-fraction", str(schedule["pulse_forward_body_fraction"]),
            "--fast-forward-event-contact-nodes", str(schedule["pulse_contact_nodes"]),
            "--fast-forward-direction-fraction", str(schedule["pulse_direction_fraction"]),
            "--fast-forward-event-target-steps", "250",
            "--fast-forward-launch-lift", str(schedule["launch_tail_lift_threshold"]),
            "--fast-forward-launch-forward", str(schedule["launch_tail_forward_threshold"]),
            "--fast-forward-launch-curl", str(schedule["launch_curl_prefix_threshold"]),
            "--fast-forward-launch-head-contact", str(schedule["launch_head_contact_threshold"]),
            "--fast-forward-launch-hold-steps", str(schedule["launch_hold_steps"]),
            "--fast-forward-stall-steps", "150",
            "--fast-forward-rotation-step-ref-degrees", "2.0",
            "--fast-forward-translation-step-ref", "0.002",
            "--no-pretrained-policy-only",
            "--no-compatible-input-expansion",
            "--buffer-storage", "tensor",
            "--no-auto-analysis",
            "--episodes", str(self.config["formal"]["batches"]),
            "--save-every", str(training["save_every"]),
            "--reward-func", reward,
            "--seed", str(seed),
            "--results-dir", str(results_dir),
            "--run-name", run_name,
        ]

    def case_spec(self, seed: int, arm: str, attempt: int) -> dict[str, Any]:
        canonical = f"formal__seed{seed}__{arm}"
        run_name = canonical if attempt == 1 else f"{canonical}__technical_retry{attempt}"
        results_dir = self.root / "formal" / "runs"
        run_dir = results_dir / run_name
        audit_path = self.root / "formal" / "initialization" / f"{run_name}.json"
        logs = self.root / "formal" / "logs"
        reward = self.config["rewards"][arm]
        command = [
            str(self.python),
            str(self.audit_wrapper),
            "--audit-output", str(audit_path),
            "--case-id", canonical,
            "--expected-reward", reward,
            "--trainer", str(self.trainer),
            *self.base_trainer_args(reward, seed, run_name, results_dir),
        ]
        return {
            "canonical_case_id": canonical,
            "run_name": run_name,
            "seed": seed,
            "arm": arm,
            "reward": reward,
            "attempt": attempt,
            "run_dir": run_dir,
            "audit_path": audit_path,
            "stdout_path": logs / f"{run_name}.stdout.log",
            "stderr_path": logs / f"{run_name}.stderr.log",
            "command": command,
        }

    @staticmethod
    def terminate_process(process: subprocess.Popen[Any]) -> None:
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)

    def validate_batch0_audit(
        self,
        path: Path,
        *,
        expected_seed: int,
        expected_reward: str,
        expected_case_id: str,
    ) -> dict[str, Any]:
        audit = read_json(path)
        if audit.get("schema") != "roll_learning_initialization_audit/v1":
            raise ContractFailure(f"Wrong initialization audit schema: {path}")
        if audit.get("case_id") != expected_case_id:
            raise ContractFailure(f"Wrong case identity in audit: {path}")
        if audit.get("seed") != expected_seed or audit.get("reward") != expected_reward:
            raise ContractFailure(f"Wrong seed/reward in audit: {path}")
        if audit.get("batch_index") != 0 or audit.get("from_scratch") is not True:
            raise ContractFailure(f"Audit is not from batch 0: {path}")
        # A wrapper may change an otherwise valid, immutable batch-0 capture to
        # ``failed`` after a later OOM/CUDA/process failure.  Accept that status
        # for pair-hash verification; endpoint acceptance below still requires
        # status=complete and trainer_exit_code=0.
        if audit.get("status") not in ("captured_before_training", "complete", "failed"):
            raise ContractFailure(f"Unexpected initialization audit status: {path}")
        runtime = audit.get("runtime_args", {})
        if runtime.get("contract_valid") is not True:
            raise ContractFailure(f"Runtime contract failed: {path}")
        expected_runtime = {
            "from_scratch": True,
            "reward_func": expected_reward,
            "seed": expected_seed,
            "episodes": int(self.config["formal"]["batches"]),
            "expected_num_envs": 10,
            "observation_func": "dth_tot_plus_friction_thdot",
            "control_mode": "formula",
            "share_policy": False,
            "per_joint_k1_k2": True,
            "share_critic": True,
            "centralised_critic": True,
            "rolling_observation": False,
            "tail_roll_observation": False,
            "fast_forward_observation": False,
        }
        for key, expected in expected_runtime.items():
            if runtime.get(key) != expected:
                raise ContractFailure(f"Runtime batch-0 contract drift in {path}: {key}")
        expected_protocol = {
            "fast_forward_launch_lift": 0.2,
            "fast_forward_launch_forward": 0.1,
            "fast_forward_launch_curl": 0.12,
            "fast_forward_launch_head_contact": 0.5,
            "fast_forward_launch_hold_steps": 8,
            "fast_forward_event_degrees": 60.0,
            "fast_forward_event_forward_fraction": 0.08,
            "fast_forward_event_contact_nodes": 1.5,
            "fast_forward_direction_fraction": 0.65,
        }
        if runtime.get("reward_protocol") != expected_protocol:
            raise ContractFailure(f"Reward protocol drift in audit: {path}")
        forbidden = runtime.get("forbidden_sources", {})
        expected_forbidden = {
            "pretrained_model_path": None,
            "pretrained_policy_only": False,
            "resume_training_state": None,
            "bc_teacher_checkpoint": None,
            "wave_bc_teacher_json": None,
            "bc_steps": 0,
            "bc_epochs": 0,
            "policy_anchor_coeff": 0.0,
            "policy_anchor_anneal_batches": 0,
        }
        for key, expected in expected_forbidden.items():
            if forbidden.get(key) != expected:
                raise ContractFailure(f"Forbidden source in audit {path}: {key}")
        environment = audit.get("environment", {})
        if environment.get("observation_shape") != [10, 8, 2] or environment.get("action_shape") != [10, 8, 2]:
            raise ContractFailure(f"Observation/action contract failed: {path}")
        expected_environment = {
            "formula_action_names": ["k1", "k2"],
            "rolling_observation_size": 0,
            "tail_roll_observation_size": 0,
            "fast_forward_observation_size": 0,
            "scratch_wr_v2_observation_size": 0,
            "control_mode": "formula",
            "k_action_scale": 100.0,
            "max_torque": 9.0,
            "expected_num_envs": 10,
        }
        for key, expected in expected_environment.items():
            if environment.get(key) != expected:
                raise ContractFailure(f"Environment contract drift in {path}: {key}")
        trainer = audit.get("trainer", {})
        if trainer.get("sha256") != sha256_file(self.trainer):
            raise ContractFailure(f"Audit used a different trainer source: {path}")
        bundle = audit.get("pair_hash_bundle")
        if not isinstance(bundle, dict) or set(bundle) != set(PAIR_HASH_KEYS):
            raise ContractFailure(f"Incomplete batch-0 hash bundle: {path}")
        scalar_keys = tuple(key for key in PAIR_HASH_KEYS if key != "torch_cuda_rng_sha256")
        if any(not isinstance(bundle[key], str) or re.fullmatch(r"[0-9a-f]{64}", bundle[key]) is None for key in scalar_keys):
            raise ContractFailure(f"Malformed scalar SHA-256 in batch-0 bundle: {path}")
        cuda_hashes = bundle["torch_cuda_rng_sha256"]
        if (
            not isinstance(cuda_hashes, list)
            or not cuda_hashes
            or any(not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None for value in cuda_hashes)
        ):
            raise ContractFailure(f"Malformed CUDA RNG hashes in batch-0 bundle: {path}")
        if audit.get("pair_hash_bundle_sha256") != stable_object_hash(bundle):
            raise ContractFailure(f"Batch-0 bundle digest mismatch: {path}")
        return audit

    def verify_completed_case(self, spec: dict[str, Any], return_code: int) -> dict[str, Any]:
        if return_code != 0:
            raise TechnicalFailure(f"Trainer exit code {return_code}: {spec['run_name']}")
        run_dir: Path = spec["run_dir"]
        batches = int(self.config["formal"]["batches"])
        checkpoint = run_dir / f"checkpoint_{batches}.pt"
        log_path = run_dir / "training_log.csv"
        metadata = run_dir / "metadata.json"
        training_summary_path = run_dir / "training_summary.json"
        audit_path: Path = spec["audit_path"]
        for path in (checkpoint, log_path, metadata, training_summary_path, audit_path):
            if not path.is_file() or path.stat().st_size <= 0:
                raise TechnicalFailure(f"Missing completed artifact: {path}")
        with log_path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != batches:
            raise TechnicalFailure(f"Training row count is not {batches}: {run_dir}")
        try:
            episodes = [int(row["episode"]) for row in rows]
            if episodes != list(range(1, batches + 1)):
                raise ContractFailure(f"Training episodes are not continuous 1..{batches}: {run_dir}")
            for row in rows:
                if row.get("algorithm") != "ppo" or row.get("robot") != "crawler" or row.get("terrain") != "flat":
                    raise ContractFailure(f"Training log scientific identity drift: {run_dir}")
                if not math.isfinite(float(row["reward_mean"])) or not math.isfinite(float(row["speed_mean"])):
                    raise TechnicalFailure(f"Non-finite primary training metric: {run_dir}")
        except (KeyError, TypeError, ValueError) as error:
            raise ContractFailure(f"Malformed training log: {run_dir}: {error}") from error
        checkpoint_numbers = sorted(
            int(path.stem.rsplit("_", 1)[-1])
            for path in run_dir.glob("checkpoint_*.pt")
            if path.stem.rsplit("_", 1)[-1].isdigit()
        )
        if any(number > batches for number in checkpoint_numbers):
            raise ContractFailure(f"Checkpoint beyond fixed endpoint {batches}: {run_dir}")
        expected_checkpoints = list(range(int(self.config["training"]["save_every"]), batches + 1, int(self.config["training"]["save_every"])))
        if checkpoint_numbers != expected_checkpoints:
            raise TechnicalFailure(f"Incomplete checkpoint inventory: {run_dir}: {checkpoint_numbers}")

        metadata_payload = read_json(metadata)
        training_args = metadata_payload.get("training_args", {})
        expected_metadata = {
            "robot": "crawler",
            "channel": "action",
            "observation_func": "dth_tot_plus_friction_thdot",
            "control_mode": "formula",
            "reward_func": spec["reward"],
            "per_joint_k1_k2": True,
            "share_policy": False,
            "rolling_observation": False,
            "tail_roll_observation": False,
            "fast_forward_observation": False,
            "compatible_input_expansion": False,
            "pretrained_policy_only": False,
            "source_checkpoint": None,
            "scratch_wr_resumed_from": None,
        }
        for key, expected in expected_metadata.items():
            if metadata_payload.get(key) != expected:
                raise ContractFailure(f"Completed metadata drift in {run_dir}: {key}")
        expected_args = {
            "seed": int(spec["seed"]),
            "reward_func": spec["reward"],
            "run_name": spec["run_name"],
            "episodes": batches,
            "episode_steps": int(self.config["training"]["episode_steps"]),
            "save_every": int(self.config["training"]["save_every"]),
            "resume_training_state": None,
            "pretrained_model_path": None,
            "pretrained_policy_only": False,
            "compatible_input_expansion": False,
            "bc_teacher_checkpoint": None,
            "wave_bc_teacher_json": None,
            "bc_steps": 0,
            "bc_epochs": 0,
            "share_policy": False,
            "per_joint_k1_k2": True,
            "share_critic": True,
            "centralised_critic": True,
        }
        for key, expected in expected_args.items():
            if training_args.get(key) != expected:
                raise ContractFailure(f"Completed training args drift in {run_dir}: {key}")

        training_summary = read_json(training_summary_path)
        if training_summary.get("status") != "complete" or int(training_summary.get("episodes", -1)) != batches:
            raise TechnicalFailure(f"Training summary is not complete at {batches}: {run_dir}")
        if Path(training_summary.get("final_checkpoint", "")).resolve() != checkpoint.resolve():
            raise ContractFailure(f"Training summary selected a different endpoint: {run_dir}")
        if Path(training_summary.get("save_dir", "")).resolve() != run_dir.resolve():
            raise ContractFailure(f"Training summary points to a different run directory: {run_dir}")
        if not math.isfinite(float(training_summary["final_reward_mean"])) or not math.isfinite(float(training_summary["final_speed_mean"])):
            raise TechnicalFailure(f"Training summary has non-finite primary metrics: {run_dir}")
        audit = self.validate_batch0_audit(
            audit_path,
            expected_seed=int(spec["seed"]),
            expected_reward=str(spec["reward"]),
            expected_case_id=str(spec["canonical_case_id"]),
        )
        if audit.get("status") != "complete":
            raise TechnicalFailure(f"Final audit status is not complete: {audit_path}")
        if audit.get("trainer_exit_code") != 0 or audit.get("trainer_error") is not None:
            raise ContractFailure(f"Final audit reports a trainer failure: {audit_path}")
        receipt = {
            "schema": "roll_learning_formal_case_receipt/v1",
            "study_id": self.config["study_id"],
            "case_id": spec["canonical_case_id"],
            "run_name": spec["run_name"],
            "seed": int(spec["seed"]),
            "arm": spec["arm"],
            "reward": spec["reward"],
            "attempt": int(spec["attempt"]),
            "batches": batches,
            "run_dir": str(run_dir),
            "audit": str(audit_path),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "batch0_pair_hash_bundle": audit["pair_hash_bundle"],
            "completed_at_utc": utc_now(),
            "from_scratch": True,
            "historical_checkpoint_loaded": False,
        }
        attempt_receipt = self.root / "formal" / "receipts" / "attempts" / f"{spec['run_name']}.json"
        receipt["attempt_receipt"] = str(attempt_receipt)
        atomic_json(attempt_receipt, receipt)
        return receipt

    def wait_for_resources(self, seed: int) -> None:
        minimum = float(self.config["resource_policy"]["minimum_available_memory_gb_before_pair"])
        while True:
            available = available_memory_gb()
            if available is None or available >= minimum:
                return
            self.state["phase"] = "WAITING_RESOURCES"
            self.state["resource_wait"] = {
                "seed": seed,
                "available_memory_gb": available,
                "minimum_required_gb": minimum,
                "checked_at_utc": utc_now(),
            }
            self.write_state()
            self.event(
                "resource_wait",
                seed=seed,
                available_memory_gb=available,
                minimum_required_gb=minimum,
            )
            time.sleep(30)

    def classify_trainer_exit(
        self,
        *,
        seed: int,
        pair_verified: bool,
        nonzero: dict[str, int],
        specs: dict[str, dict[str, Any]],
    ) -> BaseException:
        excerpts = {arm: tail_text(specs[arm]["stderr_path"]) for arm in nonzero}
        prefixes = tuple(str(value) for value in self.config["technical_retry"]["retryable_trainer_error_prefixes"])
        known_contract_errors = (
            "Traceback (most recent call last)",
            "AssertionError",
            "ValueError:",
            "TypeError:",
            "KeyError:",
            "ModuleNotFoundError:",
            "ImportError:",
            "unrecognized arguments:",
            "error: argument",
        )
        resource_error = {
            arm: any(prefix in text for prefix in prefixes) for arm, text in excerpts.items()
        }
        contract_error = {
            arm: any(marker in text for marker in known_contract_errors) for arm, text in excerpts.items()
        }
        nonretryable_contract_arms = [
            arm for arm in nonzero if contract_error[arm] and not resource_error[arm]
        ]
        if nonretryable_contract_arms:
            compact = {arm: excerpts[arm][-2000:] for arm in nonretryable_contract_arms}
            return ContractFailure(
                f"Non-retryable trainer/code/contract exit for seed {seed}: "
                f"arms={nonretryable_contract_arms}, exits={nonzero}, stderr={compact}"
            )
        abrupt_allowed = (
            pair_verified
            and self.config["technical_retry"].get("retryable_abrupt_exit_after_valid_batch0") is True
        )
        eligible = {
            arm: resource_error[arm] or (abrupt_allowed and not contract_error[arm])
            for arm in nonzero
        }
        if all(eligible.values()):
            return TechnicalFailure(
                f"Retryable resource/runtime or post-batch0 abrupt exit for seed {seed}: "
                f"exits={nonzero}, resource_error={resource_error}"
            )
        compact = {
            arm: text[-2000:] if text else "<empty stderr>"
            for arm, text in excerpts.items()
        }
        return ContractFailure(
            f"Non-retryable trainer/code/contract exit for seed {seed}: exits={nonzero}, stderr={compact}"
        )

    def run_pair_attempt(self, seed: int, attempt: int) -> tuple[dict[str, Any], dict[str, Any]]:
        specs = {arm: self.case_spec(seed, arm, attempt) for arm in ("R0", "Rroll")}
        for spec in specs.values():
            if spec["run_dir"].exists() or spec["audit_path"].exists():
                raise ContractFailure(f"Refusing to reuse formal output: {spec['run_name']}")
        self.verify_sources()
        self.state["phase"] = "FORMAL_TRAIN"
        self.state["status"] = "running"
        self.state["current_pair"] = {"seed": seed, "attempt": attempt, "started_at_utc": utc_now()}
        self.state["active_cases"] = {}
        self.write_state()
        self.event("formal_pair_started", seed=seed, attempt=attempt, arms=["R0", "Rroll"])

        creationflags = 0x08000000 if os.name == "nt" else 0
        processes: dict[str, subprocess.Popen[Any]] = {}
        handles: dict[str, tuple[Any, Any]] = {}
        last_rows = {"R0": 0, "Rroll": 0}
        last_progress = {"R0": time.monotonic(), "Rroll": time.monotonic()}
        started = time.monotonic()
        pair_verified = False
        failure: BaseException | None = None

        try:
            for arm in ("R0", "Rroll"):
                spec = specs[arm]
                spec["stdout_path"].parent.mkdir(parents=True, exist_ok=True)
                out = spec["stdout_path"].open("w", encoding="utf-8", newline="\n")
                err = spec["stderr_path"].open("w", encoding="utf-8", newline="\n")
                handles[arm] = (out, err)
                process = subprocess.Popen(
                    spec["command"],
                    cwd=str(self.code),
                    env=self.environment(evaluation=False),
                    stdout=out,
                    stderr=err,
                    text=True,
                    creationflags=creationflags,
                )
                processes[arm] = process
                self.state["active_cases"][spec["canonical_case_id"]] = {
                    "seed": seed,
                    "arm": arm,
                    "reward": spec["reward"],
                    "attempt": attempt,
                    "run_name": spec["run_name"],
                    "run_dir": str(spec["run_dir"]),
                    "pid": process.pid,
                    "alive": True,
                    "progress_rows": 0,
                    "latest_checkpoint": 0,
                    "batch0_pair_verified": False,
                    "started_at_utc": utc_now(),
                }
            self.write_state()
            self.event(
                "formal_pair_processes_started",
                seed=seed,
                attempt=attempt,
                pids={arm: process.pid for arm, process in processes.items()},
            )

            while True:
                for arm, process in processes.items():
                    spec = specs[arm]
                    rows = csv_rows(spec["run_dir"] / "training_log.csv")
                    if rows > last_rows[arm]:
                        last_rows[arm] = rows
                        last_progress[arm] = time.monotonic()
                    case = self.state["active_cases"][spec["canonical_case_id"]]
                    case["progress_rows"] = rows
                    case["latest_checkpoint"] = latest_checkpoint(spec["run_dir"])
                    case["alive"] = process.poll() is None
                    case["last_progress_utc"] = utc_now()

                self.write_state()
                codes = {arm: process.poll() for arm, process in processes.items()}
                nonzero = {arm: code for arm, code in codes.items() if code not in (None, 0)}
                if nonzero:
                    valid_capture_before_exit = pair_verified
                    audit_paths_exist = all(
                        specs[arm]["audit_path"].is_file() for arm in ("R0", "Rroll")
                    )
                    if not valid_capture_before_exit and audit_paths_exist:
                        raw_audits = {
                            arm: read_json(specs[arm]["audit_path"]) for arm in ("R0", "Rroll")
                        }
                        # Minimal failed audits created before build_components do
                        # not contain a bundle and must not be treated as a valid
                        # paired initialization.  A complete captured bundle is
                        # fully validated even when a later failure changed its
                        # status to ``failed``.
                        if all(isinstance(raw_audits[arm].get("pair_hash_bundle"), dict) for arm in raw_audits):
                            exit_audits = {
                                arm: self.validate_batch0_audit(
                                    specs[arm]["audit_path"],
                                    expected_seed=seed,
                                    expected_reward=specs[arm]["reward"],
                                    expected_case_id=specs[arm]["canonical_case_id"],
                                )
                                for arm in ("R0", "Rroll")
                            }
                            if exit_audits["R0"]["pair_hash_bundle"] != exit_audits["Rroll"]["pair_hash_bundle"]:
                                raise ContractFailure(f"Batch-0 pair hash mismatch for failed seed {seed}")
                            valid_capture_before_exit = True
                    raise self.classify_trainer_exit(
                        seed=seed,
                        pair_verified=valid_capture_before_exit,
                        nonzero={arm: int(code) for arm, code in nonzero.items()},
                        specs=specs,
                    )

                if not pair_verified and all(specs[arm]["audit_path"].is_file() for arm in ("R0", "Rroll")):
                    audits = {
                        arm: self.validate_batch0_audit(
                            specs[arm]["audit_path"],
                            expected_seed=seed,
                            expected_reward=specs[arm]["reward"],
                            expected_case_id=specs[arm]["canonical_case_id"],
                        )
                        for arm in ("R0", "Rroll")
                    }
                    bundles = {arm: audits[arm]["pair_hash_bundle"] for arm in audits}
                    if bundles["R0"] != bundles["Rroll"]:
                        raise ContractFailure(f"Batch-0 pair hash mismatch for seed {seed}")
                    pair_verified = True
                    for case in self.state["active_cases"].values():
                        case["batch0_pair_verified"] = True
                        case["pair_hash_bundle"] = bundles["R0"]
                    self.write_state()
                    receipt = {
                        "schema": "roll_learning_formal_pair_initialization/v1",
                        "study_id": self.config["study_id"],
                        "seed": seed,
                        "attempt": attempt,
                        "arms": ["R0", "Rroll"],
                        "pair_hash_bundle": bundles["R0"],
                        "identical_batch0": True,
                        "verified_at_utc": utc_now(),
                    }
                    atomic_json(
                        self.root / "formal" / "receipts" / f"pair_seed{seed}_attempt{attempt}_batch0.json",
                        receipt,
                    )
                    self.event("formal_pair_batch0_verified", seed=seed, attempt=attempt)
                if all(code is not None for code in codes.values()):
                    break
                if not pair_verified and time.monotonic() - started > float(
                    self.config["technical_retry"]["batch0_audit_timeout_seconds"]
                ):
                    raise TechnicalFailure(f"Timed out waiting for batch-0 audits: seed {seed}")
                stall_timeout = float(self.config["technical_retry"]["training_log_stall_timeout_seconds"])
                stalled = [
                    arm
                    for arm, process in processes.items()
                    if process.poll() is None and time.monotonic() - last_progress[arm] > stall_timeout
                ]
                if stalled:
                    raise TechnicalFailure(f"Training log stalled for seed {seed}: {stalled}")
                time.sleep(15)

            if not pair_verified:
                raise ContractFailure(f"Pair completed without batch-0 verification: seed {seed}")
            receipts = {
                arm: self.verify_completed_case(specs[arm], int(processes[arm].returncode))
                for arm in ("R0", "Rroll")
            }
            pair_receipt = {
                "schema": "roll_learning_formal_pair_receipt/v1",
                "study_id": self.config["study_id"],
                "seed": seed,
                "attempt": attempt,
                "identical_batch0": True,
                "pair_hash_bundle": receipts["R0"]["batch0_pair_hash_bundle"],
                "case_receipts": {arm: str(self.root / "formal" / "receipts" / f"formal__seed{seed}__{arm}.json") for arm in receipts},
                "completed_at_utc": utc_now(),
            }
            if receipts["R0"]["batch0_pair_hash_bundle"] != receipts["Rroll"]["batch0_pair_hash_bundle"]:
                raise ContractFailure(f"Final pair hash mismatch for seed {seed}")
            for arm, receipt in receipts.items():
                canonical_receipt = {
                    **receipt,
                    "selected_attempt": attempt,
                    "selected_after_pair_validation": True,
                }
                atomic_json(
                    self.root / "formal" / "receipts" / f"{specs[arm]['canonical_case_id']}.json",
                    canonical_receipt,
                )
                receipts[arm] = canonical_receipt
            atomic_json(self.root / "formal" / "receipts" / f"pair_seed{seed}.json", pair_receipt)
            self.event("formal_pair_succeeded", seed=seed, attempt=attempt)
            return receipts["R0"], receipts["Rroll"]
        except OSError as error:
            failure = TechnicalFailure(f"Unable to spawn paired trainer process for seed {seed}: {error}")
            raise failure from error
        except BaseException as error:
            failure = error
            raise
        finally:
            for process in processes.values():
                if process.poll() is None:
                    self.terminate_process(process)
            for arm, process in processes.items():
                case_id = specs[arm]["canonical_case_id"]
                if case_id in self.state["active_cases"]:
                    self.state["active_cases"][case_id]["alive"] = False
                    self.state["active_cases"][case_id]["exit_code"] = process.poll()
            for out, err in handles.values():
                out.close()
                err.close()
            if failure is not None:
                evidence_path = (
                    self.root / "formal" / "attempt_failures" / f"pair_seed{seed}_attempt{attempt}.json"
                )
                arms_evidence: dict[str, Any] = {}
                for arm, spec in specs.items():
                    process = processes.get(arm)
                    arms_evidence[arm] = {
                        "run_name": spec["run_name"],
                        "reward": spec["reward"],
                        "command": spec["command"],
                        "pid": process.pid if process is not None else None,
                        "exit_code": process.poll() if process is not None else None,
                        "run_dir": str(spec["run_dir"]),
                        "audit_path": str(spec["audit_path"]),
                        "stdout_path": str(spec["stdout_path"]),
                        "stderr_path": str(spec["stderr_path"]),
                        "run_artifacts": artifact_inventory(spec["run_dir"]),
                        "audit_artifact": artifact_inventory(spec["audit_path"]),
                        "stdout_artifact": artifact_inventory(spec["stdout_path"]),
                        "stderr_artifact": artifact_inventory(spec["stderr_path"]),
                    }
                atomic_json(
                    evidence_path,
                    {
                        "schema": "roll_learning_formal_pair_attempt_failure/v1",
                        "study_id": self.config["study_id"],
                        "seed": seed,
                        "attempt": attempt,
                        "classification": type(failure).__name__,
                        "error": f"{type(failure).__name__}: {failure}",
                        "failed_attempt_preserved": True,
                        "resume_checkpoint_used": False,
                        "captured_at_utc": utc_now(),
                        "arms": arms_evidence,
                    },
                )
                self.event(
                    "formal_pair_attempt_failed",
                    seed=seed,
                    attempt=attempt,
                    error=f"{type(failure).__name__}: {failure}",
                    pids={arm: process.pid for arm, process in processes.items()},
                    evidence_receipt=str(evidence_path),
                )
            self.write_state()

    def run_pair(self, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        attempts = int(self.config["technical_retry"]["max_pair_attempts"])
        for attempt in range(1, attempts + 1):
            self.wait_for_resources(seed)
            try:
                receipts = self.run_pair_attempt(seed, attempt)
                self.state["active_cases"] = {}
                self.state["current_pair"] = None
                for receipt in receipts:
                    self.results_by_case[receipt["case_id"]] = receipt
                    self.state["completed_cases"].append(receipt)
                self.write_state()
                return receipts
            except ContractFailure:
                raise
            except TechnicalFailure as error:
                decision = {
                    "schema": "roll_learning_formal_technical_retry/v1",
                    "study_id": self.config["study_id"],
                    "seed": seed,
                    "failed_attempt": attempt,
                    "error": f"{type(error).__name__}: {error}",
                    "failed_attempt_preserved": True,
                    "attempt_failure_receipt": str(
                        self.root / "formal" / "attempt_failures" / f"pair_seed{seed}_attempt{attempt}.json"
                    ),
                    "resume_checkpoint_used": False,
                    "whole_pair_restart_from_batch0": attempt < attempts,
                    "next_attempt": attempt + 1 if attempt < attempts else None,
                    "decided_at_utc": utc_now(),
                }
                atomic_json(
                    self.root / "formal" / "technical_retries" / f"pair_seed{seed}_attempt{attempt}.json",
                    decision,
                )
                self.state["technical_retries"].append(decision)
                self.write_state()
                self.event("technical_retry_decision", **{key: value for key, value in decision.items() if key != "schema"})
                if attempt >= attempts:
                    raise
        raise AssertionError("unreachable")

    def episode_success(self, episode: dict[str, Any]) -> bool:
        criteria = self.config["episode_success"]
        interval = episode.get("mean_roll_pulse_interval_steps")
        return bool(
            int(episode["roll_pulse_count"]) >= int(criteria["minimum_roll_pulses"])
            and float(episode["desired_net_rotation_degrees"])
            >= float(criteria["minimum_desired_net_rotation_degrees"])
            and float(episode["desired_active_rotation_fraction"])
            >= float(criteria["minimum_direction_fraction"])
            and float(episode["forward_body_lengths"])
            >= float(criteria["minimum_forward_body_lengths"])
            and interval is not None
            and float(interval) <= float(criteria["maximum_mean_inter_pulse_interval_steps"])
        )

    def evaluate_case(self, receipt: dict[str, Any]) -> dict[str, Any]:
        formal = self.config["formal"]
        case_id = receipt["case_id"]
        attempts = int(self.config["technical_retry"]["evaluation_max_attempts"])
        checkpoint = Path(receipt["checkpoint"])
        if sha256_file(checkpoint) != receipt["checkpoint_sha256"]:
            raise ContractFailure(f"Checkpoint changed before frozen evaluation: {checkpoint}")
        for attempt in range(1, attempts + 1):
            if sha256_file(checkpoint) != receipt["checkpoint_sha256"]:
                raise ContractFailure(f"Checkpoint changed before evaluation attempt {attempt}: {checkpoint}")
            output = self.root / "formal" / "evaluations" / f"{case_id}__eval_attempt{attempt}.json"
            logs = self.root / "formal" / "logs"
            command = [
                str(self.python),
                str(self.evaluator),
                "--repo", str(self.code),
                "--checkpoint", str(checkpoint),
                "--output", str(output),
                "--episodes", str(formal["evaluation_episodes"]),
                "--steps", str(formal["evaluation_steps"]),
                "--seed", str(formal["evaluation_base_seed"]),
                "--terrain", "flat",
                "--direction", "right",
                "--tail-side", "left",
                "--pulse-rotation-degrees", "60.0",
                "--pulse-forward-body-fraction", "0.08",
                "--pulse-contact-index-fraction", "0.20",
                "--quiet",
            ]
            self.state["phase"] = "FORMAL_EVAL"
            self.state["current_evaluation"] = {
                "case_id": case_id,
                "seed": receipt["seed"],
                "arm": receipt["arm"],
                "attempt": attempt,
                "checkpoint": str(checkpoint),
            }
            self.write_state()
            stdout_path = logs / f"{case_id}.eval_attempt{attempt}.stdout.log"
            stderr_path = logs / f"{case_id}.eval_attempt{attempt}.stderr.log"
            try:
                code = self.run_sync(
                    command,
                    stdout_path,
                    stderr_path,
                    f"evaluate_{case_id}_attempt{attempt}",
                    evaluation=True,
                )
            except OSError as error:
                self.event(
                    "formal_evaluation_spawn_failed",
                    case_id=case_id,
                    attempt=attempt,
                    error=f"{type(error).__name__}: {error}",
                    immutable_checkpoint=True,
                )
                continue
            if code != 0:
                stderr = tail_text(stderr_path)
                prefixes = tuple(
                    str(value) for value in self.config["technical_retry"]["retryable_trainer_error_prefixes"]
                )
                retryable_resource_error = any(prefix in stderr for prefix in prefixes)
                deterministic_error_markers = (
                    "Traceback (most recent call last)",
                    "AssertionError",
                    "ValueError:",
                    "TypeError:",
                    "KeyError:",
                    "ModuleNotFoundError:",
                    "ImportError:",
                    "metadata",
                    "terrain_contact_mode",
                    "unrecognized arguments:",
                )
                deterministic_error = any(marker in stderr for marker in deterministic_error_markers)
                if deterministic_error and not retryable_resource_error:
                    raise ContractFailure(
                        f"Non-retryable frozen evaluator/code/metadata failure for {case_id}: {stderr[-4000:]}"
                    )
                self.event(
                    "formal_evaluation_technical_exit",
                    case_id=case_id,
                    attempt=attempt,
                    exit_code=code,
                    retryable_resource_error=retryable_resource_error,
                    abrupt_exit_without_contract_trace=not deterministic_error,
                    immutable_checkpoint=True,
                )
                continue
            if code == 0 and output.is_file() and output.stat().st_size > 0:
                if sha256_file(checkpoint) != receipt["checkpoint_sha256"]:
                    raise ContractFailure(f"Checkpoint changed during frozen evaluation: {checkpoint}")
                try:
                    payload = read_json(output)
                    method = payload["method"]
                    results = payload["results"]
                    if not isinstance(results, list) or len(results) != 1:
                        raise ValueError("frozen evaluator must return exactly one result")
                    result = results[0]
                    expected_method = {
                        "schema": "fast_forward_eval/v1",
                        "name": "fast_forward_roll_v2",
                        "policy_mode": "deterministic",
                        "terrain": "flat",
                        "steps_per_episode": int(formal["evaluation_steps"]),
                        "episodes_per_checkpoint": int(formal["evaluation_episodes"]),
                        "base_seed": int(formal["evaluation_base_seed"]),
                    }
                    for key, expected in expected_method.items():
                        if method.get(key) != expected:
                            raise ValueError(f"frozen evaluator method identity changed: {key}")
                    expected_result = {
                        "checkpoint_name": checkpoint.name,
                        "run_name": receipt["run_name"],
                        "reward_func": receipt["reward"],
                        "channel": "action",
                        "control_mode": "formula",
                        "terrain_contact_mode": "legacy_flat",
                        "direction": "right",
                        "tail_side": "left",
                    }
                    for key, expected in expected_result.items():
                        if result.get(key) != expected:
                            raise ValueError(f"frozen evaluator result identity changed: {key}")
                    if Path(result["checkpoint"]).resolve() != checkpoint.resolve():
                        raise ValueError("frozen evaluator evaluated a different checkpoint")
                    episodes = result["episodes"]
                    if not isinstance(episodes, list) or len(episodes) != int(formal["evaluation_episodes"]):
                        raise ValueError("frozen evaluator returned the wrong episode count")
                    expected_seeds = [
                        int(formal["evaluation_base_seed"]) + index
                        for index in range(int(formal["evaluation_episodes"]))
                    ]
                    actual_seeds = [int(episode["seed"]) for episode in episodes]
                    if actual_seeds != expected_seeds:
                        raise ValueError(f"frozen evaluator seeds changed: {actual_seeds}")
                    if any(int(episode["steps"]) != int(formal["evaluation_steps"]) for episode in episodes):
                        raise ValueError("frozen evaluator returned the wrong step count")
                    success = [self.episode_success(episode) for episode in episodes]
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                    raise ContractFailure(
                        f"Invalid frozen evaluation output for {case_id}: {type(error).__name__}: {error}"
                    ) from error
                summary = {
                    "stage": "formal",
                    "seed": int(receipt["seed"]),
                    "arm": receipt["arm"],
                    "reward": receipt["reward"],
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": receipt["checkpoint_sha256"],
                    "evaluation_output": str(output),
                    "evaluation_attempt": attempt,
                    "evaluation_device": "cpu",
                    "evaluation_base_seed": int(formal["evaluation_base_seed"]),
                    "success_episodes": int(sum(success)),
                    "evaluation_episodes": len(success),
                    "success_rate": float(sum(success) / len(success)),
                    "episode_success": success,
                }
                self.state["current_evaluation"] = None
                self.state["evaluated_cases"].append(summary)
                self.write_state()
                self.event(
                    "formal_case_evaluated",
                    seed=summary["seed"],
                    arm=summary["arm"],
                    success_episodes=summary["success_episodes"],
                    evaluation_episodes=summary["evaluation_episodes"],
                    attempt=attempt,
                )
                return summary
            self.event(
                "formal_evaluation_attempt_failed",
                case_id=case_id,
                attempt=attempt,
                exit_code=code,
                immutable_checkpoint=True,
            )
        raise TechnicalFailure(f"Frozen evaluation failed after {attempts} attempts: {case_id}")

    def write_evaluation_summary(self) -> None:
        atomic_json(
            self.root / "formal" / "evaluations" / "evaluation_summary.json",
            {
                "schema": "roll_learning_v2_1_formal_evaluation_summary/v1",
                "study_id": self.config["study_id"],
                "updated_at_utc": utc_now(),
                "results": self.evaluation_results,
            },
        )

    def formal_result(self) -> dict[str, Any]:
        formal = self.config["formal"]
        rroll = [item for item in self.evaluation_results if item["arm"] == "Rroll"]
        r0 = [item for item in self.evaluation_results if item["arm"] == "R0"]
        minimum = int(formal["single_training_seed_success_min_episodes"])
        successful = [item for item in rroll if int(item["success_episodes"]) >= minimum]
        paired: list[dict[str, Any]] = []
        for seed in [int(value) for value in formal["training_seeds"]]:
            left = next(item for item in r0 if int(item["seed"]) == seed)
            right = next(item for item in rroll if int(item["seed"]) == seed)
            paired.append(
                {
                    "seed": seed,
                    "R0_success_episodes": int(left["success_episodes"]),
                    "Rroll_success_episodes": int(right["success_episodes"]),
                    "paired_difference_episodes": int(right["success_episodes"] - left["success_episodes"]),
                }
            )
        count = len(successful)
        reproducible = count >= int(formal["condition_reproducible_min_training_seeds"])
        robust = count >= int(formal["condition_robust_min_training_seeds"])
        return {
            "schema": "roll_learning_v2_1_formal_result/v1",
            "study_id": self.config["study_id"],
            "completed_at_utc": utc_now(),
            "training_seed_count": len(formal["training_seeds"]),
            "total_training_runs": len(self.evaluation_results),
            "endpoint_checkpoint": int(formal["batches"]),
            "successful_rroll_training_seeds": [int(item["seed"]) for item in successful],
            "successful_rroll_training_seed_count": count,
            "execution_status": "complete",
            "scientific_outcome": "robust" if robust else ("reproducible" if reproducible else "not_reproducible"),
            "reproducible": reproducible,
            "robust": robust,
            "single_seed_threshold": f">={minimum}/{formal['evaluation_episodes']}",
            "paired_reward_effect": paired,
            "from_scratch": True,
            "historical_checkpoint_loaded": False,
            "observation_channels_unchanged": True,
            "action_channels_unchanged": True,
            "maximum_concurrent_training_processes": 2,
            "technical_retries": self.state["technical_retries"],
            "results": self.evaluation_results,
        }

    def run(self) -> int:
        if self.state_path.exists() or self.events_path.exists():
            raise FileExistsError("Existing formal state/events; refusing duplicate launch")
        self.root.mkdir(parents=True, exist_ok=True)
        self.write_state()
        self.write_launcher_receipt()
        self.event(
            "formal_orchestrator_started",
            study_id=self.config["study_id"],
            from_scratch=True,
            maximum_concurrent_training_processes=2,
        )
        try:
            self.verify_static_contract()
            self.freeze_sources()
            self.verify_sources()
            self.run_prechecks()
            self.verify_sources()

            formal = self.config["formal"]
            start = {
                "schema": "roll_learning_formal_execution_started/v1",
                "study_id": self.config["study_id"],
                "started_at_utc": utc_now(),
                "approved_marker": str(self.approval_path),
                "approved_regression_gate": str(self.gate_path),
                "training_seeds": formal["training_seeds"],
                "arms": formal["arms"],
                "batches": int(formal["batches"]),
                "maximum_concurrent_training_processes": 2,
                "pair_concurrency": "same-seed R0 and Rroll",
                "from_batch0": True,
                "pretrained_loaded": False,
                "resume_loaded": False,
                "pilot_or_historical_checkpoint_loaded": False,
            }
            atomic_json(self.root / "FORMAL_EXECUTION_STARTED.json", start)
            self.state["status"] = "running"
            self.state["phase"] = "FORMAL_TRAIN"
            self.write_state()
            self.event("formal_execution_started", **{key: value for key, value in start.items() if key != "schema"})

            for seed in [int(value) for value in formal["execution_order"]]:
                r0_receipt, rroll_receipt = self.run_pair(seed)
                for receipt in (r0_receipt, rroll_receipt):
                    self.verify_sources()
                    self.evaluation_results.append(self.evaluate_case(receipt))
                    self.write_evaluation_summary()

            if len(self.evaluation_results) != 10:
                raise ContractFailure(f"Expected ten formal evaluations, got {len(self.evaluation_results)}")
            result = self.formal_result()
            atomic_json(self.root / "FORMAL_RESULT.json", result)
            self.state["formal_result"] = result
            self.state["status"] = "complete"
            self.state["phase"] = "FORMAL_COMPLETE"
            self.state["current_pair"] = None
            self.state["active_cases"] = {}
            self.state["current_evaluation"] = None
            self.write_state()
            self.event(
                "formal_complete",
                reproducible=result["reproducible"],
                robust=result["robust"],
                successful_rroll_training_seed_count=result["successful_rroll_training_seed_count"],
            )
            return 0
        except BaseException as error:
            self.state["status"] = "failed"
            self.state["phase"] = "FORMAL_FAILED"
            self.state["error"] = f"{type(error).__name__}: {error}"
            self.write_state()
            self.event("formal_execution_failed", error=self.state["error"])
            raise


def static_self_test(config_path: Path) -> None:
    config = read_json(config_path)
    assert config["schema"] == "roll_learning_reward_only_formal/v2.1"
    assert config["formal"]["training_seeds"] == [9201, 9202, 9203, 9204, 9205]
    assert config["formal"]["arms"] == ["R0", "Rroll"]
    assert config["formal"]["batches"] == 1500
    assert config["formal"]["max_concurrent_processes"] == 2
    assert config["rewards"] == {"R0": "horizontal_speed", "Rroll": "obs2_roll_repro_v2_1"}
    assert config["from_scratch_contract"]["pretrained_model_path"] is None
    assert config["from_scratch_contract"]["resume_training_state"] is None
    assert config["failure_policy"]["automatic_resume"] is False
    assert config["failure_policy"]["automatic_reward_change"] is False
    assert config["technical_retry"]["restart_from_batch0"] is True
    assert config["technical_retry"]["retry_scope"] == "whole_pair"
    assert config["episode_success"] == {
        "minimum_roll_pulses": 4,
        "minimum_desired_net_rotation_degrees": 360.0,
        "minimum_direction_fraction": 0.7,
        "minimum_forward_body_lengths": 1.0,
        "maximum_mean_inter_pulse_interval_steps": 250.0,
    }
    print("Formal v2.1 static self-test passed")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Paired-concurrent v2.1 formal supervisor")
    value.add_argument("--root", type=Path)
    value.add_argument("--config", type=Path, required=True)
    value.add_argument("--self-test", action="store_true")
    value.add_argument("--verify-only", action="store_true")
    return value


def main() -> int:
    args = parser().parse_args()
    if args.self_test:
        static_self_test(args.config)
        return 0
    if args.root is None:
        raise SystemExit("--root is required unless --self-test is used")
    supervisor = FormalSupervisor(args.root, args.config)
    if args.verify_only:
        supervisor.verify_static_contract()
        missing = [str(path) for path in supervisor.required_sources().values() if not path.is_file()]
        if missing:
            raise ContractFailure(f"Missing formal source before launch: {missing}")
        print("Formal v2.1 deployment verification passed")
        return 0
    return supervisor.run()


if __name__ == "__main__":
    raise SystemExit(main())
