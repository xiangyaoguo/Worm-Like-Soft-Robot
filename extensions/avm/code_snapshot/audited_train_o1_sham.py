"""Fail-closed initialisation gate and audit wrapper for formal HPR/O1-sham.

This wrapper compares the newly constructed batch-0 state with the archived
HPR/O2 initialisation for the same internal seed *before* the collector can
iterate or the optimiser can update.  It intentionally reuses the parent's
canonical hashing implementation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCHEMA = "formal_hpr_o1_sham_initialization_audit/v1"
DEFAULT_HASH_HELPER = Path(
    "C:\\Users\\PUBLIC_USER\\CloudStorage\\Desktop\\finalproject\\job\\roll_learning\\"
    "obs2_roll_repro_v2_1_formal_20260803_r2\\_control\\code_snapshot\\"
    "training\\audited_train_reward_only.py"
)
PAIR_FIELDS = (
    "actor_sha256",
    "critic_sha256",
    "optimizer_sha256",
    "torch_cpu_rng_sha256",
    "torch_cuda_rng_sha256",
    "numpy_rng_sha256",
    "python_rng_sha256",
)


class PreflightComplete(RuntimeError):
    pass


class InitializationMismatch(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_HASH = load_module(
    Path(os.environ.get("FORMAL_PARENT_HASH_HELPER", str(DEFAULT_HASH_HELPER))).resolve(),
    "formal_parent_initialization_hashing",
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--audit-output", type=Path, required=True)
    result.add_argument("--case-id", required=True)
    result.add_argument("--expected-seed", type=int, choices=range(9201, 9206), required=True)
    result.add_argument("--reference-audit", type=Path, required=True)
    result.add_argument("--reference-audit-sha256", required=True)
    result.add_argument("--trainer", type=Path, required=True)
    result.add_argument("--preflight-only", action="store_true")
    return result


def assert_close(name: str, actual: float, expected: float) -> None:
    if not np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-12):
        raise RuntimeError(f"Formal contract drift: {name}={actual!r}, expected {expected!r}")


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def validate_args(args: Any, seed: int) -> dict[str, Any]:
    exact = {
        "robot": "crawler",
        "terrain": "flat",
        "terrain_contact_mode": "legacy_flat",
        "num_particles": 10,
        "channel": "action",
        "observation_func": "dth_tot_plus_friction_thdot",
        "control_mode": "formula",
        "reward_func": "horizontal_speed",
        "algorithm": "ppo",
        "policy_depth": 2,
        "policy_cells": 256,
        "episode_steps": 1000,
        "frames_per_batch": 10000,
        "memory_size": 1000000,
        "minibatch_size": 128,
        "optim_steps": 10,
        "episodes": 1500,
        "save_every": 100,
        "actor_observation_mode": "spatial_only_sham",
        "critic_observation_mode": "full_o2",
    }
    for name, expected in exact.items():
        actual = getattr(args, name)
        if actual != expected:
            raise RuntimeError(f"Formal contract drift: {name}={actual!r}, expected {expected!r}")
    numerical = {
        "feedback_gain": 1.0,
        "max_control_gain": 9.0,
        "k_action_scale": 100.0,
        "passive_kappa": 4.0,
        "lr": 0.0003,
        "weight_decay": 0.0001,
        "max_grad_norm": 1.0,
        "gamma": 0.99,
        "clip_epsilon": 0.2,
        "lambda_gae": 0.9,
        "entropy_eps": 0.0001,
        "init_pos_randomness": 0.01,
        "init_angle_range_degrees": 0.0,
        "init_height_jitter": 0.0,
        "action_smoothness_weight": 0.0,
        "policy_anchor_coeff": 0.0,
        "policy_anchor_anneal_batches": 0.0,
    }
    for name, expected in numerical.items():
        assert_close(name, getattr(args, name), expected)
    if int(args.seed) != int(seed):
        raise RuntimeError(f"Formal seed mismatch: {args.seed} != {seed}")
    required_true = {
        "per_joint_k1_k2": args.per_joint_k1_k2,
        "share_critic": args.share_critic,
        "centralised_critic": args.centralised_critic,
    }
    required_false = {
        "share_policy": args.share_policy,
        "fix_k1": args.fix_k1,
        "fix_k2": args.fix_k2,
        "rolling_observation": args.rolling_observation,
        "tail_roll_observation": args.tail_roll_observation,
        "fast_forward_observation": args.fast_forward_observation,
        "compatible_input_expansion": args.compatible_input_expansion,
        "pretrained_policy_only": args.pretrained_policy_only,
        "ppo_normalize_advantage": args.ppo_normalize_advantage,
        "auto_analysis": args.auto_analysis,
    }
    for name, value in required_true.items():
        if not bool(value):
            raise RuntimeError(f"Formal contract drift: {name} must be true")
    for name, value in required_false.items():
        if bool(value):
            raise RuntimeError(f"Formal contract drift: {name} must be false")
    forbidden = {
        "pretrained_model_path": args.pretrained_model_path,
        "resume_training_state": args.resume_training_state,
        "bc_teacher_checkpoint": args.bc_teacher_checkpoint,
        "wave_bc_teacher_json": args.wave_bc_teacher_json,
        "bc_steps": args.bc_steps,
        "bc_epochs": args.bc_epochs,
    }
    for name, value in forbidden.items():
        if value not in (None, 0):
            raise RuntimeError(f"From-scratch violation: {name}={value!r}")
    return {
        "contract_valid": True,
        "seed": int(seed),
        "paper_run": int(seed) - 9201,
        "reward": "horizontal_speed",
        "actor_observation_mode": "spatial_only_sham",
        "critic_observation_mode": "full_o2",
        "from_scratch": True,
        "episodes": 1500,
        "training_args": json_safe(vars(args)),
    }


def environment_contract(env: Any, expected_num_envs: int) -> dict[str, Any]:
    obs_shape = list(env.observation_spec[("agents", "observation")].shape)
    action_shape = list(env.action_spec[env.action_key].shape)
    expected_shape = [int(expected_num_envs), 8, 2]
    if obs_shape != expected_shape or action_shape != expected_shape:
        raise RuntimeError(
            f"Unexpected observation/action shapes: {obs_shape}, {action_shape}; "
            f"expected {expected_shape}"
        )
    checks = {
        "observation_func": (str(env.observation_func), "dth_tot_plus_friction_thdot"),
        "control_mode": (str(env.control_mode), "formula"),
        "formula_action_names": (list(env.formula_action_names), ["k1", "k2"]),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise RuntimeError(f"Environment contract drift: {name}={actual!r}")
    assert_close("environment feedback_gain", env.feedback_gain, 1.0)
    assert_close("environment k_action_scale", env.k_action_scale, 100.0)
    assert_close("environment max_torque", env.max_torque, 9.0)
    if bool(env.fix_k1) or bool(env.fix_k2):
        raise RuntimeError("K1/K2 output channel was unexpectedly fixed")
    return {
        "observation_shape": obs_shape,
        "action_shape": action_shape,
        "raw_environment_observation": "full_o2",
        "actor_observation": "spatial_only_sham",
        "critic_observation": "full_o2",
        "feedback_gain": 1.0,
        "physical_k2_theta_dot_enabled": True,
        "formula_action_names": ["k1", "k2"],
        "control_mode": "formula",
    }


def load_reference(path: Path, expected_sha256: str, seed: int) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha = _HASH.sha256_file(path)
    if actual_sha != expected_sha256.lower():
        raise RuntimeError(f"Reference audit SHA-256 drift: {actual_sha} != {expected_sha256}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if int(value.get("seed", -1)) != int(seed):
        raise RuntimeError("Reference audit seed mismatch")
    if value.get("reward") != "horizontal_speed":
        raise RuntimeError("Reference audit is not the parent HPR arm")
    if value.get("status") != "complete" or value.get("trainer_exit_code") != 0:
        raise RuntimeError("Reference audit is not a complete successful formal run")
    if set(value.get("pair_hash_bundle", {})) != set(PAIR_FIELDS):
        raise RuntimeError("Reference pair hash bundle fields drifted")
    return value


def compare_initialization(
    actual_bundle: dict[str, Any],
    actor: dict[str, Any],
    critic: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    expected_bundle = reference["pair_hash_bundle"]
    comparisons = {
        name: {
            "actual": actual_bundle[name],
            "expected": expected_bundle[name],
            "match": actual_bundle[name] == expected_bundle[name],
        }
        for name in PAIR_FIELDS
    }
    comparisons["actor_total_numel"] = {
        "actual": int(actor["total_numel"]),
        "expected": int(reference["actor"]["total_numel"]),
        "match": int(actor["total_numel"]) == int(reference["actor"]["total_numel"]),
    }
    comparisons["critic_total_numel"] = {
        "actual": int(critic["total_numel"]),
        "expected": int(reference["critic"]["total_numel"]),
        "match": int(critic["total_numel"]) == int(reference["critic"]["total_numel"]),
    }
    failed = [name for name, item in comparisons.items() if not item["match"]]
    return {"all_match": not failed, "failed_fields": failed, "fields": comparisons}


def main() -> int:
    wrapper, trainer_arguments = parser().parse_known_args()
    audit_path = wrapper.audit_output.resolve()
    trainer_path = wrapper.trainer.resolve()
    if audit_path.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {audit_path}")
    if not trainer_path.is_file():
        raise FileNotFoundError(trainer_path)
    reference = load_reference(
        wrapper.reference_audit,
        wrapper.reference_audit_sha256,
        wrapper.expected_seed,
    )
    seed_tokens = [
        trainer_arguments[index + 1]
        for index, token in enumerate(trainer_arguments[:-1])
        if token == "--seed"
    ]
    if seed_tokens != [str(wrapper.expected_seed)]:
        raise ValueError("Trainer must contain exactly the locked --seed")

    random.seed(wrapper.expected_seed)
    trainer = load_module(trainer_path, f"formal_o1_sham_trainer_{os.getpid()}")
    original_parse = trainer.parse_args
    original_env = trainer.metamaterial.env
    original_build = trainer.build_components
    capture: dict[str, Any] = {"args": None, "environment": None, "audit": None}

    def wrapped_parse() -> Any:
        args = original_parse()
        capture["args"] = validate_args(args, wrapper.expected_seed)
        return args

    def wrapped_env(*args: Any, **kwargs: Any) -> Any:
        value = original_env(*args, **kwargs)
        expected_num_envs = 10
        if int(kwargs.get("num_envs", expected_num_envs)) != expected_num_envs:
            raise RuntimeError("Formal environment count drifted")
        capture["environment"] = environment_contract(value, expected_num_envs)
        return value

    def wrapped_build(env: Any, args: Any, device: Any) -> Any:
        result = original_build(env, args, device)
        policy, critic_module, optimizer = result[0], result[2], result[6]
        actor_backbones = [
            module
            for module in policy.modules()
            if hasattr(module, "_formal_actor_observation_hook_handle")
        ]
        if len(actor_backbones) != 1:
            raise RuntimeError(
                "Actor-only observation hook marker missing or ambiguous: "
                f"{len(actor_backbones)} marked backbones"
            )
        if actor_backbones[0]._formal_actor_observation_mode != "spatial_only_sham":
            raise RuntimeError("Actor observation hook is not spatial_only_sham")
        actor = _HASH.module_summary(policy)
        critic_summary = _HASH.module_summary(critic_module)
        rng = _HASH.rng_summary()
        actual_bundle = {
            "actor_sha256": actor["state_sha256"],
            "critic_sha256": critic_summary["state_sha256"],
            "optimizer_sha256": _HASH.object_hash(_HASH.optimizer_state(optimizer)),
            "torch_cpu_rng_sha256": rng["torch_cpu_sha256"],
            "torch_cuda_rng_sha256": rng["torch_cuda_sha256"],
            "numpy_rng_sha256": rng["numpy_sha256"],
            "python_rng_sha256": rng["python_sha256"],
        }
        comparison = compare_initialization(
            actual_bundle, actor, critic_summary, reference
        )
        audit = {
            "schema": SCHEMA,
            "status": "gate_passed_before_training" if comparison["all_match"] else "initialization_mismatch",
            "captured_at_utc": utc_now(),
            "capture_point": "after component construction and actor hook installation; before collector iteration or optimizer update",
            "case_id": wrapper.case_id,
            "seed": int(wrapper.expected_seed),
            "paper_run": int(wrapper.expected_seed) - 9201,
            "reward": "horizontal_speed",
            "from_scratch": True,
            "preflight_only": bool(wrapper.preflight_only),
            "runtime_args": capture["args"],
            "environment": capture["environment"],
            "actor_observation_intervention": {
                "mode": "spatial_only_sham",
                "raw_environment_input": "[s_i, theta_dot_i]",
                "actor_input": "[s_i, 0]",
                "critic_input": "[s_i, theta_dot_i]",
                "physical_torque_retains_k2_theta_dot": True,
                "parameter_free_forward_pre_hook": True,
            },
            "actor": actor,
            "critic": critic_summary,
            "pair_hash_bundle": actual_bundle,
            "pair_hash_bundle_sha256": _HASH.object_hash(actual_bundle),
            "reference_audit": {
                "path": str(wrapper.reference_audit.resolve()),
                "sha256": wrapper.reference_audit_sha256.lower(),
                "pair_hash_bundle_sha256": reference["pair_hash_bundle_sha256"],
            },
            "initialization_comparison": comparison,
            "trainer": {
                "path": str(trainer_path),
                "sha256": _HASH.sha256_file(trainer_path),
            },
        }
        capture["audit"] = audit
        _HASH.atomic_json(audit_path, audit)
        if not comparison["all_match"]:
            raise InitializationMismatch(
                "Batch-0 state differs from paired parent HPR run: "
                + ", ".join(comparison["failed_fields"])
            )
        if wrapper.preflight_only:
            raise PreflightComplete("Initialisation gate passed; no collector iteration requested")
        return result

    trainer.parse_args = wrapped_parse
    trainer.metamaterial.env = wrapped_env
    trainer.build_components = wrapped_build
    original_argv = sys.argv[:]
    exit_code = 1
    error_text: str | None = None
    final_status = "failed_before_initialization"
    try:
        sys.argv = [str(trainer_path), *trainer_arguments]
        trainer.main()
        exit_code = 0
        final_status = "complete"
    except PreflightComplete:
        exit_code = 0
        final_status = "preflight_passed"
    except BaseException as error:
        error_text = f"{type(error).__name__}: {error}"
        traceback.print_exc()
        final_status = (
            "initialization_mismatch"
            if isinstance(error, InitializationMismatch)
            else "failed"
        )
        if isinstance(error, KeyboardInterrupt):
            exit_code = 130
        elif isinstance(error, SystemExit) and isinstance(error.code, int):
            exit_code = int(error.code) or 1
    finally:
        sys.argv = original_argv
        audit = capture["audit"] or {
            "schema": SCHEMA,
            "case_id": wrapper.case_id,
            "seed": int(wrapper.expected_seed),
            "paper_run": int(wrapper.expected_seed) - 9201,
            "reward": "horizontal_speed",
            "from_scratch": True,
            "preflight_only": bool(wrapper.preflight_only),
        }
        audit["finished_at_utc"] = utc_now()
        audit["trainer_exit_code"] = int(exit_code)
        audit["trainer_error"] = error_text
        audit["status"] = final_status
        _HASH.atomic_json(audit_path, audit)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
