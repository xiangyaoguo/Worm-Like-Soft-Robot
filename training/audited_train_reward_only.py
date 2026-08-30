from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import struct
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


SCHEMA = "roll_learning_initialization_audit/v1"
ALLOWED_REWARDS = {
    "horizontal_speed",
    "obs2_roll_repro_v1",
    "obs2_roll_repro_v2",
    "obs2_roll_repro_v2_1",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _length(digest: Any, payload: bytes) -> None:
    digest.update(struct.pack(">Q", len(payload)))
    digest.update(payload)


def _update(digest: Any, value: Any) -> None:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"T")
        _length(digest, str(tensor.dtype).encode("ascii"))
        _length(digest, json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        _length(digest, tensor.view(torch.uint8).numpy().tobytes(order="C"))
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"A")
        _length(digest, str(array.dtype).encode("ascii"))
        _length(digest, json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
        _length(digest, array.tobytes(order="C"))
    elif isinstance(value, dict):
        digest.update(b"D")
        items = sorted(value.items(), key=lambda pair: repr(pair[0]))
        digest.update(struct.pack(">Q", len(items)))
        for key, item in items:
            _update(digest, key)
            _update(digest, item)
    elif isinstance(value, (list, tuple)):
        digest.update(b"L" if isinstance(value, list) else b"Q")
        digest.update(struct.pack(">Q", len(value)))
        for item in value:
            _update(digest, item)
    elif value is None:
        digest.update(b"N")
    elif isinstance(value, bool):
        digest.update(b"B1" if value else b"B0")
    elif isinstance(value, int):
        digest.update(b"I")
        _length(digest, str(value).encode("ascii"))
    elif isinstance(value, float):
        digest.update(b"F")
        digest.update(struct.pack(">d", value))
    elif isinstance(value, str):
        digest.update(b"U")
        _length(digest, value.encode("utf-8"))
    else:
        raise TypeError(f"Unsupported hash object: {type(value).__name__}")


def object_hash(value: Any) -> str:
    digest = hashlib.sha256()
    _update(digest, value)
    return digest.hexdigest()


def module_summary(module: Any) -> dict[str, Any]:
    state = module.state_dict()
    nonfinite = 0
    total = 0
    for value in state.values():
        if torch.is_tensor(value):
            total += int(value.numel())
            if value.dtype.is_floating_point or value.dtype.is_complex:
                nonfinite += int((~torch.isfinite(value.detach())).sum().item())
    return {
        "class": f"{type(module).__module__}.{type(module).__qualname__}",
        "state_sha256": object_hash(state),
        "total_numel": total,
        "nonfinite_element_count": nonfinite,
        "all_tensors_finite": nonfinite == 0,
    }


def optimizer_state(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): optimizer_state(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [optimizer_state(item) for item in value]
    return value.state_dict()


def rng_summary() -> dict[str, Any]:
    algorithm, keys, position, has_gauss, cached = np.random.get_state()
    numpy_payload = {
        "algorithm": str(algorithm),
        "keys_sha256": object_hash(keys),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian_hex": float(cached).hex(),
    }
    py_version, py_state, py_gauss = random.getstate()
    python_payload = {
        "version": int(py_version),
        "state_sha256": object_hash(tuple(int(item) for item in py_state)),
        "gaussian": None if py_gauss is None else float(py_gauss).hex(),
    }
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    return {
        "torch_cpu_sha256": object_hash(torch.random.get_rng_state()),
        "torch_cuda_sha256": [object_hash(item) for item in cuda_states],
        "numpy_sha256": object_hash(numpy_payload),
        "python_sha256": object_hash(python_payload),
    }


def pair_bundle(audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "actor_sha256": audit["actor"]["state_sha256"],
        "critic_sha256": audit["critic"]["state_sha256"],
        "optimizer_sha256": audit["optimizer_sha256"],
        "torch_cpu_rng_sha256": audit["rng"]["torch_cpu_sha256"],
        "torch_cuda_rng_sha256": audit["rng"]["torch_cuda_sha256"],
        "numpy_rng_sha256": audit["rng"]["numpy_sha256"],
        "python_rng_sha256": audit["rng"]["python_sha256"],
    }


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"roll_learning_trainer_{os.getpid()}", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def validate_args(args: Any, reward: str, seed: int) -> dict[str, Any]:
    forbidden = {
        "pretrained_model_path": args.pretrained_model_path,
        "pretrained_policy_only": bool(args.pretrained_policy_only),
        "resume_training_state": args.resume_training_state,
        "bc_teacher_checkpoint": args.bc_teacher_checkpoint,
        "wave_bc_teacher_json": args.wave_bc_teacher_json,
        "bc_steps": int(args.bc_steps),
        "bc_epochs": int(args.bc_epochs),
        "policy_anchor_coeff": float(args.policy_anchor_coeff),
        "policy_anchor_anneal_batches": int(args.policy_anchor_anneal_batches),
    }
    violations: list[str] = []
    if str(args.reward_func) != reward:
        violations.append("reward mismatch")
    if int(args.seed) != seed:
        violations.append("seed mismatch")
    if args.pretrained_model_path is not None or args.pretrained_policy_only:
        violations.append("pretrained policy/model is forbidden")
    if args.resume_training_state is not None:
        violations.append("resume is forbidden")
    if args.bc_teacher_checkpoint is not None or args.wave_bc_teacher_json is not None:
        violations.append("behavior-cloning teacher is forbidden")
    if args.bc_steps != 0 or args.bc_epochs != 0:
        violations.append("behavior cloning is forbidden")
    if args.policy_anchor_coeff != 0 or args.policy_anchor_anneal_batches != 0:
        violations.append("policy anchor is forbidden")
    if args.robot != "crawler" or args.num_particles != 10:
        violations.append("crawler10 contract changed")
    if args.channel != "action" or args.control_mode != "formula":
        violations.append("formula action contract changed")
    if args.observation_func != "dth_tot_plus_friction_thdot":
        violations.append("obs2 contract changed")
    if args.share_policy or not args.per_joint_k1_k2:
        violations.append("independent per-joint actor contract changed")
    if not args.share_critic or not args.centralised_critic:
        violations.append("centralized shared critic contract changed")
    if args.rolling_observation or args.tail_roll_observation or args.fast_forward_observation:
        violations.append("global reward features leaked into observations")
    if args.scratch_wr_v2:
        violations.append("Scratch-WR controller is forbidden")
    if args.compatible_input_expansion:
        violations.append("compatible input expansion must be disabled")
    reward_protocol = {
        "fast_forward_launch_lift": 0.20,
        "fast_forward_launch_forward": 0.10,
        "fast_forward_launch_curl": 0.12,
        "fast_forward_launch_head_contact": 0.50,
        "fast_forward_launch_hold_steps": 8,
        "fast_forward_event_degrees": 60.0,
        "fast_forward_event_forward_fraction": 0.08,
        "fast_forward_event_contact_nodes": 1.5,
        "fast_forward_direction_fraction": 0.65,
    }
    for name, expected in reward_protocol.items():
        actual = getattr(args, name)
        if isinstance(expected, int):
            matches = int(actual) == expected
        else:
            matches = bool(np.isclose(float(actual), expected, rtol=0.0, atol=1e-12))
        if not matches:
            violations.append(f"reward protocol changed: {name}={actual!r}, expected {expected!r}")
    if violations:
        raise RuntimeError("; ".join(violations))
    return {
        "contract_valid": True,
        "from_scratch": True,
        "forbidden_sources": {key: None if value is None else value for key, value in forbidden.items()},
        "reward_func": str(args.reward_func),
        "seed": int(args.seed),
        "episodes": int(args.episodes),
        "expected_num_envs": max(1, int(args.frames_per_batch) // int(args.episode_steps)),
        "observation_func": str(args.observation_func),
        "control_mode": str(args.control_mode),
        "share_policy": bool(args.share_policy),
        "per_joint_k1_k2": bool(args.per_joint_k1_k2),
        "share_critic": bool(args.share_critic),
        "centralised_critic": bool(args.centralised_critic),
        "rolling_observation": bool(args.rolling_observation),
        "tail_roll_observation": bool(args.tail_roll_observation),
        "fast_forward_observation": bool(args.fast_forward_observation),
        "reward_protocol": reward_protocol,
    }


def environment_contract(env: Any, expected_num_envs: int) -> dict[str, Any]:
    obs_shape = list(env.observation_spec[("agents", "observation")].shape)
    action_shape = list(env.action_spec[env.action_key].shape)
    value = {
        "observation_shape": obs_shape,
        "action_shape": action_shape,
        "formula_action_names": list(env.formula_action_names),
        "rolling_observation_size": int(env.rolling_observation_size),
        "tail_roll_observation_size": int(env.tail_roll_observation_size),
        "fast_forward_observation_size": int(env.fast_forward_observation_size),
        "scratch_wr_v2_observation_size": int(env.scratch_wr_v2_observation_size),
        "control_mode": str(env.control_mode),
        "k_action_scale": float(env.k_action_scale),
        "max_torque": float(env.max_torque),
        "expected_num_envs": int(expected_num_envs),
    }
    expected_shape = [int(expected_num_envs), 8, 2]
    if obs_shape != expected_shape or action_shape != expected_shape:
        raise RuntimeError(f"Unexpected observation/action shapes: {obs_shape}, {action_shape}")
    if value["formula_action_names"] != ["k1", "k2"]:
        raise RuntimeError("Formula action names changed")
    if any(value[name] != 0 for name in (
        "rolling_observation_size", "tail_roll_observation_size",
        "fast_forward_observation_size", "scratch_wr_v2_observation_size",
    )):
        raise RuntimeError("Additional observation feature block detected")
    return value


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--audit-output", type=Path, required=True)
    result.add_argument("--case-id", required=True)
    result.add_argument("--expected-reward", choices=sorted(ALLOWED_REWARDS), required=True)
    result.add_argument("--trainer", type=Path, required=True)
    return result


def main() -> int:
    wrapper, trainer_arguments = parser().parse_known_args()
    audit_path = wrapper.audit_output.resolve()
    trainer_path = wrapper.trainer.resolve()
    if audit_path.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {audit_path}")
    if not trainer_path.is_file():
        raise FileNotFoundError(trainer_path)
    reward = str(wrapper.expected_reward)
    seed_tokens = [trainer_arguments[index + 1] for index, token in enumerate(trainer_arguments[:-1]) if token == "--seed"]
    if len(seed_tokens) != 1:
        raise ValueError("Exactly one --seed is required")
    seed = int(seed_tokens[0])
    random.seed(seed)

    trainer = load_module(trainer_path)
    original_parse = trainer.parse_args
    original_env = trainer.metamaterial.env
    original_build = trainer.build_components
    capture: dict[str, Any] = {"args": None, "environment": None, "audit": None}

    def wrapped_parse():
        args = original_parse()
        capture["args"] = validate_args(args, reward, seed)
        return args

    def wrapped_env(*args: Any, **kwargs: Any):
        value = original_env(*args, **kwargs)
        expected_num_envs = int(capture["args"]["expected_num_envs"])
        if "num_envs" in kwargs and int(kwargs["num_envs"]) != expected_num_envs:
            raise RuntimeError(
                f"Trainer num_envs {kwargs['num_envs']} != expected {expected_num_envs}"
            )
        capture["environment"] = environment_contract(value, expected_num_envs)
        return value

    def wrapped_build(env: Any, args: Any, device: Any):
        result = original_build(env, args, device)
        policy, critic, optimizer = result[0], result[2], result[6]
        actor = module_summary(policy)
        critic_summary = module_summary(critic)
        if not actor["all_tensors_finite"] or not critic_summary["all_tensors_finite"]:
            raise FloatingPointError("Non-finite batch0 network")
        audit = {
            "schema": SCHEMA,
            "status": "captured_before_training",
            "captured_at_utc": utc_now(),
            "capture_point": "after build_components and before collector iteration / optimizer update",
            "case_id": wrapper.case_id,
            "seed": seed,
            "reward": reward,
            "batch_index": 0,
            "from_scratch": True,
            "runtime_args": capture["args"],
            "environment": capture["environment"],
            "actor": actor,
            "critic": critic_summary,
            "optimizer_sha256": object_hash(optimizer_state(optimizer)),
            "rng": rng_summary(),
            "trainer": {"path": str(trainer_path), "sha256": sha256_file(trainer_path)},
        }
        audit["pair_hash_bundle"] = pair_bundle(audit)
        audit["pair_hash_bundle_sha256"] = object_hash(audit["pair_hash_bundle"])
        capture["audit"] = audit
        atomic_json(audit_path, audit)
        return result

    trainer.parse_args = wrapped_parse
    trainer.metamaterial.env = wrapped_env
    trainer.build_components = wrapped_build
    original_argv = sys.argv[:]
    exit_code = 1
    error_text = None
    try:
        sys.argv = [str(trainer_path), *trainer_arguments]
        trainer.main()
        exit_code = 0
    except BaseException as error:
        error_text = f"{type(error).__name__}: {error}"
        traceback.print_exc()
        if isinstance(error, KeyboardInterrupt):
            exit_code = 130
        elif isinstance(error, SystemExit) and isinstance(error.code, int):
            exit_code = int(error.code) or 1
    finally:
        sys.argv = original_argv
        audit = capture["audit"] or {
            "schema": SCHEMA,
            "case_id": wrapper.case_id,
            "seed": seed,
            "reward": reward,
            "from_scratch": True,
        }
        audit["finished_at_utc"] = utc_now()
        audit["trainer_exit_code"] = int(exit_code)
        audit["trainer_error"] = error_text
        audit["status"] = "complete" if exit_code == 0 else "failed"
        atomic_json(audit_path, audit)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
