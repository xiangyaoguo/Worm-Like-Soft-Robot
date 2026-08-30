from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from condition_matrix import Condition, build_conditions, canonical_sha256
from mechanism_rollout import ROOT, SeedRuntime, atomic_json


def independent_expected(
    condition: Condition,
    r0: np.ndarray,
    roll: np.ndarray,
    calibration: dict[str, np.ndarray],
    permutation: np.ndarray,
    step: int,
) -> np.ndarray:
    spec = condition.spec
    op = str(spec["op"])
    result = r0.copy()
    if op == "source_mix":
        result[list(spec["k1_roll_joints"]), 0] = roll[
            list(spec["k1_roll_joints"]), 0
        ]
        result[list(spec["k2_roll_joints"]), 1] = roll[
            list(spec["k2_roll_joints"]), 1
        ]
    elif op == "k1_zero":
        result = roll.copy()
        result[:, 0] = 0.0
    elif op == "k1_sign":
        result = roll.copy()
        result[:, 0] = np.asarray(spec["signs"]) * np.abs(roll[:, 0])
    elif op == "k1_reverse":
        result = roll.copy()
        result[:, 0] = roll[::-1, 0]
    elif op == "k2_scale":
        result = roll.copy()
        result[:, 1] = float(spec["alpha"]) * roll[:, 1]
    elif op == "k2_sign_force":
        result = roll.copy()
        result[:, 1] = float(spec["sign"]) * np.abs(roll[:, 1])
    elif op == "k2_region":
        result = roll.copy()
        keep = set(int(value) for value in spec["keep_joints"])
        result[[joint for joint in range(8) if joint not in keep], 1] = 0.0
    elif op == "k2_calibration_static_mean":
        result = roll.copy()
        result[:, 1] = calibration["static"]
    elif op == "k2_calibration_time_template":
        result = roll.copy()
        result[:, 1] = calibration["template"][step]
    elif op == "k2_calibration_permuted_template":
        result = roll.copy()
        result[:, 1] = calibration["template"][permutation[step]]
    else:
        raise AssertionError(op)
    return result


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    conditions = build_conditions()
    runtime = SeedRuntime(9201, "identity", environment_arm="Rroll")
    try:
        runtime.frozen_eval._self_test()
        runtime.calibration = {
            "static": np.linspace(-0.8, 0.8, 8, dtype=np.float32),
            "template": np.arange(runtime.steps * 8, dtype=np.float32).reshape(
                runtime.steps, 8
            )
            / 10000.0,
        }
        r0_np = np.stack(
            [np.linspace(-0.9, 0.7, 8), np.linspace(0.8, -0.6, 8)], axis=1
        ).astype(np.float32)
        roll_np = np.stack(
            [np.linspace(0.95, -0.75, 8), np.linspace(-0.85, 0.65, 8)], axis=1
        ).astype(np.float32)
        r0 = torch.from_numpy(r0_np).unsqueeze(0)
        roll = torch.from_numpy(roll_np).unsqueeze(0)
        maximum_transform_error = 0.0
        for condition in conditions:
            actual = runtime.apply_condition(condition, r0, roll, step=17)[0].numpy()
            expected = independent_expected(
                condition,
                r0_np,
                roll_np,
                runtime.calibration,
                runtime.permutation,
                17,
            )
            error = float(np.max(np.abs(actual - expected)))
            maximum_transform_error = max(maximum_transform_error, error)
            if error != 0.0:
                raise RuntimeError(f"Transform mismatch for {condition.id}: {error}")

        td = runtime.env.reset()
        r0_live, roll_live, observation_error = runtime.actor_actions(td)
        by_id = {condition.id: condition for condition in conditions}
        c00 = runtime.apply_condition(by_id["C00"], r0_live, roll_live, 0)
        c11 = runtime.apply_condition(by_id["C11"], r0_live, roll_live, 0)
        if not torch.equal(c00, r0_live):
            raise RuntimeError("C00 endpoint is not bit-exact R0")
        if not torch.equal(c11, roll_live):
            raise RuntimeError("C11 endpoint is not bit-exact Rroll")
        if observation_error != 0.0:
            raise RuntimeError("R0/Rroll did not receive the same observation")

        invalid = Condition(
            id="INVALID_SIGN_VECTOR",
            module="C",
            family="negative_test",
            description="Must fail.",
            spec={"op": "k1_sign", "signs": [1] * 7},
        )
        failed_closed = False
        try:
            runtime.apply_condition(invalid, r0_live, roll_live, 0)
        except RuntimeError:
            failed_closed = True
        if not failed_closed:
            raise RuntimeError("Invalid sign-vector adversary did not fail closed")

        immutability = runtime.verify_unchanged()
        payload = {
            "schema": "obs2_v2_1_k_smoke/v1",
            "passed": True,
            "condition_count": len(conditions),
            "condition_matrix_sha256": canonical_sha256(conditions),
            "frozen_evaluator_self_test": "passed",
            "maximum_transform_error": maximum_transform_error,
            "C00_endpoint_bit_exact": True,
            "C11_endpoint_bit_exact": True,
            "same_observation_error": observation_error,
            "invalid_sign_vector_failed_closed": failed_closed,
            "immutability": immutability,
        }
        atomic_json(ROOT / "SMOKE_TEST_PASS.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
