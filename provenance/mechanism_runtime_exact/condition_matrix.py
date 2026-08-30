"""Canonical 59-condition matrix for the frozen obs2 v2.1 K-mechanism study.

Every condition consumes two deterministic actor proposals evaluated on the
same current state: ``r0_action`` and ``roll_action``.  Joint indices inside
``Condition.spec`` are zero-based for direct tensor indexing.  Human-readable
IDs and descriptions use the one-based labels J01..J08.

This module is declarative only: it does not load checkpoints, run an
environment, or write experiment outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence


SCHEMA = "obs2_v2_1_k_mechanism/condition_matrix/v1"
PROPOSAL_CONTRACT = "r0_action_and_roll_action_evaluated_on_the_same_state"
JOINTS = tuple(range(8))
JOINT_LABELS = tuple(f"J{joint + 1:02d}" for joint in JOINTS)
EXPECTED_MODULE_COUNTS: Mapping[str, int] = MappingProxyType(
    {"A": 4, "B": 32, "C": 13, "D": 10}
)
EXPECTED_TOTAL = sum(EXPECTED_MODULE_COUNTS.values())
CALIBRATION_SOURCE = "C11_Rroll_actions_on_identity_gate_episodes"
FIXED_TIME_PERMUTATION_SEED = 20264301


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, set):
        return tuple(sorted(_deep_freeze(item) for item in value))
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


@dataclass(frozen=True)
class Condition:
    """One immutable, JSON-serializable evaluation-time intervention."""

    id: str
    module: str
    family: str
    description: str
    spec: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.id or not isinstance(self.id, str):
            raise ValueError("Condition.id must be a non-empty string")
        if self.module not in EXPECTED_MODULE_COUNTS:
            raise ValueError(f"Unknown module {self.module!r} for {self.id}")
        if not self.family or not isinstance(self.family, str):
            raise ValueError(f"Condition.family must be non-empty for {self.id}")
        if not self.description or not isinstance(self.description, str):
            raise ValueError(f"Condition.description must be non-empty for {self.id}")
        if not isinstance(self.spec, Mapping):
            raise TypeError(f"Condition.spec must be a mapping for {self.id}")
        object.__setattr__(self, "spec", _deep_freeze(dict(self.spec)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "module": self.module,
            "family": self.family,
            "description": self.description,
            "spec": _deep_thaw(self.spec),
        }


def _condition(
    condition_id: str,
    module: str,
    family: str,
    description: str,
    spec: Mapping[str, Any],
) -> Condition:
    return Condition(
        id=condition_id,
        module=module,
        family=family,
        description=description,
        spec=spec,
    )


def _source_mix(
    k1_roll_joints: Iterable[int],
    k2_roll_joints: Iterable[int],
) -> dict[str, Any]:
    return {
        "op": "source_mix",
        "k1_roll_joints": sorted(int(joint) for joint in k1_roll_joints),
        "k2_roll_joints": sorted(int(joint) for joint in k2_roll_joints),
    }


def _joint_list_text(joints: Sequence[int]) -> str:
    return ",".join(JOINT_LABELS[joint] for joint in joints)


def build_conditions() -> tuple[Condition, ...]:
    """Build and validate the canonical ordered tuple of exactly 59 conditions."""

    conditions: list[Condition] = []

    # Module A: complete 2x2 K1/K2 source factorial.
    for condition_id, k1_roll, k2_roll, description in (
        ("C00", (), (), "K1 and K2 both come from R0 (recipient identity control)."),
        ("C10", JOINTS, (), "All K1 comes from Rroll; all K2 remains from R0."),
        ("C01", (), JOINTS, "All K1 remains from R0; all K2 comes from Rroll."),
        ("C11", JOINTS, JOINTS, "K1 and K2 both come from Rroll (rolling identity control)."),
    ):
        conditions.append(
            _condition(
                condition_id,
                "A",
                "channel_factorial",
                description,
                _source_mix(k1_roll, k2_roll),
            )
        )

    # Module B: per-joint K1 sufficiency and necessity.
    for joint in JOINTS:
        label = JOINT_LABELS[joint]
        conditions.append(
            _condition(
                f"K1_SUFF_{label}",
                "B",
                "k1_joint_sufficiency",
                f"Only {label} K1 comes from Rroll; every other K1 and all K2 come from R0.",
                _source_mix((joint,), ()),
            )
        )
    for joint in JOINTS:
        label = JOINT_LABELS[joint]
        conditions.append(
            _condition(
                f"K1_NEC_{label}",
                "B",
                "k1_joint_necessity",
                f"All K1 except {label} comes from Rroll; {label} K1 and all K2 come from R0.",
                _source_mix((candidate for candidate in JOINTS if candidate != joint), ()),
            )
        )

    # Module B: per-joint K2 sufficiency and necessity.  K2 necessity is
    # assessed against the complete rolling controller, hence K1 is all Rroll.
    for joint in JOINTS:
        label = JOINT_LABELS[joint]
        conditions.append(
            _condition(
                f"K2_SUFF_{label}",
                "B",
                "k2_joint_sufficiency",
                f"Only {label} K2 comes from Rroll; every other K2 and all K1 come from R0.",
                _source_mix((), (joint,)),
            )
        )
    for joint in JOINTS:
        label = JOINT_LABELS[joint]
        conditions.append(
            _condition(
                f"K2_NEC_{label}",
                "B",
                "k2_joint_necessity",
                f"K1 is all Rroll; K2 is Rroll except {label}, whose K2 comes from R0.",
                _source_mix(JOINTS, (candidate for candidate in JOINTS if candidate != joint)),
            )
        )

    # Module C: four preregistered multi-joint K1 sufficiency combinations.
    for joints in ((1, 2), (1, 4), (2, 4), (1, 2, 4)):
        labels = "_".join(JOINT_LABELS[joint] for joint in joints)
        conditions.append(
            _condition(
                f"K1_SUFF_{labels}",
                "C",
                "k1_subset_sufficiency",
                f"Only K1 at {_joint_list_text(joints)} comes from Rroll; all other K1 and all K2 come from R0.",
                _source_mix(joints, ()),
            )
        )

    # Module C: K1 sign and spatial-organization controls; K2 stays Rroll.
    conditions.append(
        _condition(
            "K1_ZERO_ALL",
            "C",
            "k1_sign_space",
            "Set K1 to zero at J01..J08 while retaining all Rroll K2.",
            {"op": "k1_zero", "k2_source": "Rroll"},
        )
    )

    sign_conditions = (
        (
            "K1_SIGN_J01_POS_J02_J08_NEG",
            (1, -1, -1, -1, -1, -1, -1, -1),
            "Force J01 K1 positive and J02..J08 K1 negative using sign*abs(Rroll K1).",
        ),
        (
            "K1_SIGN_J01_J02_POS_J03_J08_NEG",
            (1, 1, -1, -1, -1, -1, -1, -1),
            "Force J01..J02 K1 positive and J03..J08 K1 negative using sign*abs(Rroll K1).",
        ),
        (
            "K1_SIGN_J01_J03_POS_J04_J08_NEG",
            (1, 1, 1, -1, -1, -1, -1, -1),
            "Force J01..J03 K1 positive and J04..J08 K1 negative using sign*abs(Rroll K1).",
        ),
        (
            "K1_SIGN_J01_J04_POS_J05_J08_NEG",
            (1, 1, 1, 1, -1, -1, -1, -1),
            "Force J01..J04 K1 positive and J05..J08 K1 negative using sign*abs(Rroll K1).",
        ),
        (
            "K1_SIGN_ALL_POS",
            (1, 1, 1, 1, 1, 1, 1, 1),
            "Force J01..J08 K1 positive using abs(Rroll K1).",
        ),
        (
            "K1_SIGN_ALL_NEG",
            (-1, -1, -1, -1, -1, -1, -1, -1),
            "Force J01..J08 K1 negative using -abs(Rroll K1).",
        ),
        (
            "K1_SIGN_ALTERNATING_J01_POS",
            (1, -1, 1, -1, 1, -1, 1, -1),
            "Force alternating K1 signs +,-,+,-,+,-,+,- from J01 to J08.",
        ),
    )
    for condition_id, signs, description in sign_conditions:
        conditions.append(
            _condition(
                condition_id,
                "C",
                "k1_sign_space",
                description,
                {"op": "k1_sign", "signs": signs, "k2_source": "Rroll"},
            )
        )

    conditions.append(
        _condition(
            "K1_SIGN_MIRROR_CANONICAL_J08_POS",
            "C",
            "k1_sign_space",
            "Mirror the canonical learned sign template: force J01..J07 negative and J08 positive on each target joint's own abs(Rroll K1); this is not a policy-function or observation mirror.",
            {
                "op": "k1_sign",
                "signs": (-1, -1, -1, -1, -1, -1, -1, 1),
                "k2_source": "Rroll",
            },
        )
    )

    # Module D: K2 dose, sign, region, and calibrated temporal controls.
    for condition_id, alpha, description in (
        ("K2_SCALE_0", 0.0, "Set all K2 to zero while retaining all Rroll K1."),
        ("K2_SCALE_0P5", 0.5, "Apply 0.5 times Rroll K2 while retaining all Rroll K1."),
        ("K2_SCALE_1P5", 1.5, "Apply 1.5 times Rroll K2 while retaining all Rroll K1."),
    ):
        conditions.append(
            _condition(
                condition_id,
                "D",
                "k2_amplitude",
                description,
                {"op": "k2_scale", "alpha": alpha, "k1_source": "Rroll"},
            )
        )

    for condition_id, sign, description in (
        ("K2_FORCE_POSITIVE", 1, "Force K2 positive as abs(Rroll K2); retain all Rroll K1."),
        ("K2_FORCE_NEGATIVE", -1, "Force K2 negative as -abs(Rroll K2); retain all Rroll K1."),
    ):
        conditions.append(
            _condition(
                condition_id,
                "D",
                "k2_sign",
                description,
                {
                    "op": "k2_sign_force",
                    "source": "Rroll",
                    "sign": sign,
                    "k1_source": "Rroll",
                },
            )
        )

    for condition_id, keep_joints, description in (
        (
            "K2_TAIL_ONLY_J01_J02",
            (0, 1),
            "Keep Rroll K2 only at tail-proximal J01..J02; set J03..J08 K2 to zero.",
        ),
        (
            "K2_BODY_ONLY_J03_J08",
            (2, 3, 4, 5, 6, 7),
            "Set J01..J02 K2 to zero and keep Rroll K2 at body joints J03..J08.",
        ),
    ):
        conditions.append(
            _condition(
                condition_id,
                "D",
                "k2_region",
                description,
                {
                    "op": "k2_region",
                    "source": "Rroll",
                    "keep_joints": keep_joints,
                    "fill": 0.0,
                    "k1_source": "Rroll",
                },
            )
        )

    conditions.extend(
        (
            _condition(
                "K2_CALIBRATION_STATIC_MEAN",
                "D",
                "k2_temporal_calibration",
                "Use the per-joint static mean K2 calibrated only from C11 identity-gate episodes.",
                {
                    "op": "k2_calibration_static_mean",
                    "k1_source": "Rroll",
                    "template_source": CALIBRATION_SOURCE,
                    "aggregation": "per_joint_mean_over_calibration_episodes_and_time",
                    "application": "constant_per_joint_for_entire_episode",
                },
            ),
            _condition(
                "K2_CALIBRATION_TIME_TEMPLATE",
                "D",
                "k2_temporal_calibration",
                "Use the fixed per-step, per-joint K2 time template calibrated only from C11 identity-gate episodes.",
                {
                    "op": "k2_calibration_time_template",
                    "k1_source": "Rroll",
                    "template_source": CALIBRATION_SOURCE,
                    "aggregation": "per_step_per_joint_mean_across_calibration_episodes",
                    "application": "open_loop_by_control_step",
                },
            ),
            _condition(
                "K2_CALIBRATION_PERMUTED_TEMPLATE",
                "D",
                "k2_temporal_calibration",
                "Apply one fixed time permutation of the calibrated K2 template; retain its values but destroy timing.",
                {
                    "op": "k2_calibration_permuted_template",
                    "k1_source": "Rroll",
                    "template_source": CALIBRATION_SOURCE,
                    "base_template": "K2_CALIBRATION_TIME_TEMPLATE",
                    "permutation_axis": "control_step",
                    "permutation_seed": FIXED_TIME_PERMUTATION_SEED,
                    "permutation_rng": "numpy.random.PCG64",
                    "fixed_across_training_and_evaluation_seeds": True,
                },
            ),
        )
    )

    result = tuple(conditions)
    validate_conditions(result)
    return result


def _validate_joint_indices(condition: Condition, key: str, value: Any) -> None:
    if key not in {"k1_roll_joints", "k2_roll_joints", "keep_joints", "joint_order"}:
        return
    indices = tuple(value)
    if any(not isinstance(index, int) or index not in JOINTS for index in indices):
        raise ValueError(f"{condition.id}.{key} contains an invalid zero-based joint index")
    if len(indices) != len(set(indices)):
        raise ValueError(f"{condition.id}.{key} contains duplicate joint indices")


def validate_conditions(conditions: Sequence[Condition]) -> None:
    """Raise on count, ID, module, or basic spec-contract violations."""

    if len(conditions) != EXPECTED_TOTAL:
        raise ValueError(f"Expected {EXPECTED_TOTAL} conditions, got {len(conditions)}")

    ids = [condition.id for condition in conditions]
    duplicates = sorted(condition_id for condition_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate condition IDs: {duplicates}")

    module_counts = Counter(condition.module for condition in conditions)
    if dict(module_counts) != dict(EXPECTED_MODULE_COUNTS):
        raise ValueError(
            f"Expected module counts {dict(EXPECTED_MODULE_COUNTS)}, got {dict(module_counts)}"
        )

    for condition in conditions:
        spec = condition.spec
        op = spec.get("op")
        if not isinstance(op, str) or not op:
            raise ValueError(f"{condition.id} has no valid spec.op")
        for key, value in spec.items():
            _validate_joint_indices(condition, key, value)
        if op == "source_mix":
            expected_keys = {"op", "k1_roll_joints", "k2_roll_joints"}
            if set(spec) != expected_keys:
                raise ValueError(f"{condition.id} has invalid source_mix keys: {sorted(spec)}")
        if op == "k1_sign":
            signs = tuple(spec.get("signs", ()))
            if len(signs) != len(JOINTS) or any(sign not in {-1, 1} for sign in signs):
                raise ValueError(f"{condition.id} must contain exactly eight -1/+1 signs")


def canonical_payload(conditions: Sequence[Condition] | None = None) -> dict[str, Any]:
    selected = build_conditions() if conditions is None else tuple(conditions)
    validate_conditions(selected)
    return {
        "schema": SCHEMA,
        "proposal_contract": PROPOSAL_CONTRACT,
        "joint_indexing": {
            "spec": "zero_based_0_to_7",
            "ids_and_descriptions": "one_based_J01_to_J08",
        },
        "expected_module_counts": dict(EXPECTED_MODULE_COUNTS),
        "condition_count": len(selected),
        "conditions": [condition.to_dict() for condition in selected],
    }


def canonical_json(conditions: Sequence[Condition] | None = None) -> str:
    return json.dumps(
        canonical_payload(conditions),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(conditions: Sequence[Condition] | None = None) -> str:
    return hashlib.sha256(canonical_json(conditions).encode("utf-8")).hexdigest()


def _summary(conditions: Sequence[Condition]) -> str:
    counts = Counter(condition.module for condition in conditions)
    lines = [
        f"schema={SCHEMA}",
        f"proposal_contract={PROPOSAL_CONTRACT}",
        f"condition_count={len(conditions)}",
        "module_counts=" + ",".join(f"{module}:{counts[module]}" for module in "ABCD"),
        f"canonical_sha256={canonical_sha256(conditions)}",
        "conditions:",
    ]
    lines.extend(
        f"  {condition.module}  {condition.id}  [{condition.family}]  {condition.description}"
        for condition in conditions
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--format",
        choices=("summary", "json", "pretty", "hash"),
        default="summary",
        help="Output a human summary, canonical compact JSON, pretty JSON, or only the SHA-256 hash.",
    )
    args = parser.parse_args(argv)
    conditions = build_conditions()

    if args.format == "hash":
        print(canonical_sha256(conditions))
    elif args.format == "json":
        print(canonical_json(conditions))
    elif args.format == "pretty":
        payload = canonical_payload(conditions)
        payload["canonical_sha256"] = canonical_sha256(conditions)
        print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        print(_summary(conditions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
