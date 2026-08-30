"""Single canonical contract for the R0_V21core_abs360_v1 reward."""

from __future__ import annotations

import hashlib
import json


REWARD_ID = "r0_v21core_abs360_v1"

REWARD_CONTRACT = {
    "version": "frozen_named_definition_v1",
    "reward": "100*vx+400*dM+1600*q*dE/(2*pi)",
    "launch_inputs": (
        "uL=clip(tail_lift/.20),uF=clip(tail_forward/.10),"
        "uC=clip(ordered_curl/.12),"
        "uH=sigmoid(clip((head_contact-.50)/.10,-20,20))"
    ),
    "launch_core": (
        "0.70*min(uL,uF,uC)+0.30*harmonic_mean_eps1e-4(uL,uF,uC)"
    ),
    "prepare_score": (
        "episode_latched_physical_contact(clearance<=max(1e-4*body_length,1e-5))"
        "*clip((uH*launch_core-.02)/.98,0,1)"
    ),
    "dM": "new per-episode prepare-score high-water progress",
    "rotation": (
        "E=max_history(abs(unwrapped_centered_Procrustes_angle)); "
        "dE=clip(E-Eprev,0,10deg)"
    ),
    "q": ".25+.75*clip(zero_filled_mean20(max(vx,0))/.02,0,1)",
    "phase_or_milestone": "none",
}


def contract_sha256(contract: object = REWARD_CONTRACT) -> str:
    canonical = json.dumps(
        contract,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


REWARD_CONTRACT_SHA256 = contract_sha256()
