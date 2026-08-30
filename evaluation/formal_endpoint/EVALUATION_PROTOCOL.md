# Unified Frozen Checkpoint-1500 Endpoint-Evaluation Protocol for the Six Formal Configurations

> **Final thesis convention:** This file preserves the historical field names used by the protocol executed on 2026-08-13, in which direction-independent rotation span was called the "primary endpoint." The final thesis text and the principal results in this release use only the strict three-gate common-kinematics criterion (net rotation, direction ratio, and forward body-length displacement), for which HPR-O2-JS is 47/100. Follow Section 8 of the repository-root `README.md` for the authoritative command and interpretation. The historical lenient result of 48/100 must not be reported as formal thesis rolling.

This protocol was frozen before formal evaluation began. The evaluation covers six formal controller-reward configurations. Each configuration contains formal runs 0-4 (internal training seeds 9201-9205), for a total of 30 independent training runs. Only `checkpoint_1500.pt` from each run is evaluated; checkpoints are not selected according to evaluation performance.

## Common Rollout Contract

- Every policy uses the same frozen code snapshot, `flat` terrain, and the `legacy_flat` contact implementation.
- Each frozen policy uses reset seeds 20264101-20264120, giving 20 paired initial states.
- Each rollout lasts 1000 control steps; the action is the deterministic location of the policy distribution.
- The three direct-torque configurations restore a one-dimensional bounded active-torque action. The three O2 configurations restore two-dimensional unbounded policy coordinates and then execute them through `K1=100u1`, `K2=100u2`, and the frozen environment feedback law. The evaluator must not interpret direct-torque actions as K1/K2.
- Cross-configuration comparisons use only shared physical trajectory quantities; raw HPR and SGRR rewards are not compared directly.

## Endpoint Hierarchy

The primary rolling-identification endpoint is the direction-independent unwrapped whole-body rotation span:

`A^Theta = max_t Theta_t - min_t Theta_t >= 360 degrees`.

This gate does not use direction, displacement, pulse, or contact conditions. An independent training run is considered to have discovered rolling only if at least 10 of its 20 nested rollouts satisfy this gate.

The secondary strict common-kinematics characterization jointly requires at least 360 degrees of net rotation in the expected direction, an expected-direction step ratio of at least 0.70 among active rotational increments, and forward displacement of at least one initial body length in the expected direction. The active-increment threshold is fixed at 0.05 degrees. This characterization does not use a pulse/contact gate; the SGRR C11 five-gate mechanism endpoint is not part of this unified evaluation.

The common horizontal-motion quantity is:

`scaled signed horizontal progress = 100 * desired-direction COM displacement / 1000`.

It is scaled signed horizontal progress in simulator units, not a percentage and not a calibrated SI velocity.

## Inference Unit and Prespecified Comparisons

The 20 reset rollouts are first aggregated within each policy to produce one run-level estimate. The inference unit for between-configuration comparisons is the five independent training runs per configuration, with formal run index serving as the pairing block. Report all five points, mean+/-SD, median, and range. The five local comparisons use two-sided exact paired sign-flip tests and report the B-A paired difference. At n=5, the smallest attainable two-sided p-value is 0.0625; do not exaggerate its interpretation using a 0.05 threshold.

## Fail-Closed Checks and Artifacts

Before launch, validate all 30 training summaries, contiguous 1,500-row training logs, checkpoint payloads/metadata, finite weights, and frozen source-file SHA-256 digests. Each task must atomically write a JSON validation receipt and an NPZ containing all 20 complete trajectories. For the common R0/Rroll metrics, the existing official 20 x 1000 evaluation must be reproduced within an absolute error of 1e-6. Neither the checkpoint nor the loaded policy-state hash may change during evaluation. Existing incompatible artifacts must not be overwritten; after interruption, a completed task may be reused only if it passes the full signature and hash checks.

The physical environment, checkpoint restoration, action selection, and legacy common metrics always come from the original formal code snapshot. The direction-independent rotation span calls only the independently boundary-tested pure functions `_best_fit_rotation` and `_rotation_excursion_metrics`. Their frozen file path and SHA-256 digest are recorded in the configuration and final receipt; the evaluator does not import that file's environment or K1/K2-specific evaluation logic.
