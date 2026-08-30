# Supplementary frozen-policy validation for paper-facing HPR run 2

## Purpose and evidence status

This evaluation-only supplement asks whether the channel and multi-joint
dependencies tested previously in paper-facing HPR runs 0 and 4 are also
observable in paper-facing HPR run 2.  Run 2 is the formal HPR checkpoint whose
internal reproducibility identifier is seed 9203.  No policy is retrained, no
checkpoint is selected after observing intervention outcomes, and no formal
artifact is modified.

Run 2 is not a consistently successful policy under every paired reset: its
locked formal endpoint baseline satisfies the common kinematic rolling criterion
in exactly 7 of 20 resets.  Therefore, paired continuous changes in desired
rotation, forward displacement, and directional consistency are interpreted
together with thresholded success counts.  A thresholded success change alone
is not treated as a complete mechanism result.

## Frozen object and identity gate

- Paper-facing identifier: HPR run 2.
- Internal identifier: formal training seed 9203.
- Endpoint: `checkpoint_1500.pt` only.
- Checkpoint SHA-256:
  `0428d9a86b6622d924738c68fe09df4c6ab922e2a3225a5c24ba41e96eb1c4b8`.
- Deterministic evaluation; 1000 control steps.
- Reset seeds: 20264101--20264120 for every condition.
- Baseline acceptance: exactly 7/20 common-criterion successes; identical
  per-reset success classifications; and absolute error no greater than 1e-6
  for initial body length, forward displacement, forward body lengths, net
  best-fit rotation, desired-direction net rotation, and desired active-rotation
  fraction relative to the locked formal endpoint evaluation.
- Checkpoint-file and in-memory policy hashes must be unchanged after execution.

No intervention result is admissible unless the baseline identity gate passes
first.

## Locked common kinematic endpoint

A rollout is classified as rolling only when all three requirements hold:

1. desired-direction net best-fit body rotation is at least 360 degrees;
2. desired active-rotation fraction is at least 0.70; and
3. forward displacement is at least one initial body length.

No pulse or contact field is imputed.

## Locked 36-condition matrix

The condition identifiers, order, intervention point, and JSON schema are
identical to the accepted runs 0/4 formal freeze study:

1. `BASELINE`;
2. `GLOBAL_K1_OFF`, `GLOBAL_K2_OFF`, `GLOBAL_BOTH_OFF`;
3. `J01_BOTH_OFF`--`J08_BOTH_OFF`;
4. `J01_ONLY`--`J08_ONLY`;
5. `J01_K1_OFF`--`J08_K1_OFF`;
6. `J01_K2_OFF`--`J08_K2_OFF`.

The intervention is applied after deterministic joint-specific actor-network
inference and before the environment step.  Observations and network parameters
remain unchanged.  The full matrix comprises 36 conditions x 20 paired resets =
720 rollouts.

## Planned contrasts

- Global dependence: each global channel intervention versus baseline.
- Joint necessity: each whole-joint ablation versus baseline.
- Single-joint sufficiency: each single-joint-retention condition versus
  baseline.
- Channel localisation: each joint-specific K1 or K2 ablation versus baseline.
- Cross-policy replication: effect-pattern comparison across paper-facing HPR
  runs 0, 2 and 4.

For every contrast, report the condition success count and paired mean changes
in desired revolutions, forward body lengths and desired active-rotation
fraction.  Bootstrap intervals resample the 20 paired reset-level differences
within a frozen policy; they quantify within-policy reset sensitivity and do not
turn the resets into independent training runs.

## Interpretation boundary

The independent training units are paper-facing HPR runs 0, 2 and 4.  The 20
reset episodes are nested paired conditions.  Agreement across these three
policies strengthens cross-training-run replication within the tested formal
configuration, but it does not establish a universal HPR mechanism, hardware
validity, terrain generalisation, or the superiority of independently
parameterised actor networks over parameter sharing.

Because the intervention family is large and this supplement extends an
existing preregistered matrix, joint ranks and individual threshold crossings
are treated as effect-pattern evidence, not as isolated confirmatory discoveries
without multiplicity qualification.
