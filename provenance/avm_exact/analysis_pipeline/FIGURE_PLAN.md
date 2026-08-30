# Comprehensive figure plan (not limited by the thesis figure allowance)

The pipeline deliberately produces a broad evidence library. A smaller subset can be
selected for the thesis only after results are known. No figure is selected on the
basis of whether it favours O2.

## Protocol figures (available before training)

1. **P00 Study design and provenance.** Archived O2-HPR and newly trained O1-sham
   branches, paired by formal initialisation and brought into one frozen evaluator.
2. **P01 Capacity-matched information ablation.** The actor-only angular-velocity mask,
   full O2 shared centralised critic, and invariant two-channel physical controller.
3. **P02 Confirmatory evidence hierarchy.** Initialization gate, training gate,
   checkpoint evaluation, endpoint primary analysis, sensitivity and mechanism layers.

## Required result figures

1. **R01 Completeness matrix.** Arm x run x checkpoint coverage and 20-reset counts.
2. **R02 Initialization pairing audit.** Actor, critic, optimizer and RNG hash equality
   for each matched run. This is a QA figure, not an outcome figure.
3. **R03 Training HPR trajectories.** Raw and predeclared 50-batch rolling means, with
   each matched run shown separately.
4. **R04 Training speed trajectories.** Same layout as R03; this guards against a reward
   logging artefact.
5. **R05 PPO diagnostics.** Approximate KL and completed PPO updates by run and arm.
6. **R06 Primary endpoint paired success.** Success count out of 20 at checkpoint 1500,
   with one line per matched formal run.
7. **R07 Endpoint paired effect plot.** O2 minus O1-sham success proportion for each
   run; the five run-level effects are the inferential units.
8. **R08 Endpoint episode outcome matrix.** All 200 paired endpoint decisions, arranged
   by run and reset seed; colour is paired with a success/failure symbol.
9. **R09 Endpoint kinematic components.** Net desired rotation, direction fraction and
   forward body lengths as run-faceted paired episode distributions.
10. **R10 Criterion geometry.** Desired rotation versus direction fraction, marker size
    proportional to forward displacement; common thresholds are drawn explicitly.
11. **R11 Rolling-discovery trajectories.** Success count at every checkpoint for each
    paired run, without choosing a favourable intermediate checkpoint.
12. **R12 Checkpoint success heat maps.** Arm-specific run x checkpoint matrices.
13. **R13 Discovery and persistence.** First checkpoint meeting >=10/20 and the number
    of later checkpoints retaining that threshold.
14. **R14 Aggregate criterion sensitivity.** The full 3 x 3 x 3 threshold grid for both
    arms and their paired difference.
15. **R15 Run-specific criterion sensitivity.** Difference surfaces for runs 0-4, to
    disclose whether an aggregate result is driven by one training run.
16. **R16 Reward-behaviour alignment.** Checkpoint HPR return against independently
    measured rolling success, labelled by run and checkpoint.
17. **R17 Joint-wise gain maps.** Mean absolute K1 and K2 by actor arm, formal run and
    joint at the endpoint.
18. **R18 Torque saturation maps.** Saturated-step fraction by arm, run and joint.
19. **R19 Controller-component balance.** Distributions of `K1*s` and `K2*theta_dot`
    contributions, showing whether observation removal changes use of the preserved K2
    physical channel.
20. **R20 Representative time series.** Rotation, displacement, direction fraction,
    K1/K2 contributions and torque saturation for a predeclared reset.
21. **R21 COM trajectories.** Paired x-y trajectories for the same reset, with common
    axes and start/end labels.
22. **R22 Morphology/contact montage.** Ten fixed normalized times for the same paired
    reset. Node positions are data, not artist reconstruction.
23. **R23 Actor angular-velocity probe.** K1 and K2 output versus angular velocity at
    fixed spatial-difference values for each joint and run.
24. **R24 Sham invariance audit.** O1-sham outputs must be constant across the angular-
    velocity sweep; any non-zero range is a hard implementation failure.
25. **R25 Exact five-pair randomisation distribution.** All 32 sign flips of the mean
    paired endpoint effect, with the observed statistic marked. This explains why five
    pairs cannot provide a two-sided p-value below 0.0625.

## Optional diagnostic images

- representative rolling and non-rolling episode contact sheets for every run;
- one video per arm/run for the fixed reset used in R20-R22;
- per-joint phase portraits (`spatial_difference`, `angular_velocity`);
- checkpoint-to-checkpoint gain-map differences;
- actor-output correlation matrices;
- bootstrap stability displays labelled strictly as descriptive/nested;
- failed-criterion decomposition (rotation-only, direction-only, displacement-only);
- wall-clock throughput, GPU memory and checkpoint integrity diagnostics.

## Predeclared representative episode

R20-R22 use paired reset block 1, not the most visually impressive rollout. Paper-facing
figures display only paired reset labels 1–20; the internal reset-seed mapping remains in
the source manifest. If that rollout is corrupt, the whole corresponding evaluation is
rerun; it is not replaced by a favourable episode.
