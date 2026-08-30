# Frozen paired O1-sham/O2 evaluator

This evaluator is frozen before the five formal O1-sham training runs.  It
evaluates the archived HPR/O2 runs and the paired HPR/O1-sham extension with
the same deterministic protocol.

## Locked matrix

- Internal training seeds: 9201--9205; paper-facing run IDs: 0--4.
- Checkpoints: batches 100, 200, ..., 1500.  Batch 1500 is the primary
  endpoint; earlier checkpoints describe discovery trajectories only.
- Reset seeds: 20264101--20264120, paired across every arm, run and checkpoint.
- Rollout duration: 1,000 control steps on the frozen flat/legacy-flat model.
- Common rolling criterion: desired net best-fit rotation >= 360 degrees,
  desired active-rotation fraction >= 0.70, and forward displacement >= one
  initial body length.  No pulse or contact gate is used.

## Observation intervention

The raw environment and controller always retain the complete observation
`[s_i, theta_dot_i]`.  For a checkpoint whose immutable metadata records
`actor_observation_mode=spatial_only_sham`, the evaluator creates an actor-only
TensorDict copy and changes only its second channel to exact zeros.  The
unmodified TensorDict is used for environment stepping and torque
reconstruction.  For archived O2 checkpoints, the actor receives the complete
two-channel observation.

The evaluator refuses an O1-sham checkpoint if its metadata omits or changes
the observation-mode fields.  The only legacy inference permitted is for an
archived parent HPR/O2 checkpoint with the exact locked run name and contract.

## Exports

Every arm/run/checkpoint task writes one atomic JSON manifest and one compressed
NPZ trace archive.  The archive contains all 20 trajectories, observations,
actor inputs, actions, physical K1/K2 gains, active-torque components, clipped
active torque, saturation flags, support index and ground-contact strength.
These fields support endpoint comparisons, discovery curves, trajectory
montages, gain/torque heat maps, phase portraits and saturation diagnostics
without rerunning a selected subset.

For archived O2 checkpoint 1500, the evaluator must reproduce the original
formal endpoint metrics for every reset within absolute tolerance 1e-6.  A
failure blocks the whole comparison.

## Commands

Contract scan only (default; no rollouts):

```powershell
python frozen_evaluator.py --config evaluator_config.json --contract-only
```

Complete locked evaluation after all five O1-sham runs have completed:

```powershell
python frozen_evaluator.py --config evaluator_config.json --execute --workers 2
```

`--execute` always means the complete 2 x 5 x 15 x 20 matrix.  There is no
command-line mechanism for changing the seeds, checkpoints, rollout length or
criterion.  Existing incompatible output is never overwritten.

