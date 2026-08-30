# v2.1 reward-only formal protocol

## Scope

- Study: `obs2_reward_only_roll_reproduction_v2_1_formal_20260803_r2`
- This is a preserved, from-batch-0 technical relaunch. The original formal root failed before any training process, log, or checkpoint was created because CloudStorage temporarily denied an atomic state-file replacement.
- New training seeds: `9201` through `9205`
- Arms per seed: `R0=horizontal_speed`, `Rroll=obs2_roll_repro_v2_1`
- Ten runs total, each exactly 1500 batches and starting from batch 0.
- Within each seed, R0 and Rroll run concurrently as one fixed pair. Pairs run serially.
- Maximum concurrent training processes: 2.

## Permanently frozen interface

- Observation per joint: `[dth_tot, theta_dot]`.
- Action per joint: `[K1, K2]`.
- Eight independent per-joint actors and one shared centralised critic.
- Torque: `clip(K1*dth_tot + K2*theta_dot, -9, 9)`.
- Rolling, tail-roll, and fast-forward observation additions remain disabled.
- PPO, physics, terrain, initialisation, network, action scale, and event thresholds remain unchanged.
- No pretrained model, resume state, teacher, BC, anchor, input expansion, pilot checkpoint, regression checkpoint, or historical checkpoint may be loaded.

## Frozen reward

The Rroll arm uses the regression-approved `obs2_roll_repro_v2_1`. Relative to v2, its only semantic change is the rolling-phase motion-quality coefficient `0.08 -> 0.16`. Preparation-phase coefficients and all thresholds remain unchanged.

## Pairing and concurrency

- Both arms of a seed use the same seed and must have identical batch-0 actor, critic, optimizer, Torch CPU/CUDA RNG, NumPy RNG, and Python RNG hashes.
- A mismatch is a non-repairable contract failure and stops the experiment.
- A technical failure may trigger at most one whole-pair retry. Both arms restart from batch 0 in new attempt directories; failed attempts are preserved. Checkpoint resume is forbidden.
- Scientific failure or a weak curve never triggers retry, reward changes, seed replacement, extension, or intermediate-checkpoint selection.

## End point and evaluation

- Checkpoints are saved every 100 batches. Checkpoints 300/600/900/1200 are diagnostic only.
- The sole primary training endpoint is checkpoint 1500.
- Every endpoint is evaluated deterministically on CPU for 20 episodes of 1000 steps with base seed 20264101.
- An episode succeeds only when all five conditions hold: pulse count >= 4, desired net rotation >= 360 degrees, direction fraction >= 0.70, forward displacement >= 1 body length, and mean inter-pulse interval <= 250 steps.
- A training seed succeeds at >= 10/20 Rroll episodes. Three of five successful seeds means reproducible; four of five means robust.

## Evidence and failure handling

- State, events, source hashes, launcher receipt, approval marker, batch-0 audits, pair receipts, checkpoint hashes, logs, raw evaluation JSON, and the final result are retained.
- Evaluation may be retried once only against the same immutable endpoint and fixed episode seeds after a technical evaluator failure.
- No evaluator, reward, observation, action, PPO, physics, seed, or success threshold may be changed after formal execution starts.
