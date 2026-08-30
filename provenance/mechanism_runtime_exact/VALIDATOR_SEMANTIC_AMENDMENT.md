# Validator Amendment for Pulse-Boundary Semantics

## Observation

During the first post-hoc integrity validation, evaluation seed `20264206` for `seed9201/C00` reported a mismatch between per-pulse `tail_launch_step` values and episode-level `tail_launch_steps`.

## Cause

The frozen evaluator assigns launch events to pulses using the closed interval `previous_pulse_end <= step <= pulse_end`. Adjacent pulses share a boundary step, so one physical launch exactly on that boundary can be recorded in both neighboring pulses. The episode-level field then explicitly stores distinct events using `sorted(set(...))` in the frozen code.

The per-pulse sequence for this episode was:

```text
[7, 144, 144, 342, 432, 675, 944]
```

The episode-level distinct events were:

```text
[7, 144, 342, 432, 675, 944]
```

A complete read-only scan of 5,900 episodes found this valid shared-boundary duplication in 185 episodes. Direct list comparison therefore produced 185 false positives, whereas sorted-set comparison agreed for all 5,900 episodes.

## Amendment

Only this validator comparison:

```python
pulse_launch_steps != episode["tail_launch_steps"]
```

was changed to:

```python
sorted(set(pulse_launch_steps)) != episode["tail_launch_steps"]
```

All checks of pulse start/end steps, temporal links, candidate containment, episode count/Boolean consistency, and all five success thresholds remain intact. No primary result, checkpoint, frozen evaluator, condition matrix, or trajectory file was changed.

After the amendment, `VALIDATION_PASS.json` passed all 5 x 59 x 20 = 5,900 episodes and recorded the current validator source hash.
