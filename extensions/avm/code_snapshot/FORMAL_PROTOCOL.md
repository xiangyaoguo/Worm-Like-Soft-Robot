# Formal HPR O1-sham observation-ablation extension

Status: **implementation and pre-registration draft; full training is not authorised by this file**.

## Scientific contrast

This extension reuses the five archived HPR/O2 training runs and adds five
from-scratch, seed-paired HPR/O1-sham runs.  Paper-facing run identifiers are
`run 0`--`run 4`; the internal reproducibility seeds remain `9201`--`9205`.

For joint `i`, the archived O2 actor input is `[s_i, theta_dot_i]`, whereas the
new O1-sham actor input is `[s_i, 0]`.  The tensor shape remains two, so actor
capacity and parameter count are unchanged.  The shared centralised critic
continues to receive the unmodified O2 observation, and the simulator continues
to apply the real `K2 * theta_dot` term in the physical torque law.

The implementation installs a parameter-free forward pre-hook on the actor
backbone only.  It deliberately does **not** set `feedback_gain` to zero and
does not modify the archived environment source.

## Locked design

- Parent study: `obs2_reward_only_roll_reproduction_v2_1_formal_20260803_r2`.
- Parent HPR arm: internally named `R0`, reward function `horizontal_speed`.
- New arm: `HPR_O1_sham`.
- Seeds: `9201, 9202, 9203, 9204, 9205`, mapped in the paper to runs `0--4`.
- Robot: crawler, ten particles, eight controlled joints, flat terrain,
  `legacy_flat` contact.
- Controller: independently parameterised joint-specific actor networks,
  shared centralised critic, per-joint `K1/K2`, scale 100, torque clip 9.
- PPO and training budget: unchanged from the parent configuration; 1,500
  collection batches, 10,000 frames per batch, checkpoint every 100 batches.
- No pretraining, warm start, resume, behaviour cloning, parameter anchoring,
  checkpoint selection, seed replacement, or post-hoc budget extension.
- Maximum training concurrency: two processes on CUDA device 0.

## Fail-closed gates

Before collector iteration, every new run must exactly match its paired parent
HPR run for all seven fields in `pair_hash_bundle`: actor, critic, optimiser,
Torch CPU RNG, Torch CUDA RNG, NumPy RNG, and Python RNG.  The actor and critic
parameter counts must also match.  A mismatch writes a failed audit and exits
before training.

The source files and parent initialisation audits are pinned by SHA-256 in
`extension_config.json` and `reference_initializations.json`.  The launcher
refuses to overwrite any run or audit directory.  Full training additionally
requires a frozen code manifest, five successful preflight receipts, and an
explicit `FORMAL_APPROVAL.json` bound to that manifest.

## Storage contract

Only run products are written into a segregated extension subtree of the parent
formal tree.  The original ten-arm `formal/runs` directory is never modified:

```text
C:\Users\PUBLIC_USER\CloudStorage\Desktop\finalproject\job\roll_learning\
  obs2_roll_repro_v2_1_formal_20260803_r2\formal\extensions\
    formal_o1_sham_hpr_20260811\
      runs\formal__seed92XX__HPR__O1sham\
      initialization\formal__seed92XX__HPR__O1sham.json
```

All extension control records, preflights, logs, receipts, analysis, figures,
tables and reports are written under:

```text
E:\finalproject\result\formal_hpr_o1_sham_vs_o2_20260811
```

## Required execution order

1. Run `verify_extension.py` and `test_actor_observation_sham.py`.
2. Run launcher phase `freeze`.
3. Run launcher phase `preflight` for all five seeds.
4. Independently inspect the frozen manifest and preflight receipts.
5. Create the explicit approval marker described in the launcher README.
6. Run launcher phase `launch`; never run the trainer directly.
7. Evaluate checkpoint 1500 with the locked common kinematic criterion.

The historical O1/warm-start development runs are excluded from the formal
comparison.
