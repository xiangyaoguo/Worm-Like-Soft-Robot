# obs2 v2.1 Per-K Causal Completion (Frozen Checkpoints, Read-Only Evaluation)

This directory completes the evidence needed to determine the causal role of every joint's K1 and K2 in rolling. It reads only the formal experiment's `checkpoint_1500.pt` files. It never trains or resumes training, modifies a checkpoint, or writes to the formal experiment or pre-existing mechanism-study directories.

## Frozen Experimental Contract

- Formal policy seeds: 9201-9205.
- Observations remain `[delta_theta, theta_dot]` per joint; actions remain `[K1, K2]` per joint.
- Policies use deterministic CPU inference. Physics, PPO, environment, torque equation and clipping, the 1,000-step endpoint, and the five-part joint success threshold remain unchanged.
- Calibration set: 20264101-20264120, used only to generate a `[1000,8,2]` two-channel temporal template and `[8,2]` static mean from C11 policy actions, never to estimate intervention effects.
- Main causal-evaluation set: 20264401-20264420, disjoint from calibration.
- The independent units are five independently trained seeds. Each seed's 20 shared initial states support paired comparisons and are not represented as 100 independent policies.

`study_contract.json` froze the following adjudication rules before study results were observed:

- Necessary contribution: at least a 30-percentage-point success-rate drop from C11 and degradation in at least 4/5 training seeds.
- Strong necessity: at least a 50-percentage-point drop and degradation in at least 4/5 training seeds.
- Timing-critical: static-mean or fixed time-permutation intervention reduces success from C11 by at least 30 percentage points and degrades at least 4/5 seeds.
- Single-channel or joint-pair sufficiency: at least a 30-percentage-point gain over C00, with at least 3/5 seeds reaching `>=10/20` successes.
- `Equivalent/redundant` is allowed only when success is within +/-5 percentage points of the reference, all primary continuous metrics are within +/-0.2 reference SD, and there is no consistent seed-level degradation.

## 113 Frozen Conditions

1. `C11`: full Rroll K1+K2 baseline (one condition).
2. On the full C11 background, each of 16 individual K channels is separately zeroed, scaled by 0.5, scaled by 1.5, sign-flipped, replaced by its calibration static mean, or subjected to a fixed calibration time permutation (16 x 6 = 96 conditions).
3. On the full C11 background, both K1+K2 are zeroed together at each joint to test joint-pair necessity (eight conditions).
4. On the full C00/R0 background, Rroll K1+K2 are transplanted together at each joint to test joint-pair sufficiency (eight conditions).

The main evaluation therefore contains `113 x 5 x 20 = 11,300` episodes. The separate two-channel calibration contains `5 x 20 = 100` episodes and does not enter causal conclusions.

## Pre-Run Technical Smoke Test

This study depends on TorchRL from the project virtual environment. The following PowerShell commands show the validated entry point; creating this README did not launch an evaluation:

```powershell
$env:PYTHONPATH='C:\Users\PUBLIC_USER\CloudStorage\Desktop\finalproject\job\RLMetamaterialLocomotion-main\RLMetamaterialLocomotion-main\.venv\Lib\site-packages'
$py='C:\Users\PUBLIC_USER\AppData\Local\Programs\Python\Python311\python.exe'
& $py '.\run_causal_completion.py' --stage verify
& $py '.\run_causal_completion.py' --stage smoke --training-seed 9201
```

`verify` checks SHA-256 for this script, the configuration, contract, reused source, frozen evaluator, and ten checkpoints, then prints all 113 normalized conditions. `smoke` uses only dedicated initial state 20264601, writes to `smoke/`, and explicitly marks the output as nonscientific.

## Formal Read-Only Reevaluation

After smoke testing passes, one command can generate calibration for all five seeds in parallel and then run the main evaluation for all five seeds:

```powershell
& $py '.\run_causal_completion.py' --stage all --workers 5
```

Stages can also be run separately for easier diagnosis:

```powershell
& $py '.\run_causal_completion.py' --stage calibration --training-seed 9201
& $py '.\run_causal_completion.py' --stage main --training-seed 9201
& $py '.\run_causal_completion.py' --stage summarize
```

`--stage main` accepts repeated `--condition CONDITION_ID` arguments for a contract-constrained subset. An existing result is skipped only when its study, seed, normalized-condition hash, and episode count all match. Incompatible files cause a fail-closed exit and are never overwritten to imitate a valid resume.

## Evidence Outputs

- `calibration/seed*.npz`: two-channel action templates, static means, and physical-K versions; companion JSON receipts record initial states, shapes, hashes, and checkpoint immutability.
- `results/seed*/CONDITION.json`: 20 episodes per condition, frozen-evaluator metrics, joint five-part success labels, per-joint K/torque/power-boundary statistics, and checkpoint/policy/evaluator hashes.
- `traces/seed*/`: all 20 C11 trajectories and the first trajectory for every other condition. Every step records observations, R0/Rroll/applied action, physical K, unclipped K1/K2 torque terms, total and clipped torque, exact two-channel Shapley torque/power-boundary decomposition, position, support, contact, and fitted rotation.
- `analysis/episode_results.csv`: unified episode-level long table and five-part failure decomposition.
- `analysis/condition_summary.csv/json`: cross-seed condition summary; `complete=true` only when all 113 conditions and 11,300 episodes are present.
- `progress/`: auditable progress and final immutability receipts for each training seed.

## Interpretation Boundary and Implementation Risks

1. K1/K2 torque and power are auditable control-boundary proxies computed as `clip(K1*delta_theta + K2*theta_dot, -9,9)`. The original environment also has physics substeps within each control step, so these boundary proxies are not exact per-substep actuator power.
2. Static and permuted interventions alter the closed-loop state distribution. Their results are the total effect of the channel's time-varying closed-loop rule, not a controlled direct effect of changing only the current step.
3. Only the first episode trajectory is stored for each nonbaseline condition to localize propagation. Formal success rates and continuous endpoint effects must use all 20 episodes.
4. Failure of a single-joint K1+K2 transplant to create rolling rejects only independent sufficiency of that joint pair on the C00 background; it does not reject multijoint cooperative sufficiency.
5. On Windows, every worker loads the environment and both policy sets separately. `--workers 5` reduces latency but uses more memory; reducing it to two or three does not change the scientific contract.
