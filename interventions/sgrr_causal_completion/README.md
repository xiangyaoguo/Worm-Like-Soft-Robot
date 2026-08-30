# obs2 v2.1 Per-K Causal Completion (Frozen Checkpoints, Read-Only Evaluation)

> **Portable entry point:** The author-machine absolute paths later in this document are retained only for historical auditing. On a new machine, run `python run.py intervene-sgrr --component causal --stage verify` from the repository root and then follow Section 12 of the root `README.md`; paths are supplied by `configs/paths.local.json`.

This directory completes the evidence needed to determine the causal contribution of each joint's K1/K2 to rolling. It only reads `checkpoint_1500.pt` from the formal experiments. It does not train, resume training, modify checkpoints, or write to the formal-experiment directory or the existing mechanism-study directory.

## Frozen Experimental Contract

- Formal policy seeds: 9201-9205.
- Observations permanently remain `[delta_theta, theta_dot]` per joint, and actions permanently remain `[K1, K2]` per joint.
- Policies are CPU deterministic; physics, PPO, environment, torque formula and clipping, the 1000-step endpoint, and the five joint success gates are unchanged.
- Calibration set: 20264101-20264120. It is used only to generate `[1000,8,2]` two-channel temporal templates and `[8,2]` static means from C11 policy actions, not to estimate intervention effects.
- Main causal-evaluation set: 20264401-20264420, disjoint from the calibration set.
- The inference unit is the five independent training seeds. The same 20 initial states within each seed provide paired comparisons and must not be presented as 100 independent policies.

`study_contract.json` froze the decision rules before the study results were observed:

- Necessary contribution: success rate falls by at least 30 percentage points relative to C11, with deterioration in at least 4/5 training seeds.
- Strong necessity: success rate falls by at least 50 percentage points, with deterioration in at least 4/5 training seeds.
- Temporal criticality: a calibrated static mean or fixed time permutation lowers success by at least 30 percentage points relative to C11, with deterioration in at least 4/5 seeds.
- Single-channel/joint-pair sufficiency: at least a 30-percentage-point improvement relative to C00, with at least 3/5 seeds attaining `>=10/20` successes.
- A condition may be called "equivalent/redundant" only if its success rate is within +/-5 percentage points of the reference, every primary continuous metric is within +/-0.2 SD of the reference, and there is no consistent per-seed deterioration.

## 113 Frozen Conditions

1. `C11`: complete Rroll K1+K2 baseline (1 condition).
2. In the complete C11 background, each of the 16 individual K channels receives six interventions: zero, 0.5 x, 1.5 x, sign flip, calibrated static mean, and calibrated fixed time permutation (16 x 6 = 96 conditions).
3. In the complete C11 background, K1+K2 are simultaneously set to zero at each joint to test joint-pair necessity (8 conditions).
4. In the complete C00/R0 background, Rroll K1+K2 are transplanted simultaneously at each joint to test joint-pair sufficiency (8 conditions).

The total is `113 x 5 x 20 = 11,300` main evaluation episodes. Two-channel calibration adds `5 x 20 = 100` episodes, which do not contribute to the causal conclusions.

## Technical Smoke Tests Before Execution

This study depends on TorchRL in the project virtual environment. The recommended author-machine commands use the project's installed dependencies. They only document the execution entry point; no evaluation was started when this README was created:

```powershell
$env:PYTHONPATH='C:\Users\PUBLIC_USER\CloudStorage\Desktop\finalproject\job\RLMetamaterialLocomotion-main\RLMetamaterialLocomotion-main\.venv\Lib\site-packages'
$py='C:\Users\PUBLIC_USER\AppData\Local\Programs\Python\Python311\python.exe'
& $py '.\run_causal_completion.py' --stage verify
& $py '.\run_causal_completion.py' --stage smoke --training-seed 9201
```

`verify` checks the SHA-256 digests of this script, its configuration, the contract, reused source, frozen evaluator, and ten checkpoints, then prints the 113 normalized conditions. `smoke` runs only one dedicated initial state (20264601), writes to `smoke/`, and explicitly labels the output as nonscientific.

## Formal Read-Only Reevaluation

After smoke tests pass, one command first generates calibration data for the five seeds in parallel and then executes the five main seed evaluations in parallel:

```powershell
& $py '.\run_causal_completion.py' --stage all --workers 5
```

The stages may also be run separately for easier troubleshooting:

```powershell
& $py '.\run_causal_completion.py' --stage calibration --training-seed 9201
& $py '.\run_causal_completion.py' --stage main --training-seed 9201
& $py '.\run_causal_completion.py' --stage summarize
```

`--stage main` may receive repeated `--condition CONDITION_ID` arguments for a contract-constrained subset run. An existing result is skipped only when its study, seed, condition-specification hash, and episode count all match exactly. An incompatible file causes a fail-closed stop and is never overwritten to masquerade as a valid resumed result.

## Output Evidence

- `calibration/seed*.npz`: two-channel action templates, static means, and physical-K versions. The corresponding JSON receipts record initial states, shapes, file hashes, and checkpoint immutability.
- `results/seed*/CONDITION.json`: frozen-evaluator metrics for the 20 episodes of each condition, five-gate joint success, per-joint K/torque/power-boundary statistics, and checkpoint/policy/evaluator hashes.
- `traces/seed*/`: all 20 C11 trajectories and the first episode of every other condition. Each step contains observations, R0/Rroll/applied actions, physical K, unclipped K1/K2 torque terms, total torque and clipping, exact two-channel Shapley torque/power-boundary decompositions, position, support, contact, and fitted rotation.
- `analysis/episode_results.csv`: unified per-episode long table and five-gate failure decomposition.
- `analysis/condition_summary.csv/json`: cross-seed condition summary. `complete=true` only when all 113 conditions and 11,300 episodes are present.
- `progress/`: auditable progress and final immutability receipts for each training seed.

## Interpretation Boundaries and Implementation Risks

1. K1/K2 torque and power are reproducible control-boundary proxies computed as `clip(K1*delta_theta + K2*theta_dot, -9,9)`. The original environment has physics substeps within each control step; do not present the boundary proxy as exact actuator power at every substep.
2. Static/permuted interventions alter the closed-loop state distribution. Their results are therefore the total effect of that channel's time-varying closed-loop rule, not a controlled direct effect that changes only the current step.
3. For each nonbaseline condition, a step-by-step trace is stored only for the first episode and is used to locate the propagation chain. Formal success rates and continuous endpoint effects must use all 20 episodes.
4. If transplanting one joint's K1+K2 is insufficient to produce rolling, this rejects only "independent sufficiency of one joint pair in the C00 background"; it does not reject sufficiency through multijoint coordination.
5. Under Windows multiprocessing, each worker loads the environment and both policies separately. `--workers 5` lowers latency but uses more memory; it may be reduced to 2-3 without changing the scientific contract.
