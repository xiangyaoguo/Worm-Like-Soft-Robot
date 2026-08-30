# Acceptance Report for the Unified Frozen Checkpoint-1500 Endpoint Evaluation of Six Configurations

> **Historical acceptance report, not the final thesis metric definition.** The term "primary endpoint" below retains the historical direction-independent rotation-span naming. The final thesis main table reads only the strict three-gate common-kinematics fields, for which HPR-O2-JS is 47/100. Run `python run.py evaluate ...` from the repository root, and treat Section 8 of the root `README.md` as the sole usage guide. The old-machine absolute path below records provenance only and is not a portable command; its Chinese directory names have been translated for this English copy.

## 1. Acceptance Conclusion

The shared checkpoint-1500 endpoint evaluation of the six formal controller-reward configurations has been completed and passed the final validation receipt. The formal matrix consists of 6 configurations x 5 independent training runs x 20 paired initial states x 1000 control steps: 30 frozen policies, 600 nested rollouts, and 600,000 control steps. The final status is `PASS`, with no failed checks.

The formal output root was:

`C:\Users\PUBLIC_USER\Documents\Graduate-Thesis-Project\05_Experimental-Data-and-Code\formal_six_config_endpoint_eval_20260813\evaluation_final`

Key completion evidence:

- `EVALUATION_COMPLETE.json`: 30/30 tasks, 600/600 rollouts, and `complete_locked_matrix=true`.
- `FINAL_VALIDATION.json`: all 17 final checks are true.
- `STUDY_MANIFEST.json`: the complete task set, configuration summaries, local comparisons, and hashes of all artifacts.
- Thirty per-run JSON validation receipts and 30 full-trajectory NPZ files.
- `episode_results.csv` with 600 rows, `run_results.csv` with 30 rows, and `configuration_summary.csv` with 6 rows.
- `paired_run_differences.csv` with 25 rows and `pairwise_local_comparisons.csv` with 25 rows.

The complete formal execution took 791.81 s. The 30 NPZ files total 146,107,971 bytes. No `.tmp` files remain.

## 2. Frozen Evaluation Contract

- Checkpoint: every run uses `checkpoint_1500.pt`; models are not selected according to evaluation results.
- Reset seeds: 20264101-20264120, paired exactly across all six configurations.
- Replay: CPU, deterministic location of the action distribution, flat/legacy-flat, and 1000 control steps per rollout.
- Primary endpoint: direction-independent unwrapped whole-body rotation span `A^Theta >= 360 degrees`.
- Run-level rolling discovery: at least 10/20 rollouts from the same frozen policy satisfy the primary endpoint.
- Secondary strict common-kinematics characterization: at least 360 degrees of net rotation in the expected direction, an expected-direction step ratio of at least 0.70 among active rotational increments, and forward displacement of at least one initial body length.
- Common horizontal endpoint: `100 x desired-direction COM displacement / 1000`, called scaled signed horizontal progress in the thesis text, not a percentage or SI velocity.
- Inference unit: five independent training runs per configuration; the 20 rollouts measure only within-policy repeatability.

The one-dimensional actor actions of the three direct configurations are interpreted as physical active torques. The two-dimensional actions of the three formula configurations are interpreted as `K1=100u1` and `K2=100u2`. The unified evaluator did not mislabel direct actions as normalized actions or as K1/K2.

## 3. Completeness and Identity Gates

Final acceptance confirmed that:

- All 30 runs across the six configurations have a complete summary, contiguous training logs from 1-1500, and a checkpoint.
- Policy and critic weights are finite in all 30 checkpoints.
- None of the 600 `(configuration, formal run, reset seed)` identities is duplicated or missing.
- All primary trajectory arrays contain finite values. Protocol-permitted NaNs occur only in optional support/contact diagnostics not exported by the legacy environment.
- Checkpoint SHA-256 and policy-state hashes did not change before versus after evaluation.
- Across the six configurations and 20 reset seeds, the maximum absolute difference in initial positions is 0.
- Repeated calls to each of the six deterministic policies produce actions identical element by element.
- The formally imported environment file is the frozen nested environment file, with SHA-256 `1f0335b65dab64569ddba7536df4a0ae2ab31c54904d0a041a68741988d32900`.
- Across the ten R0/Rroll runs, the maximum per-rollout absolute error for all six common metrics against the existing official 20 x 1000 evaluation is 0, better than the prespecified `1e-6` threshold.

The evaluator SHA-256 is `3a6607051b5d03e6f89e25daf1876cd41bec5d49cdfad5578cc73c425239de38`; the frozen configuration SHA-256 is `8a8daaa8bb72cf8d57ba8ef9ee286d59c570279dfcf38192b2b580c588242336`; and the formal protocol SHA-256 is `dd22dad857d2cdc0f1e1b0dcbd5e724f45ef7b7c9dd84898402c27c6aa1e6985`.

The first parallel launch failed closed at 0/30 completed tasks because of the environment-path identity gate; it produced no formal task result. The cause, fix, and old launch-level record are preserved in `PRELAUNCH_GUARD_NOTE.md` and the old `evaluation/` directory. After the fix, separate subprocesses completed one-step smoke tests for direct and formula policies. Only results under `evaluation_final/` are formal.

## 4. Run-Level Results for the Six Configurations

For the horizontal endpoint below, the mean is first taken across the 20 rollouts of each policy and then summarized across the five independent training runs of each configuration. `x/100` is a nested-rollout count, not 100 independent samples.

| Formal configuration | Scaled signed horizontal progress, run-level mean +/- SD | Median [range] | Runs discovering primary endpoint | Primary-endpoint nested rollouts | Secondary-strict runs | Secondary-strict nested rollouts |
|---|---:|---:|---:|---:|---:|---:|
| HPR-DTH-PS | 0.5275 +/- 1.1710 | 0.0051 [0.0000, 2.6224] | 0/5 | 0/100 | 0/5 | 0/100 |
| HPR-THDOT-PS | 1.1514 +/- 1.5748 | 0.0051 [-0.0002, 2.9280] | 0/5 | 0/100 | 0/5 | 0/100 |
| HPR-OBS-PS | 1.0008 +/- 1.3756 | 0.0000 [0.0000, 2.6721] | 0/5 | 0/100 | 0/5 | 0/100 |
| HPR-O2-PS | 2.8350 +/- 0.0922 | 2.8635 [2.6728, 2.8997] | 0/5 | 0/100 | 0/5 | 0/100 |
| HPR-O2-JS | 2.0618 +/- 0.5363 | 1.7536 [1.5868, 2.8163] | 2/5 | 48/100 | 2/5 | 47/100 |
| SGRR-O2-JS | 2.9521 +/- 0.4824 | 2.9600 [2.4263, 3.5278] | 5/5 | 99/100 | 5/5 | 99/100 |

The descriptive ordering is SGRR-O2-JS, HPR-O2-PS, HPR-O2-JS, HPR-THDOT-PS, HPR-OBS-PS, and HPR-DTH-PS. This ordering must not be interpreted as a single "channel effect" because the six configurations also differ in observation representation, action parameterization, actor parameter sharing, and reward. Formal inference uses only the five prespecified local comparisons below.

## 5. Five Prespecified Local Comparisons

Differences are consistently defined as B-A. The table reports the five run-level paired differences in scaled signed horizontal progress. For the two-sided exact sign-flip test, n is 5 and the smallest attainable p-value is 0.0625. The formal table actually calculates tests separately for five types of run-level outcomes in each of the five prespecified comparisons, giving 25 tests. No Holm or other multiple-comparison correction was applied. The thesis text must not describe the table below as "all tests" or recast the uncorrected p-values as evidence of significance.

| Prespecified comparison A -> B | B-A mean +/- SD | Median [range] | Exact p |
|---|---:|---:|---:|
| HPR-DTH-PS -> HPR-THDOT-PS | +0.6239 +/- 2.3158 | 0.0000 [-2.6225, 2.9280] | 0.5000 |
| HPR-DTH-PS -> HPR-OBS-PS | +0.4732 +/- 1.0391 | -0.0051 [-0.0051, 2.3316] | 0.5000 |
| HPR-OBS-PS -> HPR-O2-PS | +1.8342 +/- 1.4154 | +2.8568 [0.2276, 2.8820] | 0.0625 |
| HPR-O2-PS -> HPR-O2-JS | -0.7732 +/- 0.5238 | -0.9574 [-1.2953, -0.0472] | 0.0625 |
| HPR-O2-JS -> SGRR-O2-JS | +0.8903 +/- 0.7958 | +0.8870 [-0.2938, 1.8124] | 0.1250 |

For the primary lenient rolling success rate, the mean B-A run-level paired differences and exact p-values are: DTH->THDOT 0.00, p=1.00; DTH->OBS 0.00, p=1.00; OBS->O2-PS 0.00, p=1.00; O2-PS->O2-JS +0.48, p=0.25; and HPR-O2-JS->SGRR-O2-JS +0.51, p=0.25. These tests use the success proportion across the 20 rollouts within each independent run, not the configuration-level `x/5` discovery indicator. Report the results using an estimation-first formulation; do not recast p=0.0625 as "approaching significance" or inflate n with the nested rollouts.

## 6. Focused HPR-O2-JS Formal Run 0 Case

The unified endpoint-evaluation results for internal archive `formal__seed9201__R0` are:

- Primary lenient rotation span: 20/20.
- Secondary strict common-kinematics characterization: 20/20.
- Scaled signed horizontal progress: 2.8163 +/- 0.3137 across 20 nested rollouts.
- Forward displacement: mean 3.1285 initial body lengths.
- Direction-independent rotation span: mean 817.75 degrees.
- Per-rollout reproduction error for common fields from the original formal evaluation: 0.

This supports continued use of formal run 0 as the prespecified mechanism case, with replication on formal run 4. It does not elevate a single policy to the population-level inference unit or replace the five-run comparison of the six configurations.

## 7. Boundaries for Thesis Writing

- It is acceptable to state: "All 30/30 independent training runs and the common checkpoint-1500 evaluations were included; final validation PASS."
- The Chapter 4.1 table may report the six-configuration run-level values, the number of primary-lenient rolling discoveries, and the secondary-strict results.
- The primary-lenient HPR-O2-JS results are fixed for formal runs 0-4 as 20, 0, 8, 0, 20/20: 48/100 total and 2/5 run-level discoveries.
- The primary-lenient SGRR-O2-JS results are fixed as 19, 20, 20, 20, 20/20: 99/100 total and 5/5 run-level discoveries.
- `HPR-O2-PS` has high and highly consistent horizontal progress but 0/100 primary-lenient rolling. Horizontal advance and rolling must therefore be reported as distinct outcomes.
- SGRR is associated with a higher and more repeatable discovery of rolling, but the exact run-level results do not support wording such as "guarantees," "inevitably," or "proven superior to all methods."
- The raw HPR and SGRR rewards still must not be compared directly; this report compares only common physical endpoints.
- The five AVM policies are not part of this 30-policy main matrix. If Chapter 4 needs their primary-lenient span, they must be evaluated separately under the same evaluation contract.

## 8. Independent Review Conclusion

Three mutually independent read-only reviews found no blocking issue. Every shared kinematic endpoint and success label for all 600 trajectories can be recomputed from the raw position trajectories in the NPZ files. The 30-run aggregation, six configuration statistics, 25 paired-difference rows, and 25 two-sided exact sign-flip tests all agree with the CSV files. Direct and formula action semantics, dimensions, gain mappings, and the active-torque diagnostic identity all pass. One nonblocking limitation remains: the evaluator's built-in initial-state pairing gate actively tests run 0 of each configuration, while independent review of all 30 final NPZ files confirmed elementwise-identical initial states for every run. Also, formula field `tau_active_pre_step_*` represents only the diagnostic reconstruction at the beginning of each control step, not the complete torque trajectory over the ten physics substeps; it must not be used for energy-consumption or substep-level torque conclusions.
