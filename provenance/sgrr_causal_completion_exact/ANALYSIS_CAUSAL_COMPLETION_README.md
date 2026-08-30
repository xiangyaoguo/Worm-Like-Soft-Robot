# 113-Condition Causal-Completion Analyzer

Analyzer: `analyze_causal_completion_results.py`

## When to Run

Run the analyzer only after the new matrix contains all of the following evidence:

- 113 conditions;
- five training seeds (9201-9205);
- 20 fixed-initial-state episodes per condition-seed;
- 565 result JSON files and 11,300 new episodes in total; and
- a passing `matched_c00/MATCHED_C00_COMPLETE.json`, with
  `matched_c00/results/seed{9201..9205}/C00.json` containing exactly 5 x 20 = 100
  strictly paired C00 episodes for initial states 20264401-20264420.

The script also verifies that the archived legacy study contains exactly 59 conditions x five seeds x 20 initial states = 5,900 episodes. Missing or extra files, drift in study/hash/seed metadata, or disagreement between the saved success labels and the five joint thresholds causes a fail-closed exit with no conclusions generated.

## Commands

Validate integrity only:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\PUBLIC_USER\AppData\Local\Programs\Python\Python311\python.exe' `
  'C:\Users\PUBLIC_USER\Documents\Graduate_Thesis_Project\obs2_v2_1_k_causal_completion_20260804\analyze_causal_completion_results.py' `
  --validate-only
```

After the main matrix is complete, generate all tables, classifications, and figures:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\PUBLIC_USER\AppData\Local\Programs\Python\Python311\python.exe' `
  'C:\Users\PUBLIC_USER\Documents\Graduate_Thesis_Project\obs2_v2_1_k_causal_completion_20260804\analyze_causal_completion_results.py'
```

If that Python environment lacks `pandas`, `numpy`, or `matplotlib`, use the project's validated dependency directory:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='C:\Users\PUBLIC_USER\CloudStorage\Desktop\finalproject\job\RLMetamaterialLocomotion-main\RLMetamaterialLocomotion-main\.venv\Lib\site-packages'
& 'C:\Users\PUBLIC_USER\AppData\Local\Programs\Python\Python311\python.exe' `
  'C:\Users\PUBLIC_USER\Documents\Graduate_Thesis_Project\obs2_v2_1_k_causal_completion_20260804\analyze_causal_completion_results.py'
```

## Fixed Outputs

Output root: `analysis_causal_completion`

- `episode_long.csv`: unified 17,300-row long table containing 11,300 new-matrix episodes, 100 matched-C00 episodes, and 5,900 legacy episodes;
- `condition_seed_summary.csv`: condition x training seed, where each row is one independent policy unit;
- `condition_summary.csv`: condition overview with success rates equally weighted over five training seeds;
- `per_k_mechanism_cards.csv`: 16 mechanism cards for J01-J08 x K1/K2;
- `failure_decomposition.csv`: failure decomposition and joint failure patterns for the five-threshold success rule;
- `joint_pair_effects.csv`: joint K1+K2 necessity and joint sufficiency for every joint;
- `classification.json` and `classification.md`: machine- and human-readable adjudications produced only by the frozen contract thresholds;
- `figures/fig_01_16_channel_effect_heatmap.png`;
- `figures/fig_02_channel_dose_response.png`;
- `figures/fig_03_channel_timing_effects.png`;
- `figures/fig_04_joint_pair_effects.png`;
- `figures/fig_05_zero_failure_decomposition.png`; and
- `analysis_manifest.json`: hashes for the analyzer, contract, archived inputs, and every output.

## Inference Boundary

1. The independent inference units are permanently fixed as the five training seeds. The 20 evaluation initial states are paired repeats of the same policy.
2. Strong necessity and necessary contribution are adjudicated only from `zero` conditions on the full C11 background; timing-critical status is adjudicated only from `static_mean` or `time_permuted` conditions.
3. Single-channel sufficiency is read from the archived matrix's `K1_SUFF_Jxx` and `K2_SUFF_Jxx` conditions, which share initial states with legacy C00.
4. Joint K1+K2 sufficiency must use the new `matched::C00`: candidate and reference use the same training seed and ordered initial states 20264401-20264420. The analyzer refuses to produce results if any seed/episode, checkpoint/policy/trace hash, immutability receipt, or success-recomputation gate is missing. Legacy C00 remains only as a historical reproduction control.
5. The primary continuous metrics for equivalence/redundancy are fixed as forward body lengths, target net rotation, direction ratio, rolling-pulse count, and mean pulse interval. Every metric must lie within 0.2 episode-level reference SD, the absolute success-rate difference must be no greater than five percentage points, and there must be no consistent degradation in 4/5 seeds.
6. Dose and sign-flip curves are mechanistic descriptions. Unless an existing equivalence rule in the contract is satisfied, the analyzer does not invent post hoc labels such as `optimal` or `critical`.
