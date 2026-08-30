# Analyzer for the 113-Condition Causal-Completion Study

> **Portable entry point:** Use `python run.py intervene-sgrr --component analysis --validate-only`. Set the root of the legacy 59-condition dataset through `sgrr_legacy_root` in `configs/paths.local.json`. The old absolute paths below are retained only as a historical record; the historical Chinese workspace segment is rendered as the semantic English placeholder `GraduateThesisProject`.

Analyzer file: `analyze_causal_completion_results.py`

## When to Run

Run the analyzer only after the new matrix has produced all of the following evidence:

- 113 conditions.
- Five training seeds (9201-9205).
- Twenty fixed-initial-state episodes for each condition-seed pair.
- A total of 565 result JSON files and 11,300 new episodes.
- `matched_c00/MATCHED_C00_COMPLETE.json` is PASS, and
  `matched_c00/results/seed{9201..9205}/C00.json` contains exactly the same set
  of 20264401-20264420 initial states, giving 5 x 20 = 100 strictly paired C00 episodes.

The script also verifies that the sealed legacy study contains exactly 59 conditions x 5 seeds x 20 initial states = 5,900 episodes. If anything is missing, an extra file is present, a study/hash/seed drifts, or a success label disagrees with the five gates, the script fails closed and produces no conclusion.

## Commands

First validate completeness only:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\PUBLIC_USER\AppData\Local\Programs\Python\Python311\python.exe' `
  'C:\Users\PUBLIC_USER\Documents\GraduateThesisProject\obs2_v2_1_k_causal_completion_20260804\analyze_causal_completion_results.py' `
  --validate-only
```

After the main matrix is complete, generate all tables, classifications, and figures:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\PUBLIC_USER\AppData\Local\Programs\Python\Python311\python.exe' `
  'C:\Users\PUBLIC_USER\Documents\GraduateThesisProject\obs2_v2_1_k_causal_completion_20260804\analyze_causal_completion_results.py'
```

If that Python installation lacks `pandas/numpy/matplotlib`, use the project's validated dependency directory:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTHONPATH='C:\Users\PUBLIC_USER\CloudStorage\Desktop\finalproject\job\RLMetamaterialLocomotion-main\RLMetamaterialLocomotion-main\.venv\Lib\site-packages'
& 'C:\Users\PUBLIC_USER\AppData\Local\Programs\Python\Python311\python.exe' `
  'C:\Users\PUBLIC_USER\Documents\GraduateThesisProject\obs2_v2_1_k_causal_completion_20260804\analyze_causal_completion_results.py'
```

## Fixed Outputs

Output root: `analysis_causal_completion`

- `episode_long.csv`: a unified long table with 11,300 new-matrix episodes, 100 matched-C00 episodes, and 5,900 legacy-evidence episodes, for 17,300 rows total.
- `condition_seed_summary.csv`: condition x training seed; each row is one independent policy unit.
- `condition_summary.csv`: condition overview, with success rates weighting the five training seeds equally.
- `per_k_mechanism_cards.csv`: 16 mechanism cards for J01-J08 x K1/K2.
- `failure_decomposition.csv`: decomposition of failures under the five joint gates and their joint failure modes.
- `joint_pair_effects.csv`: joint necessity and joint sufficiency of K1+K2 at each joint.
- `classification.json` / `classification.md`: machine-readable and human-readable decisions generated only from the frozen contract thresholds.
- `figures/fig_01_16_channel_effect_heatmap.png`.
- `figures/fig_02_channel_dose_response.png`.
- `figures/fig_03_channel_timing_effects.png`.
- `figures/fig_04_joint_pair_effects.png`.
- `figures/fig_05_zero_failure_decomposition.png`.
- `analysis_manifest.json`: hashes of the script, contract, sealed input evidence, and every output.

## Inference Boundaries

1. The independent inference unit is permanently fixed as the five training seeds. The 20 evaluation initial states are only paired replicates of the same policy.
2. Strong necessity and necessary contribution are judged only from the `zero` condition in the complete C11 background; temporal criticality is judged only from `static_mean` or `time_permuted`.
3. Single-channel sufficiency reads `K1_SUFF_Jxx` / `K2_SUFF_Jxx` from the sealed legacy matrix; these conditions share the same initial states as legacy C00.
4. Joint K1+K2 sufficiency must reference the added `matched::C00`: candidate and reference use the same training seed and the same ordered initial states 20264401-20264420. If any seed/episode, checkpoint/policy/trace hash, immutability receipt, or success-recomputation gate is missing, the analyzer refuses to generate results. Legacy C00 is retained only as a historical reproduction reference.
5. The primary continuous metrics for equivalence/redundancy are fixed as forward body lengths, target net rotation, direction ratio, rolling pulse count, and mean pulse interval. Every metric must lie within 0.2 episode-level SD of the reference condition, the absolute success-rate difference must be no greater than 5 percentage points, and there must not be consistent deterioration in 4/5 seeds.
6. Continuous dose and sign-flip curves are mechanism descriptions. Unless the existing equivalence rule in the contract is satisfied, the analyzer does not invent post hoc labels such as "optimal" or "critical."
