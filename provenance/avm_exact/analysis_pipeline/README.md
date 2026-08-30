# O1-sham versus archived O2 formal analysis pipeline

This folder contains the preregistered, fail-closed analysis and figure pipeline for the
capacity-matched observation ablation. It does **not** contain training results and it
does not create placeholder result figures.

## Scientific contrast

- O2 actor observation: `[spatial_difference, angular_velocity]`.
- O1-sham actor observation: `[spatial_difference, 0]`.
- The shared centralised critic retains the full O2 observation in both arms.
- The physical controller retains both `K1 * spatial_difference` and
  `K2 * angular_velocity` in both arms.
- Paper run labels are `run 0` to `run 4`; internal reproducibility seeds are
  `9201` to `9205` and appear only in manifests.

## Scripts

- `scripts/render_protocol_figures.py`: creates only protocol/architecture figures;
  these require no outcome data and cannot be mistaken for results.
- `scripts/ingest_archived_o2.py`: normalises the five archived O2-HPR formal runs.
- `scripts/assemble_paired_package.py`: combines the archived O2 import, frozen O2
  all-checkpoint re-evaluation and new O1-sham exports; it also enforces endpoint
  regression against the archived 100 O2 episodes.
- `scripts/validate_formal_inputs.py`: validates pairing, completeness, thresholds,
  hashes, checkpoint coverage, reset pairing and numeric bounds.
- `scripts/generate_formal_figures.py`: creates the full result-figure set only after
  validation passes.
- `scripts/build_figure_manifest.py`: hashes every generated figure and its source
  data and writes an auditable manifest.

## Formal data package

The default paired-full profile requires the following files in `DATA_ROOT`:

1. `run_manifest.csv`
2. `training_metrics.csv`
3. `checkpoint_episode_metrics.csv`
4. `trajectory_joint.csv`
5. `trajectory_node.csv`
6. `actor_probe.csv`
7. `study_contract.json`

The exact fields are defined in `schemas/column_contracts.json`. The validator requires
two arms, five matched runs, all 15 checkpoints (`100, ..., 1500`) and 20 paired reset
seeds per checkpoint. A missing arm, checkpoint, reset, hash or required numeric value
causes a non-zero exit. The plotting script refuses to run without a validation receipt.

## Recommended execution order

Use the project's existing Python 3.11 runtime and site-packages. On this workstation,
set the runtime before invoking any script:

```powershell
$env:PYTHONPATH='C:\Users\PUBLIC_USER\CloudStorage\Desktop\finalproject\job\RLMetamaterialLocomotion-main\RLMetamaterialLocomotion-main\.venv\Lib\site-packages'
$PY311='C:\Users\PUBLIC_USER\AppData\Local\Programs\Python\Python311\python.exe'
```

```powershell
& $PY311 scripts/render_protocol_figures.py --out E:\finalproject\result\formal_hpr_o1_sham_vs_o2_20260811\figures\protocol

& $PY311 scripts/ingest_archived_o2.py `
  --formal-root C:\Users\PUBLIC_USER\CloudStorage\Desktop\finalproject\job\roll_learning\obs2_roll_repro_v2_1_formal_20260803_r2 `
  --data-root E:\finalproject\result\formal_hpr_o1_sham_vs_o2_20260811\analysis\archived_o2_import

# After the frozen evaluator has exported all O2 checkpoints and the O1-sham runs:
& $PY311 scripts/assemble_paired_package.py `
  --o2-import-root E:\finalproject\result\formal_hpr_o1_sham_vs_o2_20260811\analysis\archived_o2_import `
  --o2-eval-root E:\finalproject\result\formal_hpr_o1_sham_vs_o2_20260811\analysis\o2_frozen_evaluation `
  --o1-root E:\finalproject\result\formal_hpr_o1_sham_vs_o2_20260811\analysis\o1_sham_exports `
  --out E:\finalproject\result\formal_hpr_o1_sham_vs_o2_20260811\analysis\normalized

# Validate the assembled package:
& $PY311 scripts/validate_formal_inputs.py `
  --data-root E:\finalproject\result\formal_hpr_o1_sham_vs_o2_20260811\analysis\normalized `
  --receipt E:\finalproject\result\formal_hpr_o1_sham_vs_o2_20260811\qa\ANALYSIS_INPUT_VALIDATION_PASS.json

& $PY311 scripts/generate_formal_figures.py `
  --data-root E:\finalproject\result\formal_hpr_o1_sham_vs_o2_20260811\analysis\normalized `
  --receipt E:\finalproject\result\formal_hpr_o1_sham_vs_o2_20260811\qa\ANALYSIS_INPUT_VALIDATION_PASS.json `
  --out E:\finalproject\result\formal_hpr_o1_sham_vs_o2_20260811\figures\results

& $PY311 scripts/build_figure_manifest.py `
  --figure-root E:\finalproject\result\formal_hpr_o1_sham_vs_o2_20260811\figures `
  --data-root E:\finalproject\result\formal_hpr_o1_sham_vs_o2_20260811\analysis\normalized `
  --out E:\finalproject\result\formal_hpr_o1_sham_vs_o2_20260811\qa\FIGURE_MANIFEST.json
```

## Fail-closed rules

- No result figure is produced when validation fails.
- No zero filling, forward filling, mean imputation or synthetic placeholder is allowed.
- Existing `success_common` values are recomputed from the three raw kinematic metrics.
- Episode-level rows are not treated as independent training replicates.
- Internal seeds never replace paper-facing run labels on figures.
- A partial O2-only import is labelled `incomplete` and is insufficient for plotting.
- Figures are saved as both 300-dpi PNG and vector PDF.
