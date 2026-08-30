# Paper-Facing HPR Run 2 Frozen-Policy Supplement

> **Portable entry point:** This release combines runs 0/2/4 under `python run.py intervene-hpr`. The legacy drive-letter commands below are retained only to audit the original run-2 supplement and should not be copied directly on a new machine. The historical Chinese workspace segment is rendered as the semantic English placeholder `GraduateThesisProject`.

This directory contains a non-destructive adapter for the previously validated
`formal_hpr_freeze_validation_20260810` runner. It evaluates paper-facing HPR
run 2 (internal training seed 9203) at `checkpoint_1500.pt` under the same 36
frozen-policy interventions and the same 20 paired reset seeds used for runs 0
and 4.

The adapter never edits the formal checkpoint, formal evaluation payload, code
snapshot, or the existing runs 0/4 study. Pilot and full outputs are separated
into `pilot_data/` and `data/`.

## Locked Study Identity

- Paper-facing run: 2
- Internal reproducibility seed: 9203
- Checkpoint: `formal__seed9203__R0/checkpoint_1500.pt`
- Checkpoint SHA-256:
  `0428d9a86b6622d924738c68fe09df4c6ab922e2a3225a5c24ba41e96eb1c4b8`
- Paired reset seeds: 20264101--20264120
- Rollout length: 1000 control steps
- Intervention matrix: 36 conditions
- Full evaluation size: 36 x 20 = 720 rollouts
- Baseline gate: exactly 7/20 common-criterion successes and all six continuous
  baseline identity metrics within absolute tolerance 1e-6 for every reset.

## Commands

Run from PowerShell:

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\PUBLIC_USER\AppData\Local\Programs\Python\Python311\python.exe' `
  'C:\Users\PUBLIC_USER\Documents\GraduateThesisProject\_work_formal_supplement_20260811\run2_freeze\run_formal_hpr_run2.py' --pilot

& 'C:\Users\PUBLIC_USER\AppData\Local\Programs\Python\Python311\python.exe' `
  'C:\Users\PUBLIC_USER\Documents\GraduateThesisProject\_work_formal_supplement_20260811\run2_freeze\run_formal_hpr_run2.py' --full

& 'C:\Users\PUBLIC_USER\AppData\Local\Programs\Python\Python311\python.exe' `
  'C:\Users\PUBLIC_USER\Documents\GraduateThesisProject\_work_formal_supplement_20260811\run2_freeze\validate_run2_study.py'

$env:MPLCONFIGDIR='C:\Users\PUBLIC_USER\Documents\GraduateThesisProject\_work_formal_supplement_20260811\run2_freeze\mplconfig'
& 'C:\Users\PUBLIC_USER\AppData\Local\Programs\Python\Python311\python.exe' `
  'C:\Users\PUBLIC_USER\Documents\GraduateThesisProject\_work_formal_supplement_20260811\run2_freeze\plot_run2_study.py'

& 'C:\Users\PUBLIC_USER\AppData\Local\Programs\Python\Python311\python.exe' `
  'C:\Users\PUBLIC_USER\Documents\GraduateThesisProject\_work_formal_supplement_20260811\run2_freeze\validate_run2_study.py' --require-figures
```

The runner refuses to overwrite a completed full matrix unless `--replace` is
explicitly supplied. `--replace` archives the old `data/` directory by rename;
it does not delete it.

## Output Contract

- `data/raw/seed9203/*.json`: 36 condition payloads plus identity and seed
  manifests, preserving the runs 0/4 condition schema.
- `data/episode_results.csv`: 720 rows.
- `data/condition_policy_summary.csv`: 36 rows.
- `data/STUDY_MANIFEST.json`: base runner study manifest.
- `data/RUN2_EXECUTION_MANIFEST.json`: paper/internal identity mapping, source
  hashes, code-snapshot tree hash, and a content-hash ledger for raw outputs.
- `data/paired_effects.csv`: paired effects relative to the run 2 baseline.
- `data/ANALYSIS_SUMMARY.json`: figure and effect summary.
- `data/VALIDATION_PASS.json`: final integrity receipt.

Paper text and figures should use **HPR run 2**. The internal seed 9203 belongs
only in reproducibility mappings and manifests.
