# Reinforcement Learning for a Soft Multi-Joint Robot: Official Thesis Reproducibility Package

This repository collects the official programs actually used for the final thesis. It covers the two-dimensional soft-robot simulator, six PPO/MAPPO-style CTDE training configurations, the AVM extension, frozen checkpoint-1500 replay, the unified strict rolling evaluation, automated gait classification, HPR frozen-policy interventions, the 113-condition SGRR causal-completion study, and the K1/K2 response-surface analysis used in the thesis.

> **Public-release identity:** This repository is a privacy-sanitized, fully English public derivative of the private frozen thesis archive. Local account names, messaging identifiers, temporary clipboard references, direct supervisor-discussion evidence, IDE state, and nonessential backup artifacts have been removed or replaced with neutral placeholders. Output-path metadata has been normalized in all 35 public checkpoints. These changes alter file-level SHA-256 digests but do not alter checkpoint tensors, scientific constants, seeds, thresholds, or reported numerical results. Tensor fingerprints are recorded in `CHECKPOINT_TRANSLATION_AUDIT.csv`. The byte-exact pre-sanitization archive is retained privately by the author and is not distributed in this repository.

Public sanitization, scientific invariants, omitted-file categories, validation boundaries, citation, and current license status are summarized in `PUBLIC_RELEASE_NOTES.md`.

The repository includes **30 official checkpoint-1500 runs for the six main configurations** and **5 AVM checkpoint-1500 runs**. After installing the Python environment, users can load models, run simulations, and perform endpoint evaluations without locating training results from the original computer. All checkpoint output-path metadata is repository-relative, and all portable entry points obtain paths from `configs/paths.local.json`.

> Important: the thesis main results use the three strict kinematic rolling criteria. Earlier training supervisors and the SGRR causal-completion study retain their historical five-gate pulse criterion as part of the archived protocol. Neither may replace the unified six-configuration evaluation used by the thesis.

## 1. Quick Start on Windows

Recommended: Windows 10/11, 64-bit Python 3.11, and at least 16 GB of memory. CPU execution is sufficient for replay and evaluation; an NVIDIA GPU is strongly recommended for complete official training.

Run the following in PowerShell:

```powershell
git clone https://github.com/xiangyaoguo/Worm-Like-Soft-Robot.git
cd Worm-Like-Soft-Robot

# CPU environment; dependencies are downloaded on the first run.
.\setup_windows.ps1 -Compute cpu

.\.venv\Scripts\Activate.ps1
python run.py verify
python run.py simulate --arm SGRR-O2-JS --seed 9201 --preflight
python run.py simulate --arm SGRR-O2-JS --seed 9201
```

If PowerShell blocks local scripts, relax the policy for the current window only:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

With an NVIDIA GPU, select an official PyTorch wheel supported by the installed driver, for example:

```powershell
.\setup_windows.ps1 -Compute cu126
```

The script also supports `cu130` and `cu132`. If uncertain, use the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) and select Windows, Pip, Python, and the CUDA platform supported by the machine. This project does not depend on `torchvision` or `torchaudio`.

## 2. Manual Environment Installation

If the automated script fails, install the environment in the following order. Do not reuse the thesis author's old `.venv`, and do not manually add its `site-packages` directory to `PYTHONPATH`.

### 2.1 Create a Python 3.11 Virtual Environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

### 2.2 Install PyTorch

CPU:

```powershell
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
```

CUDA 12.6 example:

```powershell
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu126
```

### 2.3 Install the Remaining Dependencies and Local Simulator Package

```powershell
python -m pip install -r requirements.txt
python -m pip install -e packages\metamaterial_envs
python scripts\configure_paths.py --create
python scripts\verify_install.py --quick
python scripts\verify_install.py
```

The final command reconstructs the SGRR seed-9201 policy and executes one simulation step in headless mode.

### 2.4 Linux/macOS

```bash
chmod +x setup_unix.sh
./setup_unix.sh
source .venv/bin/activate
python run.py verify
```

The official experiments were conducted on Windows with Python 3.11. Linux CPU execution can normally be used for replay and offline analysis, but floating-point differences across operating systems, BLAS implementations, and PyTorch/CUDA versions mean that bitwise-identical results are not guaranteed.

## 3. Edit Only One Path File

Initial setup creates:

```text
configs/paths.local.json
```

All defaults are relative to the repository root and normally require no changes. To place new training or evaluation outputs on a larger drive, edit only this file:

```json
{
  "schema": "thesis_repro_paths/v1",
  "python": "auto",
  "training_output_root": "D:/thesis_outputs/training",
  "formal_checkpoint_root": "checkpoints/formal_six/runs",
  "avm_checkpoint_root": "checkpoints/avm/runs",
  "formal_evaluation_output": "D:/thesis_outputs/formal_endpoint",
  "avm_evaluation_output": "D:/thesis_outputs/avm_endpoint",
  "gait_output": "D:/thesis_outputs/gait_classification",
  "hpr_output": "D:/thesis_outputs/hpr_freeze",
  "sgrr_output": "D:/thesis_outputs/sgrr_causal_completion",
  "sgrr_legacy_root": "data_external/sgrr_legacy_evidence",
  "figure_output": "analysis/response_surfaces/generated",
  "cuda_visible_devices": "0",
  "maximum_workers": 2,
  "threads_per_worker": 1
}
```

Notes:

- `python: "auto"` uses the Python interpreter in the currently activated environment and should normally remain unchanged.
- Relative paths are resolved from the repository root; absolute paths may be written as `D:/...`.
- `formal_checkpoint_root` and `avm_checkpoint_root` point to the read-only official endpoint models by default and must not be used as output directories for new training.
- `sgrr_legacy_root` is used only for the complete SGRR integrated analysis. See `docs/DATA_BUNDLES.md` for the required directory structure. Main training, replay, and the 113-condition run do not depend on it.
- `figure_output` controls the output location for the K1/K2 shared-colour-scale response figure.
- On multi-GPU systems, edit `cuda_visible_devices` as needed.
- On Windows, use `/` in JSON paths or escape every backslash as `\\`.

Display the fully resolved configuration:

```powershell
python run.py configure --show
```

An alternative configuration can also be selected temporarily through an environment variable:

```powershell
$env:THESIS_REPRO_PATHS = 'D:\configs\paths.local.json'
python run.py verify
```

## 4. Repository Layout

```text
training/                         Official trainer, replay tools, and shared analysis code
packages/metamaterial_envs/       Editable-install two-dimensional simulation environment
formal_training/                  Original official supervisors, protocols, approvals, and source inventory
evaluation/formal_endpoint/       Unified checkpoint-1500 evaluation for the six configurations
evaluation/gait_classification/   Automated gait classification for 600 trajectories
interventions/hpr_freeze/         HPR 36-condition frozen-policy intervention
interventions/mechanism_runtime/  Shared frozen-replay runtime for HPR/SGRR
interventions/sgrr_causal_completion/  SGRR 113-condition programs
extensions/avm/                   AVM (historical code name: O1-sham)
analysis/                         Policy analyses and thesis figures
checkpoints/formal_six/runs/      30 official checkpoint-1500 runs
checkpoints/avm/runs/             5 AVM checkpoint-1500 runs
reference_results/                Official summaries, CSV files, protocols, and compact validation data
provenance/                       Privacy-sanitized English provenance and official evidence
scripts/                          Recommended portable entry points
tests/                            Release-layout and scientific-configuration tests
docs/                             Program index, data bundles, and reproducibility notes
```

The original launchers in `provenance/` and `formal_training/`, as well as operator documents explicitly marked as historical within individual studies, may contain absolute paths from the original computer. Normal use must begin from `run.py` or `scripts/`; historical files are retained for audit and are not portable entry points.

## 5. Official Thesis Configurations

| Thesis ID | Observation | Action | Actor | Checkpoint directory tag |
|---|---|---|---|---|
| HPR-DTH-PS | Two adjacent joint-angle deviations | Direct torque | Shared | `DTH` |
| HPR-THDOT-PS | Adjacent deviations + local joint angular velocity | Direct torque | Shared | `THDOT` |
| HPR-OBS-PS | Spatial difference `s_i` | Direct torque | Shared | `OBS` |
| HPR-O2-PS | `[s_i, theta_dot_i]` | K1/K2 feedback | Shared | `HPR__O2shared` |
| HPR-O2-JS | `[s_i, theta_dot_i]` | K1/K2 feedback | Independent per joint | `R0` |
| SGRR-O2-JS | `[s_i, theta_dot_i]` | K1/K2 feedback | Independent per joint | `Rroll` |
| HPR-O2-AVM-JS | Actor sees `[s_i, 0]`; critic and physical term remain O2 | K1/K2 feedback | Independent per joint | `HPR__O1sham` |

Each of the six main configurations uses seeds `9201–9205`, corresponding to thesis runs 0–4. AVM is a separate extension and is not part of the primary six-configuration table.

## 6. Simulation and Checkpoint Replay

Deterministic visualisation:

```powershell
python run.py simulate --arm HPR-O2-JS --seed 9201
python run.py simulate --arm SGRR-O2-JS --seed 9205 --steps 1000
python run.py simulate --arm HPR-O2-AVM-JS --seed 9201
```

Load a model and execute one headless simulation step only:

```powershell
python run.py simulate --arm HPR-DTH-PS --seed 9201 --preflight
```

By default, `run_simulation.py` uses checkpoint metadata as the authority for reconstructing the robot, observations, actions, network architecture, and terrain settings. Except when debugging, do not override `--channel`, `--control-mode`, or the particle count.

Call the low-level replay program directly:

```powershell
python training\demo_metamaterial.py `
  --checkpoint checkpoints\formal_six\runs\formal__seed9201__Rroll\checkpoint_1500.pt `
  --policy-mode deterministic `
  --max-steps 1000 `
  --follow-camera `
  --no-pause
```

On a server without a display, set:

```powershell
$env:SDL_VIDEODRIVER = 'dummy'
$env:MPLBACKEND = 'Agg'
```

## 7. Training (Portable Scientific Rerun)

### 7.1 Print the Official Commands First

```powershell
python run.py train --mode commands --commands-for formal --arm main6 --seeds 9201
python run.py train --mode commands --commands-for formal --arm HPR-O2-AVM-JS --seeds 9201
```

### 7.2 Small-Scale Smoke Training

Smoke mode validates the complete data flow and must not be used for thesis conclusions. It reduces the number of batches to 1, episode length to 100 steps, and frames per batch to 1000.

```powershell
python run.py train --mode smoke --arm main6 --seeds 9201 --workers 1
python run.py train --mode smoke --arm HPR-O2-AVM-JS --seeds 9201 --workers 1
```

### 7.3 Complete Official Training for the Six Configurations

```powershell
python run.py train --mode formal --arm main6 --workers 2
```

Train one configuration only:

```powershell
python run.py train --mode formal --arm SGRR-O2-JS --seeds 9201 9202 9203 9204 9205 --workers 2
```

Train the six configurations plus AVM:

```powershell
python run.py train --mode formal --arm all --workers 2
```

Here, `formal` means training from scratch with the scientific parameters frozen by the thesis. For AVM, this is a portable scientific rerun and is not equivalent to the archive-exact operational approval chain used on the original machine. The original AVM workflow also requires bitwise reference-initialisation hashes, freeze/preflight checks, and human approval; a privacy-sanitized English derivative is under `provenance/avm_exact/`, while the byte-exact material is retained in the private archive. A stable PyTorch wheel or different GPU may not pass those bitwise gates, so portable AVM retraining must not be described as a bitwise reproduction of the original initialisation.

Official parameters are locked in `configs/formal_training.json`:

- PPO / MAPPO-style CTDE with 8 joint agents;
- actor and critic both use 2x256 tanh networks;
- `clip=0.2`, `gamma=0.99`, and GAE `lambda=0.9`;
- Adam `lr=3e-4` and weight decay `1e-4`;
- entropy `1e-4` and gradient norm `1.0`;
- no advantage normalisation and no target-KL early stopping;
- 10 parallel environments x 1000 steps = `10000` team frames per batch;
- minibatch `128` and `10` optimisation steps;
- 1500 batches = `15,000,000` team frames per run;
- save every 100 batches; checkpoint 1500 is the only official endpoint;
- no pretraining, BC, teacher, anchor, or intermediate-checkpoint selection.

Official training time depends on the GPU. The 35 runs may require tens to hundreds of GPU-hours and produce approximately 0.7 GB of intermediate checkpoints plus additional logs. Ensure that the output drive has sufficient capacity before starting. The launcher refuses to overwrite a non-empty run with the same name.

## 8. Unified Official Evaluation of the Six Configurations (Thesis Main Results)

Recommended order:

```powershell
python run.py evaluate --mode contract
python run.py evaluate --mode self-test
python run.py evaluate --mode process-smoke
python run.py evaluate --mode execute --workers 2
python run.py evaluate --mode validate
```

Evaluation is fixed to checkpoint 1500, the 20 shared reset seeds `20264101–20264120`, 1000 control steps per trajectory, and deterministic policy output.

Official thesis rolling requires all three conditions:

1. desired-direction net rotation of at least `360°`;
2. a desired-direction fraction of at least `0.70` among active rotation steps satisfying `|delta_phi| >= 0.05°`;
3. forward displacement of at least `1` initial body length.

A training run is considered to have discovered rolling only if at least 10 of its 20 trajectories satisfy the criterion.

Read these output fields:

- `success_secondary_strict_common_kinematic`
- `strict_success_count`
- `training_run_discovered_strict_common_kinematic`

The historical field named `primary_lenient_rotation_span` uses only direction-independent rotation span. It is not the thesis main criterion and must not replace the strict result.

Reference results are in `reference_results/formal_endpoint/`. Thesis acceptance values are:

| Configuration | Scaled progress (5-run mean +/- SD) | Strict rolling |
|---|---:|---:|
| HPR-DTH-PS | 0.5275+/-1.1710 | 0/100 |
| HPR-THDOT-PS | 1.1514+/-1.5748 | 0/100 |
| HPR-OBS-PS | 1.0008+/-1.3756 | 0/100 |
| HPR-O2-PS | 2.8350+/-0.0922 | 0/100 |
| HPR-O2-JS | 2.0618+/-0.5363 | 47/100 |
| SGRR-O2-JS | 2.9521+/-0.4824 | 99/100 |

## 9. AVM Endpoint Evaluation

The repository includes five AVM checkpoint-1500 runs and the five matching O2 endpoints, so it provides a portable endpoint-only evaluation:

```powershell
python run.py evaluate-avm --mode contract
python run.py evaluate-avm --mode execute --workers 2
```

Expected results are `0/100` rolling for AVM and `47/100` for matched HPR-O2-JS.

The original frozen AVM protocol also evaluates the complete checkpoint 100–1500 curve, requiring 150 O2/AVM checkpoints in total. The repository contains the 10 endpoints, so 140 intermediate checkpoints remain missing. This public copy contains a privacy-sanitized English derivative of the original program in `provenance/avm_exact/`; the byte-exact program is retained privately by the author. The endpoint-only mode in this repository does not claim to reproduce the complete curve. See `docs/DATA_BUNDLES.md` for the required large-data bundle.

## 10. Automated Gait Classification

First complete the evaluation in Section 8 so that it produces 30 NPZ trajectory archives, then run:

```powershell
python run.py classify-gait
```

The script processes 6 x 5 x 20 = `600` trajectories and generates all anonymous review images plus two blank rater forms. Automated labels include formal rolling, rocking, partial roll, crawling candidate, sliding candidate, and failed/other.

When analysing 600 trajectories from newly trained policies, do not require the original thesis counts of 146/147:

```powershell
python run.py classify-gait --portable-new-results
```

Notes:

- `crawling_candidate` is an automated candidate label and does not indicate completed human blind review.
- The generated `PRIVATE_unblinded_code_key.csv` is for local adjudication only, is excluded by `.gitignore`, and must never be published.
- Official automated-classification reference results are in `reference_results/gait_classification/`.

## 11. HPR Frozen-Policy Interventions

The thesis uses HPR-O2-JS runs 0, 2, and 4, corresponding to seeds 9201, 9203, and 9205. Each policy is evaluated under 36 conditions with the same 20 reset states:

```powershell
python run.py intervene-hpr --workers 2
```

Run a one-condition pilot, still using all 20 paired reset states:

```powershell
python run.py intervene-hpr --training-seeds 9201 --workers 1 --pilot
```

After a complete run, the program automatically validates 2160 episode rows and 108 policy-condition cells and writes `THREE_RUN_VALIDATION_PASS.json`.

The original main study evaluated only runs 0 and 4; run 2 was later completed as an official supplement. Both historical programs and their results are retained under `interventions/hpr_freeze/` and `reference_results/hpr_freeze/`. The portable entry point handles all three according to the final thesis definition.

The provenance versions of `analyse_and_plot.py`, `validate_study.py`, and the bilingual report builder form the historical runs-0/4 Figure 4.4 workflow. The active English-only report is generated by `interventions/hpr_freeze/scripts/build_english_report.py` as `Formal_HPR_Rolling_Mechanism_Validation_English_20260810.docx`. These tools support `THESIS_HPR_OUTPUT`, but must not be presented as a unified three-run plotting workflow. The final numerical summary for runs 0/2/4 is generated by `summarize_three_run_study.py`; historical supplemental figures for run 2 are retained under `interventions/hpr_freeze/run2/`.

## 12. SGRR 113-Condition Causal Completion

First verify the frozen contract and the 10 R0/Rroll checkpoints:

```powershell
python run.py intervene-sgrr --component causal --stage verify
```

Single-condition smoke test:

```powershell
python run.py intervene-sgrr --component causal --stage smoke --training-seed 9201
```

Complete 113 conditions x 5 policies x 20 reset states:

```powershell
python run.py intervene-sgrr --component causal --stage all --workers 5
python run.py intervene-sgrr --component matched --stage all --workers 5
```

The causal matrix contains one identity condition; 16 joint-channel combinations with 6 transformations each, giving 96 conditions; 8 joint-pair necessity conditions; and 8 joint-pair sufficiency conditions, for 113 in total.

The outcome in this study is post-hoc `causal-completion`, defined by the five pulse, rotation, direction, forward, and interval gates. It is neither the internal 60° training-reward event nor the unified three-gate formal rolling criterion used by the thesis. Reference summaries are under `reference_results/sgrr_causal_completion/`.

The complete historical 59-condition legacy evidence is an audit dependency of the final integrated analyser, and its large raw dataset is not copied by default. Copy the original contents of the `obs2_v2_1_k_mechanism_20260804` root unchanged into `sgrr_legacy_root`. That directory must directly contain `VALIDATION_PASS.json`, `FINAL_EVIDENCE_SEAL.json`, and `analysis/`. Verify its hashes as described in `docs/DATA_BUNDLES.md`, then run:

```powershell
python run.py intervene-sgrr --component analysis --validate-only
python run.py intervene-sgrr --component analysis
```

## 13. Thesis Figures and Analysis

The selected data required for the K1/K2 shared-colour-scale response surfaces are included:

```powershell
python run.py figure-response
```

Output defaults to the `figure_output` value in `paths.local.json`, initially `analysis/response_surfaces/generated/`. HPR intervention plotting, SGRR heatmap/bar plots, AVM analysis, policy heatmaps, and related programs are located in their corresponding study directories. See `docs/PROGRAM_INDEX.md` for the mapping between final thesis figure numbers and historical source-file numbers.

## 14. Two Reproducibility Levels

### Portable Rerun (Recommended)

- Python 3.11 with stable PyTorch 2.12.1;
- paths bound through one local JSON file;
- scientific parameters, official checkpoints, and reset panel remain unchanged;
- appropriate for subsequent training, replay, evaluation, and extension;
- bitwise identity with the original nightly PyTorch/CUDA environment is not guaranteed.

### Archive Traceability and the Private Byte-Exact Archive

- Privacy-sanitized English derivatives of the official source snapshots, configurations, approvals, hash inventories, and evaluation identity JSON files are under `provenance/`; byte-exact originals are retained in the private archive and are not distributed here;
- the recorded original environment is Python 3.11.9, `torch 2.12.0.dev20260408+cu128`, and TorchRL 0.12.0;
- neutralized historical path labels and fail-closed runtime hashes in historical programs are part of the evidence;
- bitwise initialisation comparison requires the same software and hardware stack, and a stable-wheel environment may not pass the original RNG/hash gates.

## 15. Testing and Acceptance

Static release tests:

```powershell
python -m unittest discover -s tests -v
```

Complete installation tests:

```powershell
python scripts\verify_install.py
python run.py evaluate --mode contract
python run.py evaluate --mode self-test
python run.py evaluate --mode process-smoke
python scripts\verify_english_release.py
python scripts\verify_public_release.py
python scripts\build_release_manifest.py --verify
```

At minimum, subsequent modifications should pass these checks:

1. every Python file parses successfully as an AST;
2. all 35 checkpoint-1500 runs and their metadata/log/summary files are complete; `CHECKPOINT_SHA256.csv` verifies the current public checkpoint files, `CHECKPOINT_TRANSLATION_AUDIT.csv` maps archival-to-public checkpoint hashes and proves that tensor fingerprints remain unchanged for all 35 files, and `TRANSLATION_AUDIT.csv` records the status and hashes of every non-self-referential public payload file;
3. environment and policy preflight;
4. six-configuration evaluator contract, self-test, and subprocess smoke test;
5. any scientific change to training or evaluation parameters receives a new study ID and never overwrites the historical official configuration.

## 16. Troubleshooting

### `ModuleNotFoundError: metamaterial_envs`

```powershell
python -m pip install -e packages\metamaterial_envs
```

### `ModuleNotFoundError: torchrl` or `tensordict`

Confirm that `.venv` is activated, then run:

```powershell
python -m pip install -r requirements.txt
```

### CUDA Is Unavailable

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If the result is `False`, a CPU wheel may be installed or the driver may not support the selected CUDA wheel. Replay and evaluation can continue on CPU; complete training should use a PyTorch wheel compatible with the installed driver.

### A Checkpoint Cannot Be Loaded

- Use the repository's Python 3.11/TorchRL 0.12 environment.
- Do not manually alter the observation dimension, share-policy setting, or control mode.
- An AVM checkpoint requires the actor's `theta_dot=0` pre-hook; the repository replay program restores it automatically.
- Do not treat the legacy simulation-command text files as new entry points; they contain historical path labels and are not portable launchers.

### Pygame or Remote-Server Display Error

Use `--preflight` or set `SDL_VIDEODRIVER=dummy`. The official evaluator itself is headless.

### Paths Contain Non-ASCII Characters or Spaces

Python entry points use `pathlib` and argument lists and therefore support Unicode paths. Do not place an unescaped single backslash in JSON.

### The Evaluator Rejects an Existing Output Directory

This fail-closed behaviour prevents different studies from being mixed. Configure a new output directory for a new run; do not delete or overwrite sealed official results.

## 17. Archival and Release Principles

- The public release omits direct supervisor-discussion evidence, IDE state, and nonessential historical backup files. The release manifest excludes `.git`, `.venv`, `__pycache__`, and generated local outputs. Early exploratory branches and the wave/yaw/constant-gain studies explicitly excluded by the thesis are also outside the release. Byte-level pre-sanitization provenance is retained only in the private archive.
- No blind-review private key is included.
- All 35 official endpoints are retained; intermediate checkpoints 100–1400 are not retained by default.
- Large trajectories, complete checkpoint curves, and complete intervention evidence use external data bundles. See `docs/DATA_BUNDLES.md` for sources and sizes.
- Checkpoint tensors and scientific state are unchanged. Output-path metadata was normalized in all 35 public checkpoints, which changes serialized file SHA-256 values; `CHECKPOINT_TRANSLATION_AUDIT.csv` records the archival-to-public mapping and unchanged tensor fingerprints.
- `TRANSLATION_AUDIT.csv` records the available archival hash, current public hash, and transformation status for each non-self-referential public payload path. Run `python scripts\verify_english_release.py` and `python scripts\verify_public_release.py` to check language, privacy, checkpoint metadata, tensor fingerprints, NumPy metadata, secrets, file sizes, and complete audit coverage.
- `requirements-reference.txt` is only an audit record of the original training environment and includes a Python-version line that pip cannot install. Do not run `pip install -r` on it.
- Before release, run `python scripts\build_release_manifest.py --verify` to verify the complete file tree. Directories named `provenance/*_exact/` preserve frozen-archive lineage but are privacy-sanitized derivatives in this public release; byte-exact source remains in the private archive.

For a more detailed mapping from thesis sections to programs, inputs, and outputs, see `docs/PROGRAM_INDEX.md`. For source directories and known protocol differences, see `docs/REPRODUCIBILITY_NOTES.md`.

## 18. Citation and License Status

If this repository contributes to your work, cite Version 1.0.0 using `CITATION.cff` or GitHub's **Cite this repository** function:

> Guo, Xiangyao. (2026). *Reinforcement Learning for a Soft Multi-Joint Robot: Official Thesis Reproducibility Package* (Version 1.0.0) [Computer software]. GitHub. https://github.com/xiangyaoguo/Worm-Like-Soft-Robot

No DOI has been assigned yet. If a DOI is created for an archived release, this section and `CITATION.cff` will be updated.

No open-source license has currently been granted. Licensing terms are awaiting confirmation with the thesis supervisor and institution. Until a `LICENSE` file is added, all rights are reserved, subject to applicable law and GitHub's terms. Citation metadata does not itself grant permission to modify or redistribute the source code or trained models.
