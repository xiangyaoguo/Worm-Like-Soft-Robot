# Reproducibility Boundaries, Protocol Differences, and Provenance

## 1. Thesis Source

The release was organized against the following file (Chinese directory name translated here):

```text
C:\Users\PUBLIC_USER\CloudStorage\Desktop\Paper-Revisions\8.29\Word5000_Draft_8.27.2.docx
```

The original file was open in Word at the time, so it was parsed from a read-only Word snapshot; the thesis itself was not modified. The thesis specifies the scientific methods and results, but not the Python version, filenames, or installation commands. Environment versions were therefore recovered from the formal code, virtual environment, and checkpoint compatibility.

## 2. Formal Source Provenance

### Core Formal-Training Snapshot

```text
C:\Users\PUBLIC_USER\CloudStorage\Desktop\finalproject\job\roll_learning\
obs2_roll_repro_v2_1_formal_20260803_r2\_control\code_snapshot
```

This snapshot contains the thesis SGRR reward, `obs2_roll_repro_v2_1`, whereas newer live repositories do not necessarily retain its complete formal reward semantics. This public English derivative stores a privacy-sanitized translated mirror in `provenance/formal_core_exact/`; the byte-exact copy is retained in the author's private archive and is not distributed in this repository.

### Training for the Other Four Configurations

```text
C:\Users\PUBLIC_USER\Documents\Graduate-Thesis-Project\05_Experimental-Data-and-Code\
formal_four_channel_training_20260811
```

### Unified Evaluation of the Six Configurations

```text
C:\Users\PUBLIC_USER\Documents\Graduate-Thesis-Project\05_Experimental-Data-and-Code\
formal_six_config_endpoint_eval_20260813
```

### Automated Gait Classifier

```text
C:\Users\PUBLIC_USER\Documents\Graduate-Thesis-Project\
formal_gait_classification_tooling_20260824
```

### HPR Frozen Intervention

```text
C:\Users\PUBLIC_USER\Documents\Graduate-Thesis-Project\formal_hpr_freeze_validation_20260810
E:\finalproject\result\formal_hpr_thesis_supplement_20260811\02_run2_freeze
```

### SGRR Causal Completion

```text
C:\Users\PUBLIC_USER\Documents\Graduate-Thesis-Project\obs2_v2_1_k_causal_completion_20260804
```

### AVM

```text
E:\finalproject\result\formal_hpr_o1_sham_vs_o2_20260811\_control\code_snapshot
```

### Unified Checkpoint Copy Source

```text
E:\Official-Training-Results\runs
```

The official sources and the E-drive mirrors for R0, Rroll, and AVM were compared by SHA-256 in 75 checks; all matched.

### Active Patches and Frozen-Archive Mirrors

Path resolution, output directories, and compatibility gates for new results have been made portable in the recommended entry points. English translated mirrors of the corresponding frozen files are stored in:

- `provenance/gait_classification_exact/`
- `provenance/hpr_freeze_exact/`
- `provenance/mechanism_runtime_exact/`
- `provenance/sgrr_causal_completion_exact/`
- `provenance/response_surfaces_exact/`

In addition, `provenance/formal_core_exact/`, `provenance/formal_parent_exact/`, and `provenance/avm_exact/` mirror the primary frozen training and evaluation snapshots. The portable-layer changes address local machine paths, output isolation, missing dependencies, and the choice between archived-result and new-result modes; this public English derivative additionally translates source and documentation text and removes private identifiers. Machine-local output-path metadata was normalized to repository-relative paths in all 35 public checkpoints. Reserialization changes their whole-file SHA-256 digests, while their tensor payloads and deterministic tensor fingerprints remain unchanged and are recorded in `CHECKPOINT_TRANSLATION_AUDIT.csv`. The frozen scientific contract and thesis reset panel were not rewritten. Byte-exact source and checkpoint files are retained in the author's private archive and are not distributed in this repository.

## 3. Three Different "Rolling Events" Must Not Be Conflated

### Internal SGRR 60-Degree Event

This event belongs to the training-reward state machine and its event bonus. It is a shaping event with a repeatable reset anchor, not the final gait label.

### Historical Five-Gate Success / Causal-Completion Criterion

This criterion jointly requires the pulse count, 360-degree rotation, direction ratio, forward body-length displacement, and mean pulse interval. The formal-training monitor and the SGRR mechanism programs use it for historical auditing.

### Thesis Three-Gate Formal Rolling Criterion

This criterion requires only 360 degrees of net rotation in the expected direction, an effective-rotation direction ratio of 0.70, and one body length of forward displacement. Thesis Table 4.1, formal gait rolling, and the principal AVM result all use this definition.

## 4. Legacy Evaluator Naming

The historical six-configuration evaluator config calls direction-independent rotation span `primary_lenient_rotation_span` and the thesis three-gate criterion `secondary_strict_common_kinematic`. These names are part of the execution history and the original fields have not been rewritten. Thesis results must be read from `strict_*` / `success_secondary_strict_common_kinematic`.

## 5. HPR Runs 0/2/4

The original HPR main package contained only runs 0 and 4; run 2 was executed independently in a later formal supplement. The final thesis text uses runs 0, 2, and 4. This release:

- Preserves both original programs and result sets.
- Adds the seed 9203 checkpoint hash to the portable runner.
- Uses a dynamic three-run summarizer to verify 2,160 episodes and 108 cells.
- Does not present the old plotting QA, which applies only to two seeds, as a three-run validator.

## 6. Gait Classifier

For archived thesis results, the historical classifier locks strict=146 and lenient=147 and fixes the final automated class counts. The release adds `--portable-new-results`: newly trained policies must still provide the complete 600 episodes / 30 NPZ files, but they are not required to reproduce the exact archived thesis counts.

The `crawling` label in automated output may be described only as a `crawling candidate`. The current archive status is automated complete, human blind review pending.

## 7. AVM

Historical code names are `O1_sham` / `spatial_only_sham`. The physical environment and critic still receive `[s, theta_dot]`; only the second input channel of the actor backbone is set exactly to zero. The physical K2 x theta_dot term remains present.

The original AVM evaluator requires 15 checkpoints x 2 arms x 5 seeds. This release includes only the ten endpoints (O2 plus AVM) by default. Therefore:

- `run.py evaluate-avm` is explicitly labeled endpoint-only.
- A privacy-sanitized translated mirror of the original full-curve evaluator remains in `provenance/`; its byte-exact copy is retained in the author's private archive and is not distributed in this repository.
- Until the intermediate checkpoints are supplied, do not claim to reproduce the complete AVM learning curve.

## 8. Runtime Levels

Original training environment: Python 3.11.9, development PyTorch 2.12 + CUDA 12.8, TorchRL 0.12.0, and TensorDict 0.12.4. The stable release environment uses PyTorch 2.12.1.

Checkpoint state dictionaries can generally be loaded across these two PyTorch 2.12 builds, but the following may vary:

- RNG initialization hashes.
- CUDA kernel ordering and nondeterminism.
- BLAS/NumPy reductions.
- Pygame/Matplotlib rendering.
- Multiprocessing scheduling and log timestamps.

Consequently, a portable rerun targets the same scientific protocol and comparable results. The archive-exact gates provide traceability to the original execution; they do not imply that every new machine must reproduce every bit exactly.

## 9. Items Not Fully Defined in the Thesis

- The written definition of the rocking "lenient span test" is incomplete; the frozen code values are authoritative.
- The thesis body does not enumerate every combination in the 81-setting gait-threshold sweep; the program and reference `sensitivity_summary.csv` preserve the combinations.
- The current Appendix D primarily contains SGRR reward pseudocode; the frozen program/contract is authoritative for the complete causal-completion implementation.
- Algorithm and table numbering in Appendix D contains a layout legacy that duplicates some Appendix B numbers; this does not affect the code.
