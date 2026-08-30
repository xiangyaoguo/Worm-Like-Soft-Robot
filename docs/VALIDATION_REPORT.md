# Release Validation Record (2026-08-30)

This record lists the checks that were actually run before release. The thesis DOCX was parsed only through a read-only copy; the original file was not modified. Smoke and pilot outputs produced during validation are not part of the release manifest and were excluded from the final copy according to `RELEASE_MANIFEST.csv`.

> **Public-release note:** `CHECKPOINT_SHA256.csv` records the current file hashes in this public release. Machine-local output-path metadata was normalized to repository-relative paths in all 35 checkpoints, so every public checkpoint was reserialized and its whole-file SHA-256 may differ from the archival file. `CHECKPOINT_TRANSLATION_AUDIT.csv` maps the archival and public file hashes and verifies unchanged tensor fingerprints for all 35 checkpoints.

## Executed and Passed

- All Python source files passed AST parsing; PowerShell installation scripts parsed with zero errors; all JSON configurations parsed successfully.
- The final English-only and public-release validators check every readable payload covered by the current audit, all 35 checkpoints through restricted loading, all bundled NumPy files through non-pickle loading, and complete current audit coverage. The authoritative file and row counts are the values printed by the final validation commands and recorded in the current manifests, rather than pre-public staging counts.
- The release-structure suite passed under `python -m unittest discover -s tests -v`.
- All 13 core dependencies imported successfully in the original thesis environment: Python 3.11.9, Torch `2.12.0.dev20260408+cu128`, and TorchRL 0.12.0.
- All 30 six-configuration endpoints and five AVM endpoints contain a nonempty checkpoint, metadata, a 1,500-row training log, and a `complete` summary.
- The SHA-256 digests of all 35 public-release endpoints match the current `CHECKPOINT_SHA256.csv`; comparison with the archival files is reconciled through `CHECKPOINT_TRANSLATION_AUDIT.csv` as described above.
- `RELEASE_MANIFEST.csv` verifies the complete current public payload file set, and `TRANSLATION_AUDIT.csv` records the current transformation status and hashes for every non-self-referential public payload file. Final counts are taken from these regenerated files and their successful verification output.
- The SGRR seed 9201 checkpoint successfully reconstructed its policy and executed one headless simulation step; the AVM seed 9201 actor-only sham hook also completed a one-step reconstruction.
- SGRR-O2-JS completed smoke training for one batch and 1,000 frames; the trainer, environment, PPO optimization, and checkpoint-saving chain exited with code 0.
- For the formal six-configuration evaluator, the contract scan, rotation-span boundary self-test, and DTH/O2-JS two-subprocess smoke test all passed.
- The AVM endpoint-only evaluator passed the contracts for all 10/10 O2 plus AVM endpoints.
- For SGRR causal completion, the 113-condition canonical hash, ten R0/Rroll checkpoints, and portable source manifest all passed.
- For SGRR matched-C00, the five exact source-code hashes from the historical amendment, the current portable manifest, and the parent 113-condition contract all passed.
- In the HPR pilot, the seed 9201 baseline was strict 20/20 and seed 9203 was 7/20 under the archived definition; the maximum absolute error across the six continuous trajectory fields against the official JSON was 0 for both.
- After merging the archived CSV files for HPR runs 0/2/4, the 2,160-episode / 108-policy-condition-cell summary contract passed; the respective baselines were 20/7/20.
- The shared-scale K1/K2 response figure was successfully reconstructed as PNG, PDF, SVG, and manifest files from the bundled selected NPZ files.

## Not Repeated During Release Validation

- From-scratch retraining of all 35 formal runs for 15,000,000 frames.
- Complete reevaluation of the 600 six-configuration episodes and 11,300 SGRR episodes.
- The complete learning curve for AVM checkpoints 100-1400.
- The final integrated SGRR figure, which requires the external legacy large-data bundle.

These tasks are time-consuming or depend on large datasets that were not copied into the source repository. The repository provides their formal frozen programs, endpoint checkpoints, reference results, hash contracts, and explicit commands. Smoke or endpoint-only checks must not be described as reproduction of the complete curves.

## Subsequent Acceptance Commands

```powershell
python scripts\verify_install.py
python run.py evaluate --mode contract
python run.py evaluate --mode self-test
python run.py evaluate --mode process-smoke
python run.py evaluate-avm --mode contract
python run.py intervene-sgrr --component causal --stage verify
python run.py intervene-sgrr --component matched --stage verify
python scripts\verify_english_release.py
python scripts\verify_public_release.py
python scripts\build_release_manifest.py --verify
```
