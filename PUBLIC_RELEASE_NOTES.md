# Public Release Notes

## Release Identity

- Repository: `https://github.com/xiangyaoguo/Worm-Like-Soft-Robot`
- Version: `1.0.0`
- Release date: `2026-08-30`
- Scope: official thesis code, 35 endpoint checkpoints, reference results, evaluation programs, and reproducibility documentation

This repository is a privacy-sanitized, fully English derivative of the private frozen thesis archive. It is intended for public inspection, citation, model replay, evaluation, portable retraining, and subsequent research. It is not a byte-exact copy of the private workstation archive.

## Privacy Sanitization

The public tree replaces or removes:

- the original workstation account name and personal storage prefixes;
- messaging-account identifiers and temporary clipboard references;
- direct supervisor-discussion and chat-report evidence; and
- nonessential IDE state, before-fix snapshots, backup copies, and duplicated historical `training/old/` scripts.

Exactly 27 files were omitted: 5 private communication/report-evidence files and 22 backup or IDE artifacts. Machine-local paths retained only as historical labels use neutral placeholders. Portable use does not depend on those labels; configure current paths through `configs/paths.local.json` or documented command-line arguments.

No blind-review private key is included. The public-release verifier also checks for known local identifiers, common private-key or access-token signatures, prohibited private files, and GitHub's 100 MiB single-file limit.

## Scientific Invariants

Privacy sanitization does not change:

- any tensor value in the 35 official `checkpoint_1500.pt` files;
- tensor object paths, dtypes, shapes, or counts;
- policy architecture or observation/action contracts;
- study IDs, training seeds, evaluation seeds, frozen thresholds, or condition definitions;
- numeric result tables, NumPy result arrays, bundled figures, or reported success counts; or
- the distinction between the thesis's strict three-criterion rolling evaluation and historical five-gate pulse studies.

All 35 checkpoints were reserialized only to replace machine-local output-path metadata with repository-relative paths. Their whole-file SHA-256 digests therefore changed, while their deterministic tensor fingerprints remained unchanged. See `CHECKPOINT_TRANSLATION_AUDIT.csv`.

## Hash and Provenance Boundaries

Use these files to validate the public distribution:

- `RELEASE_MANIFEST.csv` for the current public file set;
- `CHECKPOINT_SHA256.csv` for current public checkpoint files;
- `CHECKPOINT_TRANSLATION_AUDIT.csv` for archival-to-public checkpoint mapping and tensor identity; and
- `TRANSLATION_AUDIT.csv` for non-self-referential public payload coverage and transformation status.

Manifests and seals inside `provenance/*_exact/` belong to historical study layers. They may refer to pretranslation bytes, private workstation paths, or objects intentionally omitted from the public tree. They are retained for traceability and must not be presented as validators of current public bytes. See `provenance/TRANSLATION_IMPACT.md` for the complete boundary.

## Validation

After installing the environment described in `README.md`, run from the repository root:

```powershell
python -m unittest discover -s tests -v
python scripts\verify_install.py
python scripts\verify_english_release.py
python scripts\verify_public_release.py
python scripts\build_release_manifest.py --verify
python run.py evaluate --mode contract
```

For the complete evaluator smoke checks, also run:

```powershell
python run.py evaluate --mode self-test
python run.py evaluate --mode process-smoke
```

The verification scripts print results to the terminal and do not create a self-referential audit report inside the repository.

## Citation

Use `CITATION.cff` or GitHub's **Cite this repository** function. The initial software citation is:

> Guo, Xiangyao. (2026). *Reinforcement Learning for a Soft Multi-Joint Robot: Official Thesis Reproducibility Package* (Version 1.0.0) [Computer software]. GitHub. https://github.com/xiangyaoguo/Worm-Like-Soft-Robot

No DOI is assigned in this release. Citation metadata can be updated after a release DOI is created.

## License Status

No open-source license has currently been granted. Licensing terms are awaiting confirmation with the thesis supervisor and institution. Until a `LICENSE` file is added, all rights are reserved, subject to applicable law and GitHub's terms. Citation metadata does not itself grant permission to modify or redistribute the source code or trained models.
