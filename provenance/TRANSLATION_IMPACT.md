# Translation, Privacy Sanitization, and Hash-Contract Notice

## Purpose

This `provenance/` tree is a public, English-language derivative of the frozen thesis archive. Two transformations separate it from that private archive:

1. source and documentation text was translated into English; and
2. the public tree was privacy-sanitized by replacing machine-local identifiers and omitting private communication evidence, IDE state, and nonessential backup artifacts.

These transformations deliberately change file bytes. The public tree is therefore not a byte-exact copy of the private archive, even where directory names retain an `_exact` suffix to describe historical lineage. The byte-exact archive is retained privately by the author and is not distributed in this repository.

## Public Privacy Derivative

The public-release sanitization removes local account names, personal storage prefixes, messaging identifiers, temporary clipboard references, and direct supervisor-discussion evidence. Historical machine paths that remain useful as provenance labels are represented by neutral placeholders and are not portable runtime defaults. Current programs obtain paths through repository-relative defaults, command-line arguments, or `configs/paths.local.json`.

Exactly 27 nonessential files present in the pre-public English archive are omitted from this public tree:

- 5 files associated with direct supervisor-discussion or chat-report evidence; and
- 22 backup or IDE artifacts: 2 workspace XML files, 4 before-fix snapshots, 12 `.bak`/`.bak_*` snapshots, and 4 duplicated scripts under historical `training/old/` directories.

The omissions do not remove any of the 35 official endpoint checkpoints, active portable entry points, scientific configuration files, reported numerical result tables, or bundled reference figures and arrays. Historical manifests may still record the names or hashes of omitted private-archive objects. Such entries are preserved as historical evidence; they do not assert that the omitted objects are part of the public release.

## Checkpoint Metadata Normalization

All 35 public `checkpoint_1500.pt` files were loaded with PyTorch's restricted weights-only loader, had only machine-local output-path metadata normalized to repository-relative paths, and were then reserialized. Reserialization changes each checkpoint's whole-file SHA-256 digest.

The tensor payload is unchanged. `CHECKPOINT_TRANSLATION_AUDIT.csv` records, for every checkpoint:

- the archival and public file SHA-256 digests;
- the tensor count; and
- a deterministic tensor fingerprint computed from tensor object paths, dtypes, shapes, and raw bytes.

The public verifier requires all 35 current file hashes and tensor fingerprints to match the audit. A public checkpoint file hash must never be substituted for a historical hash without identifying the release layer to which it belongs.

## Text and Filename Translation

English translation preserves program structure, numerical constants, condition identifiers, seed sets, thresholds, and scientific meaning. Some source and documentation files were renamed so that public paths are English-only, and internal references were updated accordingly. Generated English report names are descriptions of reproducible outputs; they do not imply that a private report binary is included in this repository.

Hashes recorded before translation or privacy sanitization remain historical evidence for their corresponding archival objects. They must not be used to validate public bytes unless a public-release audit explicitly maps the archival object to the public derivative.

## Historical Hash Records

Frozen manifests and seals under `provenance/` were created for the original study environment. They may hash source files whose bytes later changed through translation, path sanitization, or public-file omission. Examples include:

- `formal_parent_exact/_control/source_manifest.json`;
- `sgrr_causal_completion_exact/SOURCE_MANIFEST.json`;
- `sgrr_causal_completion_exact/FINAL_REPORT_MANIFEST.json`; and
- `mechanism_runtime_exact/FINAL_EVIDENCE_SEAL.json` together with its referenced contract and execution receipts.

Their embedded checkpoint, condition, result, seed, and scientific-identity hashes remain historical data-integrity evidence for the layer in which they were created. The containing manifest's old whole-file hash, and any entry for a translated, sanitized, or omitted text object, is not a validator for the current public bytes. Historical records must not be rewritten in place to make an old seal appear current.

## Public Validation Layer

Use the root-level public-release artifacts for the distributed tree:

- `RELEASE_MANIFEST.csv` records the current public payload paths, byte sizes, and SHA-256 digests;
- `CHECKPOINT_SHA256.csv` records the 35 current public checkpoint files;
- `CHECKPOINT_TRANSLATION_AUDIT.csv` maps archival-to-public checkpoint hashes and proves tensor identity; and
- `TRANSLATION_AUDIT.csv` records the available archival-to-public transformation status and hashes for each non-self-referential public payload file.

Run the following from the repository root:

```powershell
python scripts\verify_english_release.py
python scripts\verify_public_release.py
python scripts\build_release_manifest.py --verify
python -m unittest discover -s tests -v
```

`TRANSLATION_AUDIT.csv` excludes itself and `RELEASE_MANIFEST.csv`; `RELEASE_MANIFEST.csv` excludes itself. A verifier or audit report must not silently add a timestamped file inside the repository and thereby change the file set it is validating.

## Frozen Scientific Values

Translation and privacy sanitization do not alter checkpoint tensor payloads, NumPy result arrays, figures, numeric result tables, condition identifiers, evaluation seeds, thresholds, success counts, or scientific interpretations. Hashes covering unchanged binary or data artifacts remain valid for those exact bytes. Hashes covering a translated or sanitized containing file do not.

Programs that compare their own public source bytes with a historical private-archive manifest may fail closed. That failure is expected and must not be bypassed. Use the current public validation layer for the distributed tree, and retain historical manifests solely for traceability to the layer in which they were generated.
