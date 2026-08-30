# Large Data and Checkpoint Bundles

The source-code repository includes, by default, the 35 endpoint runs required to reproduce the main results. Together with their metadata, training logs, and summaries, these occupy approximately 66.6 MiB; the 35 `.pt` files account for approximately 46.0 MiB. The large datasets below are not programs and therefore are not duplicated by default. To audit full learning curves or individual trajectories, create a separate data bundle from the original working drive.

The Chinese directory names in the historical source paths below are translated for this English release; the directory tails uniquely identify the source artifacts.

| Data | Original location | Approx. size | Default status | Purpose |
|---|---|---:|---|---|
| All intermediate checkpoints for the six configurations and AVM | `E:\Official-Training-Results\runs` | 709.9 MiB | Endpoints only | Checkpoint 100-1500 curves |
| Full trajectories from the unified endpoint evaluation of the six configurations | `C:\Users\PUBLIC_USER\Documents\Graduate-Thesis-Project\05_Experimental-Data-and-Code\formal_six_config_endpoint_eval_20260813\evaluation_final` | 149.5 MiB | Top-level CSV/manifest only | Rerun gait classification directly, without first rerunning evaluation |
| Full gait-review assets | `E:\finalproject\formal_gait_classification_20260824` | Approx. 75 MiB | Data/protocol/report only | 600 anonymized images; the private key must not be published |
| Full-checkpoint AVM evaluation trajectories | `E:\finalproject\result\formal_hpr_o1_sham_vs_o2_20260811` | Approx. 1.19 GiB | Endpoints only | O2/AVM checkpoints 100-1500 |
| Complete evidence for the 113-condition SGRR study | `C:\Users\PUBLIC_USER\Documents\Graduate-Thesis-Project\obs2_v2_1_k_causal_completion_20260804` | Approx. 780 MiB | Programs and summaries included | 11,300 episodes, traces, and Figure 4.5 |
| Legacy 59-condition mechanism evidence | `C:\Users\PUBLIC_USER\Documents\Graduate-Thesis-Project\obs2_v2_1_k_mechanism_20260804` | More than several hundred MiB | Minimal runtime/calibration included | Sealed dependency of the integrated SGRR analyzer |
| Full K1/K2 policy atlas | `C:\Users\PUBLIC_USER\Documents\Graduate-Thesis-Project\05_Experimental-Data-and-Code\formal10_initial3_k1k2_atlas_20260804` | Approx. 404 MiB | Selected NPZ files for Figure 4.3 included | Full response surfaces for checkpoints 100-1500 |

## Recommended Data-Bundle Structure

```text
data_external/
  formal_all_checkpoints/
  formal_endpoint_traces/
  avm_all_checkpoints_and_traces/
  gait_review_assets/
  sgrr_legacy_evidence/
  k1k2_full_atlas/
  SHA256SUMS.csv
```

`sgrr_legacy_evidence/` must mirror the root of the original `obs2_v2_1_k_mechanism_20260804` directory: `VALIDATION_PASS.json`, `FINAL_EVIDENCE_SEAL.json`, `analysis/`, and the other root-level items must sit directly below it. The `sgrr_legacy_root` key in `configs/paths.local.json` binds this path.

`data_external/` is excluded by `.gitignore`. Store large files in a release asset, object store, or institutional data repository rather than committing them to ordinary Git. If Git LFS is used, configure its tracking rules before the first large-file commit.

## Copying Rules

1. Preserve the original relative directory structure and do not modify checkpoint contents.
2. Record the byte count and SHA-256 digest of every file.
3. After copying, recompute every SHA-256 digest and compare it item by item with the source manifest.
4. Gait file `PRIVATE_unblinded_code_key.csv` must never be included in a public data bundle.
5. `obs2_v2_1_roll_full_research_20260805` is a later 1,248-condition exploration and is not part of the thesis; do not mislabel it as formal thesis data.

## Endpoint-Only Versus Full-Curve Reproduction

- The six-configuration main thesis table depends only on checkpoint 1500, which this repository fully supports.
- The principal AVM result, `0/100`, also depends only on checkpoint 1500, and the repository includes an endpoint-only evaluator for it.
- Figure E.4 and the AVM learning-curve protocol require intermediate checkpoints. Without the corresponding data bundle, do not claim to have reproduced the complete curves.
