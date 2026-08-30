# Formal gait-classification experiment report

## Outcome

The locked six-configuration archive was processed successfully. All 600 trajectories passed the input-identity and finite-array gates. The analysis retained the thesis formal rolling labels unchanged and then applied a locked, conservative crawling-candidate definition based on sustained forward progress, limited whole-body rotation, repeated internal deformation, and repeated geometric support migration in four post-settling windows.

This report is an automated candidate classification. The anonymous review package has been generated, but its rater fields are intentionally blank. Human-confirmed crawling rates must not be claimed until two independent ratings and adjudication are completed.

## Overall automated counts

| Category | Rollouts / 600 |
|---|---|
| formal rolling | 146 |
| crawling candidate | 242 |
| partial roll | 9 |
| rocking | 0 |
| sliding candidate | 0 |
| failed other | 203 |
| technical exclusion | 0 |

Kinematic crawling/sliding candidates before the shape and support gates: **242/600**.

## Configuration-level summary

| Configuration | Crawling candidates / 100 | Per-run range / 20 | Stable-candidate runs / 5 | Formal roll / 100 |
|---|---|---|---|---|
| HPR-DTH-PS | 20 | 0-20 | 1 | 0 |
| HPR-THDOT-PS | 40 | 0-20 | 2 | 0 |
| HPR-OBS-PS | 40 | 0-20 | 2 | 0 |
| HPR-O2-PS | 100 | 20-20 | 5 | 0 |
| HPR-O2-JS | 42 | 0-20 | 2 | 47 |
| SGRR-O2-JS | 0 | 0-0 | 0 | 99 |

## Representative trajectories

- Crawling-candidate medoid: `HPR_O2_PS__run2__reset20264111`.
- Formal rolling medoid: `HPR_O2_JS__run4__reset20264103`.

## Contact-proxy validation

The geometric support proxy reconstructed from particle positions was checked against the valid archived SGRR logs over 100 trajectories. Maximum absolute errors were 8.34e-06 for support index and 1.79e-07 for contact strength. This supports numerical equivalence to the environment's geometric weighting, but it remains a geometric contact proxy rather than a measured physical normal force.

## Exact effective-behaviour equivalence

All 30 checkpoint SHA-256 identities are unique, while the complete run-level position matrices contain 23 distinct effective behaviours. Two exact-equivalence groups contain five and four independently trained checkpoints respectively. These runs were retained because the checkpoint identities differ; the equality is disclosed as convergence to identical deterministic effective behaviour, not treated as a reason to delete observations.

## Interpretation boundary

The configuration summaries describe learned-policy outcomes under the fixed endpoint protocol. They do not establish that a controller configuration causally creates a gait, and they do not turn the twenty nested reset states into independent training replications. Automated crawling must be labelled as such until the blind review is completed.
