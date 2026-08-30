# Summary of the Crawling Supplement Results

## Study Status

Automated candidate classification has been completed: 6 configurations, 5 independent training runs per configuration, and 20 paired resets per frozen policy, giving 600 trajectories of 1,000 steps each. The original checkpoint-1500 trajectories were read-only, and input hashes matched before and after analysis. Two independent anonymous reviewers must still complete blind review before a `crawling candidate` can be promoted to `human-confirmed crawling`.

## Overall Automated-Stage Results

- Formal rolling: 146/600.
- Crawling candidates: 242/600.
- Partial roll: 9/600.
- Rocking: 0/600.
- Sliding candidates: 0/600.
- Failed/other: 203/600.
- Technical exclusion: 0/600.

| Configuration | Crawling candidates /100 | Formal rolling /100 | Partial roll /100 | Failed/other /100 | Stable-crawl runs |
|---|---:|---:|---:|---:|---:|
| HPR-DTH-PS | 20 | 0 | 0 | 80 | 1/5 |
| HPR-THDOT-PS | 40 | 0 | 0 | 60 | 2/5 |
| HPR-OBS-PS | 40 | 0 | 0 | 60 | 2/5 |
| HPR-O2-PS | 100 | 0 | 0 | 0 | 5/5 |
| HPR-O2-JS | 42 | 47 | 8 | 3 | 2/5 |
| SGRR-O2-JS | 0 | 99 | 1 | 0 | 0/5 |

## Conclusions That May Be Safely Reported in the Thesis

Under the locked automated criteria, all five independent HPR-O2-PS training runs produced a crawling candidate on every one of their 20 resets (100/100), with no formal rolling. By contrast, SGRR-O2-JS primarily produced formal rolling (99/100) and no crawling candidate. HPR-O2-JS produced crawling candidates, formal rolling, and partial rolls, indicating that changing joint sharing differentiated the gait composition of the learned policies. These are descriptive results for frozen policies under a fixed evaluation protocol and must not be presented as causal conclusions.

## Representative Trajectories

- Crawling-candidate medoid: `HPR_O2_PS__run2__reset20264111`.
- Formal-rolling medoid: `HPR_O2_JS__run4__reset20264103`.

## Key Limitations

The 600 rollouts are not 600 independent replicates. The independent unit for configuration comparison is the five training runs per configuration; the 20 resets are nested environmental states paired across policies. A formal crawling success rate must await anonymous labels from two independent reviewers and adjudication of disagreements. Until then, the thesis text must consistently use "crawling candidate" or "automated crawling candidate."

In addition, all 30 checkpoint SHA-256 digests differ, but the complete run-level position matrices contain only 23 distinct patterns. In two groups, 5 and 4 independent training checkpoints, respectively, converged to exactly the same deterministic effective behavior. All these runs have been retained and separately disclosed in the QA table; they must not be deleted as duplicate files.
