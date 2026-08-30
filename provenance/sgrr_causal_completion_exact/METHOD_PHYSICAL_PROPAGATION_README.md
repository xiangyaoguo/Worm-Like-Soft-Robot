# Cross-Joint Physical-Propagation and Timing Analyzer

Script: [analyze_physical_propagation.py](./analyze_physical_propagation.py)

This script only reads frozen traces from the causal-completion experiment. It does not train, run the evaluator, load or modify checkpoints, or alter existing results. It requires all 20 `C11` baseline traces for each of five training seeds, plus the first-episode trace for every single-K `ZERO` and `SIGN_FLIP` condition. Every trace must pass the SHA-256 check in its result receipt. If inputs are incomplete, the script fails without creating a directory that could be mistaken for a final analysis.

Core definitions:

- Each intervention begins at action step 0.
- Lag t for `observation`, physical `K`, and `tau` refers to action step t.
- Lag t for position/contact/support refers to state t+1 after action t; lag t for rotation is cumulative desired-direction rotation including transition t.
- The 20 C11 trajectories for a training seed estimate the pointwise standard deviation. Crossing `0.5 SD` for three consecutive steps defines clear separation; the first one-step threshold crossing is also retained.
- Positions J01-J08 are the x and y displacements of the eight internal particles/joint nodes relative to their initial states; center-of-mass position is reported separately.
- The formal actor consists of eight local 2-to-2 networks. Its direct 16 x 16 actor Jacobian is block diagonal, so direct cross-joint derivatives are exactly zero. Differences at other joints for lag > 0 are therefore interpreted as closed-loop physical propagation, not as the policy directly reading other joints' observations.

The default output directory is `analysis_physical_propagation`, containing:

- `per_channel_propagation.csv`
- `lag_response.csv`
- `causal_chain_order.csv`
- `contact_rotation_first_change.csv`
- `stage_timing_summary.csv`
- `first_separation_heatmaps.png`
- `lag_response_by_joint_distance.png`
- `contact_rotation_change_order.png`
- `METHOD.md`
- `ANALYSIS_AUDIT.json`

Important limitation: each intervention condition has only one first-episode trace. These outputs explain where a difference appears first and how it propagates to contact and rotation; by themselves they cannot establish success rate, necessity, or cross-seed robustness. Overall causal judgments must combine them with the frozen endpoint evaluation of 5 x 20 episodes per condition.
