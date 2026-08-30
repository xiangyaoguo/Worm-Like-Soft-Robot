# Thesis Section-to-Program Index

This table follows the current final thesis. In this public English derivative, `provenance/` is a privacy-sanitized translated mirror of the original frozen files; byte-exact originals are retained in the author's private archive and are not distributed in this repository. The "recommended entry point" is the portable layer that can be run after editing `configs/paths.local.json`.

| Thesis content | Recommended entry point | Core implementation / historical program | Main output |
|---|---|---|---|
| Section 3.1: robot model and two-dimensional simulation | `run.py simulate` | `packages/metamaterial_envs/metamaterial_envs/env/metamaterial.py` | Pygame replay and particle/COM state |
| Section 3.2: PPO / MAPPO-style CTDE | `run.py train` | `training/train_metamaterial.py`, `training/rlmm_common.py` | Checkpoints, metadata, and training logs |
| Section 3.3: HPR-DTH/THDOT/OBS | `run.py train --arm ...` | `formal_training/launch_four_channel_training.py` | 4 x 5 formal runs |
| Section 3.4: O2 and PS/JS | `run.py train --arm HPR-O2-PS HPR-O2-JS` | `training/train_metamaterial.py` | K1/K2 policies |
| Section 3.5: SGRR | `run.py train --arm SGRR-O2-JS` | `obs2_roll_repro_v2_1` in the formal environment; `formal_training/run_formal_v2_1.py` | Five Rroll runs |
| Section 3.6: frozen endpoint evaluation | `run.py evaluate` | `evaluation/formal_endpoint/six_config_endpoint_evaluator.py` | 600 episodes, strict/lenient metrics, and 30 NPZ files |
| Section 3.7: gait classification | `run.py classify-gait` | `evaluation/gait_classification/run_gait_classification.py` | Gait CSV, 81-setting sensitivity analysis, and anonymized review assets |
| Section 3.8: AVM | `run.py train --arm HPR-O2-AVM-JS`; `run.py evaluate-avm` | `extensions/avm/code_snapshot/actor_observation_shim.py` | Five AVM endpoints and 200 endpoint episodes |
| Section 3.8: HPR frozen intervention | `run.py intervene-hpr` | `interventions/hpr_freeze/scripts/run_formal_hpr_freeze_study.py` | 3 x 36 x 20 episodes |
| Section 3.8: SGRR causal completion | `run.py intervene-sgrr` | `interventions/sgrr_causal_completion/run_causal_completion.py` | 5 x 113 x 20 episodes |
| Figure 4.3: K1/K2 response surfaces | `run.py figure-response` | `analysis/response_surfaces/build_shared_scale_response_surfaces.py` | PNG/PDF/SVG/manifest |
| Figure 4.4: HPR interventions | Historical analysis entry point | `interventions/hpr_freeze/scripts/analyse_and_plot.py` plus the run-2 supplement | Global/joint intervention figure |
| Figure 4.5: SGRR interventions | SGRR analysis component | `interventions/sgrr_causal_completion/analyze_causal_completion_results.py` | Channel heatmap and necessity/sufficiency bars |
| Appendix B: physics | Simulation package | `packages/metamaterial_envs/.../metamaterial.py` | Environment dynamics |
| Appendix C: PPO settings | `configs/formal_training.json` | Formal launch manifests | Locked hyperparameters |
| Appendix D: SGRR state machine | Formal environment | `obs2_roll_repro_v2_1` reward implementation | LAUNCH-to-ROLL state/event logs |
| Release integrity | `scripts/build_release_manifest.py --verify` | `CHECKPOINT_SHA256.csv`, `RELEASE_MANIFEST.csv` | SHA-256 verification of 35 endpoints and the complete tree |

## Six-Configuration Directory Mapping

| Thesis ID | `archive_tag` | Example run directory |
|---|---|---|
| HPR-DTH-PS | `DTH` | `formal__seed9201__DTH` |
| HPR-THDOT-PS | `THDOT` | `formal__seed9201__THDOT` |
| HPR-OBS-PS | `OBS` | `formal__seed9201__OBS` |
| HPR-O2-PS | `HPR__O2shared` | `formal__seed9201__HPR__O2shared` |
| HPR-O2-JS | `R0` | `formal__seed9201__R0` |
| SGRR-O2-JS | `Rroll` | `formal__seed9201__Rroll` |
| HPR-O2-AVM-JS | `HPR__O1sham` | `formal__seed9201__HPR__O1sham` |

## Historical Figure-Number Aliases

The final thesis figure numbers were reordered after some programs and embedded files had already been named. When searching the original drives, use both names:

- Current Figure 4.4: old source name `Figure_4_7_hpr_global_and_joint_channel_interventions`.
- Current Figure E.3: old source name `Figure_4_4_run0_actual_k_timeline`.
- Current Figure E.4: old source name `Figure_4_3_run0_checkpoint_evolution`.
- In the code, AVM is often named `O1_sham` or `spatial_only_sham`; the thesis consistently calls it AVM.

## Outside the Current Thesis Mainline

The following legacy programs are not exposed through the default portable entry point: active wave, yaw, tail-wave teacher, constant-gain sweep, `k1_k2_wave_study`, the old O1 atlas, the Scratch-WR multi-terrain study, the DDPG comparison, and the 1,248-condition full research study. They did not contribute to the current six-configuration thesis results.
