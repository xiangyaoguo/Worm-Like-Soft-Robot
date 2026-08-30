# RL Metamaterial Locomotion

This project trains, replays, and analyzes locomotion-control policies for a two-dimensional robotic metamaterial. The current main workflow supports two robot morphologies, three terrains, the PPO/DDPG algorithm families, multiple observation/control channels, shared policies, and independent per-joint policies.

The active-control equation for the `action` channel is:

```text
tau_active_i = K1_i * (delta_theta_(i+1) - delta_theta_(i-1))
             + K2_i * theta_dot_i
```

The environment also adds physical terms such as passive hinge stiffness, damping, contact forces, and gravity. `feedback_gain / F` is fixed at `1.0` in this project, and the final active torque is clipped to the actuator range `[-9, 9]`.

## 1. Latest Feature: Independent Per-Joint K1/K2

By default, the action policy uses parameter sharing: every joint uses the same actor function,
`[K1_i, K2_i] = f(observation_i)`. Outputs can still differ when different joints have different observations, but a shared actor produces the same output for identical observations.

New argument:

```powershell
--per-joint-k1-k2
```

When enabled, each controlled joint has independent actor parameters:

```text
joint 0: [K1_0, K2_0] = f_0(observation_0)
joint 1: [K1_1, K2_1] = f_1(observation_1)
...
joint n: [K1_n, K2_n] = f_n(observation_n)
```

Thus, two joints can learn different K1/K2 mappings even when they receive the same observation. This mode supports PPO, PPO no-clip, DDPG, and DDPG update-clip.

Minimal example:

```powershell
python .\training\train_metamaterial.py `
  --robot crawler `
  --terrain flat `
  --channel action `
  --algorithm ppo `
  --num-particles 13 `
  --per-joint-k1-k2 `
  --episodes 1500 `
  --episode-steps 1000 `
  --save-every 100 `
  --run-name crawler_flat_action_ppo_per_joint_seed0
```

The equivalent lower-level form is:

```powershell
--channel action --no-share-policy
```

For new experiments, prefer the semantically clearer `--per-joint-k1-k2`. `--no-share-policy` remains available for compatibility with existing commands and other control channels.

Important notes:

- This feature learns locally state-dependent functions `K1_i(o_i)` and `K2_i(o_i)`, not one pair of static constants per joint.
- "Independent" means that parameters are not shared across joints; it does not require the trained values to be unequal at every time step.
- `--fixed-k1` and `--fixed-k2` remain scalar values broadcast to every joint; a fixed coefficient does not become a per-joint value.
- If both K1 and K2 are fixed, a per-joint actor has no practical effect on physical control.
- The parameter count of a nonshared actor grows approximately in proportion to the number of controlled joints and generally requires more samples and training time.
- The critic remains shared and centralized by default; per-joint K1/K2 does not require disabling critic sharing.

## 2. Current Feature Overview

| Category | Supported features |
|---|---|
| Robot | `crawler`, `ring` |
| Terrain | `flat`, `stairs`, `tunnel` |
| Algorithm | `ppo`, `ppo_noclip`, `ddpg`, `ddpg_clip` |
| Direct-torque channels | `dth`, `thdot`, `obs` |
| Formula channel | `action`: learns K1/K2; either coefficient can be fixed, bounded, or numerically rescaled |
| Nonreciprocal channel | `paper`: learns `kappa_alpha` |
| Sign-constrained channels | `k2_positive`, `k2_negative` |
| Actor | Shared across joints by default; independent per-joint actors supported |
| Critic | Shared or nonshared; centralized or decentralized |
| Checkpoint | Saves policy, critic, and complete metadata; supports initializing a per-joint policy from a legacy shared checkpoint |
| Automated analysis | Training curves, policy heatmaps, K1/K2 evolution, per-joint K1/K2, cross-terrain evaluation, motion frames, and baseline |
| Demonstration | Single or batched checkpoints, specified terrain, and deterministic or sampled policies |

## 3. Recommended Entry Points and Project Structure

```text
.
|-- metamaterial_envs/
|   |-- env/metamaterial.py
|   `-- metamaterial_envs/env/metamaterial.py
|-- training/
|   |-- train_metamaterial.py
|   |-- demo_metamaterial.py
|   |-- demo_metamaterial_batch.py
|   |-- analyze_training_results.py
|   |-- analyze_policy_heatmaps.py
|   |-- verify_environment_config.py
|   `-- rlmm_common.py
|-- policy_analysis_tools/
|-- results/
|-- requirements_minimal.txt
`-- README.md
```

Primary entry points:

| File | Purpose |
|---|---|
| `training/train_metamaterial.py` | Unified training entry point |
| `training/demo_metamaterial.py` | Single-checkpoint replay |
| `training/demo_metamaterial_batch.py` | Batch replay and cross-terrain evaluation |
| `training/analyze_training_results.py` | Complete automated analysis |
| `training/analyze_policy_heatmaps.py` | Generate policy heatmaps separately |
| `training/verify_environment_config.py` | Validate the environment, action shape, and per-joint actor |

The project contains two synchronized copies of `metamaterial.py` for compatibility with different import paths. The main training script uses `add_env_package_to_path()` to load the environment implementation from the local Python package.

The following legacy scripts are not recommended for action/per-joint checkpoints:

```text
training/replay_checkpoint.py
training/checkpoint_heatmap.py
training/sim.py
training/train.py
training/train_gpu.py
```

They are retained for reproducing early experiments, and some paths still assume a scalar action or shared actor. Use the primary entry points in the table above for all new experiments.

## 4. Installation and Validation

Open PowerShell in the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements_minimal.txt
```

View arguments:

```powershell
python .\training\train_metamaterial.py --help
python .\training\demo_metamaterial.py --help
python .\training\analyze_training_results.py --help
```

Run the complete environment check:

```powershell
python .\training\verify_environment_config.py
```

The validation covers:

- Thesis `dth` / `thdot` observation dimensions.
- `feedback_gain` fixed at `1.0`.
- Action dimensions when the action channel learns or fixes K1 and K2.
- Whether K1/K2 are unbounded by default.
- Whether crawler actions retain shape `[num_envs, num_joints, 2]`.
- Whether a shared actor produces the same mapping for identical observations.
- Whether an independent actor has per-joint parameters and the correct output shape.
- Whether `reset -> rand_action -> step` succeeds.

## 5. Robots and Terrains

### 5.1 Robot Morphology

```powershell
--robot crawler
--robot ring
```

- `crawler`: open-chain structure. With `N` particles, the number of controlled joints/agents is `N-2`.
- `ring`: closed-loop structure. With `N` particles, the number of controlled joints/agents is `N`.
- Set the particle count with `--num-particles`.

### 5.2 Terrain

```powershell
--terrain flat
--terrain stairs
--terrain tunnel
```

Stair parameters:

```powershell
--start-stairs 5
--step-width 5
--step-height 0.2
--steps 10
```

Tunnel parameters:

```powershell
--tunnel-start 10
--tunnel-slope 5
--tunnel-slope-height 1
--tunnel-length 10
--tunnel-height 5
```

## 6. Control Channels

| Channel | Observation | Policy output | Environment control |
|---|---|---|---|
| `dth` | `[dtheta_prev, dtheta_next]` | Active torque | Direct torque |
| `thdot` | `[dtheta_prev, dtheta_next, theta_dot_i]` | Active torque | Direct torque |
| `obs` | `dtheta_next-dtheta_prev` | Active torque | Direct torque |
| `action` | `dtheta_next-dtheta_prev` | `[K1_i,K2_i]` or the unfixed coefficient | K1/K2 formula control |
| `paper` | `dtheta_next-dtheta_prev` | `kappa_alpha_i` | Nonreciprocal/odd-elasticity formula |
| `k2_positive` | `dtheta_next-dtheta_prev` | Positive K2 | Fixed K1; learns only positive K2 |
| `k2_negative` | `dtheta_next-dtheta_prev` | Negative K2 | Fixed K1; learns only negative K2 |

The TensorDict action for the `action` channel always retains the agent dimension:

```text
[num_envs, num_controlled_joints, action_dim]
```

When K1 and K2 are both learned, `action_dim=2`; when either coefficient is fixed, `action_dim=1`.

## 7. K1/K2 Configuration

### 7.1 Default: Learn Both K1 and K2

```powershell
--channel action
```

By default, K1/K2 are unbounded:

```text
K1 in [-inf, +inf]
K2 in [-inf, +inf]
```

The policy first outputs a raw value `u`, and the environment applies:

```text
K = k_action_scale * u
```

The default is `--k-action-scale 1.0`. Even when K is unbounded, the final active torque is still clipped to `[-9,9]`.

### 7.2 Set K1/K2 Bounds

```powershell
--k1-min -20 --k1-max 20
--k2-min -5  --k2-max 5
```

Complete example:

```powershell
python .\training\train_metamaterial.py `
  --robot ring `
  --terrain stairs `
  --channel action `
  --algorithm ppo `
  --k1-min -20 --k1-max 20 `
  --k2-min -5 --k2-max 5 `
  --episodes 1500 `
  --run-name ring_stairs_action_bounded_seed0
```

### 7.3 Adjust the Numeric Scale of an Unbounded Action

```powershell
--k-action-scale 100
```

This argument changes the scale between K and the actor's raw output; it does not change final torque clipping. When comparing different scales, keep all other training parameters, seeds, and evaluation methods identical.

### 7.4 Fix K1 and Learn Only K2

```powershell
--fix-k1 --fixed-k1 -5
```

Explicitly passing `--fixed-k1` also enables fixed K1 automatically. To record this value while continuing to learn K1, add `--no-fix-k1`.

### 7.5 Fix K2 and Learn Only K1

```powershell
--fix-k2 --fixed-k2 0
```

Similarly, use `--no-fix-k2` to disable automatic fixing.

### 7.6 Fix Both K1 and K2

```powershell
--fix-k1 --fixed-k1 -5 --fix-k2 --fixed-k2 0
```

In this case, a 1-dimensional placeholder action is retained to satisfy the TorchRL spec, but its value does not participate in the K1/K2 calculation.

### 7.7 Positive/Negative K2 Channels

```powershell
--channel k2_positive --fixed-k1 -5
--channel k2_negative --fixed-k1 -5
```

Default ranges:

```text
k2_positive: [min_k2_magnitude, max_control_gain]
k2_negative: [-max_control_gain, -min_k2_magnitude]
```

The bounds can also be overridden with `--k2-min/--k2-max`, but the corresponding sign must be preserved.

## 8. Policy and Critic Configuration

### 8.1 Shared Actor by Default

```powershell
--share-policy
```

This is the default, so legacy training commands behave as before. A shared actor is more sample-efficient and better matches the assumption of homogeneous local controllers.

### 8.2 Independent Per-Joint K1/K2 Actors

Recommended:

```powershell
--channel action --per-joint-k1-k2
```

Compatible form:

```powershell
--channel action --no-share-policy
```

Do not pass `--per-joint-k1-k2` and `--share-policy` together; the training script reports a conflict. Existing batch scripts that explicitly include `--share-policy` still train a shared policy. For a per-joint experiment, remove that argument, add `--per-joint-k1-k2`, and preferably use a new run name.

### 8.3 Critic

```powershell
--share-critic / --no-share-critic
--centralised-critic / --no-centralised-critic
```

For per-joint K1/K2 experiments, initially retain the default shared centralized critic so that actor parameter sharing is the only changed factor.

## 9. Algorithms

```powershell
--algorithm ppo
--algorithm ppo_noclip
--algorithm ddpg
--algorithm ddpg_clip
```

- `ppo`: standard clipped PPO.
- `ppo_noclip`: approximates unclipped PPO with a very wide ratio range, for ablation.
- `ddpg`: standard DDPG.
- `ddpg_clip`: limits the relative displacement of actor parameters in each update; set the threshold with `--ddpg-policy-update-clip`.

## 10. Common Training Commands

### 10.1 Quick Smoke Test

```powershell
python .\training\train_metamaterial.py `
  --robot crawler `
  --terrain flat `
  --channel action `
  --algorithm ppo `
  --per-joint-k1-k2 `
  --episodes 1 `
  --episode-steps 2 `
  --frames-per-batch 2 `
  --optim-steps 1 `
  --minibatch-size 1 `
  --memory-size 10 `
  --num-particles 6 `
  --force-cpu `
  --no-auto-analysis `
  --run-name smoke_per_joint_k1_k2
```

### 10.2 Per-Joint PPO for a Crawler on Flat Terrain

```powershell
python .\training\train_metamaterial.py `
  --robot crawler `
  --terrain flat `
  --channel action `
  --algorithm ppo `
  --num-particles 13 `
  --per-joint-k1-k2 `
  --k-action-scale 100 `
  --episodes 1500 `
  --episode-steps 1000 `
  --save-every 100 `
  --seed 0 `
  --run-name crawler13_flat_action_ppo_per_joint_scale100_seed0
```

### 10.3 Per-Joint DDPG for a 20-Node Crawler

```powershell
python .\training\train_metamaterial.py `
  --robot crawler `
  --terrain flat `
  --channel action `
  --algorithm ddpg `
  --num-particles 20 `
  --per-joint-k1-k2 `
  --k-action-scale 100 `
  --episodes 1500 `
  --episode-steps 1000 `
  --save-every 100 `
  --seed 0 `
  --run-name crawler20_flat_action_ddpg_per_joint_scale100_seed0
```

### 10.4 Per-Joint PPO for a Ring in a Tunnel

```powershell
python .\training\train_metamaterial.py `
  --robot ring `
  --terrain tunnel `
  --channel action `
  --algorithm ppo `
  --num-particles 13 `
  --per-joint-k1-k2 `
  --tunnel-height 5 `
  --episodes 1500 `
  --episode-steps 1000 `
  --seed 0 `
  --run-name ring13_tunnel_action_ppo_per_joint_seed0
```

### 10.5 Initialize Per-Joint Training from a Legacy Shared Checkpoint

```powershell
python .\training\train_metamaterial.py `
  --robot crawler `
  --terrain flat `
  --channel action `
  --algorithm ppo `
  --num-particles 13 `
  --per-joint-k1-k2 `
  --pretrained-model-path .\results\old_shared_run\checkpoint_1500.pt `
  --episodes 500 `
  --run-name crawler_flat_action_ppo_shared_to_per_joint
```

On load, the legacy shared actor parameters are copied to every joint as identical initial values and then updated separately during subsequent training. The reverse operation, forcing an independent checkpoint into a shared actor, is not currently supported. When replaying an independent checkpoint, let the demo reconstruct the architecture automatically from metadata.

## 11. Output Files and Checkpoint Metadata

Default output directory:

```text
results/<run_name>/
```

Primary files:

```text
checkpoint_<episode>.pt
checkpoint_final.pt
metadata.json
training_log.csv
training_summary.json
simulation_command.txt
analysis/
```

After training, `simulation_command.txt` is generated automatically in the run's result directory. The file contains a single command line that can be copied directly into
PowerShell. It uses the final checkpoint and automatically restores the robot,
algorithm, control channel, K bounds, and policy-sharing architecture from checkpoint metadata. It uses a deterministic policy and enables the tracking camera by default.

Example per-joint metadata:

```json
{
  "control_mode": "formula",
  "formula_action_names": ["k1", "k2"],
  "share_parameters_policy": false,
  "share_policy": false,
  "per_joint_k1_k2": true,
  "policy_parameter_sharing": "independent_per_joint",
  "num_controlled_joints": 11,
  "k1_min": -20.0,
  "k1_max": 20.0,
  "k2_min": -5.0,
  "k2_max": 5.0,
  "k_action_scale": 1.0
}
```

The demo and main analysis script reconstruct a shared or independent actor automatically from the metadata.

## 12. Automated Analysis

Automated analysis runs after training by default. Disable it with:

```powershell
--no-auto-analysis
```

Common arguments:

```powershell
--analysis-every 100
--analysis-terrains all
--analysis-grid-size 81
--analysis-theta-dot-slices 9
--analysis-eval-episodes 3
--analysis-eval-steps 300
--analysis-motion-steps 300
--analysis-motion-frames 8
--analysis-dpi 180
--analysis-no-baseline
```

The analysis directory for action training contains:

```text
training_curve.png
policy_heatmaps/
k1_k2_evolution.png
k1_k2_evolution_summary.csv
k1_k2_evolution_grid.npz
k1_k2_per_joint_evolution.png
k1_k2_per_joint_summary.csv
cross_terrain_evaluation.csv
cross_terrain_evaluation.png
motion_frames.png
analysis_summary.json
```

New files:

- `k1_k2_per_joint_summary.csv`: mean, standard deviation, minimum, and maximum K1/K2 values for every checkpoint and joint.
- `k1_k2_per_joint_evolution.png`: per-joint checkpoint evolution and the final checkpoint's K1/K2 response plot.
- `k1_k2_evolution_grid.npz`: retains the original aggregate keys and adds:
  - `joint_indices`
  - `k1_by_checkpoint_dtheta_joint`
  - `k2_by_checkpoint_dtheta_joint`

The complete per-joint array shape is:

```text
[checkpoint, dtheta_grid, joint]
```

Analyze an existing checkpoint manually:

```powershell
python .\training\analyze_training_results.py `
  --checkpoint .\results\crawler_flat_action_ppo_per_joint_seed0\checkpoint_1500.pt `
  --terrains all `
  --grid-size 151 `
  --dpi 300
```

Generate only policy heatmaps:

```powershell
python .\training\analyze_policy_heatmaps.py `
  --checkpoint .\results\crawler_flat_action_ppo_per_joint_seed0\checkpoint_1500.pt `
  --output-dir .\results\crawler_flat_action_ppo_per_joint_seed0\analysis\manual_heatmap `
  --grid-size 151 `
  --theta-dot-slices 9
```

Note: the policy heatmap primarily displays the final active torque after the formula is evaluated. Inspect the new per-joint CSV, NPZ, and evolution plot for the per-joint K1/K2 coefficients themselves.

## 13. Replay and Batch Evaluation

Use the latest checkpoint:

```powershell
python .\training\demo_metamaterial.py --checkpoint latest --policy-mode deterministic --follow-camera
```

Specify a checkpoint:

```powershell
python .\training\demo_metamaterial.py `
  --checkpoint .\results\crawler_flat_action_ppo_per_joint_seed0\checkpoint_1500.pt `
  --policy-mode deterministic `
  --follow-camera
```

Override only the terrain for a generalization test:

```powershell
python .\training\demo_metamaterial.py `
  --checkpoint .\results\crawler_flat_action_ppo_per_joint_seed0\checkpoint_1500.pt `
  --terrain stairs `
  --follow-camera
```

Batch evaluation:

```powershell
python .\training\demo_metamaterial_batch.py `
  --checkpoint .\results\run_a\checkpoint_1500.pt .\results\run_b\checkpoint_1500.pt `
  --terrain all `
  --mode all `
  --eval-episodes 5 `
  --eval-steps 500 `
  --motion-frames 10
```

Available `--mode` values:

| Mode | Meaning |
|---|---|
| `human` | Open a window and demonstrate checkpoints one by one |
| `frames` | Save motion-frame figures |
| `evaluate` | Evaluate speed and save a CSV |
| `all` | Save both motion frames and evaluation results |

Do not casually override a checkpoint's robot, channel, algorithm, fixed-K settings, or policy sharing; these configurations affect the network and action shape.

## 14. Frequently Asked Questions

### 14.1 Why Do the Per-Joint Results Still Look Identical?

Check:

1. Whether `per_joint_k1_k2` is `true` in the metadata.
2. Whether `share_parameters_policy` is `false`.
3. Whether a legacy batch script that explicitly includes `--share-policy` was used accidentally.
4. Whether only a legacy joint-averaged plot was inspected instead of the `k1_k2_per_joint_*` files.
5. Whether K1/K2 were fixed as scalar values shared by all joints.
6. Whether training ran long enough for independent actors copied from the same pretrained weights to diverge.

Independent networks permit different functions, but optimization may still converge to similar functions. That is a training outcome, not a dimension-broadcasting error.

### 14.2 Conflicting `--per-joint-k1-k2` Arguments

The following form causes an error:

```powershell
--per-joint-k1-k2 --share-policy
```

Remove `--share-policy`.

### 14.3 `state_dict size mismatch`

Common causes:

- Forcing an independent checkpoint into a shared network.
- Reconstructing a PPO checkpoint as DDPG.
- A crawler/ring or particle-count mismatch.
- Reconstructing an action checkpoint as a direct-torque channel.
- A fixed K changing the action dimension relative to the checkpoint.

Prefer `demo_metamaterial.py` and the checkpoint's own metadata.

### 14.4 Slower Training or Increased GPU/System Memory Use

A per-joint actor contains multiple sets of network parameters. Initially, you can:

- Reduce `--policy-cells`.
- Shorten the smoke test.
- Keep the critic shared.
- Validate the workflow with a robot containing fewer particles.
- Then scale gradually to 20 nodes and multiple seeds.

### 14.5 Large K Values but No Further Torque Increase

K1/K2 can be unbounded, but final active torque is clipped to `[-9,9]`. Analyze K values, torque-saturation rate, locomotion speed, energy use, and generalization together rather than comparing only absolute K magnitude.

## 15. Recommended Controlled Experiment

To determine whether a per-joint policy provides real value, change only actor sharing and keep every other condition identical:

| Group | Key argument | Purpose |
|---|---|---|
| Shared baseline | `--share-policy` | Original homogeneous local controller |
| Per-joint experiment | `--per-joint-k1-k2` | Independent K1_i/K2_i mappings |

At minimum, keep the following identical between groups:

```text
robot / num_particles
terrain / terrain settings
algorithm
K1/K2 bounds and k_action_scale
episodes / episode_steps
network depth / cells
seed set
evaluation terrains / steps
```

Report the multi-seed mean and standard deviation, speed, cross-terrain generalization, per-joint K1/K2 differences, torque-saturation rate, and energy metrics together. Because the per-joint model has more parameters, improvement must not be judged from only the single best speed.
