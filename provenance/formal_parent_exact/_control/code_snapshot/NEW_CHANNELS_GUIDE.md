# Thesis Channels, Contact Parameters, and Fixed Feedback Gain

In addition to the original project channels, this version adds the two direct-torque channels actually used in thesis training:

| `--channel` | Policy observation | Policy output | Total environment torque |
|---|---|---|---|
| `dth` | Deflections of two neighboring nodes, `dth_neighbours` | Active torque `u_i` | `tau_i = -kappa*delta_theta_i + u_i` |
| `thdot` | Two neighboring-node deflections and the joint's own angular velocity, `dth_neighbours_plus_thdot` | Active torque `u_i` | `tau_i = -kappa*delta_theta_i + u_i` |

Aliases:

```text
paper_dth   -> dth
paper_thdot -> thdot
```

The thesis's `simple non-reciprocity` controller is a hand-designed baseline, not a reinforcement-learning channel. Automated analysis still produces its baseline curve and heatmap.

## Environment parameters

The physical environment was restored to the contact parameters used by the thesis implementation:

```text
background_friction = 0.0
ground_stiffness    = 1000.0
ground_damping      = 5.0
```

Here, `ground_damping` is the principal ground-contact damping or "friction" term in the original program.

The feedback gain is fixed in all training, demonstration, and analysis runs:

```text
feedback_gain_value = 1.0
```

`--feedback-gain` is retained only for compatibility with older commands; passing any value other than `1.0` raises an error.

## Thesis-channel training examples

```powershell
python .\training\train_metamaterial.py --robot ring --terrain stairs --channel dth --algorithm ppo --seed 0 --episodes 1500 --run-name ring_stairs_dth_ppo_seed0

python .\training\train_metamaterial.py --robot ring --terrain stairs --channel thdot --algorithm ppo --seed 0 --episodes 1500 --run-name ring_stairs_thdot_ppo_seed0

python .\training\train_metamaterial.py --robot crawler --terrain stairs --channel dth --algorithm ppo --seed 0 --episodes 1500 --run-name crawler_stairs_dth_ppo_seed0
```

## Quick verification

```powershell
python .\training\train_metamaterial.py --help

python .\training\train_metamaterial.py --robot ring --terrain flat --channel dth --algorithm ppo --episodes 1 --episode-steps 20 --frames-per-batch 20 --run-name smoke_dth --no-auto-analysis

python .\training\train_metamaterial.py --robot ring --terrain flat --channel thdot --algorithm ppo --episodes 1 --episode-steps 20 --frames-per-batch 20 --run-name smoke_thdot --no-auto-analysis
```
