# Modification Report

## 1. Environment contact parameters restored to the thesis implementation

```text
background_friction = 0.0
ground_stiffness    = 1000.0
ground_damping      = 5.0
```

Both compatibility copies of the environment were synchronized:

```text
metamaterial_envs/env/metamaterial.py
metamaterial_envs/metamaterial_envs/env/metamaterial.py
```

## 2. `feedback_gain_value` is always 1

The environment no longer derives this value from `ground_damping` or a command-line argument:

```text
feedback_gain_value = 1.0
```

The training and demonstration scripts retain `--feedback-gain` for compatibility with older commands, but reject values other than `1.0`. Checkpoint metadata also records a fixed value of `1.0`.

## 3. Channels used by the thesis training were added

### dth

```text
--channel dth
observation_func = dth_neighbours
control_mode     = direct
observation dim  = 2
action dim       = 1
```

### thdot

```text
--channel thdot
observation_func = dth_neighbours_plus_thdot
control_mode     = direct
observation dim  = 3
action dim       = 1
```

Aliases:

```text
paper_dth   -> dth
paper_thdot -> thdot
```

The existing `obs`, `action`, `paper`, `k2_positive`, and `k2_negative` channels remain available.

## 4. Verification

```powershell
python .\training\verify_environment_config.py
```

Short training tests:

```powershell
python .\training\train_metamaterial.py --robot ring --terrain flat --channel dth --algorithm ppo --episodes 1 --episode-steps 20 --frames-per-batch 20 --run-name smoke_dth --no-auto-analysis

python .\training\train_metamaterial.py --robot ring --terrain flat --channel thdot --algorithm ppo --episodes 1 --episode-steps 20 --frames-per-batch 20 --run-name smoke_thdot --no-auto-analysis
```
