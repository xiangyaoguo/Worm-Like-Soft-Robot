# Policy analysis figures

Place `analyze_policy.py` into your project at:

```text
training/analyze_policy.py
```

Then generate figures from the latest checkpoint:

```powershell
python .\training\analyze_policy.py --checkpoint latest
```

Or from a specific checkpoint:

```powershell
python .\training\analyze_policy.py --checkpoint .\results\20260611_230813_crawler_stairs\checkpoint_500.pt
```

The script saves files next to the checkpoint, for example:

```text
results\20260611_230813_crawler_stairs\analysis_checkpoint_500\
  policy_torque_heatmap_2d.png
  policy_torque_slices_theta_dot.png
  simple_nonreciprocity_baseline.png
  gait_snapshots.png
  rollout_speed.png
  analysis_summary.json
```

For PowerShell, train then analyze in one line:

```powershell
python .\training\train_metamaterial.py --robot crawler --terrain stairs --episodes 500 --episode-steps 400 --save-every 50 --start-stairs 5 --step-width 5 --step-height 0.2 --steps 10; if ($LASTEXITCODE -eq 0) { python .\training\analyze_policy.py --checkpoint latest }
```
