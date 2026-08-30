# Policy analysis heatmap tools

Copy the `training/` folder from this zip into your project root. It adds:

```text
training/analyze_policy_heatmaps.py
training/train_then_analyze.py
```

## After training

```powershell
python .\training\analyze_policy_heatmaps.py --checkpoint latest
```

Figures will be saved in:

```text
results\<run_name>\analysis\
```

## Train and analyze in one command

```powershell
python .\training\train_then_analyze.py --robot crawler --terrain stairs --episodes 500 --episode-steps 400 --save-every 50 --step-height 0.2
```

Extra analysis options can be put after `--`:

```powershell
python .\training\train_then_analyze.py --robot ring --terrain flat --episodes 200 -- --grid-size 151 --theta-dot-slices 9
```
