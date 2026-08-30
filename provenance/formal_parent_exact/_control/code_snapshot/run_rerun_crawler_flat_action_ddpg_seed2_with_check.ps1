# run_rerun_crawler_flat_action_ddpg_seed2_with_check.ps1
#
# Purpose:
#   Re-run crawler + flat + action + DDPG with seed=2 after fixing the DDPG actor update/saving issue.
#
# What this script does:
#   1. Backs up the existing results/crawler_flat_action_ddpg_seed2 directory if it exists.
#   2. Runs a very short DDPG preflight test and checks whether the saved policy parameters are non-zero.
#   3. Runs the full 1500-episode training with the original environment/training settings.
#   4. Checks the final checkpoint policy parameter magnitude again.
#
# Put this file in the project root directory, then run:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\run_rerun_crawler_flat_action_ddpg_seed2_with_check.ps1

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$Python = "python"
$ResultsDir = Join-Path $ProjectRoot "results"

$Seed = "2"
$RunName = "crawler_flat_action_ddpg_seed2"
$PreflightRunName = "ddpg_action_savecheck_seed2"

function Get-PolicyMaxAbs {
    param (
        [Parameter(Mandatory=$true)]
        [string]$CheckpointPath
    )

    $PythonCode = @'
import sys
import torch

checkpoint_path = sys.argv[1]
ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
state = ck.get("policy", ck)

values = []
if isinstance(state, dict):
    for value in state.values():
        if torch.is_tensor(value):
            values.append(float(value.detach().abs().max().item()))
else:
    try:
        for parameter in state.parameters():
            values.append(float(parameter.detach().abs().max().item()))
    except Exception:
        pass

if not values:
    print("nan")
else:
    print(f"{max(values):.8e}")
'@

    $Output = ($PythonCode | & $Python - $CheckpointPath)
    return [double]($Output.Trim())
}

function Backup-ResultDirectory {
    param (
        [Parameter(Mandatory=$true)]
        [string]$Name
    )

    $Dir = Join-Path $ResultsDir $Name
    if (Test-Path $Dir) {
        $Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $BackupDir = Join-Path $ResultsDir "${Name}_backup_${Timestamp}"
        Write-Host "Existing result directory found. Moving:" -ForegroundColor Yellow
        Write-Host "  $Dir" -ForegroundColor Yellow
        Write-Host "to:" -ForegroundColor Yellow
        Write-Host "  $BackupDir" -ForegroundColor Yellow
        Move-Item -Path $Dir -Destination $BackupDir
    }
}

$CommonArgs = @(
    ".\training\train_metamaterial.py",
    "--robot", "crawler",
    "--terrain", "flat",
    "--channel", "action",
    "--algorithm", "ddpg",
    "--seed", $Seed,
    "--num-particles", "10",
    "--episode-steps", "1000",
    "--save-every", "100",
    "--frames-per-batch", "10000",
    "--memory-size", "1000000",
    "--minibatch-size", "128",
    "--optim-steps", "10",
    "--lr", "0.0003",
    "--weight-decay", "0.0001",
    "--max-grad-norm", "1.0",
    "--gamma", "0.99",
    "--polyak-tau", "0.005",
    "--expl-noise-start", "0.9",
    "--expl-noise-end", "0.1",
    "--policy-depth", "2",
    "--policy-cells", "256",
    "--feedback-gain", "1.0",
    "--passive-kappa", "4.0",
    "--max-control-gain", "9.0"
)

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Step 1/3: Backup old seed2 result if it exists" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Backup-ResultDirectory -Name $RunName
Backup-ResultDirectory -Name $PreflightRunName

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Step 2/3: Short preflight run to verify DDPG policy saving" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

& $Python @CommonArgs `
    "--episodes" "1" `
    "--episode-steps" "20" `
    "--frames-per-batch" "20" `
    "--save-every" "1" `
    "--run-name" $PreflightRunName `
    "--no-auto-analysis"

if ($LASTEXITCODE -ne 0) {
    throw "Preflight DDPG run failed."
}

$PreflightCheckpoint = Join-Path $ResultsDir "$PreflightRunName\checkpoint_1.pt"
if (!(Test-Path $PreflightCheckpoint)) {
    throw "Preflight checkpoint was not created: $PreflightCheckpoint"
}

$PreflightMaxAbs = Get-PolicyMaxAbs -CheckpointPath $PreflightCheckpoint
Write-Host "Preflight policy max |parameter| = $PreflightMaxAbs" -ForegroundColor Green

if ([double]::IsNaN($PreflightMaxAbs) -or $PreflightMaxAbs -lt 1e-8) {
    throw @"
The saved policy parameters are still almost zero in the preflight checkpoint.
This means the DDPG actor update/saving issue is probably not fixed yet.
Please fix the DDPG branch before running the full training.
"@
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Step 3/3: Full training run" -ForegroundColor Cyan
Write-Host "Run: crawler + flat + action + DDPG + seed2" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

& $Python @CommonArgs `
    "--episodes" "1500" `
    "--run-name" $RunName

if ($LASTEXITCODE -ne 0) {
    throw "Full DDPG training failed: $RunName"
}

$FinalCheckpoint = Join-Path $ResultsDir "$RunName\checkpoint_1500.pt"
if (!(Test-Path $FinalCheckpoint)) {
    throw "Final checkpoint was not created: $FinalCheckpoint"
}

$FinalMaxAbs = Get-PolicyMaxAbs -CheckpointPath $FinalCheckpoint
Write-Host ""
Write-Host "Final policy max |parameter| = $FinalMaxAbs" -ForegroundColor Green

if ([double]::IsNaN($FinalMaxAbs) -or $FinalMaxAbs -lt 1e-8) {
    throw @"
The final saved policy parameters are still almost zero.
The training finished, but the checkpoint is probably invalid.
Do not use this result as a valid DDPG experiment.
"@
}

Write-Host ""
Write-Host "DDPG seed2 re-run completed successfully." -ForegroundColor Green
Write-Host "Result directory:" -ForegroundColor Green
Write-Host "  $(Join-Path $ResultsDir $RunName)" -ForegroundColor Green
