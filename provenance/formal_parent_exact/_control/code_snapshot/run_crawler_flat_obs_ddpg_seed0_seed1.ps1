param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$ResultsDir = Join-Path $ProjectRoot "resulits_k"
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

$CommonArgs = @(
    ".\training\train_metamaterial.py",
    "--robot", "crawler",
    "--terrain", "flat",
    "--num-particles", "10",
    "--channel", "obs",
    "--algorithm", "ddpg",
    "--episodes", "1500",
    "--episode-steps", "1000",
    "--save-every", "100",
    "--frames-per-batch", "10000",
    "--memory-size", "1000000",
    "--minibatch-size", "128",
    "--optim-steps", "10",
    "--results-dir", $ResultsDir,
    "--analysis-terrains", "training"
)

$Runs = @(
    @{
        Seed = "1"
        Name = "crawler_flat_obs_ddpg_seed1"
    },
    @{
        Seed = "0"
        Name = "crawler_flat_obs_ddpg_seed0"
    }
)

foreach ($Run in $Runs) {
    $RunDir = Join-Path $ResultsDir $Run.Name
    $FinalCheckpoint = Join-Path $RunDir "checkpoint_1500.pt"

    if (Test-Path -LiteralPath $FinalCheckpoint) {
        Write-Host "Skip completed run: $($Run.Name)" -ForegroundColor Yellow
        continue
    }

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "Starting run: $($Run.Name)" -ForegroundColor Cyan
    Write-Host "channel=obs, robot=crawler, terrain=flat, algorithm=ddpg, seed=$($Run.Seed)" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan

    & $Python @CommonArgs `
        "--seed" $Run.Seed `
        "--run-name" $Run.Name

    if ($LASTEXITCODE -ne 0) {
        throw "Training failed: $($Run.Name), exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "All crawler flat obs DDPG runs finished." -ForegroundColor Green
