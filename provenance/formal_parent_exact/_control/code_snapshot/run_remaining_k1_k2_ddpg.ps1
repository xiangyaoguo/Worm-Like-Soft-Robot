param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$ResultsDir = Join-Path $ProjectRoot "resulits_k1_k2"
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

$CommonArgs = @(
    ".\training\train_metamaterial.py",
    "--robot", "crawler",
    "--terrain", "flat",
    "--num-particles", "10",
    "--channel", "action",
    "--algorithm", "ddpg",
    "--episodes", "1500",
    "--episode-steps", "1000",
    "--save-every", "100",
    "--frames-per-batch", "10000",
    "--memory-size", "1000000",
    "--minibatch-size", "128",
    "--optim-steps", "10",
    "--seed", "0",
    "--results-dir", $ResultsDir,
    "--analysis-terrains", "training"
)

$Runs = @(
    @{
        Name = "crawler_flat_action_ddpg_k1_fixed_pos1_k2_range_neg1_to_0_seed0"
        FixedK1 = "1"
        K2Min = "-1"
        K2Max = "0"
    },
    @{
        Name = "crawler_flat_action_ddpg_k1_fixed_neg1_k2_range_0_to_pos1_seed0"
        FixedK1 = "-1"
        K2Min = "0"
        K2Max = "1"
    },
    @{
        Name = "crawler_flat_action_ddpg_k1_fixed_neg1_k2_range_neg1_to_0_seed0"
        FixedK1 = "-1"
        K2Min = "-1"
        K2Max = "0"
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
    Write-Host "K1 fixed to $($Run.FixedK1), K2 range [$($Run.K2Min), $($Run.K2Max)]" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan

    & $Python @CommonArgs `
        "--run-name" $Run.Name `
        "--fix-k1" `
        "--fixed-k1" $Run.FixedK1 `
        "--k2-min" $Run.K2Min `
        "--k2-max" $Run.K2Max

    if ($LASTEXITCODE -ne 0) {
        throw "Training failed: $($Run.Name), exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "All remaining K1/K2 DDPG runs finished." -ForegroundColor Green
