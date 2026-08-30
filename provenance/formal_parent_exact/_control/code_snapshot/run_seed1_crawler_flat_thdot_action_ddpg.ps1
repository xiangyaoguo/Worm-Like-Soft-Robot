# run_seed1_crawler_flat_thdot_action_ddpg.ps1
# Sequentially trains:
#   1) crawler + flat + thdot  + DDPG + seed1
#   2) crawler + flat + action + DDPG + seed1
#
# Put this file in the project root directory, then run:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\run_seed1_crawler_flat_thdot_action_ddpg.ps1

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$Python = "python"
$ResultsDir = Join-Path $ProjectRoot "results"

$CommonArgs = @(
    ".\training\train_metamaterial.py",
    "--robot", "crawler",
    "--terrain", "flat",
    "--algorithm", "ddpg",
    "--seed", "1",
    "--num-particles", "10",
    "--episodes", "1500",
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

$Runs = @(
    @{
        Channel = "thdot"
        RunName = "crawler_flat_thdot_ddpg_seed1"
    },
    @{
        Channel = "action"
        RunName = "crawler_flat_action_ddpg_seed1"
    }
)

foreach ($Run in $Runs) {
    $RunName = $Run.RunName
    $Channel = $Run.Channel
    $FinalCheckpoint = Join-Path $ResultsDir "$RunName\checkpoint_1500.pt"

    if (Test-Path $FinalCheckpoint) {
        Write-Host "Skipping $RunName because checkpoint_1500.pt already exists." -ForegroundColor Yellow
        continue
    }

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host "Starting run: $RunName" -ForegroundColor Cyan
    Write-Host "Channel: $Channel | Robot: crawler | Terrain: flat | Algorithm: ddpg | Seed: 1" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""

    & $Python @CommonArgs "--channel" $Channel "--run-name" $RunName

    if ($LASTEXITCODE -ne 0) {
        throw "Training failed: $RunName"
    }

    Write-Host ""
    Write-Host "Finished run: $RunName" -ForegroundColor Green
    Write-Host ""
}

Write-Host "All requested runs are complete." -ForegroundColor Green
