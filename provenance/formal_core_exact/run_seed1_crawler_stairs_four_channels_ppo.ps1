# Run from the project root:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\run_seed1_crawler_stairs_four_channels_ppo.ps1

$ErrorActionPreference = "Stop"

$ProjectRoot = (Get-Location).Path
$TrainScript = Join-Path $ProjectRoot "training\train_metamaterial.py"
if (-not (Test-Path $TrainScript)) {
    throw "Run this script from the project root. Missing: $TrainScript"
}

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    $Python = "python"
}

$ResultsDir = Join-Path $ProjectRoot "results"
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

# Common settings are identical to the previous stairs/PPO/seed1 comparison runs.
$CommonArgs = @(
    $TrainScript,
    "--robot", "crawler",
    "--terrain", "stairs",
    "--algorithm", "ppo",
    "--control-mode", "auto",
    "--num-particles", "10",

    "--start-stairs", "5.0",
    "--step-width", "5.0",
    "--step-height", "0.6",
    "--steps", "10",

    "--tunnel-start", "10.0",
    "--tunnel-slope", "5.0",
    "--tunnel-slope-height", "1.0",
    "--tunnel-length", "10.0",
    "--tunnel-height", "4.0",

    "--feedback-gain", "1.0",
    "--max-control-gain", "9.0",
    "--fixed-k1", "-5.0",
    "--min-k2-magnitude", "0.001",
    "--passive-kappa", "4.0",

    "--seed", "1",
    "--episodes", "1500",
    "--episode-steps", "1000",
    "--frames-per-batch", "10000",
    "--save-every", "100",

    "--memory-size", "1000000",
    "--minibatch-size", "128",
    "--optim-steps", "10",
    "--lr", "0.0003",
    "--weight-decay", "0.0001",
    "--max-grad-norm", "1.0",
    "--gamma", "0.99",
    "--polyak-tau", "0.005",
    "--clip-epsilon", "0.2",
    "--lambda-gae", "0.9",
    "--entropy-eps", "0.0001",
    "--expl-noise-start", "0.9",
    "--expl-noise-end", "0.1",

    "--policy-depth", "2",
    "--policy-cells", "256",
    "--share-policy",
    "--share-critic",
    "--centralised-critic",
    "--normal-scale-lb", "0.0001",
    "--buffer-storage", "tensor",

    "--results-dir", $ResultsDir,
    "--auto-analysis",
    "--analysis-every", "0",
    "--analysis-terrains", "all",
    "--analysis-grid-size", "101",
    "--analysis-theta-dot-slices", "9",
    "--analysis-eval-episodes", "10",
    "--analysis-eval-steps", "1000",
    "--analysis-motion-steps", "1000",
    "--analysis-motion-frames", "8",
    "--analysis-dpi", "250"
)

$Experiments = @(
    @{ Channel = "dth";    RunName = "crawler_stairs_dth_ppo_seed1" },
    @{ Channel = "thdot";  RunName = "crawler_stairs_thdot_ppo_seed1" },
    @{ Channel = "action"; RunName = "crawler_stairs_action_ppo_seed1" },
    @{ Channel = "paper";  RunName = "crawler_stairs_paper_ppo_seed1" }
)

Write-Host "Project root: $ProjectRoot"
Write-Host "Python:       $Python"
Write-Host "Results:      $ResultsDir"
Write-Host "Experiments:  $($Experiments.Count)"

foreach ($Experiment in $Experiments) {
    $RunName = $Experiment["RunName"]
    $FinalCheckpoint = Join-Path $ResultsDir "$RunName\checkpoint_1500.pt"

    if (Test-Path $FinalCheckpoint) {
        Write-Host "[SKIP] $RunName already has checkpoint_1500.pt" -ForegroundColor Yellow
        continue
    }

    $ArgsForRun = $CommonArgs + @(
        "--channel", $Experiment["Channel"],
        "--run-name", $RunName
    )

    Write-Host ""
    Write-Host ("=" * 88)
    Write-Host "[START] $RunName" -ForegroundColor Cyan
    Write-Host ("=" * 88)

    & $Python @ArgsForRun
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment failed: $RunName (exit code $LASTEXITCODE)"
    }

    Write-Host "[DONE] $RunName" -ForegroundColor Green
}

Write-Host "All four experiments are complete." -ForegroundColor Green
