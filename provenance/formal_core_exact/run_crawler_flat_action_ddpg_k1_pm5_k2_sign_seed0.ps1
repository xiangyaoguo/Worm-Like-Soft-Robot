param(
    [string]$Python = "python",
    [string]$MaxControlGain = "9"
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$ResultsDir = Join-Path $ProjectRoot "resulits_k1_k2"
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null
$GainLabel = ($MaxControlGain.Trim() -replace "\+", "pos" -replace "-", "neg" -replace "\.", "p")

# The training environment uses a bounded action space. Here "posinf/neginf"
# means no explicit K2 upper/lower bound is passed; the practical bound is
# +/- $MaxControlGain, which defaults to the program's standard value 9.
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
    "--max-control-gain", $MaxControlGain,
    "--results-dir", $ResultsDir,
    "--analysis-terrains", "training"
)

$Runs = @(
    @{
        Name = "crawler_flat_action_ddpg_k1_fixed_pos5_k2_range_0_to_posinf_max$($GainLabel)_seed0"
        FixedK1 = "5"
        K2Min = "0"
        K2Max = $null
        Description = "K1 = +5, K2 constrained to [0, +inf) as [0, +MaxControlGain]"
    },
    @{
        Name = "crawler_flat_action_ddpg_k1_fixed_pos5_k2_range_neginf_to_0_max$($GainLabel)_seed0"
        FixedK1 = "5"
        K2Min = $null
        K2Max = "0"
        Description = "K1 = +5, K2 constrained to (-inf, 0] as [-MaxControlGain, 0]"
    },
    @{
        Name = "crawler_flat_action_ddpg_k1_fixed_neg5_k2_range_0_to_posinf_max$($GainLabel)_seed0"
        FixedK1 = "-5"
        K2Min = "0"
        K2Max = $null
        Description = "K1 = -5, K2 constrained to [0, +inf) as [0, +MaxControlGain]"
    },
    @{
        Name = "crawler_flat_action_ddpg_k1_fixed_neg5_k2_range_neginf_to_0_max$($GainLabel)_seed0"
        FixedK1 = "-5"
        K2Min = $null
        K2Max = "0"
        Description = "K1 = -5, K2 constrained to (-inf, 0] as [-MaxControlGain, 0]"
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
    Write-Host $Run.Description -ForegroundColor Cyan
    Write-Host "Actual finite K2 cap: +/- $MaxControlGain" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan

    $RunArgs = @(
        "--run-name", $Run.Name,
        "--fix-k1",
        "--fixed-k1", $Run.FixedK1
    )

    if ($null -ne $Run.K2Min) {
        $RunArgs += @("--k2-min", $Run.K2Min)
    }
    if ($null -ne $Run.K2Max) {
        $RunArgs += @("--k2-max", $Run.K2Max)
    }

    & $Python @CommonArgs @RunArgs

    if ($LASTEXITCODE -ne 0) {
        throw "Training failed: $($Run.Name), exit code $LASTEXITCODE"
    }
}

Write-Host ""
Write-Host "All K1 +/-5 and signed K2 action/DDPG runs finished." -ForegroundColor Green
