$ErrorActionPreference = "Stop"

$common = @(
    "--terrain", "stairs",
    "--algorithm", "ppo",
    "--seed", "0",
    "--episodes", "1500",
    "--episode-steps", "1000",
    "--num-particles", "10",
    "--start-stairs", "5.0",
    "--step-width", "5.0",
    "--step-height", "0.6",
    "--steps", "10"
)

$experiments = @(
    @{ Robot = "crawler"; Channel = "dth"   },
    @{ Robot = "crawler"; Channel = "thdot" },
    @{ Robot = "ring";    Channel = "dth"   },
    @{ Robot = "ring";    Channel = "thdot" }
)

foreach ($exp in $experiments) {
    $runName = "$($exp.Robot)_stairs_$($exp.Channel)_ppo_seed0"
    Write-Host "`n=== Running $runName ===" -ForegroundColor Cyan
    python .\training\train_metamaterial.py `
        --robot $exp.Robot `
        --channel $exp.Channel `
        --run-name $runName `
        @common
    if ($LASTEXITCODE -ne 0) {
        throw "Experiment failed: $runName"
    }
}
