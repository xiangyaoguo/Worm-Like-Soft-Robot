[CmdletBinding()]
param(
    [ValidateSet('cpu', 'cu126', 'cu130', 'cu132')]
    [string]$Compute = 'cpu',
    [string]$PythonCommand = 'py'
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Native command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
    }
}

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot

if ($PythonCommand -eq 'py') {
    Invoke-NativeChecked -FilePath 'py' -Arguments @('-3.11', '-m', 'venv', '.venv')
} else {
    Invoke-NativeChecked -FilePath $PythonCommand -Arguments @('-m', 'venv', '.venv')
}

$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Virtual-environment Python was not created: $Python"
}

Invoke-NativeChecked -FilePath $Python -Arguments @('-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel')
Invoke-NativeChecked -FilePath $Python -Arguments @('-m', 'pip', 'install', 'torch==2.12.1', '--index-url', "https://download.pytorch.org/whl/$Compute")
Invoke-NativeChecked -FilePath $Python -Arguments @('-m', 'pip', 'install', '-r', 'requirements.txt')
Invoke-NativeChecked -FilePath $Python -Arguments @('-m', 'pip', 'install', '-e', 'packages\metamaterial_envs')
Invoke-NativeChecked -FilePath $Python -Arguments @('scripts\configure_paths.py', '--create')
Invoke-NativeChecked -FilePath $Python -Arguments @('scripts\verify_install.py', '--quick')

Write-Host ''
Write-Host 'Environment is ready.' -ForegroundColor Green
Write-Host 'Activate with: .\.venv\Scripts\Activate.ps1'
Write-Host 'Run a checkpoint preflight with: python scripts\run_simulation.py --preflight'
