[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Raw,
    [Parameter(Mandatory = $true)]
    [string]$RunRoot,
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-z][a-z0-9_]{0,40}$')]
    [string]$Case,
    [ValidateSet('AdaptiveSensor', 'ReliableLidar', 'VisualTargetDiagnostic')]
    [string]$PointProfile = 'AdaptiveSensor',
    [ValidateSet('sensors', 'front', 'complete')]
    [string]$StopAfter = 'complete',
    [int]$GaussianIterations = 12000,
    [int]$PgsrIterations = 15000,
    [string]$BodyCameraCalibration
)

$ErrorActionPreference = 'Stop'
$packageRoot = $PSScriptRoot
& (Join-Path $packageRoot 'VERIFY_PACKAGE.ps1') -Output (Join-Path $packageRoot 'PACKAGE_VERIFY_RESULT.json') -RequireRuntime | Out-Null
$python = Join-Path $packageRoot 'payload\05_envs\sfm_py311_v1\Scripts\python.exe'
$entry = Join-Path $packageRoot 'payload\02_code\reconstruction_dual_v1_20261009\run_target_delivery.py'
$arguments = @(
    $entry,
    '--run-root', ([IO.Path]::GetFullPath($RunRoot)),
    '--raw', ([IO.Path]::GetFullPath($Raw)),
    '--case', $Case,
    '--point-profile', $PointProfile,
    '--stop-after', $StopAfter,
    '--gaussian-iterations', [string]$GaussianIterations,
    '--pgsr-iterations', [string]$PgsrIterations
)
if ($BodyCameraCalibration) {
    $arguments += @('--body-camera-calibration', ([IO.Path]::GetFullPath($BodyCameraCalibration)))
}
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PIP_NO_INDEX = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:WANDB_MODE = 'disabled'
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw "Dual reconstruction failed. Evidence is retained in $RunRoot" }
