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
    [string]$BodyCameraCalibration,
    [ValidateRange(3, 1200)]
    [int]$InitialFrameBudget
)

$ErrorActionPreference = 'Stop'
$packageRoot = $PSScriptRoot
& (Join-Path $packageRoot 'VERIFY_PACKAGE.ps1') -Output (Join-Path $packageRoot 'PACKAGE_VERIFY_RESULT.json') -RequireRuntime | Out-Null
$python = Join-Path $packageRoot 'payload\05_envs\sfm_py311_v1\Scripts\python.exe'
$entry = Join-Path $packageRoot 'payload\02_code\reconstruction_dual_v1_20261009\run_target_delivery.py'
$resolvedRaw = [IO.Path]::GetFullPath($Raw)
$resolvedRunRoot = [IO.Path]::GetFullPath($RunRoot)
$resolvedCalibration = if ($BodyCameraCalibration) { [IO.Path]::GetFullPath($BodyCameraCalibration) } else { $null }
$arguments = @(
    $entry,
    '--run-root', $resolvedRunRoot,
    '--raw', $resolvedRaw,
    '--case', $Case,
    '--point-profile', $PointProfile,
    '--stop-after', $StopAfter,
    '--gaussian-iterations', [string]$GaussianIterations,
    '--pgsr-iterations', [string]$PgsrIterations
)
if ($BodyCameraCalibration) {
    $arguments += @('--body-camera-calibration', $resolvedCalibration)
}
if ($PSBoundParameters.ContainsKey('InitialFrameBudget')) {
    $arguments += @('--initial-frame-budget', [string]$InitialFrameBudget)
}
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PIP_NO_INDEX = '1'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:WANDB_MODE = 'disabled'
$startedUtc = [DateTime]::UtcNow
& $python @arguments
$childExitCode = $LASTEXITCODE
$endedUtc = [DateTime]::UtcNow
if (Test-Path -LiteralPath $resolvedRunRoot) {
    $receipt = [ordered]@{
        status = if ($childExitCode -eq 0) { 'package_entry_completed' } else { 'package_entry_failed' }
        package_id = 'offline_dual_reconstruction_github_v0_1_20261110'
        entry = (Join-Path $packageRoot 'RUN.ps1')
        raw = $resolvedRaw
        run_root = $resolvedRunRoot
        case = $Case
        point_profile = $PointProfile
        stop_after = $StopAfter
        gaussian_iterations = $GaussianIterations
        pgsr_iterations = $PgsrIterations
        initial_frame_budget = if ($PSBoundParameters.ContainsKey('InitialFrameBudget')) { $InitialFrameBudget } else { $null }
        body_camera_calibration_provided = [bool]$BodyCameraCalibration
        body_camera_calibration = $resolvedCalibration
        package_verification = (Join-Path $packageRoot 'PACKAGE_VERIFY_RESULT.json')
        runtime_language_model_required = $false
        runtime_downloads_allowed = $false
        child_exit_code = $childExitCode
        started_utc = $startedUtc.ToString('o')
        ended_utc = $endedUtc.ToString('o')
        elapsed_seconds = ($endedUtc - $startedUtc).TotalSeconds
    }
    $receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $resolvedRunRoot 'PACKAGE_ENTRY_RESULT.json') -Encoding utf8
}
if ($childExitCode -ne 0) { throw "Dual reconstruction failed. Evidence is retained in $RunRoot" }
