param(
    [Parameter(Mandatory = $true)][string]$RawMcap,
    [Parameter(Mandatory = $true)][string]$ExpectedMcapSha256,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [ValidateSet('Auto', 'ReliableLidar', 'LegacyPgsr')][string]$Profile = 'Auto',
    [string]$RuntimeConfig = (Join-Path $PSScriptRoot 'config\local_runtime.json')
)

$ErrorActionPreference = 'Stop'
$configPath = [System.IO.Path]::GetFullPath($RuntimeConfig)
if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "Runtime configuration does not exist: $configPath. Run INSTALL.ps1 first or pass -RuntimeConfig."
}
$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
if ($config.schema_version -ne 'e26_local_runtime_v1') {
    throw "Unsupported runtime configuration schema: $($config.schema_version)"
}

$mcap = [System.IO.Path]::GetFullPath($RawMcap)
$run = [System.IO.Path]::GetFullPath($OutputRoot)
$runExistedBefore = Test-Path -LiteralPath $run
$selectedProfile = $Profile
$routing = $null
$previousNoBytecode = $env:PYTHONDONTWRITEBYTECODE
$env:PYTHONDONTWRITEBYTECODE = '1'
try {
if ($Profile -eq 'Auto') {
    $initialPythonPath = $env:PYTHONPATH
    $env:PYTHONPATH = if ([string]::IsNullOrWhiteSpace($initialPythonPath)) {
        [string]$config.rosbags_vendor
    } else {
        "$($config.rosbags_vendor);$initialPythonPath"
    }
    try {
        $routingText = & ([string]$config.sfm_python) `
            (Join-Path $PSScriptRoot 'code\check_mcap_capabilities.py') --mcap $mcap
        if ($LASTEXITCODE -ne 0) { throw 'Automatic profile routing failed' }
        $routing = $routingText | ConvertFrom-Json
        $selectedProfile = [string]$routing.selected_profile
    } finally {
        $env:PYTHONPATH = $initialPythonPath
    }
} else {
    $routing = [pscustomobject]@{
        schema_version = 'e26_unattended_profile_routing_v1'
        status = 'explicit_profile'
        mcap = $mcap
        selected_profile = $selectedProfile
        language_model_used = $false
        visual_language_model_used = $false
        manual_visual_decision_used = $false
        routing_rule = 'Profile supplied by the installation configuration or command line.'
    }
}

$common = @{
    RawMcap = $mcap
    ExpectedMcapSha256 = $ExpectedMcapSha256
    OutputRoot = $run
    Colmap = [string]$config.colmap
    ExtrinsicJson = [string]$config.extrinsic_json
    SfmPython = [string]$config.sfm_python
    GenericPython = [string]$config.generic_python
    RosbagsVendor = [string]$config.rosbags_vendor
    WslDistro = [string]$config.wsl_distro
    PgsrSourceWsl = [string]$config.pgsr_source_wsl
    PgsrPythonWsl = [string]$config.pgsr_python_wsl
}

if ($selectedProfile -eq 'ReliableLidar') {
    $arguments = $common.Clone()
    $arguments.CandidateCount = [int]$config.reliable_lidar.candidate_count
    $arguments.ImageCount = [int]$config.reliable_lidar.image_count
    $arguments.StartFraction = [double]$config.reliable_lidar.start_fraction
    $arguments.EndFraction = [double]$config.reliable_lidar.end_fraction
    & (Join-Path $PSScriptRoot 'RUN_RELIABLE_LIDAR_S00_S08.ps1') @arguments
} elseif ($selectedProfile -eq 'LegacyPgsr') {
    $arguments = $common.Clone()
    $arguments.CandidateCount = [int]$config.legacy_pgsr.candidate_count
    $arguments.ImageCount = [int]$config.legacy_pgsr.image_count
    & (Join-Path $PSScriptRoot 'RUN_LEGACY_PGSR_S00_S08.ps1') @arguments
} else {
    throw "Unsupported selected profile: $selectedProfile"
}
if ($LASTEXITCODE -ne 0) {
    throw "Selected profile failed with exit code $LASTEXITCODE"
}

$routingPath = Join-Path $run '00_contract\unattended_routing_receipt.json'
$routing | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $routingPath -Encoding utf8
Write-Host (Join-Path $run 'INTEGRATED_FINAL_STATUS.json')
} catch {
    $failureArguments = @(
        (Join-Path $PSScriptRoot 'code\write_pipeline_failure_receipt.py'),
        '--output-root', $run,
        '--profile', $selectedProfile,
        '--raw-mcap', $mcap,
        '--error-type', $_.Exception.GetType().FullName,
        '--error-message', $_.Exception.Message
    )
    if ($runExistedBefore) {
        $failureArguments += '--output-root-was-preexisting'
    }
    & ([string]$config.sfm_python) @failureArguments
    if ($LASTEXITCODE -ne 0) {
        Write-Warning 'The pipeline failed and the failure receipt writer also failed.'
    }
    throw
} finally {
    $env:PYTHONDONTWRITEBYTECODE = $previousNoBytecode
}
