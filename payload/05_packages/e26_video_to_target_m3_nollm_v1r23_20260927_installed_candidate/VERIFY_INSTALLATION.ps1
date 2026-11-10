param(
    [string]$RuntimeConfig = (Join-Path $PSScriptRoot 'config\local_runtime.json'),
    [string]$OutputReceipt = (Join-Path $PSScriptRoot 'INSTALL_VERIFY.json')
)

$ErrorActionPreference = 'Stop'
$checks = [System.Collections.Generic.List[object]]::new()
function Add-Check {
    param([string]$Name, [bool]$Passed, [string]$Detail)
    $checks.Add([pscustomobject]@{ name = $Name; passed = $Passed; detail = $Detail })
}

$configPath = [System.IO.Path]::GetFullPath($RuntimeConfig)
if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "Runtime configuration does not exist: $configPath"
}
$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
Add-Check 'runtime_config_schema' ($config.schema_version -eq 'e26_local_runtime_v1') ([string]$config.schema_version)

foreach ($item in @(
    @('colmap', [string]$config.colmap, 'Leaf'),
    @('extrinsic_json', [string]$config.extrinsic_json, 'Leaf'),
    @('sfm_python', [string]$config.sfm_python, 'Leaf'),
    @('generic_python', [string]$config.generic_python, 'Leaf'),
    @('rosbags_vendor', [string]$config.rosbags_vendor, 'Container')
)) {
    $passed = Test-Path -LiteralPath $item[1] -PathType $item[2]
    Add-Check $item[0] $passed $item[1]
}

$initialPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = [string]$config.rosbags_vendor
    $sfmOutput = & ([string]$config.sfm_python) -c "import cv2,numpy,scipy,PIL; from rosbags.highlevel import AnyReader; print('sfm-runtime-ok')" 2>&1
    Add-Check 'windows_sfm_imports' ($LASTEXITCODE -eq 0) (($sfmOutput | Out-String).Trim())
} catch {
    Add-Check 'windows_sfm_imports' $false $_.Exception.Message
} finally {
    $env:PYTHONPATH = $initialPythonPath
}

try {
    $geometryOutput = & ([string]$config.generic_python) -c "import cv2,numpy,scipy,open3d,plotly; from scipy.sparse.csgraph import maximum_flow; print('geometry-runtime-ok')" 2>&1
    Add-Check 'windows_geometry_imports' ($LASTEXITCODE -eq 0) (($geometryOutput | Out-String).Trim())
} catch {
    Add-Check 'windows_geometry_imports' $false $_.Exception.Message
}

$node = Get-Command node -ErrorAction SilentlyContinue
Add-Check 'node_offline_viewer_validator' ($null -ne $node) $(if ($null -ne $node) { $node.Source } else { 'Node.js unavailable' })

try {
    $colmapOutput = & ([string]$config.colmap) -h 2>&1 | Select-Object -First 1
    Add-Check 'colmap_executable' ($LASTEXITCODE -eq 0) (($colmapOutput | Out-String).Trim())
} catch {
    Add-Check 'colmap_executable' $false $_.Exception.Message
}

try {
    $wslCommand = "test -d '$($config.pgsr_source_wsl)' && test -x '$($config.pgsr_python_wsl)' && '$($config.pgsr_python_wsl)' -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 3)'"
    $wslOutput = & wsl -d ([string]$config.wsl_distro) -- bash -lc $wslCommand 2>&1
    Add-Check 'wsl_pgsr_cuda_runtime' ($LASTEXITCODE -eq 0) (($wslOutput | Out-String).Trim())
} catch {
    Add-Check 'wsl_pgsr_cuda_runtime' $false $_.Exception.Message
}

$failed = @($checks | Where-Object { -not $_.passed })
$result = [ordered]@{
    schema_version = 'e26_installation_verification_v1'
    status = if ($failed.Count -eq 0) { 'pass' } else { 'failed' }
    package_root = $PSScriptRoot
    runtime_config = $configPath
    language_model_used = $false
    checks = $checks
    failed_check_count = $failed.Count
}
$result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $OutputReceipt -Encoding utf8
if ($failed.Count -ne 0) {
    $failedNames = ($failed | ForEach-Object { $_.name }) -join ', '
    throw "Installation verification failed: $failedNames"
}
Write-Host 'pass'
