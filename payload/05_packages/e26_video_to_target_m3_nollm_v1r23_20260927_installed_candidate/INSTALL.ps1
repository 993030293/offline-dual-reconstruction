param(
    [Parameter(Mandatory = $true)][string]$InstallRoot,
    [string]$WorkspaceRoot = '',
    [string]$Colmap = '',
    [string]$ExtrinsicJson = '',
    [string]$SfmPython = '',
    [string]$GenericPython = '',
    [string]$RosbagsVendor = '',
    [string]$WslDistro = 'E26-C2-LIV-20260907-134802',
    [string]$PgsrSourceWsl = '/opt/e26_thin_surface_v1/pgsr',
    [string]$PgsrPythonWsl = '/opt/e26_thin_surface_v1/venv/bin/python'
)

$ErrorActionPreference = 'Stop'
$source = $PSScriptRoot
$destination = [System.IO.Path]::GetFullPath($InstallRoot)
if (Test-Path -LiteralPath $destination) {
    throw "Install root already exists: $destination"
}
if ([string]::IsNullOrWhiteSpace($WorkspaceRoot)) {
    $packagesRoot = Split-Path -Parent $source
    if ((Split-Path -Leaf $packagesRoot) -eq '05_packages') {
        $WorkspaceRoot = Split-Path -Parent $packagesRoot
    } else {
        $WorkspaceRoot = 'C:\lidar-camera-fusion-mask-robust-downsampling'
    }
}
$workspace = [System.IO.Path]::GetFullPath($WorkspaceRoot)
if ([string]::IsNullOrWhiteSpace($Colmap)) { $Colmap = Join-Path $workspace '05_envs\colmap-4.1.1-cuda\bin\colmap.exe' }
if ([string]::IsNullOrWhiteSpace($ExtrinsicJson)) { $ExtrinsicJson = Join-Path $workspace '02_code\e26_orbit_visual_mvs_v1_20260818\configs\e26_trajectory_binding_v1.json' }
if ([string]::IsNullOrWhiteSpace($SfmPython)) { $SfmPython = Join-Path $workspace '05_envs\sfm_py311_v1\Scripts\python.exe' }
if ([string]::IsNullOrWhiteSpace($GenericPython)) { $GenericPython = Join-Path $workspace '05_envs\e26_generic_side_bottom_v1\Scripts\python.exe' }
if ([string]::IsNullOrWhiteSpace($RosbagsVendor)) { $RosbagsVendor = Join-Path $workspace '05_envs\e26_rosbags_vendor_v1' }

& (Join-Path $source 'VERIFY_PACKAGE.ps1') -GenericPython $GenericPython
if ($LASTEXITCODE -ne 0) { throw 'Source package verification failed' }

$manifestPath = Join-Path $source 'MANIFEST.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
New-Item -ItemType Directory -Path $destination | Out-Null
foreach ($entry in $manifest.files) {
    $relative = [string]$entry.path
    $sourceFile = Join-Path $source $relative
    $destinationFile = Join-Path $destination $relative
    $destinationParent = Split-Path -Parent $destinationFile
    if (-not (Test-Path -LiteralPath $destinationParent -PathType Container)) {
        New-Item -ItemType Directory -Path $destinationParent -Force | Out-Null
    }
    Copy-Item -LiteralPath $sourceFile -Destination $destinationFile
}
Copy-Item -LiteralPath $manifestPath -Destination (Join-Path $destination 'MANIFEST.json')

$localConfig = [ordered]@{
    schema_version = 'e26_local_runtime_v1'
    colmap = [System.IO.Path]::GetFullPath($Colmap)
    extrinsic_json = [System.IO.Path]::GetFullPath($ExtrinsicJson)
    sfm_python = [System.IO.Path]::GetFullPath($SfmPython)
    generic_python = [System.IO.Path]::GetFullPath($GenericPython)
    rosbags_vendor = [System.IO.Path]::GetFullPath($RosbagsVendor)
    wsl_distro = $WslDistro
    pgsr_source_wsl = $PgsrSourceWsl
    pgsr_python_wsl = $PgsrPythonWsl
    reliable_lidar = [ordered]@{
        candidate_count = 960
        image_count = 180
        start_fraction = 0.10
        end_fraction = 0.68
    }
    legacy_pgsr = [ordered]@{
        candidate_count = 480
        image_count = 180
    }
}
$configPath = Join-Path $destination 'config\local_runtime.json'
$localConfig | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $configPath -Encoding utf8

& (Join-Path $destination 'VERIFY_PACKAGE.ps1') -GenericPython $GenericPython -SkipTests
if ($LASTEXITCODE -ne 0) { throw 'Installed package integrity verification failed' }
& (Join-Path $destination 'VERIFY_INSTALLATION.ps1') -RuntimeConfig $configPath
if ($LASTEXITCODE -ne 0) { throw 'Installed runtime verification failed' }

$manifestHash = (Get-FileHash -LiteralPath (Join-Path $destination 'MANIFEST.json') -Algorithm SHA256).Hash.ToLowerInvariant()
$receipt = [ordered]@{
    schema_version = 'e26_install_receipt_v1'
    status = 'installed_and_runtime_verified'
    installed_at_utc = [DateTime]::UtcNow.ToString('o')
    source_package = $source
    install_root = $destination
    package_id = $manifest.package_id
    package_manifest_sha256 = $manifestHash
    installed_manifest_file_count = $manifest.file_count
    runtime_config = $configPath
    runtime_verification = (Join-Path $destination 'INSTALL_VERIFY.json')
    language_model_used = $false
}
$receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $destination 'INSTALL_RECEIPT.json') -Encoding utf8
Write-Host $destination
