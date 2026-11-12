[CmdletBinding()]
param(
    [string]$RuntimeRoot = 'C:\lidar-camera-fusion-mask-robust-downsampling',
    [switch]$RefreshLocalBindings
)

$ErrorActionPreference = 'Stop'
$packageRoot = $PSScriptRoot
$config = Get-Content -LiteralPath (Join-Path $packageRoot 'config\runtime.binding.json') -Raw | ConvertFrom-Json
$resolvedRuntime = [IO.Path]::GetFullPath($RuntimeRoot).TrimEnd('\')
$supportedRuntime = [IO.Path]::GetFullPath($config.supported_runtime_root).TrimEnd('\')
if ($resolvedRuntime -ne $supportedRuntime) {
    throw "This release is verified only with runtime root $supportedRuntime. Relocation has not been accepted."
}

$payloadRoot = Join-Path $packageRoot 'payload'
$bound = @()
foreach ($link in $config.links) {
    $source = [IO.Path]::GetFullPath((Join-Path $resolvedRuntime $link.source))
    $destination = [IO.Path]::GetFullPath((Join-Path $payloadRoot $link.relative))
    if (-not $destination.StartsWith($payloadRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Runtime link escaped the package payload: $destination"
    }
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Required installed runtime is missing: $source"
    }
    $parent = Split-Path -Parent $destination
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    if (Test-Path -LiteralPath $destination) {
        $item = Get-Item -LiteralPath $destination -Force
        if (-not ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
            throw "Package runtime destination already contains a real file or directory: $destination"
        }
        $actualTarget = [string]($item.Target | Select-Object -First 1)
        if ([IO.Path]::GetFullPath($actualTarget).TrimEnd('\') -ne $source.TrimEnd('\')) {
            throw "Existing runtime link points elsewhere: $destination"
        }
    } else {
        New-Item -ItemType Junction -Path $destination -Target $source | Out-Null
    }
    $bound += [ordered]@{ destination = $destination; source = $source }
}

$browser = $config.browser_runtime
if ($null -eq $browser) {
    throw 'The frozen runtime binding does not declare an offline browser runtime.'
}
$node = [IO.Path]::GetFullPath([string]$browser.node)
$playwright = [IO.Path]::GetFullPath([string]$browser.playwright_package)
if (-not (Test-Path -LiteralPath $node -PathType Leaf)) {
    throw "The declared offline Node executable is missing: $node"
}
if (-not (Test-Path -LiteralPath (Join-Path $playwright 'index.js') -PathType Leaf)) {
    throw "The declared offline Playwright package is missing: $playwright"
}
$localBindingPath = Join-Path $packageRoot 'payload\02_code\reconstruction_dual_v1_20261009\OFFLINE_RUNTIME.local.json'
$localBinding = [ordered]@{
    source = [string]$browser.source
    bundle_version = [string]$browser.bundle_version
    node = $node.Replace('\', '/')
    playwright_package = $playwright.Replace('\', '/')
    runtime_download_allowed = $false
}
$localBindingWritten = $false
$localBindingReplaced = $false
if (Test-Path -LiteralPath $localBindingPath) {
    $existing = Get-Content -LiteralPath $localBindingPath -Raw | ConvertFrom-Json
    $existingNode = [IO.Path]::GetFullPath([string]$existing.node)
    $existingPlaywright = [IO.Path]::GetFullPath([string]$existing.playwright_package)
    $matches = (
        $existingNode -eq $node -and
        $existingPlaywright -eq $playwright -and
        [string]$existing.source -eq [string]$browser.source -and
        [string]$existing.bundle_version -eq [string]$browser.bundle_version -and
        -not [bool]$existing.runtime_download_allowed
    )
    if (-not $matches) {
        if (-not $RefreshLocalBindings) {
            throw "Existing machine-local browser binding differs from the frozen package declaration. Re-run with -RefreshLocalBindings only after reviewing the frozen paths: $localBindingPath"
        }
        $localBinding | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $localBindingPath -Encoding utf8
        $localBindingWritten = $true
        $localBindingReplaced = $true
    }
} else {
    $localBinding | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $localBindingPath -Encoding utf8
    $localBindingWritten = $true
}

$target = $config.target_geometry_runtime
if ($null -eq $target) {
    throw 'The frozen runtime binding does not declare the target geometry runtime.'
}
$targetPackage = [IO.Path]::GetFullPath((Join-Path (Join-Path $packageRoot 'payload') ([string]$target.package_relative)))
if (-not $targetPackage.StartsWith($payloadRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Target geometry runtime path escaped the package payload: $targetPackage"
}
$targetConfigPath = Join-Path $targetPackage 'config\local_runtime.json'
$targetLocalBinding = [ordered]@{
    schema_version = [string]$target.schema_version
    colmap = [IO.Path]::GetFullPath((Join-Path $resolvedRuntime ([string]$target.colmap_source)))
    extrinsic_json = [IO.Path]::GetFullPath((Join-Path (Join-Path $packageRoot 'payload') ([string]$target.extrinsic_payload_relative)))
    sfm_python = [IO.Path]::GetFullPath((Join-Path $resolvedRuntime ([string]$target.sfm_python_source)))
    generic_python = [IO.Path]::GetFullPath((Join-Path $resolvedRuntime ([string]$target.generic_python_source)))
    rosbags_vendor = [IO.Path]::GetFullPath((Join-Path $resolvedRuntime ([string]$target.rosbags_vendor_source)))
    wsl_distro = [string]$target.wsl_distro
    pgsr_source_wsl = [string]$target.pgsr_source_wsl
    pgsr_python_wsl = [string]$target.pgsr_python_wsl
    reliable_lidar = [ordered]@{
        candidate_count = [int]$target.reliable_lidar.candidate_count
        image_count = [int]$target.reliable_lidar.image_count
        start_fraction = [double]$target.reliable_lidar.start_fraction
        end_fraction = [double]$target.reliable_lidar.end_fraction
    }
    legacy_pgsr = [ordered]@{
        candidate_count = [int]$target.legacy_pgsr.candidate_count
        image_count = [int]$target.legacy_pgsr.image_count
    }
}
$targetRequirements = @(
    @($targetLocalBinding.colmap, 'Leaf'),
    @($targetLocalBinding.extrinsic_json, 'Leaf'),
    @($targetLocalBinding.sfm_python, 'Leaf'),
    @($targetLocalBinding.generic_python, 'Leaf'),
    @($targetLocalBinding.rosbags_vendor, 'Container')
)
foreach ($requirement in $targetRequirements) {
    if (-not (Test-Path -LiteralPath $requirement[0] -PathType $requirement[1])) {
        throw "A declared target geometry runtime dependency is missing: $($requirement[0])"
    }
}
$targetBindingWritten = $false
$targetBindingReplaced = $false
if (Test-Path -LiteralPath $targetConfigPath) {
    $existingTarget = Get-Content -LiteralPath $targetConfigPath -Raw | ConvertFrom-Json
    $targetMatches = (
        [string]$existingTarget.schema_version -eq $targetLocalBinding.schema_version -and
        [IO.Path]::GetFullPath([string]$existingTarget.colmap) -eq $targetLocalBinding.colmap -and
        [IO.Path]::GetFullPath([string]$existingTarget.extrinsic_json) -eq $targetLocalBinding.extrinsic_json -and
        [IO.Path]::GetFullPath([string]$existingTarget.sfm_python) -eq $targetLocalBinding.sfm_python -and
        [IO.Path]::GetFullPath([string]$existingTarget.generic_python) -eq $targetLocalBinding.generic_python -and
        [IO.Path]::GetFullPath([string]$existingTarget.rosbags_vendor) -eq $targetLocalBinding.rosbags_vendor -and
        [string]$existingTarget.wsl_distro -eq $targetLocalBinding.wsl_distro -and
        [string]$existingTarget.pgsr_source_wsl -eq $targetLocalBinding.pgsr_source_wsl -and
        [string]$existingTarget.pgsr_python_wsl -eq $targetLocalBinding.pgsr_python_wsl -and
        [int]$existingTarget.reliable_lidar.candidate_count -eq $targetLocalBinding.reliable_lidar.candidate_count -and
        [int]$existingTarget.reliable_lidar.image_count -eq $targetLocalBinding.reliable_lidar.image_count -and
        [double]$existingTarget.reliable_lidar.start_fraction -eq $targetLocalBinding.reliable_lidar.start_fraction -and
        [double]$existingTarget.reliable_lidar.end_fraction -eq $targetLocalBinding.reliable_lidar.end_fraction -and
        [int]$existingTarget.legacy_pgsr.candidate_count -eq $targetLocalBinding.legacy_pgsr.candidate_count -and
        [int]$existingTarget.legacy_pgsr.image_count -eq $targetLocalBinding.legacy_pgsr.image_count
    )
    if (-not $targetMatches) {
        if (-not $RefreshLocalBindings) {
            throw "Existing target geometry runtime binding differs from the frozen package declaration. Re-run with -RefreshLocalBindings only after reviewing the frozen paths: $targetConfigPath"
        }
        $targetLocalBinding | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $targetConfigPath -Encoding utf8
        $targetBindingWritten = $true
        $targetBindingReplaced = $true
    }
} else {
    $targetLocalBinding | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $targetConfigPath -Encoding utf8
    $targetBindingWritten = $true
}

$receipt = [ordered]@{
    status = 'same_machine_runtime_bound'
    package_id = 'offline_dual_reconstruction_github_v0_1_20261110'
    runtime_root = $resolvedRuntime
    links = $bound
    browser_binding = [ordered]@{
        local_config = $localBindingPath
        node = $node
        playwright_package = $playwright
        generated = $localBindingWritten
        replaced = $localBindingReplaced
        runtime_download_allowed = $false
    }
    target_geometry_binding = [ordered]@{
        local_config = $targetConfigPath
        generated = $targetBindingWritten
        replaced = $targetBindingReplaced
        extrinsic_json = $targetLocalBinding.extrinsic_json
        runtime_download_allowed = $false
    }
    third_party_payload_copied = $false
    portable_relocation_verified = $false
    completed_utc = [DateTime]::UtcNow.ToString('o')
}
$receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $packageRoot 'RUNTIME_BINDING_RECEIPT.json') -Encoding utf8
Write-Host 'Runtime binding completed.'
