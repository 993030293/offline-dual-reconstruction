[CmdletBinding()]
param(
    [string]$Output,
    [switch]$RequireRuntime
)

$ErrorActionPreference = 'Stop'
$packageRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $packageRoot 'PACKAGE_VERIFY_RESULT.json'
}
$resolvedOutput = [IO.Path]::GetFullPath($Output)
$outputParent = Split-Path -Parent $resolvedOutput
New-Item -ItemType Directory -Path $outputParent -Force | Out-Null
$python = Join-Path $packageRoot 'payload\05_envs\sfm_py311_v1\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Run BIND_RUNTIME.ps1 before package verification.'
}

$publicationOutput = Join-Path $outputParent 'PUBLICATION_VERIFY_RESULT.json'
& $python (Join-Path $packageRoot 'tools\verify_publication.py') --root $packageRoot --output $publicationOutput
if ($LASTEXITCODE -ne 0) {
    throw 'Publication content verification failed.'
}
$publication = Get-Content -LiteralPath $publicationOutput -Raw | ConvertFrom-Json

$runtimeChecks = @()
$runtimeFailures = @()
if ($RequireRuntime) {
    $binding = Get-Content -LiteralPath (Join-Path $packageRoot 'config\runtime.binding.json') -Raw | ConvertFrom-Json
    foreach ($link in $binding.links) {
        $expected = [IO.Path]::GetFullPath((Join-Path $binding.supported_runtime_root $link.source)).TrimEnd('\')
        $path = [IO.Path]::GetFullPath((Join-Path (Join-Path $packageRoot 'payload') $link.relative))
        $exists = Test-Path -LiteralPath $path
        $isLink = $false
        $target = $null
        $matches = $false
        if ($exists) {
            $item = Get-Item -LiteralPath $path -Force
            $isLink = [bool]($item.Attributes -band [IO.FileAttributes]::ReparsePoint)
            if ($isLink) {
                $target = [string]($item.Target | Select-Object -First 1)
                $matches = [IO.Path]::GetFullPath($target).TrimEnd('\') -eq $expected
            }
        }
        $runtimeChecks += [ordered]@{
            path = $path
            expected_target = $expected
            exists = $exists
            is_runtime_link = $isLink
            target_matches = $matches
        }
        if (-not ($exists -and $isLink -and $matches)) {
            $runtimeFailures += $path
        }
    }
    $localBindingPath = Join-Path $packageRoot 'payload\02_code\reconstruction_dual_v1_20261009\OFFLINE_RUNTIME.local.json'
    $localBindingExists = Test-Path -LiteralPath $localBindingPath -PathType Leaf
    $browserMatches = $false
    $declaredNode = $null
    $declaredPlaywright = $null
    $actualNode = $null
    $actualPlaywright = $null
    if ($null -ne $binding.browser_runtime -and $localBindingExists) {
        try {
            $localBinding = Get-Content -LiteralPath $localBindingPath -Raw | ConvertFrom-Json
            $declaredNode = [IO.Path]::GetFullPath([string]$binding.browser_runtime.node)
            $declaredPlaywright = [IO.Path]::GetFullPath([string]$binding.browser_runtime.playwright_package)
            $actualNode = [IO.Path]::GetFullPath([string]$localBinding.node)
            $actualPlaywright = [IO.Path]::GetFullPath([string]$localBinding.playwright_package)
            $browserMatches = (
                $actualNode -eq $declaredNode -and
                $actualPlaywright -eq $declaredPlaywright -and
                -not [bool]$binding.browser_runtime.runtime_download_allowed -and
                -not [bool]$localBinding.runtime_download_allowed -and
                (Test-Path -LiteralPath $actualNode -PathType Leaf) -and
                (Test-Path -LiteralPath (Join-Path $actualPlaywright 'index.js') -PathType Leaf)
            )
        } catch {
            $browserMatches = $false
        }
    }
    $runtimeChecks += [ordered]@{
        kind = 'offline_browser_binding'
        path = $localBindingPath
        exists = $localBindingExists
        declared_node = $declaredNode
        actual_node = $actualNode
        declared_playwright = $declaredPlaywright
        actual_playwright = $actualPlaywright
        target_matches = $browserMatches
        runtime_download_allowed = $false
    }
    if (-not $browserMatches) {
        $runtimeFailures += $localBindingPath
    }
    $target = $binding.target_geometry_runtime
    $targetConfigPath = $null
    $targetMatches = $false
    $targetDependencyFailures = @()
    if ($null -ne $target) {
        $targetPackage = [IO.Path]::GetFullPath((Join-Path (Join-Path $packageRoot 'payload') ([string]$target.package_relative)))
        $targetConfigPath = Join-Path $targetPackage 'config\local_runtime.json'
        if (Test-Path -LiteralPath $targetConfigPath -PathType Leaf) {
            try {
                $targetLocal = Get-Content -LiteralPath $targetConfigPath -Raw | ConvertFrom-Json
                $expectedTargetPaths = [ordered]@{
                    colmap = [IO.Path]::GetFullPath((Join-Path $binding.supported_runtime_root ([string]$target.colmap_source)))
                    extrinsic_json = [IO.Path]::GetFullPath((Join-Path (Join-Path $packageRoot 'payload') ([string]$target.extrinsic_payload_relative)))
                    sfm_python = [IO.Path]::GetFullPath((Join-Path $binding.supported_runtime_root ([string]$target.sfm_python_source)))
                    generic_python = [IO.Path]::GetFullPath((Join-Path $binding.supported_runtime_root ([string]$target.generic_python_source)))
                    rosbags_vendor = [IO.Path]::GetFullPath((Join-Path $binding.supported_runtime_root ([string]$target.rosbags_vendor_source)))
                }
                foreach ($entry in $expectedTargetPaths.GetEnumerator()) {
                    $pathType = if ($entry.Key -eq 'rosbags_vendor') { 'Container' } else { 'Leaf' }
                    if (-not (Test-Path -LiteralPath $entry.Value -PathType $pathType)) {
                        $targetDependencyFailures += $entry.Value
                    }
                }
                $targetMatches = (
                    [string]$targetLocal.schema_version -eq [string]$target.schema_version -and
                    [IO.Path]::GetFullPath([string]$targetLocal.colmap) -eq $expectedTargetPaths.colmap -and
                    [IO.Path]::GetFullPath([string]$targetLocal.extrinsic_json) -eq $expectedTargetPaths.extrinsic_json -and
                    [IO.Path]::GetFullPath([string]$targetLocal.sfm_python) -eq $expectedTargetPaths.sfm_python -and
                    [IO.Path]::GetFullPath([string]$targetLocal.generic_python) -eq $expectedTargetPaths.generic_python -and
                    [IO.Path]::GetFullPath([string]$targetLocal.rosbags_vendor) -eq $expectedTargetPaths.rosbags_vendor -and
                    [string]$targetLocal.wsl_distro -eq [string]$target.wsl_distro -and
                    [string]$targetLocal.pgsr_source_wsl -eq [string]$target.pgsr_source_wsl -and
                    [string]$targetLocal.pgsr_python_wsl -eq [string]$target.pgsr_python_wsl -and
                    [int]$targetLocal.reliable_lidar.candidate_count -eq [int]$target.reliable_lidar.candidate_count -and
                    [int]$targetLocal.reliable_lidar.image_count -eq [int]$target.reliable_lidar.image_count -and
                    [double]$targetLocal.reliable_lidar.start_fraction -eq [double]$target.reliable_lidar.start_fraction -and
                    [double]$targetLocal.reliable_lidar.end_fraction -eq [double]$target.reliable_lidar.end_fraction -and
                    [int]$targetLocal.legacy_pgsr.candidate_count -eq [int]$target.legacy_pgsr.candidate_count -and
                    [int]$targetLocal.legacy_pgsr.image_count -eq [int]$target.legacy_pgsr.image_count -and
                    $targetDependencyFailures.Count -eq 0
                )
            } catch {
                $targetMatches = $false
            }
        }
    }
    $runtimeChecks += [ordered]@{
        kind = 'target_geometry_runtime_binding'
        path = $targetConfigPath
        exists = [bool]($targetConfigPath -and (Test-Path -LiteralPath $targetConfigPath -PathType Leaf))
        target_matches = $targetMatches
        dependency_failures = $targetDependencyFailures
        runtime_download_allowed = $false
    }
    if (-not $targetMatches) {
        if ($targetConfigPath) {
            $runtimeFailures += $targetConfigPath
        } else {
            $runtimeFailures += 'config/runtime.binding.json#target_geometry_runtime'
        }
    }
}

$result = [ordered]@{
    status = if ($publication.status -eq 'passed' -and $runtimeFailures.Count -eq 0) { 'package_verified' } else { 'package_verification_failed' }
    package_id = 'offline_dual_reconstruction_github_v0_1_20261110'
    publication_result = $publicationOutput
    publication_status = $publication.status
    publication_file_count = $publication.file_count
    runtime_required = [bool]$RequireRuntime
    runtime_checks = $runtimeChecks
    runtime_failures = $runtimeFailures
    runtime_downloads_allowed = $false
    runtime_language_model_required = $false
    portable_relocation_verified = $false
}
$result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $resolvedOutput -Encoding utf8
if ($result.status -ne 'package_verified') {
    throw 'Package verification failed.'
}
Get-Content -LiteralPath $resolvedOutput -Raw
