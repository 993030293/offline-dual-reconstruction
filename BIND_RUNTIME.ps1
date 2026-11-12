[CmdletBinding()]
param(
    [string]$RuntimeRoot = 'C:\lidar-camera-fusion-mask-robust-downsampling'
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
        throw "Existing machine-local browser binding differs from the frozen package declaration: $localBindingPath"
    }
} else {
    $localBinding | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $localBindingPath -Encoding utf8
    $localBindingWritten = $true
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
        runtime_download_allowed = $false
    }
    third_party_payload_copied = $false
    portable_relocation_verified = $false
    completed_utc = [DateTime]::UtcNow.ToString('o')
}
$receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $packageRoot 'RUNTIME_BINDING_RECEIPT.json') -Encoding utf8
Write-Host 'Runtime binding completed.'
