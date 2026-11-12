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

$receipt = [ordered]@{
    status = 'same_machine_runtime_bound'
    package_id = 'offline_dual_reconstruction_github_v0_1_20261110'
    runtime_root = $resolvedRuntime
    links = $bound
    third_party_payload_copied = $false
    portable_relocation_verified = $false
    completed_utc = [DateTime]::UtcNow.ToString('o')
}
$receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $packageRoot 'RUNTIME_BINDING_RECEIPT.json') -Encoding utf8
Write-Host 'Runtime binding completed.'
