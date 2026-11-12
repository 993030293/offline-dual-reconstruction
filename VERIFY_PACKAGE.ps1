[CmdletBinding()]
param(
    [string]$Output = (Join-Path $PSScriptRoot 'PACKAGE_VERIFY_RESULT.json'),
    [switch]$RequireRuntime
)

$ErrorActionPreference = 'Stop'
$packageRoot = $PSScriptRoot
$python = Join-Path $packageRoot 'payload\05_envs\sfm_py311_v1\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Run BIND_RUNTIME.ps1 before package verification.'
}

$publicationOutput = Join-Path ([IO.Path]::GetFullPath((Split-Path -Parent $Output))) 'PUBLICATION_VERIFY_RESULT.json'
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
$result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Output -Encoding utf8
if ($result.status -ne 'package_verified') {
    throw 'Package verification failed.'
}
Get-Content -LiteralPath $Output -Raw
