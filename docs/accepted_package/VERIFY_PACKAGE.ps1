[CmdletBinding()]
param(
    [string]$Output = (Join-Path $PSScriptRoot 'PACKAGE_VERIFY_RESULT.json'),
    [switch]$RequireRuntime
)

$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot 'payload\05_envs\sfm_py311_v1\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Run BIND_RUNTIME.ps1 before package verification.'
}
$arguments = @(
    (Join-Path $PSScriptRoot 'tools\verify_package.py'),
    '--package-root', $PSScriptRoot,
    '--output', $Output
)
if ($RequireRuntime) { $arguments += '--require-runtime' }
& $python @arguments
if ($LASTEXITCODE -ne 0) { throw 'Package verification failed.' }
Get-Content -LiteralPath $Output -Raw
