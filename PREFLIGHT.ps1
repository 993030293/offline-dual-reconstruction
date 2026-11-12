[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$OutputRoot
)

$ErrorActionPreference = 'Stop'
$packageRoot = $PSScriptRoot
$resolvedOutput = [IO.Path]::GetFullPath($OutputRoot)
New-Item -ItemType Directory -Path $resolvedOutput -Force | Out-Null
& (Join-Path $packageRoot 'VERIFY_PACKAGE.ps1') -Output (Join-Path $resolvedOutput 'PACKAGE_VERIFY_RESULT.json') -RequireRuntime | Out-Null
$python = Join-Path $packageRoot 'payload\05_envs\sfm_py311_v1\Scripts\python.exe'
$preflight = Join-Path $packageRoot 'payload\02_code\reconstruction_dual_v1_20261009\preflight.py'
& $python $preflight --run-root $resolvedOutput --output (Join-Path $resolvedOutput 'PREFLIGHT.json')
if ($LASTEXITCODE -ne 0) { throw 'Production dependency preflight failed.' }
Get-Content -LiteralPath (Join-Path $resolvedOutput 'PREFLIGHT.json') -Raw
