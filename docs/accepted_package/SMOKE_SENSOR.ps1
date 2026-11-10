[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Raw,
    [Parameter(Mandatory = $true)]
    [string]$RunRoot,
    [string]$Case = 'package_sensor_smoke'
)

$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'RUN.ps1') -Raw $Raw -RunRoot $RunRoot -Case $Case -StopAfter sensors
$state = Get-Content -LiteralPath (Join-Path $RunRoot 'STATE.json') -Raw | ConvertFrom-Json
if ($state.status -ne 'sensor_preflight_complete_reconstruction_pending') {
    throw "Unexpected sensor smoke status: $($state.status)"
}
$state | ConvertTo-Json -Depth 8
