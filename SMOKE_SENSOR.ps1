[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Raw,
    [Parameter(Mandatory = $true)]
    [string]$RunRoot,
    [string]$Case = 'package_sensor_smoke',
    [string]$BodyCameraCalibration
)

$ErrorActionPreference = 'Stop'
$arguments = @{
    Raw = $Raw
    RunRoot = $RunRoot
    Case = $Case
    StopAfter = 'sensors'
}
if ($BodyCameraCalibration) {
    $arguments.BodyCameraCalibration = $BodyCameraCalibration
}
& (Join-Path $PSScriptRoot 'RUN.ps1') @arguments
$state = Get-Content -LiteralPath (Join-Path $RunRoot 'STATE.json') -Raw | ConvertFrom-Json
if ($state.status -ne 'sensor_preflight_complete_reconstruction_pending') {
    throw "Unexpected sensor smoke status: $($state.status)"
}
$entry = Get-Content -LiteralPath (Join-Path $RunRoot 'PACKAGE_ENTRY_RESULT.json') -Raw | ConvertFrom-Json
if ($entry.status -ne 'package_entry_completed' -or $entry.stop_after -ne 'sensors') {
    throw "Unexpected package entry receipt: $($entry.status)"
}
$state | ConvertTo-Json -Depth 8
