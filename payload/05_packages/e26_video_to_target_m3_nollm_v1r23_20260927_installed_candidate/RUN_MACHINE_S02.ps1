param(
    [Parameter(Mandatory = $true)][string]$ImageManifest,
    [Parameter(Mandatory = $true)][string]$ImageRoot,
    [Parameter(Mandatory = $true)][string]$OutputReceipt,
    [string]$Python = 'python',
    [switch]$SkipHashVerification
)

$argsList = @(
    (Join-Path $PSScriptRoot 'code\runtime\machine_frame_inventory.py'),
    '--image-manifest', ([System.IO.Path]::GetFullPath($ImageManifest)),
    '--image-root', ([System.IO.Path]::GetFullPath($ImageRoot)),
    '--output', ([System.IO.Path]::GetFullPath($OutputReceipt))
)
if (-not $SkipHashVerification) { $argsList += '--verify-hashes' }
& $Python @argsList
if ($LASTEXITCODE -ne 0) { throw "Machine S02 failed with exit code $LASTEXITCODE" }
