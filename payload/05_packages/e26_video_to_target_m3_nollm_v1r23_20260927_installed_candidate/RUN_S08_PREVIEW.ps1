param(
    [Parameter(Mandatory = $true)][string]$RawS08Mesh,
    [Parameter(Mandatory = $true)][string]$GroundFilterReceipt,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [int]$ExpectedTriangles = 0,
    [int]$PointCount = 1000000,
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
$resolvedOutput = [System.IO.Path]::GetFullPath($OutputRoot)
if (Test-Path -LiteralPath $resolvedOutput) {
    throw "Output already exists: $resolvedOutput"
}
New-Item -ItemType Directory -Path $resolvedOutput | Out-Null
$runtime = Join-Path $PSScriptRoot 'code\runtime'
$viewer = Join-Path $PSScriptRoot 'code\viewer\build_pointcloud_interactive_v3.py'
$validator = Join-Path $PSScriptRoot 'code\viewer\validate_pointcloud_interactive_v3.py'
$mesh = Join-Path $resolvedOutput 's08_ground_removed_observed_surface_mesh.ply'
$ply = Join-Path $resolvedOutput 's08_after_ground_removal_before_component_filter_xyzrgb_f32_1m.ply'
$previewReceipt = Join-Path $resolvedOutput 's08_preview_derivative_receipt.json'
$builderArgs = @(
    (Join-Path $runtime 'build_s08_preview_input.py'),
    '--raw-mesh', ([System.IO.Path]::GetFullPath($RawS08Mesh)),
    '--ground-receipt', ([System.IO.Path]::GetFullPath($GroundFilterReceipt)),
    '--output-mesh', $mesh,
    '--output-ply', $ply,
    '--output-receipt', $previewReceipt,
    '--sample-count', $PointCount
)
if ($ExpectedTriangles -gt 0) {
    $builderArgs += @('--expected-triangles', $ExpectedTriangles)
}
& $Python @builderArgs
if ($LASTEXITCODE -ne 0) { throw "S08 preview input construction failed with exit code $LASTEXITCODE" }

$html = Join-Path $resolvedOutput 's08_after_ground_removal_before_component_filter.html'
$viewerReceipt = Join-Path $resolvedOutput 's08_offline_viewer_receipt.json'
& $Python $viewer `
    --input-ply $ply `
    --output-html $html `
    --output-receipt $viewerReceipt `
    --title 'S08 - Ground removed, before component filtering'
if ($LASTEXITCODE -ne 0) { throw "S08 offline viewer construction failed with exit code $LASTEXITCODE" }

$node = Get-Command node -ErrorAction SilentlyContinue
if ($null -ne $node) {
    & $Python $validator `
        --viewer-script $viewer `
        --input-ply $ply `
        --viewer-html $html `
        --viewer-receipt $viewerReceipt `
        --node-executable $node.Source `
        --output-validation (Join-Path $resolvedOutput 's08_offline_viewer_validation.json')
    if ($LASTEXITCODE -ne 0) { throw "S08 viewer validation failed with exit code $LASTEXITCODE" }
} else {
    Write-Warning 'Node.js was not found; JavaScript syntax validation was skipped.'
}
Write-Host $html
