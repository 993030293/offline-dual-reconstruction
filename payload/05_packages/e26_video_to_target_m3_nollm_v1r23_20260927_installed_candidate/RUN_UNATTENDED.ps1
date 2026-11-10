param(
    [Parameter(Mandatory = $true)][string]$RawMcap,
    [Parameter(Mandatory = $true)][string]$ExpectedMcapSha256,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [ValidateSet('Auto', 'ReliableLidar', 'LegacyPgsr')][string]$Profile = 'Auto',
    [string]$RuntimeConfig = (Join-Path $PSScriptRoot 'config\local_runtime.json')
)

$ErrorActionPreference = 'Stop'
$root = [System.IO.Path]::GetFullPath($OutputRoot)
if (Test-Path -LiteralPath $root) { throw "Output root already exists: $root" }
$upstreamRoot = Join-Path $root 'S00_S08'
$m3Root = Join-Path $root '09_ground_target_m3'
$stage = 'initialize'
New-Item -ItemType Directory -Path $root -Force | Out-Null
$log = Join-Path $root 'PIPELINE_RUN_LOG.jsonl'
$previousNoBytecode = $env:PYTHONDONTWRITEBYTECODE
$env:PYTHONDONTWRITEBYTECODE = '1'
function Write-Stage {
    param([string]$Name, [string]$Status)
    $row = [ordered]@{ at_utc = [DateTime]::UtcNow.ToString('o'); stage = $Name; status = $Status }
    Add-Content -LiteralPath $log -Value ($row | ConvertTo-Json -Compress) -Encoding utf8
}
try {
    if ($ExpectedMcapSha256 -notmatch '^[a-fA-F0-9]{64}$') {
        throw 'ExpectedMcapSha256 must be a 64-digit SHA-256 value.'
    }
    $config = Get-Content -LiteralPath ([System.IO.Path]::GetFullPath($RuntimeConfig)) -Raw | ConvertFrom-Json
    if ($config.schema_version -ne 'e26_local_runtime_v1') { throw 'Runtime configuration schema is invalid.' }
    Write-Stage 'S00_S08' 'running'
    $stage = 'S00_S08'
    & (Join-Path $PSScriptRoot 'RUN_UPSTREAM_S00_S08.ps1') `
        -RawMcap $RawMcap -ExpectedMcapSha256 $ExpectedMcapSha256 `
        -OutputRoot $upstreamRoot -Profile $Profile -RuntimeConfig $RuntimeConfig
    $upstreamStatusPath = Join-Path $upstreamRoot 'INTEGRATED_FINAL_STATUS.json'
    if (-not (Test-Path -LiteralPath $upstreamStatusPath -PathType Leaf)) {
        throw 'S00-S08 did not produce its integrated final status.'
    }
    $upstream = Get-Content -LiteralPath $upstreamStatusPath -Raw | ConvertFrom-Json
    if ($upstream.status -ne 'complete_s08_observed_surface' -or $upstream.language_model_used -ne $false) {
        throw 'S00-S08 did not meet the completed machine-only contract.'
    }
    if ($upstream.profile -eq 'ReliableLidar') {
        $tsdfPath = Join-Path $upstreamRoot '07_reliable_lidar_tsdf\reliable_lidar_tsdf_receipt.json'
    } elseif ($upstream.profile -eq 'LegacyPgsr') {
        $tsdfPath = Join-Path $upstreamRoot '07_roi_tsdf\roi_tsdf_receipt.json'
    } else {
        throw "Unsupported completed profile: $($upstream.profile)"
    }
    $tsdf = Get-Content -LiteralPath $tsdfPath -Raw | ConvertFrom-Json
    foreach ($flag in @('language_model_used', 'semantic_model_used', 'object_mask_used')) {
        if ($tsdf.$flag -ne $false) { throw "TSDF does not prove $flag=false" }
    }
    $meshName = [System.IO.Path]::GetFileName(([string]$tsdf.mesh.path).Replace('\', '/'))
    if ([string]::IsNullOrWhiteSpace($meshName) -or $meshName -notlike '*.ply') {
        throw 'TSDF receipt does not name a PLY mesh.'
    }
    $rawMesh = Join-Path (Split-Path -Parent $tsdfPath) $meshName
    if (-not (Test-Path -LiteralPath $rawMesh -PathType Leaf)) { throw "TSDF raw mesh missing: $rawMesh" }
    Write-Stage 'S00_S08' 'complete'

    $stage = 'M3_ground_target'
    Write-Stage $stage 'running'
    & (Join-Path $PSScriptRoot 'RUN_FROM_S07_M3.ps1') `
        -RawMesh $rawMesh -TsdfReceipt $tsdfPath -OutputRoot $m3Root `
        -Python ([string]$config.generic_python) -RawMcap $RawMcap `
        -RuntimeConfig $RuntimeConfig
    $auditPath = Join-Path $m3Root 'FINAL_AUDIT.json'
    $extractionPath = Join-Path $m3Root 'EXTRACTION_RECEIPT.json'
    $audit = Get-Content -LiteralPath $auditPath -Raw | ConvertFrom-Json
    $extraction = Get-Content -LiteralPath $extractionPath -Raw | ConvertFrom-Json
    if ($audit.status -ne 'pass' -or $extraction.input.upstream_provenance.status -ne 'explicit_machine_only_flags') {
        throw 'M3 audit or explicit no-LLM upstream provenance failed.'
    }
    Write-Stage $stage 'complete'
    $stage = 'target_support_quality'
    Write-Stage $stage 'running'
    $qualityPath = Join-Path $m3Root 'TARGET_SUPPORT_AUDIT.json'
    $metricSparse = Join-Path $upstreamRoot '06_pgsr_inlier\input\sparse\points3D.txt'
    & ([string]$config.generic_python) (Join-Path $PSScriptRoot 'code\audit_target_preservation.py') `
        --metric-sparse $metricSparse --raw-mesh $rawMesh `
        --target-mesh ([string]$extraction.output.mesh) `
        --tsdf-receipt $tsdfPath --output $qualityPath
    if ($LASTEXITCODE -ne 0) { throw 'Target support quality audit failed.' }
    $quality = Get-Content -LiteralPath $qualityPath -Raw | ConvertFrom-Json
    if ($quality.status -ne 'pass') { throw 'Target support quality audit did not pass.' }
    Write-Stage $stage 'complete'
    $final = [ordered]@{
        schema_version = 'e26_video_to_target_m3_final_v2'
        status = 'complete_machine_only_observed_target_preview'
        profile = $upstream.profile
        method = 'm3_grid_mrf'
        raw_mcap = [System.IO.Path]::GetFullPath($RawMcap)
        upstream_status = $upstreamStatusPath
        tsdf_receipt = $tsdfPath
        m3_receipt = $extractionPath
        m3_audit = $auditPath
        target_support_audit = $qualityPath
        observed_target_mesh = [string]$extraction.output.mesh
        offline_preview_html = (Join-Path $m3Root 'ground_removed_target_preview.html')
        language_model_used = $false
        visual_language_model_used = $false
        manual_visual_decision_used = $false
        glass_processing_used = $false
        evaluation_boundary = 'Engineering preview with metric sparse support preservation gate; no independent geometry ground truth or heldout ground-purity metric.'
    }
    $final | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $root 'FINAL_STATUS.json') -Encoding utf8
    Write-Output $final.offline_preview_html
} catch {
    Write-Stage $stage 'failed'
    $upstreamFailure = Join-Path $upstreamRoot 'PIPELINE_FAILURE.json'
    if (-not (Test-Path -LiteralPath $upstreamFailure -PathType Leaf)) {
        $upstreamFailure = "$upstreamRoot.FAILURE.json"
    }
    $m3Failure = Join-Path $m3Root 'WRAPPER_FAILURE.json'
    if (-not (Test-Path -LiteralPath $m3Failure -PathType Leaf)) {
        $m3Failure = "$m3Root.FAILURE.json"
    }
    $failure = [ordered]@{
        schema_version = 'e26_video_to_target_m3_failure_v1'
        status = 'failed_closed'
        stage = $stage
        error_type = $_.Exception.GetType().FullName
        error = $_.Exception.Message
        raw_mcap = [System.IO.Path]::GetFullPath($RawMcap)
        upstream_failure = $upstreamFailure
        upstream_failure_exists = (Test-Path -LiteralPath $upstreamFailure -PathType Leaf)
        m3_failure = $m3Failure
        m3_failure_exists = (Test-Path -LiteralPath $m3Failure -PathType Leaf)
        derived_output_reusable = $false
    }
    $failure | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $root 'PIPELINE_FAILURE.json') -Encoding utf8
    throw
} finally {
    $env:PYTHONDONTWRITEBYTECODE = $previousNoBytecode
}
