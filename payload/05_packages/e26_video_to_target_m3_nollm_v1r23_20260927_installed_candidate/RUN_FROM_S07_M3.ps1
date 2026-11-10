param(
    [Parameter(Mandatory = $true)][string]$RawMesh,
    [Parameter(Mandatory = $true)][string]$TsdfReceipt,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [string]$Python = '',
    [string]$RuntimeConfig = (Join-Path $PSScriptRoot 'config\local_runtime.json'),
    [string]$RawMcap = '',
    [string]$RegisteredScanReceipt = '',
    [int]$PointCount = 1000000
)

$ErrorActionPreference = 'Stop'
$package = $PSScriptRoot
$ground = Join-Path $package 'ground'
$output = [System.IO.Path]::GetFullPath($OutputRoot)
$runtimeConfig = [System.IO.Path]::GetFullPath($RuntimeConfig)
$previousThreads = @{
    OMP_NUM_THREADS = $env:OMP_NUM_THREADS
    OPENBLAS_NUM_THREADS = $env:OPENBLAS_NUM_THREADS
    MKL_NUM_THREADS = $env:MKL_NUM_THREADS
    PYTHONDONTWRITEBYTECODE = $env:PYTHONDONTWRITEBYTECODE
    PYTHONPATH = $env:PYTHONPATH
}
if (Test-Path -LiteralPath $output) { throw "Output root already exists: $output" }
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
try {
    if ($PointCount -ne 1000000) { throw 'The offline viewer contract requires exactly 1000000 sampled points.' }
    if ([string]::IsNullOrWhiteSpace($Python)) {
        if (-not (Test-Path -LiteralPath $runtimeConfig -PathType Leaf)) {
            throw 'No installed runtime config; supply -Python or run INSTALL.ps1.'
        }
        $Python = [string](Get-Content -LiteralPath $runtimeConfig -Raw | ConvertFrom-Json).generic_python
    }
    $pythonPath = [System.IO.Path]::GetFullPath($Python)
    foreach ($required in @($pythonPath, $RawMesh, $TsdfReceipt, (Join-Path $ground 'config.json'))) {
        if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Required input missing: $required" }
    }
    $mcapArguments = @()
    if (-not [string]::IsNullOrWhiteSpace($RawMcap)) {
        $mcap = [System.IO.Path]::GetFullPath($RawMcap)
        if (-not (Test-Path -LiteralPath $mcap -PathType Leaf)) { throw "Required input missing: $mcap" }
        if (-not (Test-Path -LiteralPath $runtimeConfig -PathType Leaf)) {
            throw 'Raw MCAP support requires a runtime configuration.'
        }
        $runtime = Get-Content -LiteralPath $runtimeConfig -Raw | ConvertFrom-Json
        if ($runtime.schema_version -ne 'e26_local_runtime_v1') {
            throw 'Runtime configuration schema is invalid.'
        }
        $vendor = [string]$runtime.rosbags_vendor
        if (-not (Test-Path -LiteralPath $vendor -PathType Container)) {
            throw "Registered LiDAR decoder dependency missing: $vendor"
        }
        $env:PYTHONPATH = if ([string]::IsNullOrWhiteSpace($env:PYTHONPATH)) {
            $vendor
        } else {
            "$vendor;$env:PYTHONPATH"
        }
        $mcapArguments = @('--raw-mcap', $mcap)
    }
    $node = Get-Command node -ErrorAction SilentlyContinue
    if (-not [string]::IsNullOrWhiteSpace($RegisteredScanReceipt)) {
        if ([string]::IsNullOrWhiteSpace($RawMcap)) {
            throw 'Generated registered scans require their original MCAP.'
        }
        $scanReceipt = [System.IO.Path]::GetFullPath($RegisteredScanReceipt)
        if (-not (Test-Path -LiteralPath $scanReceipt -PathType Leaf)) {
            throw "Registered scan receipt missing: $scanReceipt"
        }
        $mcapArguments += @('--registered-scan-receipt', $scanReceipt)
    }
    if ($null -eq $node) { throw 'Node.js is required to validate the offline viewer.' }
    & $pythonPath (Join-Path $ground 'code\extract_m3_target.py') `
        --raw-mesh ([System.IO.Path]::GetFullPath($RawMesh)) `
        @mcapArguments `
        --tsdf-receipt ([System.IO.Path]::GetFullPath($TsdfReceipt)) `
        --config (Join-Path $ground 'config.json') `
        --output-root $output --sample-count $PointCount
    if ($LASTEXITCODE -ne 0) { throw "M3 extraction failed with exit code $LASTEXITCODE" }

    $pointCloud = Join-Path $output 'ground_removed_target_xyzrgb_f32_1m.ply'
    $html = Join-Path $output 'ground_removed_target_preview.html'
    $viewerReceipt = Join-Path $output 'offline_viewer_receipt.json'
    $viewer = Join-Path $ground 'code\viewer\build_pointcloud_interactive_v3.py'
    $validator = Join-Path $ground 'code\viewer\validate_pointcloud_interactive_v3.py'
    $layer = 'ground_removed_target_roi_after_m3_machine_component_filter'
    $subtitle = 'Observed target ROI surface | read-only display | no surface completion'
    $note = 'M3 ground removal, contact protection and geometry component filtering completed. Historical S09/S26 semantic review and surface completion were not run.'
    $boundary = 'The preview shows observed surface inside the metric ROI after frozen M3 ground removal. It is not a semantic instance guarantee, complete surface, watertight mesh, or ground-truth validated segmentation.'
    & $pythonPath $viewer `
        --input-ply $pointCloud --output-html $html --output-receipt $viewerReceipt `
        --title 'M3 ground-removed target ROI' --layer-state $layer --subtitle $subtitle `
        --boundary-heading 'Machine-only M3 target extraction' --boundary-note $note `
        --claim-boundary $boundary --component-filter-state 'applied_machine_geometry'
    if ($LASTEXITCODE -ne 0) { throw "Offline viewer generation failed with exit code $LASTEXITCODE" }
    & $pythonPath $validator `
        --viewer-script $viewer --input-ply $pointCloud --viewer-html $html `
        --viewer-receipt $viewerReceipt --node-executable $node.Source `
        --output-validation (Join-Path $output 'offline_viewer_validation.json') `
        --expected-layer-state $layer --expected-subtitle $subtitle `
        --expected-boundary-note $note --expected-claim-boundary $boundary `
        --expected-component-filter-state 'applied_machine_geometry'
    if ($LASTEXITCODE -ne 0) { throw "Offline viewer validation failed with exit code $LASTEXITCODE" }

    $delivery = [ordered]@{
        schema_version = 'e26_m3_ground_target_delivery_v1'
        status = 'complete'
        method = 'm3_grid_mrf'
        language_model_used = $false
        visual_language_model_used = $false
        manual_visual_decision_used = $false
        glass_processing_used = $false
        connected_component_filter_applied = $true
        historical_s09_executed = $false
        preview_html = $html
        preview_validation = (Join-Path $output 'offline_viewer_validation.json')
        extraction_receipt = (Join-Path $output 'EXTRACTION_RECEIPT.json')
    }
    $delivery | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $output 'DELIVERY.json') -Encoding utf8
    & $pythonPath (Join-Path $ground 'code\audit_m3_delivery.py') --output-root $output --package-root $package
    if ($LASTEXITCODE -ne 0) { throw "M3 delivery audit failed with exit code $LASTEXITCODE" }
    Write-Output $html
} catch {
    $failure = [ordered]@{
        schema_version = 'e26_m3_wrapper_failure_v1'
        status = 'failed_closed'
        error_type = $_.Exception.GetType().FullName
        error = $_.Exception.Message
        output_root = $output
        derived_output_reusable = $false
    }
    $failurePath = if (Test-Path -LiteralPath $output) {
        Join-Path $output 'WRAPPER_FAILURE.json'
    } else {
        "$output.FAILURE.json"
    }
    New-Item -ItemType Directory -Path (Split-Path -Parent $failurePath) -Force | Out-Null
    $failure | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $failurePath -Encoding utf8
    throw
} finally {
    $env:OMP_NUM_THREADS = $previousThreads.OMP_NUM_THREADS
    $env:OPENBLAS_NUM_THREADS = $previousThreads.OPENBLAS_NUM_THREADS
    $env:MKL_NUM_THREADS = $previousThreads.MKL_NUM_THREADS
    $env:PYTHONDONTWRITEBYTECODE = $previousThreads.PYTHONDONTWRITEBYTECODE
    $env:PYTHONPATH = $previousThreads.PYTHONPATH
}
