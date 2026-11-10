param(
    [Parameter(Mandatory = $true)][string]$RawMcap,
    [Parameter(Mandatory = $true)][string]$ExpectedMcapSha256,
    [Parameter(Mandatory = $true)][string]$OutputRoot,
    [string]$Colmap = 'C:\lidar-camera-fusion-mask-robust-downsampling\05_envs\colmap-4.1.1-cuda\bin\colmap.exe',
    [string]$ExtrinsicJson = 'C:\lidar-camera-fusion-mask-robust-downsampling\02_code\e26_orbit_visual_mvs_v1_20260818\configs\e26_trajectory_binding_v1.json',
    [string]$SfmPython = 'C:\lidar-camera-fusion-mask-robust-downsampling\05_envs\sfm_py311_v1\Scripts\python.exe',
    [string]$GenericPython = 'C:\lidar-camera-fusion-mask-robust-downsampling\05_envs\e26_generic_side_bottom_v1\Scripts\python.exe',
    [string]$RosbagsVendor = 'C:\lidar-camera-fusion-mask-robust-downsampling\05_envs\e26_rosbags_vendor_v1',
    [string]$WslDistro = 'E26-C2-LIV-20260907-134802',
    [string]$PgsrSourceWsl = '/opt/e26_thin_surface_v1/pgsr',
    [string]$PgsrPythonWsl = '/opt/e26_thin_surface_v1/venv/bin/python',
    [int]$CandidateCount = 480,
    [int]$ImageCount = 180
)

$ErrorActionPreference = 'Stop'
$package = $PSScriptRoot
$runtime = Join-Path $package 'code\validated_fresh_runtime'
$alignmentPolicy = Join-Path $package 'config\METRIC_ALIGNMENT_POLICY.json'
$frontend = Join-Path $package 'code\reference_sources\e26_c2_support\frontend\prepare_e26_orbit_visual_mvs_workspace.py'
$sfmTools = Join-Path $package 'code\reference_sources\e26_c2_support\frontend\tools'
$lioTools = Join-Path $package 'code\reference_sources\c0_lio_fallback\tools'
$geometry = Join-Path $package 'code\reference_sources\e26_c2_support\geometry'
$run = [System.IO.Path]::GetFullPath($OutputRoot)
$mcap = [System.IO.Path]::GetFullPath($RawMcap)
$colmapPath = [System.IO.Path]::GetFullPath($Colmap)
$extrinsic = [System.IO.Path]::GetFullPath($ExtrinsicJson)
$sfmPythonPath = [System.IO.Path]::GetFullPath($SfmPython)
$genericPythonPath = [System.IO.Path]::GetFullPath($GenericPython)
$rosbagsVendorPath = [System.IO.Path]::GetFullPath($RosbagsVendor)

foreach ($path in @($mcap, $colmapPath, $extrinsic, $frontend, $sfmPythonPath, $genericPythonPath, $alignmentPolicy)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Required file does not exist: $path" }
}
if (-not (Test-Path -LiteralPath $rosbagsVendorPath -PathType Container)) {
    throw "Required rosbags vendor directory does not exist: $rosbagsVendorPath"
}
if (Test-Path -LiteralPath $run) { throw "Output root already exists: $run" }
$initialPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = if ([string]::IsNullOrWhiteSpace($initialPythonPath)) {
    $rosbagsVendorPath
} else {
    "$rosbagsVendorPath;$initialPythonPath"
}

function Invoke-SfmPythonChecked {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $sfmPythonPath @Arguments
        $pythonExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($pythonExitCode -ne 0) { throw "SfM Python stage failed with exit code $pythonExitCode" }
}

function Invoke-GenericPythonChecked {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $genericPythonPath @Arguments
        $pythonExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($pythonExitCode -ne 0) { throw "Generic Python stage failed with exit code $pythonExitCode" }
}

function Convert-ToWslPath {
    param([Parameter(Mandatory = $true)][string]$WindowsPath)
    $portableWindowsPath = $WindowsPath.Replace('\', '/')
    $converted = & wsl -d $WslDistro -- wslpath -a $portableWindowsPath
    if ($LASTEXITCODE -ne 0) { throw "wslpath failed for $WindowsPath" }
    $normalized = $converted.Trim()
    if (-not $normalized.StartsWith('/')) { throw "wslpath returned a non-absolute path for $WindowsPath`: $normalized" }
    return $normalized
}

Invoke-SfmPythonChecked (Join-Path $runtime 'initialize_fresh_run.py') `
    --run-root $run --mcap $mcap --expected-mcap-sha256 $ExpectedMcapSha256 `
    --candidate-count $CandidateCount --image-count $ImageCount

Invoke-SfmPythonChecked $frontend --mcap $mcap --sfm-tools $sfmTools `
    --output-dir (Join-Path $run '01_visual_workspace') `
    --candidate-count $CandidateCount --image-count $ImageCount `
    --start-fraction 0 --end-fraction 1 `
    --expected-mcap-sha256 $ExpectedMcapSha256 --skip-full-mcap-hash

& (Join-Path $package 'RUN_MACHINE_S02.ps1') `
    -ImageManifest (Join-Path $run '01_visual_workspace\image_manifest.json') `
    -ImageRoot (Join-Path $run '01_visual_workspace\images') `
    -OutputReceipt (Join-Path $run '02_machine\s02_machine_frame_inventory.json') `
    -Python $sfmPythonPath
if ($LASTEXITCODE -ne 0) { throw "Machine-only S02 failed with exit code $LASTEXITCODE" }

Invoke-SfmPythonChecked (Join-Path $runtime 'run_sparse_from_extracted.py') `
    --workspace (Join-Path $run '01_visual_workspace') `
    --frontend $frontend --colmap $colmapPath `
    --s02-receipt (Join-Path $run '02_machine\s02_machine_frame_inventory.json')

$metricDetectionText = & $sfmPythonPath (Join-Path $runtime 'detect_mcap_metric_source.py') --mcap $mcap
if ($LASTEXITCODE -ne 0) { throw "Metric source detection failed with exit code $LASTEXITCODE" }
$metricDetection = $metricDetectionText | ConvertFrom-Json
$metricDetectionText | Set-Content -LiteralPath (Join-Path $run '00_contract\metric_source_detection.json') -Encoding utf8
if ($metricDetection.metric_source -eq 'embedded_odometry') {
    Invoke-SfmPythonChecked (Join-Path $runtime 'extract_e26_mcap_metric_context.py') `
        --mcap $mcap --expected-mcap-sha256 $ExpectedMcapSha256 `
        --output (Join-Path $run '03_metric_context') `
        --extrinsic-json $extrinsic --skip-full-mcap-hash
} elseif ($metricDetection.metric_source -eq 'raw_lidar_imu_fallback') {
    $lioRoot = Join-Path $run '03_lio_fallback'
    $endpointRoot = Join-Path $lioRoot 'endpoint_visual_constraint'
    $trajectoryRoot = Join-Path $lioRoot 'lio_point_to_plane_visual_rotation'
    $runtimePythonPath = $env:PYTHONPATH
    $env:PYTHONPATH = "$lioTools;$sfmTools;$rosbagsVendorPath"
    try {
        Invoke-SfmPythonChecked (Join-Path $lioTools 'endpoint_visual_constraint.py') `
            --mcap $mcap --output-dir $endpointRoot
        Invoke-SfmPythonChecked (Join-Path $lioTools 'lio_local_map.py') `
            --mcap $mcap --output-dir $trajectoryRoot `
            --rotation-constraint (Join-Path $endpointRoot 'endpoint_visual_constraint.json') `
            --point-to-plane
    } finally {
        $env:PYTHONPATH = $runtimePythonPath
    }
    Invoke-SfmPythonChecked (Join-Path $runtime 'extract_e26_mcap_metric_context.py') `
        --mcap $mcap --expected-mcap-sha256 $ExpectedMcapSha256 `
        --output (Join-Path $run '03_metric_context') `
        --extrinsic-json $extrinsic --skip-full-mcap-hash `
        --trajectory-file (Join-Path $trajectoryRoot 'lio_local_map.tum')
} else {
    throw "Unsupported metric source: $($metricDetection.metric_source)"
}
Invoke-SfmPythonChecked (Join-Path $runtime 'prepare_metric_configs.py') `
    --run-root $run --policy $alignmentPolicy

$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = "$(Join-Path $geometry 'src');$rosbagsVendorPath"
try {
    Invoke-SfmPythonChecked (Join-Path $runtime 'run_e26_trajectory_binding.py') `
        --config (Join-Path $run '03_metric_context\trajectory_binding_config.json') `
        --output (Join-Path $run '04_trajectory_binding')
    Invoke-SfmPythonChecked (Join-Path $runtime 'run_e26_sim3_initialization.py') `
        --config (Join-Path $run '03_metric_context\sim3_config.json') `
        --output (Join-Path $run '05_sim3')
    Invoke-SfmPythonChecked (Join-Path $runtime 'prepare_metric_pgsr_input_inliers.py') `
        --dense-workspace (Join-Path $run '01_visual_workspace\dense_source_workspace') `
        --registered-manifest (Join-Path $run '01_visual_workspace\registered_image_manifest.json') `
        --metric-bound-manifest (Join-Path $run '03_metric_context\registered_metric_bound_manifest.json') `
        --metric-time-support-receipt (Join-Path $run '03_metric_context\metric_time_support_receipt.json') `
        --baseline-sparse-model (Join-Path $run '01_visual_workspace\sparse\0') `
        --sim3 (Join-Path $run '05_sim3\sim3.json') `
        --sim3-residuals (Join-Path $run '05_sim3\residuals.json') `
        --pose-priors (Join-Path $run '05_sim3\pose_prior_manifest.json') `
        --policy $alignmentPolicy `
        --output (Join-Path $run '06_pgsr_inlier\input')
} finally {
    $env:PYTHONPATH = $previousPythonPath
}

$runWsl = Convert-ToWslPath $run
$trainScriptWsl = Convert-ToWslPath (Join-Path $runtime 'train_pgsr_supervised.py')
$tsdfScriptWsl = Convert-ToWslPath (Join-Path $runtime 'run_auto_roi_tsdf.py')
$trainCommand = "cd '$PgsrSourceWsl' && PYTHONDONTWRITEBYTECODE=1 '$PgsrPythonWsl' '$trainScriptWsl' --run-root '$runWsl' --source '$PgsrSourceWsl' --stage-name 06_pgsr_inlier"
& wsl -d $WslDistro -- bash -lc $trainCommand
if ($LASTEXITCODE -ne 0) { throw "PGSR training failed with exit code $LASTEXITCODE" }
$tsdfCommand = "cd '$PgsrSourceWsl' && PYTHONDONTWRITEBYTECODE=1 '$PgsrPythonWsl' '$tsdfScriptWsl' --run-root '$runWsl' --source '$PgsrSourceWsl' --stage-name 06_pgsr_inlier"
& wsl -d $WslDistro -- bash -lc $tsdfCommand
if ($LASTEXITCODE -ne 0) { throw "Automatic ROI TSDF failed with exit code $LASTEXITCODE" }

Invoke-GenericPythonChecked (Join-Path $runtime 'fit_ground_plane_receipt.py') `
    --raw-mesh (Join-Path $run '07_roi_tsdf\paperbox_auto_roi_pgsr_5mm_raw_mesh.ply') `
    --tsdf-receipt (Join-Path $run '07_roi_tsdf\roi_tsdf_receipt.json') `
    --output-receipt (Join-Path $run '07_roi_tsdf\ground_filter_receipt.json')
& (Join-Path $package 'RUN_S08_PREVIEW.ps1') `
    -RawS08Mesh (Join-Path $run '07_roi_tsdf\paperbox_auto_roi_pgsr_5mm_raw_mesh.ply') `
    -GroundFilterReceipt (Join-Path $run '07_roi_tsdf\ground_filter_receipt.json') `
    -OutputRoot (Join-Path $run '08_s08_preview') `
    -Python $genericPythonPath
if ($LASTEXITCODE -ne 0) { throw "S08 preview stage failed with exit code $LASTEXITCODE" }
Invoke-GenericPythonChecked (Join-Path $runtime 'finalize_fresh_run.py') --run-root $run
Invoke-GenericPythonChecked (Join-Path $package 'code\finalize_integrated_run.py') `
    --run-root $run --profile LegacyPgsr `
    --tsdf-stage 07_roi_tsdf --tsdf-receipt-name roi_tsdf_receipt.json `
    --preview-stage 08_s08_preview

Write-Host (Join-Path $run 'INTEGRATED_FINAL_STATUS.json')
