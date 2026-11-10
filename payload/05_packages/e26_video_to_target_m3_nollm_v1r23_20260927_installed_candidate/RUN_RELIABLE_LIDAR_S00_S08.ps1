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
    [int]$CandidateCount = 960,
    [int]$ImageCount = 180,
    [double]$StartFraction = 0.10,
    [double]$EndFraction = 0.68
)

$ErrorActionPreference = 'Stop'
$package = $PSScriptRoot
$code = Join-Path $package 'code'
$runtime = Join-Path $code 'validated_fresh_runtime'
$alignmentPolicy = Join-Path $package 'config\METRIC_ALIGNMENT_POLICY.json'
$frontend = Join-Path $code 'reference_sources\e26_c2_support\frontend\prepare_e26_orbit_visual_mvs_workspace.py'
$sfmTools = Join-Path $code 'reference_sources\e26_c2_support\frontend\tools'
$geometry = Join-Path $code 'reference_sources\e26_c2_support\geometry'
$run = [System.IO.Path]::GetFullPath($OutputRoot)
$mcap = [System.IO.Path]::GetFullPath($RawMcap)
$sfmPythonPath = [System.IO.Path]::GetFullPath($SfmPython)
$genericPythonPath = [System.IO.Path]::GetFullPath($GenericPython)
$rosbagsVendorPath = [System.IO.Path]::GetFullPath($RosbagsVendor)

foreach ($path in @($mcap, $Colmap, $ExtrinsicJson, $sfmPythonPath, $genericPythonPath, $frontend, $alignmentPolicy)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required file does not exist: $path"
    }
}
if (-not (Test-Path -LiteralPath $rosbagsVendorPath -PathType Container)) {
    throw "Required directory does not exist: $rosbagsVendorPath"
}
if (Test-Path -LiteralPath $run) {
    throw "Output root already exists: $run"
}

function Invoke-PythonChecked {
    param(
        [Parameter(Mandatory = $true)][string]$Python,
        [Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments
    )
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = 'Continue'
        & $Python @Arguments
        $pythonExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($pythonExitCode -ne 0) {
        throw "Python stage failed with exit code $pythonExitCode"
    }
}

function Convert-ToWslPath {
    param([Parameter(Mandatory = $true)][string]$WindowsPath)
    $portable = $WindowsPath.Replace('\', '/')
    $converted = & wsl -d $WslDistro -- wslpath -a $portable
    if ($LASTEXITCODE -ne 0) {
        throw "wslpath failed for $WindowsPath"
    }
    return $converted.Trim()
}

$initialPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = if ([string]::IsNullOrWhiteSpace($initialPythonPath)) {
    $rosbagsVendorPath
} else {
    "$rosbagsVendorPath;$initialPythonPath"
}
try {
    Invoke-PythonChecked $sfmPythonPath (Join-Path $runtime 'initialize_fresh_run.py') `
        --run-root $run --mcap $mcap --expected-mcap-sha256 $ExpectedMcapSha256 `
        --candidate-count $CandidateCount --image-count $ImageCount

    $metricDetectionText = & $sfmPythonPath (Join-Path $runtime 'detect_mcap_metric_source.py') --mcap $mcap
    if ($LASTEXITCODE -ne 0) { throw 'Metric source detection failed' }
    $metricDetection = $metricDetectionText | ConvertFrom-Json
    $metricDetectionText | Set-Content -LiteralPath (Join-Path $run '00_contract\metric_source_detection.json') -Encoding utf8
    if ($metricDetection.metric_source -ne 'embedded_odometry') {
        throw "Reliable LiDAR fusion requires embedded odometry and registered point clouds; detected $($metricDetection.metric_source)"
    }
    Invoke-PythonChecked $sfmPythonPath (Join-Path $runtime 'extract_e26_mcap_metric_context.py') `
        --mcap $mcap --expected-mcap-sha256 $ExpectedMcapSha256 `
        --output (Join-Path $run '03_metric_context') `
        --extrinsic-json ([System.IO.Path]::GetFullPath($ExtrinsicJson)) --skip-full-mcap-hash

    Invoke-PythonChecked $sfmPythonPath (Join-Path $code 'prepare_pose_quality_workspace.py') `
        --frontend $frontend --mcap $mcap --sfm-tools $sfmTools `
        --output-dir (Join-Path $run '01_visual_workspace') `
        --trajectory (Join-Path $run '03_metric_context\metric_odometry.tum') `
        --extrinsic (Join-Path $run '03_metric_context\T_base_link_camera_optical.json') `
        --candidate-count $CandidateCount --image-count $ImageCount `
        --start-fraction $StartFraction --end-fraction $EndFraction `
        --expected-mcap-sha256 $ExpectedMcapSha256 --skip-full-mcap-hash

    & (Join-Path $package 'RUN_MACHINE_S02.ps1') `
        -ImageManifest (Join-Path $run '01_visual_workspace\image_manifest.json') `
        -ImageRoot (Join-Path $run '01_visual_workspace\images') `
        -OutputReceipt (Join-Path $run '02_machine\s02_machine_frame_inventory.json') `
        -Python $sfmPythonPath
    if ($LASTEXITCODE -ne 0) { throw 'Machine S02 failed' }

    Invoke-PythonChecked $sfmPythonPath (Join-Path $runtime 'run_sparse_from_extracted.py') `
        --workspace (Join-Path $run '01_visual_workspace') `
        --frontend $frontend --colmap ([System.IO.Path]::GetFullPath($Colmap)) `
        --s02-receipt (Join-Path $run '02_machine\s02_machine_frame_inventory.json')
    Invoke-PythonChecked $sfmPythonPath (Join-Path $runtime 'prepare_metric_configs.py') `
        --run-root $run --policy $alignmentPolicy

    $env:PYTHONPATH = "$(Join-Path $geometry 'src');$rosbagsVendorPath"
    Invoke-PythonChecked $sfmPythonPath (Join-Path $runtime 'run_e26_trajectory_binding.py') `
        --config (Join-Path $run '03_metric_context\trajectory_binding_config.json') `
        --output (Join-Path $run '04_trajectory_binding')
    Invoke-PythonChecked $sfmPythonPath (Join-Path $runtime 'run_e26_sim3_initialization.py') `
        --config (Join-Path $run '03_metric_context\sim3_config.json') `
        --output (Join-Path $run '05_sim3')
    $env:PYTHONPATH = "$(Join-Path $geometry 'src');$rosbagsVendorPath"

    Invoke-PythonChecked $sfmPythonPath (Join-Path $runtime 'prepare_metric_pgsr_input_inliers.py') `
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
    Invoke-PythonChecked $sfmPythonPath (Join-Path $code 'derive_vehicle_roi.py') `
        --metric-sparse (Join-Path $run '06_pgsr_inlier\input\sparse') `
        --output (Join-Path $run '06_pgsr_inlier\vehicle_roi_receipt.json')
    Invoke-PythonChecked $sfmPythonPath (Join-Path $code 'extract_stable_registered_lidar.py') `
        --mcap $mcap `
        --manifest (Join-Path $run '01_visual_workspace\registered_image_manifest.json') `
        --roi-receipt (Join-Path $run '06_pgsr_inlier\vehicle_roi_receipt.json') `
        --output (Join-Path $run '06_pgsr_inlier\stable_lidar')

    $runWsl = Convert-ToWslPath $run
    $trainScriptWsl = Convert-ToWslPath (Join-Path $runtime 'train_pgsr_supervised.py')
    $fusionScriptWsl = Convert-ToWslPath (Join-Path $code 'run_reliable_lidar_tsdf.py')
    $roiWsl = Convert-ToWslPath (Join-Path $run '06_pgsr_inlier\vehicle_roi_receipt.json')
    $lidarWsl = Convert-ToWslPath (Join-Path $run '06_pgsr_inlier\stable_lidar\stable_lidar_receipt.json')
    $trainCommand = "cd '$PgsrSourceWsl' && PYTHONDONTWRITEBYTECODE=1 '$PgsrPythonWsl' '$trainScriptWsl' --run-root '$runWsl' --source '$PgsrSourceWsl' --stage-name 06_pgsr_inlier"
    & wsl -d $WslDistro -- bash -lc $trainCommand
    if ($LASTEXITCODE -ne 0) { throw 'PGSR training failed' }
    $fusionCommand = "cd '$PgsrSourceWsl' && PYTHONDONTWRITEBYTECODE=1 '$PgsrPythonWsl' '$fusionScriptWsl' --run-root '$runWsl' --source '$PgsrSourceWsl' --stage-name 06_pgsr_inlier --roi-receipt '$roiWsl' --lidar-receipt '$lidarWsl' --output-stage 07_reliable_lidar_tsdf"
    & wsl -d $WslDistro -- bash -lc $fusionCommand
    if ($LASTEXITCODE -ne 0) { throw 'Reliable LiDAR TSDF fusion failed' }

    $rawMesh = Join-Path $run '07_reliable_lidar_tsdf\tesla_reliable_lidar_pgsr_5mm_raw_mesh.ply'
    $tsdfReceipt = Join-Path $run '07_reliable_lidar_tsdf\reliable_lidar_tsdf_receipt.json'
    $groundReceipt = Join-Path $run '07_reliable_lidar_tsdf\ground_filter_receipt.json'
    Invoke-PythonChecked $genericPythonPath (Join-Path $code 'fit_vehicle_ground_plane_receipt.py') `
        --raw-mesh $rawMesh --raw-mcap $mcap `
        --tsdf-receipt $tsdfReceipt --output-receipt $groundReceipt
    & (Join-Path $package 'RUN_S08_PREVIEW.ps1') `
        -RawS08Mesh $rawMesh -GroundFilterReceipt $groundReceipt `
        -OutputRoot (Join-Path $run '08_s08_reliable_preview') -Python $genericPythonPath
    if ($LASTEXITCODE -ne 0) { throw 'S08 preview generation failed' }
    Invoke-PythonChecked $genericPythonPath (Join-Path $code 'finalize_integrated_run.py') `
        --run-root $run --profile ReliableLidar `
        --tsdf-stage 07_reliable_lidar_tsdf `
        --tsdf-receipt-name reliable_lidar_tsdf_receipt.json `
        --preview-stage 08_s08_reliable_preview
} finally {
    $env:PYTHONPATH = $initialPythonPath
}

Write-Host (Join-Path $run 'INTEGRATED_FINAL_STATUS.json')
