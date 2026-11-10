param([string]$RunRoot = "", [switch]$Resume, [ValidateSet("sensors", "front", "complete")][string]$StopAfter = "complete", [string]$RepairNote = "", [ValidateSet("AdaptiveSensor", "ReliableLidar", "VisualTargetDiagnostic")][string]$PointProfile = "AdaptiveSensor", [string]$BodyCameraCalibration = "", [ValidateSet("initial", "sfm", "recovery", "temporal", "pose_gap", "metric", "metric_sim3_renderer_recovery", "roles", "export")][string]$RetryFrontStage = "")
$ErrorActionPreference = "Stop"
$taskWorkspace = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if ($Resume -and -not $RunRoot) {
    throw "Resume requires the explicit existing RunRoot."
}
if ($RetryFrontStage -and (-not $Resume -or -not $RepairNote)) {
    throw "RetryFrontStage requires Resume and a recorded RepairNote."
}
if (-not $RunRoot) {
    $RunRoot = Join-Path $taskWorkspace ("04_outputs/dual_c1_delivery_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
}
if (-not $BodyCameraCalibration) {
    $BodyCameraCalibration = Join-Path $taskWorkspace "02_code/e26_orbit_visual_mvs_v1_20260818/configs/e26_trajectory_binding_v1.json"
}
$taskRaw = "E:/lidar_reconstruction_video_data/raw_recordings/e26_orbit_capture_20260818_v1/raw_mcap/recon_collection_20260818_202249_217576/recon_collection_20260818_202249_217576_0.mcap"
$taskPython = Join-Path $taskWorkspace "05_envs/sfm_py311_v1/Scripts/python.exe"
$taskArguments = @((Join-Path $PSScriptRoot "run_target_delivery.py"), "--run-root", $RunRoot, "--raw", $taskRaw, "--case", "c1", "--stop-after", $StopAfter, "--point-profile", $PointProfile, "--body-camera-calibration", $BodyCameraCalibration)
if ($Resume) { $taskArguments += "--resume" }
if ($RepairNote) { $taskArguments += @("--repair-note", $RepairNote) }
if ($RetryFrontStage) { $taskArguments += @("--retry-front-stage", $RetryFrontStage) }
$env:PYTHONUTF8 = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
& $taskPython @taskArguments
exit $LASTEXITCODE
