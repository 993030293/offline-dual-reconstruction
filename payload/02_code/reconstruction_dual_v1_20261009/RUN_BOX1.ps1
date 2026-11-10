param([string]$RunRoot = "", [switch]$Resume, [ValidateSet("sensors", "front", "complete")][string]$StopAfter = "complete", [string]$RepairNote = "", [ValidateSet("AdaptiveSensor", "ReliableLidar", "VisualTargetDiagnostic")][string]$PointProfile = "AdaptiveSensor")
$ErrorActionPreference = "Stop"
$taskWorkspace = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (-not $RunRoot) {
    $RunRoot = Join-Path $taskWorkspace ("04_outputs/dual_box1_delivery_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
}
$taskRaw = "E:/lidar_reconstruction_video_data/raw_recordings/e26_orbit_capture_20260820_box_pair_v1/raw_mcap/20260820A_Box1/recon_collection_20260820_170154_044592_0.mcap"
$taskPython = Join-Path $taskWorkspace "05_envs/sfm_py311_v1/Scripts/python.exe"
$taskArguments = @((Join-Path $PSScriptRoot "run_target_delivery.py"), "--run-root", $RunRoot, "--raw", $taskRaw, "--case", "box1", "--stop-after", $StopAfter, "--point-profile", $PointProfile)
if ($Resume) { $taskArguments += "--resume" }
if ($RepairNote) { $taskArguments += @("--repair-note", $RepairNote) }
$env:PYTHONUTF8 = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
& $taskPython @taskArguments
exit $LASTEXITCODE
