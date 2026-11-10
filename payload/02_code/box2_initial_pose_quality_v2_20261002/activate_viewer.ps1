$ErrorActionPreference = 'Stop'
$taskRun = 'C:\lidar-camera-fusion-mask-robust-downsampling\04_outputs\box2_initial_pose_quality_v2_20261002T193102'
$taskRoot = Join-Path $taskRun 'viewer'
$taskOldRoot = 'C:\lidar-camera-fusion-mask-robust-downsampling\04_outputs\box2_lio_guided_sfm_v1_20261002T014330\viewer'
$taskCode = 'C:\lidar-camera-fusion-mask-robust-downsampling\02_code\box2_initial_pose_quality_v2_20261002'
$taskPython = 'C:\lidar-camera-fusion-mask-robust-downsampling\05_envs\sfm_py311_v1\Scripts\python.exe'
$taskReview = Get-Content -LiteralPath (Join-Path $taskRoot 'BROWSER_VALIDATION.json') -Raw | ConvertFrom-Json
if ($taskReview.status -ne 'passed') { throw 'The new preview has not passed its browser checks.' }
$taskListeners = @(Get-NetTCPConnection -LocalPort 8195 -State Listen -ErrorAction SilentlyContinue)
$taskStopped = $null
if ($taskListeners.Count -gt 0) {
    if ($taskListeners.Count -ne 1 -or $taskListeners[0].LocalAddress -ne '127.0.0.1') {
        throw 'Port 8195 has an unexpected listener.'
    }
    $taskOwnerPid = $taskListeners[0].OwningProcess
    $taskOldState = Get-Content -LiteralPath (Join-Path $taskOldRoot 'SERVER_STATE.json') -Raw | ConvertFrom-Json
    $taskProcess = Get-CimInstance Win32_Process -Filter "ProcessId=$taskOwnerPid"
    $taskNormalized = $taskProcess.CommandLine.Replace('\', '/').ToLowerInvariant()
    $taskExpectedRoot = $taskOldRoot.Replace('\', '/').ToLowerInvariant()
    if ($taskOwnerPid -ne $taskOldState.pid -or !$taskNormalized.Contains($taskExpectedRoot) -or !$taskNormalized.Contains('serve_viewer.py')) {
        throw 'The listener is not the previously verified Box2 viewer.'
    }
    Stop-Process -Id $taskOwnerPid
    $taskStopped = @{ pid = $taskOwnerPid; root = $taskOldRoot; command_line = $taskProcess.CommandLine }
}
$taskServer = Join-Path $taskCode 'serve_viewer.py'
$taskLaunched = Start-Process -FilePath $taskPython -ArgumentList @($taskServer, '--root', $taskRoot, '--port', '8195') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskRoot 'server_8195.stdout.log') -RedirectStandardError (Join-Path $taskRoot 'server_8195.stderr.log')
$taskReceipt = @{ status = 'launched_pending_HTTP_and_browser_check'; new_root = $taskRoot; launcher_pid = $taskLaunched.Id; stopped_previous_owned_server = $taskStopped; old_models_and_outputs_modified = $false; switch_time = (Get-Date).ToString('o') }
$taskReceipt | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $taskRoot 'SERVICE_HANDOFF.json') -Encoding utf8
$taskReceipt | ConvertTo-Json -Depth 4
