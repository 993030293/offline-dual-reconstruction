# Box2 记录位姿辅助 SfM：执行与复现入口

本目录实现已有 SfM 地图的记录里程计位置先验 BA，以及按粗位姿覆盖和图像质量增加视角、注册、再次 BA、点云导出、3DGS 和留出图像评估。第一轮实际运行目录为：

`C:/lidar-camera-fusion-mask-robust-downsampling/04_outputs/box2_lio_guided_sfm_v1_20261002T014330`。

结果与局限见 [执行报告](../../04_outputs/box2_lio_guided_sfm_v1_20261002T014330/RUN_REPORT.md)和 [执行记录](../../04_outputs/box2_lio_guided_sfm_v1_20261002T014330/FINAL_RECEIPT.json)。本轮没有重放 Point-LIO，也没有从头运行 `pose_prior_mapper`；MCAP 轨迹的精确生产者版本未确认。H 已被用于本轮最终解释，不应继续称为新的盲测。

## 环境及输入

- Windows Python：`05_envs/sfm_py311_v1/Scripts/python.exe`，PyCOLMAP 4.1.1。
- 原视觉工作区：`04_outputs/e26_box2_fresh_intervention_audit_20260928/box2_a1_raw/S00_S08/01_visual_workspace`。
- 原流程与投影 / 过滤配置：`05_packages/e26_video_to_target_m3_nollm_v1r23_20260927_installed_candidate`。
- 原 MCAP：`E:/lidar_reconstruction_video_data/raw_recordings/e26_orbit_capture_20260820_box_pair_v1/raw_mcap/20260820B_Box2/recon_collection_20260820_170743_689578_0.mcap`。
- WSL：`E26-C2-LIV-20260907-134802`；GPU Python：`/opt/e26_thin_surface_v1/venv/bin/python`。
- INRIA 源码与 rasterizer 环境：`04_outputs/box2_3dgs_display_v1_20260930`，commit `54c035f7834b564019656c3e3fcc3646292f727d`。
- Playwright CLI：已实际通过 `npx.cmd --yes --package @playwright/cli playwright-cli` 调用。此主机 Node 25.7.0 使用终端方式运行 CLI，避免非 TTY 的 libuv 关闭异常。

源模型、源数据库、源图像及 MCAP 保持只读。除指定查看器元数据修复外，脚本拒绝覆盖已有阶段产物；复现实验必须使用新的运行目录。运行前需要检查资源与输入，不应直接重跑第一轮目录。

## Windows 阶段

以下是逐阶段入口，参数与真实执行入口一致。函数检查 Python 非零退出码。示例会进行真实计算，只应在需要新运行时执行。

```powershell
$taskRoot = 'C:/lidar-camera-fusion-mask-robust-downsampling'
$taskCode = Join-Path $taskRoot '02_code/box2_lio_guided_sfm_v1_20261002'
$taskPython = Join-Path $taskRoot '05_envs/sfm_py311_v1/Scripts/python.exe'
$taskRun = Join-Path $taskRoot ('04_outputs/box2_lio_guided_sfm_v1_' + (Get-Date -Format 'yyyyMMddTHHmmss'))
if (Test-Path -LiteralPath $taskRun) { throw 'Use a new run directory.' }

function Invoke-Stage {
    param([string]$Script, [string[]]$StageArgs)
    & $taskPython (Join-Path $taskCode $Script) @StageArgs
    if ($LASTEXITCODE -ne 0) { throw "Stage failed: $Script" }
}

Invoke-Stage -Script 'intake_and_freeze.py' -StageArgs @('--run', $taskRun)
Invoke-Stage -Script 'build_camera_prior_manifest.py' -StageArgs @('--run', $taskRun)
Invoke-Stage -Script 'test_prior_contracts.py' -StageArgs @('--run', $taskRun)

foreach ($taskCase in @('A1_visual', 'B_w_sigma015', 'B_s_sigma005')) {
    Invoke-Stage -Script 'run_prior_ba.py' -StageArgs @('--run', $taskRun, '--case', $taskCase)
}
Invoke-Stage -Script 'compare_ba.py' -StageArgs @('--run', $taskRun)
Invoke-Stage -Script 'test_augmented_contracts.py' -StageArgs @('--run', $taskRun)
Invoke-Stage -Script 'augment_views.py' -StageArgs @('--run', $taskRun, '--phase', 'select')
Invoke-Stage -Script 'augment_views.py' -StageArgs @('--run', $taskRun, '--phase', 'register')

foreach ($taskCase in @('A1_visual', 'B_w_sigma015')) {
    Invoke-Stage -Script 'export_and_evaluate.py' -StageArgs @('--run', $taskRun, '--case', $taskCase)
}
Invoke-Stage -Script 'export_and_evaluate.py' -StageArgs @(
    '--run', $taskRun, '--case', 'C_augmented_212',
    '--model', (Join-Path $taskRun '03_coverage_augmented/model'),
    '--images', (Join-Path $taskRun '03_coverage_augmented/images')
)
Invoke-Stage -Script 'audit_remaining_coverage.py' -StageArgs @('--run', $taskRun)
```

`register` 完成注册后会进行增补 BA，不应在正常完成后再次调用 `refine`。第一轮在注册完成、BA 开始之前遇到字段与数据库写入顺序错误；当时的恢复入口为 `augment_views.py --run <run> --phase refine --refine-attempt 2`，它复用 `registered_model`，而不是重新注册。本轮保留了失败模型 / 数据库 / receipt。代码已修正该写入顺序，正常新运行无需重走失败分支。

选择依据为冻结的训练数据工程条件。新增模型应在看 H 结果前登记为最多一次额外高斯主训练，记录其模型、图像身份、训练配置及选择来源；第一轮的登记在 `04_downstream/ADDITIONAL_TRAINING_REGISTRATION.json`。如注册或工程保护条件失败，应保存负结果，不补选第二批以强行通过。

## WSL 高斯与最终评估

继续使用同一 PowerShell 会话中的新 `$taskRun`。Windows / WSL 路径映射必须使用 `/mnt/c`；以下代码检查每次原生调用的退出码。三个主训练完成后才进入 H 评估，不依据 H 调整模型。

```powershell
$taskDistro = 'E26-C2-LIV-20260907-134802'
$taskGpuPython = '/opt/e26_thin_surface_v1/venv/bin/python'
$taskLinuxCode = '/mnt/c/lidar-camera-fusion-mask-robust-downsampling/02_code/box2_lio_guided_sfm_v1_20261002'
$taskLinuxRun = '/mnt/c' + $taskRun.Substring(2).Replace('\', '/')
$taskRuntime = '/mnt/c/lidar-camera-fusion-mask-robust-downsampling/04_outputs/box2_3dgs_display_v1_20260930'

foreach ($taskCase in @('A1_visual', 'B_w_sigma015', 'C_augmented_212')) {
    $taskCaseDirectory = "$taskLinuxRun/04_downstream/$taskCase"
    foreach ($taskPhase in @('preflight', 'main')) {
        & wsl.exe -d $taskDistro -- $taskGpuPython "$taskLinuxCode/train_gaussians.py" --case-dir $taskCaseDirectory --runtime $taskRuntime --phase $taskPhase
        if ($LASTEXITCODE -ne 0) { throw "Training stage failed: $taskCase / $taskPhase" }
    }
}

Invoke-Stage -Script 'localize_extra_frames.py' -StageArgs @('--run', $taskRun)
Invoke-Stage -Script 'validate_unused_correspondences.py' -StageArgs @('--run', $taskRun)
Invoke-Stage -Script 'project_fixed_lidar.py' -StageArgs @('--run', $taskRun)

& wsl.exe -d $taskDistro -- $taskGpuPython "$taskLinuxCode/render_and_score.py" --run $taskLinuxRun --runtime $taskRuntime
if ($LASTEXITCODE -ne 0) { throw 'Rendering failed.' }
& wsl.exe -d $taskDistro -- $taskGpuPython "$taskLinuxCode/prepare_display.py" --run $taskLinuxRun --runtime $taskRuntime
if ($LASTEXITCODE -ne 0) { throw 'Viewer export failed.' }
```

本轮所有入口均实际运行过。首次 PLY preflight 因缺少法线字段失败，使用保存原 PLY 的兼容修复后重试；路径修复后的渲染保存在 `native_renders_attempt2`。入口现已包含这两个修复。示例是一组逐阶段复现入口，最后的完整组合没有再重复运行；第一轮的每阶段记录是实际执行证据。

本轮保留的 INRIA 主点投影限制仍适用于这些原配置命令。它们复现本轮对照，不构成已修正非中心主点的新版实验。

## 查看器与收尾

查看已有第一轮结果时，运行 [START_VIEWER.ps1](../../04_outputs/box2_lio_guided_sfm_v1_20261002T014330/START_VIEWER.ps1)，然后打开 [http://127.0.0.1:8195/](http://127.0.0.1:8195/)。该脚本仅核对已有服务或在当前终端运行自己的服务，不清理其他进程。

新运行可以直接在独立终端执行：

```powershell
& $taskPython (Join-Path $taskCode 'serve_viewer.py') --root (Join-Path $taskRun 'viewer') --port 8195
```

如端口已被占用，使用另一个明确指定的端口，不终止未知服务。浏览器验证前先进入 Bw 页面并读取当前页面状态，再运行 `validate_viewer.ps1 -Run <run> -Attempt <fresh attempt>`。原验证器包含第一轮的固定模型计数，是第一轮回归入口，不能用来直接验收帧数 / 高斯数不同的新运行。

`finalize_execution.py` 重新核对原始数据身份、原观测身份、209 个实际数值先验、训练完成状态、导出及浏览器记录，并保存源码快照。本脚本也包含第一轮实际案例与状态要求，应在相应实验身份匹配并完成 browser receipt 后使用，不能给未检查的新结果自动贴上本轮通过状态。

`code_snapshot/` 保存运行完成时的源码用于追溯。`common.py` 按 `02_code/<name>/` 层级解析项目根目录，不能直接从输出下的快照目录启动；需要复现该版本时，应恢复到保持该层级的隔离代码目录并检查路径契约。

## 核心限制

所有真实数据 BA 达到迭代上限，状态为 `NO_CONVERGENCE` 且 usable。位置先验接口生效不等于位姿更准确；H 渲染分数混合了地图、PnP 与渲染；未使用的 H_eval 对应仍有误匹配；固定 LiDAR 点来自同一轨迹，不是独立真值；箱顶缺少有效高位观测。相同帧先验收益、ATE / RPE、真实表面精度及从头 prior-guided SfM 的可靠性均没有得到本轮正式确认。
