# E26 S00–S08 无大语言模型无人值守安装包 v3r1

这个包整合了两条已经实跑过的路线：旧版通用 S00–S08 PGSR 流程，以及 Tesla 实跑后加入的质量选帧、多视角深度一致性和注册 LiDAR 锚定。它提供统一安装、运行时绑定、自动配置选择、离线预览和机器回执。

v3r1 根据历史失败回执修复了米制对齐链：Sim3 只负责全局对齐和残差门，不再单独决定相机准入；相机还必须通过独立 SfM 几何支撑门。Sim3、位姿先验、注册清单、策略文件和基线稀疏模型必须证明属于同一次运行，任何不一致都会失败关闭。完整矩阵见 `FAILURE_REPAIR_MATRIX.md`，实测结果见 `REPAIR_RESULT.md`。

新版本还会把 S08 去地面后的三角网格直接保存为 `s08_ground_removed_observed_surface_mesh.ply`。预览点云从这个网格采样，不再让使用者从 XYZRGB 显示点反向猜测表面。

## 安装

安装器复制经过 manifest 校验的文件，写入本机运行时配置，并实际检查 Windows Python、Open3D、COLMAP、WSL PGSR 和 CUDA 可用性：

```powershell
& '.\INSTALL.ps1' -InstallRoot 'C:\E26\s00_s08_nollm_v3r1'
```

安装成功时会生成：

- `INSTALL_RECEIPT.json`：安装位置、manifest 和安装状态；
- `INSTALL_VERIFY.json`：每项外部运行时实测结果；
- `config\local_runtime.json`：本机 COLMAP、Python、WSL 和 PGSR 路径。

安装包不内嵌 CUDA、COLMAP、WSL 发行版或 PGSR 二进制环境。它绑定并验证这些外部依赖，避免把仅复制代码误报为完整环境安装。已验证版本记录在 `config\ENVIRONMENT_LOCK.json`。

## 无人值守运行

```powershell
& 'C:\E26\s00_s08_nollm_v3r1\RUN_UNATTENDED.ps1' `
  -RawMcap 'I:\20260820E_Tesla\recon_collection_20260820_173118_725314_0.mcap' `
  -ExpectedMcapSha256 'b0388dbe9e054db011a303249796655de7ed2a4d94bcb0e285fbb0083f29ab4d' `
  -OutputRoot 'C:\lidar-camera-fusion-mask-robust-downsampling\04_outputs\new_unattended_run' `
  -Profile Auto
```

`Auto` 只读取 MCAP topic 契约：同时存在嵌入式里程计和注册 LiDAR 时选择 `ReliableLidar`，否则选择 `LegacyPgsr`。这个路由不识别物体类别，不调用大语言模型，也不需要人工查看画面。

也可以显式指定：

- `ReliableLidar`：Tesla v2 验证路线；使用质量和视角多样性选帧、多视角 PGSR 深度门控、重复 LiDAR 回波锚定和冲突覆盖；
- `LegacyPgsr`：保留旧 S00–S08 路线，并支持嵌入式里程计或原始 LiDAR/IMU fallback。

## 每次运行的正式输出

- `INTEGRATED_FINAL_STATUS.json`：统一终态和证据边界；
- `08_*\s08_ground_removed_observed_surface_mesh.ply`：S08 三角网格；
- `08_*\s08_after_ground_removal_before_component_filter_xyzrgb_f32_1m.ply`：确定性表面采样；
- `08_*\s08_after_ground_removal_before_component_filter.html`：单文件离线预览；
- 各阶段 JSON 回执、输入身份和运行参数。

运行失败时会生成 `PIPELINE_FAILURE.json`。如果失败发生在输出根创建之前，则在请求输出路径旁生成 `输出目录名.FAILURE.json`。失败回执不会使失败阶段变成可复用成功产物；修复输入后必须使用新的输出根。

S08 网格表达已观测到且通过前序门控的表面。流程不会填补玻璃、遮挡面、背面或底面；不会因为最终输出是 PLY 就宣称水密、尺寸准确或物理完整。

## 无大语言模型边界

运行时不调用大语言模型、视觉语言模型或人工视觉审阅。COLMAP、PGSR、TSDF、RANSAC、多视角重投影和 LiDAR 体素统计属于专用视觉或几何算法。完整契约见 `contracts\NO_LLM_CONTRACT.json`。

## 验证

```powershell
& '.\VERIFY_PACKAGE.ps1' `
  -GenericPython 'C:\lidar-camera-fusion-mask-robust-downsampling\05_envs\e26_generic_side_bottom_v1\Scripts\python.exe'

& '.\VERIFY_INSTALLATION.ps1'
```

包级验证检查 manifest、PowerShell 语法、Python AST、JSON、运行时代码中的大语言模型客户端导入，以及 S08 网格和点云的聚焦回归测试。安装验证检查真实外部程序和 CUDA 状态。两者都不代替一次新的完整 MCAP 重建。

## 已验证范围

- 旧通用路线：纸箱采集从原始 MCAP 到 S08 工程预览；
- `ReliableLidar`：一个 Tesla 绕车采集从原始 MCAP 到 S08，180/180 图像注册；
- 新增网格实体化：使用已有真实 Tesla S07 TSDF 网格执行 S08 后处理验证。

Tesla 的实跑指标和限制保留在 `RERUN_RESULT.md`。当前证据不能推广到所有车辆、所有传感器配置或所有光照条件。
