# Offline Dual Reconstruction

这是一个**预发布工程基线**：从同步的单目图像、LiDAR、IMU 和里程计录制中，纯离线生成共享米制相机轨迹的两类三维结果：

1. **3D Gaussian Splatting（3DGS）**：用于自由视点渲染、采集路径回放和交互预览。
2. **PGSR 几何分支**：用于表面网格、表面采样点云和可自由移动的几何预览。

> 当前版本是**私有工程候选**，不是稳定发行版。提交 `f84c7e548eaf552c5e0c984ef1a23dc74c98a360` 已在干净副本中，从四份原始绕箱 MCAP 各以一次根 `RUN.ps1` 控制器尝试、无续跑、无人工接管完成 3DGS、PGSR 表面网格、表面采样点和双预览；同机正式根入口完成数为 **4/4**。历史失败和预定义子阶段非零结果仍保留，不能写成“四例零失败”。跨机器、独立几何精度、非箱体泛化及公开许可仍未通过，因此不创建稳定标签或正式 Release。详见 [当前验证状态](docs/VALIDATION_STATUS.md)、[Round 4 正式验收](docs/FORMAL_ACCEPTANCE_F84C7E5_20261113.md)和[正式验收轮次](docs/FORMAL_ACCEPTANCE_ROUNDS.md)。

## 当前证据边界

| 项目 | 当前状态 |
|---|---|
| 历史参考包 | 2026-10-14 的源码身份 `bcf197d...c3dfd`，曾在同一工作站完成四份私有录制；它是历史证据，不等于当前提交通过 |
| 当前候选关系 | 历史参考包、首个 Git 失败基线及后续开发修复均由提交记录区分；逐文件差异记录在 `provenance/BASELINE_DELTA.json` |
| 静态与局部回归 | 完整测试套件 443/443、7 项根入口回归、八项运行时合同、根预检和私有发布内容验证已通过；Round 4 从同一干净副本执行 |
| 当前端到端结果 | `f84c7e5` 的四份原始录制均以一次根入口控制器尝试达到 `dual_target_outputs_complete`；同机正式完成 4/4 |
| 当前预览验收 | 八个主预览均完成自动真实交互和人工查看；只访问回环地址，无页面/控制台/失败请求错误。远离采集轨迹仍有模糊、漂浮和局部几何孔洞 |
| 离线边界 | 正式算法不需要 LLM、VLM、云 API 或运行时下载 |
| 跨机器复现 | 未验证；仓库不包含操作系统、CUDA、WSL、训练数据或完整第三方运行环境 |
| 公开 GT 精度 | 未验证 |

第一次提交的目的，是冻结一个真实、可审计的失败基线。后续根因修复、回归测试和 Round 4 正式完成均使用独立提交；Round 1 至 Round 3 和预定义子阶段失败没有被改写成“一次成功”。

## 算法流程

```text
原始同步录制
  -> 传感器解码与时间对齐
  -> 图像质量评估、粗位姿和自适应选帧
  -> SfM 特征匹配、相机注册、桥接帧、联合 BA
  -> 米制尺度与共用相机接口
  -> 3DGS 训练与自由飞行预览
  -> PGSR 训练、表面提取、点云/网格与自由飞行预览
```

两条输出线路共享同一轮准入相机和米制坐标。普通单目视频本身只能恢复相对尺度；本项目的米制约束来自同步录制中的 LiDAR、IMU、里程计或等价的外部尺度来源。

## 仓库内容

- `payload/02_code/`：项目源码快照、正式入口依赖、优化模块和回归工具。
- `payload/05_packages/.../`：无大模型目标几何/点云子包；不包含私有运行证据和机器安装回执。
- `payload/04_outputs/.../RUNTIME_MANIFEST.json` 及三份 DLL：初始选帧所需、已做行为等价检查的离线原生图像质量运行库。
- `docs/accepted_package/`：历史参考包的说明与入口副本，仅供追溯，不是当前候选的通过证明。
- `patches/pgsr/`：固定 PGSR 基线提交上的数值稳定性补丁。
- `licenses/` 和 `THIRD_PARTY.md`：第三方版本、许可及再分发边界。
- `provenance/`：历史参考身份、当前显式差异、上游版本锁和逐文件源码清单。
- `tools/verify_publication.py`：离线检查清单、禁入文件、缓存、连接、语法和常见凭据模式。

## 为什么先提交预发布基线

首个失败已定位到七月录制在环绕与回访轨迹下的弱连接视觉分量和粗位姿冲突，而不是一个可以通过隐藏重试或放宽阈值绕过的问题。新的分量级米制规范随后在完整原始输入流程中进入了两条训练分支，但过程中仍暴露并修复了相机见证保留、长路径、预览模板、原生打包回执和续跑进程所有权等问题。先冻结并持续提交基线有三个作用：

1. 后续可以准确比较每个修复改变了什么；
2. 历史成功、当前失败和未来通过不会混为一谈；
3. GitHub 提交记录可以作为工程异常与修复过程的可追溯证据。

仓库继续保持 **private**。四录制同机端到端门已通过，但跨机器、独立几何精度、非箱体泛化和许可/机器路径边界仍未关闭，所以不创建稳定标签或正式 Release。

## 验证仓库内容

在仓库根目录运行：

```powershell
python .\tools\verify_publication.py --root .
```

该检查验证的是发布内容和源码身份，不会启动训练，也不能替代四视频端到端重跑。新的机器环境请按 [运行时复现清单](docs/REPRODUCIBILITY.md) 准备。

Windows 下完整源码树含深层参考路径。默认 Git 配置可能在检出阶段报 `Filename too long`；无需修改系统全局设置，使用短目标目录和仅对本次命令生效的长路径选项：

```powershell
git -c core.longpaths=true clone https://github.com/993030293/offline-dual-reconstruction.git D:\offline-dual
```

## 当前本机完整运行入口

仓库根目录现在提供当前候选的唯一完整入口 `RUN.ps1`。第一次使用时先绑定这台工作站已经安装的离线运行时，并执行真实预检：

```powershell
& .\BIND_RUNTIME.ps1
& .\PREFLIGHT.ps1 -OutputRoot 'D:\offline_dual_preflight'
```

`BIND_RUNTIME.ps1` 除了建立六项本机目录连接，还会从冻结声明生成两个被 Git 忽略的配置：主流程的 `OFFLINE_RUNTIME.local.json` 和目标几何子包的 `config/local_runtime.json`。它会核对 Node、Playwright、已安装 Chromium、COLMAP、两个 Windows Python、rosbags、包内标定和 WSL PGSR 边界。默认不会覆盖不匹配的现有本机配置；审核冻结路径后才可显式使用 `-RefreshLocalBindings`。只通过仓库内容校验而没有通过根 `PREFLIGHT.ps1`，不能启动正式视频验收。预检输出目录必须尚不存在。从原始 MCAP 完整运行：

```powershell
& .\RUN.ps1 `
  -Raw 'E:\path\to\recording.mcap' `
  -RunRoot 'D:\reconstruction_runs\object_001' `
  -Case 'object_001'
```

如果 MCAP 的 `/tf_static` 没有连接 `camera_optical` 与 `base_link`，必须显式提供同一设备的相机—机体标定。当前 E26 设备标定可以这样传入；入口只读取 `T_base_link_camera_optical` 的七个数值，不读取该 JSON 中的历史轨迹或 SfM 路径：

```powershell
& .\RUN.ps1 `
  -Raw 'E:\path\to\recording.mcap' `
  -RunRoot 'D:\reconstruction_runs\object_002' `
  -Case 'object_002' `
  -BodyCameraCalibration '.\payload\02_code\e26_orbit_visual_mvs_v1_20260818\configs\e26_trajectory_binding_v1.json'
```

`SMOKE_SENSOR.ps1` 只运行到传感器阶段，用于有界冒烟；它不能替代完整双路重建验收。`docs/accepted_package/` 仍只是历史参考包快照，当前候选必须从仓库根目录入口运行。

## 完整运行的预期输出

端到端验收通过后，每份录制应同时得到：

- 3DGS 高斯场景 PLY；
- PGSR 目标表面网格；
- 从 PGSR 表面按面积采样的显示点云；
- 两条线路共用的相机与尺度记录；
- 原生异常账本和运行报告；
- 根 `RUN.ps1` 的参数、退出码和成功/失败入口回执 `PACKAGE_ENTRY_RESULT.json`；
- 可旋转、平移、缩放、自由飞行并回放采集路径的离线预览。

提交 `f84c7e5` 已在四个全新目录中从根 `RUN.ps1` 得到这一终态。四例共享训练相机分别为 1,195、1,208、1,185、1,000 张；3DGS 均训练 12,000 次并输出 593,891、466,643、470,187、542,753 个高斯；PGSR 均训练 15,000 次并输出 684,894、360,351、390,653、457,296 个参数高斯。四例均输出 1,000,000 个表面采样点和可交互双预览。完整数字、异常及质量边界见 [Round 4 正式验收](docs/FORMAL_ACCEPTANCE_F84C7E5_20261113.md)。这些结果没有独立 GT，也没有跨机器或非箱体泛化证据。

## 数据、环境与许可

仓库不包含私有 MCAP、抽帧图像、相机数据库、模型权重、PLY、网格、Python/WSL/CUDA 环境或浏览器缓存。运行环境必须在本地显式绑定，禁止运行时下载。Windows 克隆还需要短目标目录和命令级 `core.longpaths=true`；这只影响源码检出，不改变算法。

项目原创代码尚未选择统一开源许可；INRIA 3DGS 和 PGSR 还包含研究或非商业限制。因此当前仅适合私有协作，不能因为仓库可克隆就推断为允许公开再分发或商业使用。详见 [发布边界](docs/PUBLICATION_BOUNDARY.md) 和 [第三方说明](THIRD_PARTY.md)。
