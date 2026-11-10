# 离线 3DGS + PGSR 双路重建算法包

本包把已经完成四份原始 MCAP 冷启动验收的统一流程整理成一个固定入口：

`原始同步 MCAP -> 质量与粗位姿选帧 -> SfM 与增帧 -> 米制共用相机 -> 3DGS -> PGSR 目标表面 -> 点云/网格 -> 自由飞行预览`

## 当前交付等级

- **状态**：同一台工作站可运行的离线发行候选。
- **已验证算法源码身份**：`bcf197d198d6e9ff2aef2d2dd7abfb6a025cb3f06df4739beb1a4d4190dc3dfd`。
- **端到端证据**：2026-10-14 使用该源码从四份原始 MCAP 分别生成了 3DGS、PGSR 点云/网格和两类预览，运行期间没有 LLM、VLM、云 API、自动改代码或历史派生输入。
- **移机状态**：尚未验证。本包不把 CUDA、COLMAP、Python 环境、WSL 发行版和第三方训练仓库重复塞进压缩包，而是显式绑定本机已安装的离线运行时。
- **精度边界**：没有独立 GT 的录制不能据此报告绝对尺度精度；四份结果仍需要人工做最终视觉质量验收。

## 一次性准备

在 PowerShell 7 中进入解压后的包目录，执行：

```powershell
& .\BIND_RUNTIME.ps1
& .\VERIFY_PACKAGE.ps1
& .\PREFLIGHT.ps1 -OutputRoot 'D:\offline_dual_preflight'
```

第一条命令只在包内创建指向本机既有运行环境的目录连接，不复制或修改环境。第二条逐文件核对包内容。第三条真实导入 Windows/WSL 依赖并检查 GPU、COLMAP、PGSR、预览运行时和剩余磁盘空间。

## 从原始录制运行

输出目录必须是一个尚不存在的新目录。普通录制示例：

```powershell
& .\RUN.ps1 `
  -Raw 'C:\path\to\recording.mcap' `
  -RunRoot 'D:\reconstruction_runs\object_001' `
  -Case 'object_001'
```

如果原始 MCAP 中缺少相机到机体的 TF，还必须提供与该设备匹配的数值标定：

```powershell
& .\RUN.ps1 `
  -Raw 'C:\path\to\recording.mcap' `
  -RunRoot 'D:\reconstruction_runs\object_002' `
  -Case 'object_002' `
  -BodyCameraCalibration 'C:\path\to\camera_body_calibration.json'
```

默认使用 `AdaptiveSensor`，3DGS 训练 12,000 次、PGSR 训练 15,000 次。`-StopAfter sensors` 和 `-StopAfter front` 只用于诊断，不能当作完整交付。

## 完整输出

完整运行的终态是 `dual_target_outputs_complete`，主要产物为：

- `target/delivery/gaussian_scene.ply`：真实训练得到的高斯表示。
- `target/delivery/pgsr_target_surface_mesh.ply`：PGSR 和后处理得到的目标表面网格。
- `target/delivery/pgsr_target_points.ply`：从目标网格按面积采样的显示点，不代表一百万个独立传感器观测。
- `target/delivery/cameras.json` 与 `scale.json`：两条线路共用的相机和尺度来源。
- `target/DELIVERY_RESULT.json`、`RUN_REPORT.md`、`EXCEPTION_LEDGER.json`：结果、异常、恢复和证据边界。
- `viewer/` 与 `START_VIEWER.ps1`：旋转、平移、缩放、俯视、采集路径和无人机式自由飞行预览。

## 离线与异常规则

正式入口不会下载代码、依赖或权重，也不会调用 LLM/VLM 或云 API。Python 和 WSL 阶段启用离线环境与网络拒绝记录。预编码的真实匹配、PnP、桥接帧、三角化、联合 BA 和有限恢复属于算法；未知异常会保存失败证据并停止。任何人工或开发助手修复都必须另记为干预，不能改写成算法自动成功。

## 输入边界

该版本面向当前传感器录制格式。输入需要包含相机图像以及可用的 LiDAR/IMU/里程计或当前已支持的原始 LiDAR/IMU 组合。普通手机视频没有同步传感器和尺度来源时，不能直接得到同等级的米制结果。未来物体不要求是箱子，但跨物体泛化尚未由独立公开测试集证明。

## 文件说明

- `RUN.ps1`：正式统一入口。
- `BIND_RUNTIME.ps1`：绑定这台机器的离线运行时。
- `VERIFY_PACKAGE.ps1`：验证包内容和可选的运行时连接。
- `PREFLIGHT.ps1`：执行真实依赖预检。
- `SMOKE_SENSOR.ps1`：从一份原始 MCAP 运行到传感器阶段的有界冒烟入口。
- `payload/`：算法源码和目标点云子包。
- `evidence/`：四录制验收结果、源码身份和边界说明。
- `licenses/`、`THIRD_PARTY.md`：第三方归属和许可边界。
