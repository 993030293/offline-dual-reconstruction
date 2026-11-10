# v3 整合与安装验证结果

> 本文件保留 v3 的整合基线证据。v3r1 的 Sim3、输入血缘和失败回执修复结果见 `REPAIR_RESULT.md`，不能用本文件替代 v3r1 的新验证边界。

## 整合来源

- 旧通用包：`e26_video_to_s08_preview_nollm_fresh_v2_20260921`；
- Tesla 改进包：`e26_tesla_reliable_surface_v2_20260921`；
- 新包：`e26_s00_s08_nollm_installer_v3_20260921`。

新包保留旧路线的嵌入式里程计和 LIO fallback，并加入 Tesla v2 的质量/环绕覆盖选帧、多视角 PGSR 深度门控、稳定注册 LiDAR 锚定、冲突覆盖和独立地面一致性门。统一入口通过 MCAP topic 契约选择运行配置，不识别物体语义。

## 包级测试

- Python AST、JSON 和 PowerShell 语法检查通过；
- 149 个 manifest 文件在首次安装演练时身份校验通过；
- 运行时代码未发现 OpenAI 或 Anthropic 客户端导入；
- 3 个聚焦回归测试通过：三维 ROI 高度层分离、S02 非语义清点、S08 停止边界和 mesh 实体化；
- Tesla MCAP 的自动路由实测选择 `ReliableLidar`，topic 连接数为嵌入式里程计 1、静态 TF 1、注册 LiDAR 1。

## 安装演练

首次安装演练目录：

`C:\lidar-camera-fusion-mask-robust-downsampling\04_outputs\e26_s00_s08_installer_v3_install_verify_20260921T093000Z\installed`

安装器实际通过了以下检查：

- Windows SfM Python 的 NumPy、OpenCV、SciPy、Pillow 和 rosbags；
- Windows geometry Python 的 NumPy、OpenCV、SciPy 和 Open3D；
- COLMAP 4.1.1 CUDA 可执行入口；
- WSL 发行版中的 PGSR Python、PyTorch 和 CUDA 可用性。

对应机器回执保存在 `evidence\integrated_v3\INSTALL_RECEIPT.json` 和 `INSTALL_VERIFY.json`。

## 真实 Tesla S08 mesh 重放

验证目录：

`C:\lidar-camera-fusion-mask-robust-downsampling\04_outputs\e26_installer_v3_tesla_s08_mesh_verify_v1_20260921T092000Z`

这次验证复用已完成 Tesla v2 运行的 S07 原始 TSDF 网格，只重放新的 S08 后处理，没有把它报告成新的 S00–S08 全量运行。

- 输入原始 TSDF：3,060,155 顶点、5,660,151 三角形；
- S08 删除地面或无条件低位伪影三角形：594,522；
- 输出 observed-surface mesh：2,732,133 顶点、5,065,629 三角形；
- mesh 含顶点法线和顶点颜色，大小 205,192,288 字节；
- mesh SHA-256：`d6159662aee858dffa619a79f8bee9df97605d646970586609fbb4b735af4d99`；
- 100 万点预览及单文件 HTML 机器验证通过；
- 统一终态：`complete_s08_observed_surface`；
- `language_model_used=false`、`visual_language_model_used=false`、`s09_executed=false`、`surface_completion_used=false`。

对应回执保存在 `evidence\integrated_v3\INTEGRATED_FINAL_STATUS.json`、`s08_preview_derivative_receipt.json` 和 `s08_offline_viewer_validation.json`。

## 证据边界

新包已经通过代码、安装、外部环境、自动路由和真实 Tesla S08 mesh 后处理验证。它没有使用新输入再次执行昂贵的完整 S00–S08 训练，因此不能把本次整合验证描述成第二次完整 Tesla 重建。旧通用路线与 ReliableLidar 路线各自已有一次完整运行证据，但 `Auto` 在其他物体和传感器组合上的普适性仍未建立。

S08 mesh 是观测表面产物。开放边界、玻璃缺失、遮挡区和未观测背面保持未知；没有水密性、尺寸精度或制造就绪声明。
