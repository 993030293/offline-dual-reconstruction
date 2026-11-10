# Box2 粗位姿与质量初选 v2

本目录负责 SfM 之前的帧选择、现有高斯模型的自由飞行预览，以及公开 GT 的独立评估输入核对。当前运行：`C:/lidar-camera-fusion-mask-robust-downsampling/04_outputs/box2_initial_pose_quality_v2_20261002T193102/`，完整结果见其中 RUN_REPORT.md。

`initial_selection.py` 从原始 MCAP 解码图像、录制 odom、CameraInfo 和 tf_static。已有开发分割只读 extra_frames 的角色和 ID，随后释放其他字段，不使用 core_train。扫描前冻结策略；输出位置必须尚不存在。粗轨迹只是估计值，算法生产者未确认。使用离线包围样本插值，不能宣称实时或在线算法。

`verify_initial_bundle.py` 检查全图像、完整 CameraInfo、候选图和原始数据抽样回读，并用直接闭式 SLERP 检查相机位姿变换。旧项目插值的小角度 NLERP 与精确 SLERP 不完全相同，失败诊断保留在本次输出中，没有修改选择数据或验收阈值。

`build_viewer.py` 在新的输出目录复制旧版模型及 vendor，生成新的飞行 UI。它不优化权重。`serve_viewer.py` 只监听 127.0.0.1，并记录真实服务 PID 和目录。START_VIEWER.ps1 位于运行目录，已服务时返回实际地址，端口被其他进程占用时保留该进程。

`public_gt_audit.py` 核对数据与原始 PCD 的 DATA 边界；有界点采样不构成全地图验收。`download_gt_trajectory.py` 只下载固定官方版本的小型 gt-tum 数据，写入评估目录。`public_gt_preflight.py` 核对官方坐标外参、已有独立 raw FAST-LIO 与 GT 时间支持，不计算精度。GT 不出现在选帧输入中。

依赖复用 `05_envs/sfm_py311_v1`：NumPy、SciPy、OpenCV、PyYAML，以及项目已有原始 MCAP / CDR / Bayer 解析器。未安装新训练依赖。

复现初选时，给 `initial_selection.py` 传入新的 `--output` 和此前 `split_manifest.json` 的 `--exclude-source`。不得复用已存在的初选输出目录。`python -m unittest test_initial_selection test_public_gt_layout -v` 验证插值、外参方向、开发留出排除、质量空隙、运动覆盖和资源上限。

本轮 1002 帧尚未经过真实匹配、SfM / BA 和高斯训练。默认预览仍显示上一轮 212 帧模型，另外两组 180 帧可切换。进入后左键旋转、右键平移、滚轮缩放；自由飞行模式中点击场景后 WASD 移动、Q/E 升降、Shift 加速。详情与公开数据评价规则分别见 RUN_REPORT.md 和 PUBLIC_GT_EVALUATION_PLAN.md。
