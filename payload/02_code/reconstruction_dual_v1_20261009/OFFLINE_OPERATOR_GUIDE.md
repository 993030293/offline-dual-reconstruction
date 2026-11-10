# 离线目标双输出运行说明

本入口同时交付完整3D高斯、PGSR观测表面经M3去地面后的目标网格，以及从该网格按面积采样的1000000个点。两路共用当前运行的图像、相机、完整内参与尺度接口。正式入口是run_target_delivery.py；旧run_delivery.py为全场景诊断，不代表本交付。

## 运行一份原始录制

在已有本机环境中，指定一个不存在的新输出目录。Box1快捷命令示例：

```powershell
& 'C:/lidar-camera-fusion-mask-robust-downsampling/02_code/reconstruction_dual_v1_20261009/RUN_BOX1.ps1' -RunRoot 'D:/lidar_camera_cold_stability/operator_box1_new_run' -PointProfile AdaptiveSensor
```

C1使用RUN_C1.ps1，其现有快捷配置显式加载该录制的相机/机体标定。其他录制使用通用入口，必须提供自己的原始MCAP和匹配的标定来源；普通视频没有同步传感器和尺度来源时，不能自动取得米制尺度。

```powershell
& 'C:/lidar-camera-fusion-mask-robust-downsampling/05_envs/sfm_py311_v1/Scripts/python.exe' 'C:/lidar-camera-fusion-mask-robust-downsampling/02_code/reconstruction_dual_v1_20261009/run_target_delivery.py' --run-root 'D:/lidar_camera_cold_stability/operator_new_run' --raw 'C:/path/to/original_recording.mcap' --case object_capture --point-profile AdaptiveSensor
```

示例目录为占位，不是本次已执行实验路径。根据实际文件替换raw参数，并为每次新运行使用唯一目录。默认执行到complete；sensors和front仅用于显式阶段诊断，不能计为完整交付。

AdaptiveSensor保留质量与视觉注册合格的帧，雷达只在时间和稳定观测合格时参与。ReliableLidar是严格每帧覆盖的对照；VisualTargetDiagnostic是显式诊断。后两者不能冒充AdaptiveSensor的完整验收。

## 读取结果

完整状态为dual_target_outputs_complete。结果包括：

- target/delivery/gaussian_scene.ply：完成12000迭代的完整高斯参数。
- target/delivery/pgsr_target_surface_mesh.ply：本轮PGSR、三候选选择、M3去地面和支持检查后的目标网格。
- target/delivery/pgsr_target_points.ply：对应网格的1000000面积采样点，不是100万独立雷达测量。
- target/delivery/cameras.json与scale.json：实际共同相机和尺度来源；没有独立GT时不能报告尺寸精度。
- target/DELIVERY_RESULT.json、EXCEPTION_LEDGER.json与RUN_REPORT.md：实际产物、预览地址、原停止、有限恢复、开发干预及成本。
- START_VIEWER.ps1：重启本轮本机预览服务。页面支持旋转、平移、缩放、自由飞行、升降、采集路径和无人机视角。

查看运行报告中的本轮地址。电脑重启会关闭预览服务，但不删除模型；之后执行结果目录中的START_VIEWER.ps1。重新启动服务不等于重新训练，也不改变原运行的验收时间边界。

## 中断和恢复

运行时不调用LLM/VLM、云服务或下载。缺失环境、来源不明、数值或几何不满足要求时明确停止并保存证据。开发助手修改代码与程序的预编码恢复分开记账。

v18可自动回收严格证明属于前一次Windows启动且已结束的原生COLMAP GPU所属记录：独占锁、原租约、原命令和精确进程身份均须对应。旧SfM使用自己的实际STATE补足记录。v19另允许同启动的原生监控中断，但必须有对应的原命令生命周期、实际失败监控回执、完整父子身份绑定及全部记录身份已结束的证据。WSL命令、活进程、缺少失败监控的同启动残留、隔离或未知证据仍拒绝；不能仅因PID不存在就清理。恢复另写回执，原中断租约、失败监控和未知原生命令退出码保持原样。

v19对已开始的监控心跳增加共享冲突判断。只有Windows实际删除共享探测证实文件被不兼容句柄占用，才另存原错误并延后本次心跳；最终必须发布真实完整终态，并在EXCEPTION_LEDGER中核对延后记录与最终计数。初始发布、最终发布失败、真实权限错误、超时和超内存仍失败。报告不得把这种恢复称为零失败；原始的两秒单次发布重试和算法执行预算不增加。

v20新增方向焦点的原半空间有界数值校正：只有原求解成功但原近裁面复核拒绝时触发，不改变相机、分母、95%、原容差或当前原生近裁面。几何越界、不可行或校正后仍不合格继续停止并保存失败。实际触发在ROI回执、EXCEPTION_LEDGER和RUN_REPORT中留证，不能宣称零失败。原失败输入的开发诊断与341项回归不计新冷启动成功。

v21在真实去畸变后，对原畸变图和实际针孔图同步执行原联合准入政策。源图合格不能替代最终像素空间的支撑；原候选拒绝、两个候选模型与排除图像在当前独立export中保留。完整注册图与分母、90%比例、30观测、4网格和全部最终检查保持，无法满足仍失败。异常账本重新绑定原policy与最终准入，不能宣称零失败。351项回归和旧失败输入的独立导出/内容诊断不计冷启动成功。

v22把初选无歧义特征图的连通计算改为现有SciPy稀疏图例程；输入仍是本轮真实匹配，所有支撑掩码、冲突剔除和资格统计与原标量实现逐项相同。原1800秒内部预检和3000秒外层预算不变，不减少要求或跳过困难候选。旧失败数据库的两组完整等价/局部交替计时与359项回归属于开发证据；不证明完整流程加速或新冷启动。只读初选验收明确使用保留的原标量算法重新核对。

v23将本机预览打包的两份完整显示资产及初始化记录分开保存。只有原生资产完整、浏览器关闭、所属树结束、原预览进程和交付身份仍有效，且实际错误为同源display_manifest.json或camera_path.json的指定传输错误时，才允许第二次独立浏览器尝试。两次合计保留原360秒预算；404、原生资产或数量错误、外部网络、页面程序错误、未结束进程和缺失日志均拒绝。原失败、原生命令、网络记录及总耗时写入native_packing/attempt_*/、ATTEMPTS.json、异常账本和报告，恢复后不能称零失败或已证明原网络根因。

v23的进程归属采集沿每条祖先链接核对PID和创建时间，拒绝孩子早于复用父进程的错误归属，并保存拒绝记录。它不按系统程序名终止或自动放行未知进程。v22实际误归属的微软签名系统通知只在旧失败终态记录中单列，原记录不改写；新代码仍需真实八次冷启动验证。

只读表面验收在原float32容差内，对原距离查询拒绝的点增加真实float64三角面见证；找不到见证仍失败。原单精度拒绝与精确复核分开记录，不改变模型、点、相机或容差，也不作为独立几何精度。

已有成功阶段可在同一运行的冻结输入下显式resume；失败和人工retry必须留证。它们不能当作新的无人干预冷启动。四录制两轮稳定验收必须使用八个新空根，不跨运行复用图像、数据库、位姿、ROI、模型或checkpoint。电脑重启或生产修改后应回查当前连续验收范围，不能拼接旧计数。

本说明对应已绑定的本机环境。当前稳定完成状态需读取CURRENT_PROGRESS和每例内容/终态回执；开发回归通过、进程运行、文件存在或单路训练完成均不能代替完整验收。
