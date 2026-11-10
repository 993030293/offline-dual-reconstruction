# v3r1 修复验证结果

修复版使用既有 Tesla 注册图像、稀疏模型和当前 MCAP 里程计回执作为只读输入，重新执行了米制时间支持、轨迹插值绑定、Sim3 估计和 PGSR 输入构建。

- 米制支持：180/180。
- 轨迹绑定：180/180，零 abstain。
- Sim3：180 个对应、180 个 RANSAC 内点；数值与原成功运行一致。
- Sim3 文件 SHA256 与残差回执：一致。
- dense 与 Sim3 基线模型：180 个相机、157,301 个点；相机中心、旋转矩阵和点坐标最大差均为 0。
- 独立 SfM 几何准入：180/180 通过；`sim3_only_camera_admission=false`。
- 输出 PGSR 稀疏模型：180 个相机、127,992 个点，可由 pycolmap 重新载入。
- 负向测试：把 Sim3 scale 增加 0.01 后，下游因文件哈希与残差回执不一致而拒绝，退出码为 1，没有提交输出阶段。
- 聚焦回归测试：6 项通过。

首次真实回放在私有暂存模型写出时失败，因为 COLMAP 要求目标目录预先存在。代码补齐目录创建后，从失败点重跑通过；失败没有产生已提交的 `06_pgsr_inlier`。对应回执为 `evidence/failure_receipts/V3R1_IMPLEMENTATION_ATTEMPT_01_FAILED.json`。

机器验证详情见 `evidence/failure_receipts/V3R1_METRIC_REPLAY_VERIFICATION.json`。本轮没有重新执行约 15,000 步 PGSR 训练、TSDF 融合或 S08 网格生成；这些下游算法代码没有因本次 Sim3/血缘修复发生变化，历史 v3 S08 实跑证据保留在 `evidence/integrated_v3`，不能冒充 v3r1 的新端到端实跑。

