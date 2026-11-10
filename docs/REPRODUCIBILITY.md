# 运行时复现清单

## 已确认的参考环境

参考环境来自历史同机包，不代表当前候选或其他机器已经通过：

- Windows Python 3.11.9；
- COLMAP 4.1.1 CUDA，记录提交 `a0d785f`；
- 几何环境：Open3D 0.19.0、NumPy 2.2.6、SciPy 1.15.3；
- WSL PGSR：Python 3.11.16、PyTorch 2.7.1+cu128、NumPy 1.26.4、OpenCV 4.11.0；
- INRIA Gaussian Splatting 提交 `54c035f7834b564019656c3e3fcc3646292f727d`；
- gsplat 1.5.3，提交 `937e29912570c372bed6747a5c9bf85fed877bae`；
- PGSR 提交 `de24f1a38b350387e8d8fe381b2cd70c1ae946e7` 加 `patches/pgsr/0001-projective-validity-and-alpha-output.patch`。

完整版本边界见 `provenance/EXTERNAL_SOURCE_LOCK.json`。

## 新机器复现顺序

1. 克隆本仓库并运行 `python tools/verify_publication.py --root .`。
2. 按固定提交分别准备 3DGS、gsplat 和 PGSR；在干净 PGSR 基线上应用补丁并再次检查工作树差异。
3. 建立隔离的 Windows SfM、Windows 几何和 WSL PGSR 环境；不要把环境目录提交到仓库。
4. 将机器专属路径写入本地配置；本地配置必须被 `.gitignore` 排除。
5. 运行依赖预检，确认 GPU、COLMAP、Python 导入、WSL、PGSR、预览运行时和磁盘空间。
6. 用一份新输出目录运行到传感器阶段；核对图像、时间戳、传感器和标定合同。
7. 完成一份录制的冷启动双分支运行，再扩展到四份录制。
8. 核对终态、异常账本、相机/尺度一致性、3DGS 和 PGSR 产物，并人工检查自由飞行预览。

## 成功条件

- 原始输入在运行前后保持不变；
- 运行终态为 `dual_target_outputs_complete`；
- 3DGS 与 PGSR 使用同一轮已准入相机和尺度；
- 双分支真实产物和预览均存在且内容可加载；
- 所有原生异常、有限恢复和人工干预都保留在异常账本；
- 无 LLM/VLM/云 API/运行时下载；
- 新机器结果与参考环境的结构、数量和关键合同一致。

## 失败边界

配置完成、源码编译、进程启动或文件存在都不等于跨机器复现成功。任何缺失依赖、标定不匹配、数值门失败、网络访问、输出目录污染或未知异常，都应明确失败并保留证据。

截至 2026-11-10，当前候选只通过本机局部回归和完整依赖预检；四视频冷启动在第一份录制的米制联合优化阶段失败，完成数为 0/4。复现者应先阅读 `docs/VALIDATION_STATUS.md`，不能把历史参考包的成功当成当前提交的终态。
