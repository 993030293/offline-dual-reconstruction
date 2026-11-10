# 第三方依赖与许可

本仓库包含项目原创编排/验证代码，以及对第三方软件的调用和小范围补丁。第三方成果不属于本项目原创。

| 组件 | 固定版本/提交 | 仓库 | 本仓库分发内容 | 许可边界 |
|---|---|---|---|---|
| COLMAP | 4.1.1，记录提交 `a0d785f` | https://github.com/colmap/colmap | 不分发二进制或源码 | 由上游许可约束 |
| INRIA Gaussian Splatting | `54c035f7834b564019656c3e3fcc3646292f727d` | https://github.com/graphdeco-inria/gaussian-splatting | 仅版本记录和完整许可文本 | 研究/评估、非商业限制；见 `licenses/INRIA_3DGS_LICENSE.md` |
| gsplat | 1.5.3，`937e29912570c372bed6747a5c9bf85fed877bae` | https://github.com/nerfstudio-project/gsplat | 仅版本记录和完整 Apache-2.0 文本 | 见 `licenses/GSPLAT_APACHE_2.0.txt` |
| PGSR | `de24f1a38b350387e8d8fe381b2cd70c1ae946e7` | https://github.com/zju3dv/PGSR | 版本记录、完整许可文本和本项目补丁 | 研究/教育/非营利/非商业限制；见 `licenses/PGSR_LICENSE.md` |

PGSR 补丁修复无效相机深度参与可微投影时的数值问题，并暴露渲染 alpha 输出；它必须与上述固定基线一起应用。不要把补丁单独描述成完整 PGSR 实现。

Open3D、SciPy、NumPy、OpenCV、Pillow、pycolmap、pyceres、PyTorch、Playwright、Three.js 等运行时依赖不随本仓库分发，仍受各自许可约束。

本仓库目前没有为原创项目代码选择统一开源许可。缺少许可并不意味着自动授予复制、修改或商业使用权。对外公开前应由项目所有者明确许可策略，并单独评估第三方非商业条款是否满足预期用途。
