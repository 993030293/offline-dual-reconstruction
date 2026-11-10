# 第三方软件与许可边界

本发行包的正式流程调用本机预先安装的第三方组件，不把它们的完整运行环境重新分发进压缩包：

- COLMAP 4.1.1：SfM、匹配、注册与 BA。
- INRIA Gaussian Splatting 与 gsplat 相关组件：3DGS 训练、渲染和 PLY 导出。
- PGSR：平面高斯表面训练与表面提取。
- Open3D、SciPy、NumPy、OpenCV、Pillow、pycolmap、pyceres、PyTorch、Playwright、Three.js 等：数值处理、几何、预览和验证。

PGSR 的本机许可文本已复制到 `licenses/PGSR_LICENSE.md`。该许可包含研究、非营利和非商业限制；完成技术打包不代表已经取得商业再分发或商业使用授权。COLMAP、3DGS、gsplat、PGSR 及其他依赖的权利属于各自作者。

每次完整算法输出仍会在 `target/delivery/licenses/` 和 `THIRD_PARTY.json` 中记录实际交付所使用的第三方归属。跨机器分发前需要再次核对所有第三方源码、二进制、模型和字体/浏览器资产的许可。本包当前只验收了同机离线运行绑定。
