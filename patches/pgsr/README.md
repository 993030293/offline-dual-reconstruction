# PGSR 补丁

基线：`https://github.com/zju3dv/PGSR.git`，提交 `de24f1a38b350387e8d8fe381b2cd70c1ae946e7`。

应用方式：

```bash
git checkout de24f1a38b350387e8d8fe381b2cd70c1ae946e7
git apply --check /path/to/0001-projective-validity-and-alpha-output.patch
git apply /path/to/0001-projective-validity-and-alpha-output.patch
```

修改内容：

1. 在渲染返回值中暴露 `rendered_alpha`。
2. 对非有限、零深度和相机后方点建立显式有效性掩码。
3. 避免这些点先参与除法、产生 NaN/Inf 后才被过滤。
4. 把同一有效性合同用于 PGSR 多视图几何损失和深度图采样。

补丁不包含本地生成的 `egg-info` 或 Python 缓存。应用后仍需按 PGSR 上游说明安装依赖，并遵守 `licenses/PGSR_LICENSE.md`。
