# GitHub 上传步骤

## 推荐方式

当前先创建 **private** 仓库，建议仓库名：`offline-dual-reconstruction`。不要把原工作区根目录或已验收包原目录直接设为 Git 根；只上传本发布候选目录。

## 上传前检查

在候选目录执行：

```powershell
python .\tools\verify_publication.py --root .
git status --short
```

验收标准：

- 私有模式状态为 `passed`；
- `errors` 为空；
- 明确保留三类警告：四个非生产研究语法遗留、机器专属绝对路径、项目原创代码许可未决；
- 不出现 MCAP、图像帧、PLY、模型权重、环境目录、运行证据或凭据；
- PGSR 补丁已在固定基线上完成干净回放。

## 创建私有远端

在 GitHub 网页中创建一个空的 private 仓库，不要勾选自动生成 README、`.gitignore` 或 license。随后在候选目录执行：

```powershell
git add .
git status --short
git commit -m "chore: publish offline dual reconstruction prerelease baseline"
git remote add origin https://github.com/<account>/offline-dual-reconstruction.git
git push -u origin main
```

第一次 `git add` 后必须再次查看状态，确认没有忽略规则之外的大文件或私有数据。提交和推送属于外部发布动作；只有在仓库名称、账号和可见性明确后执行。

## 上传后复核

1. 在一个新目录重新克隆远端仓库。
2. 运行 `python tools/verify_publication.py --root .`。
3. 确认 GitHub Actions 的 `source-integrity` 通过。
4. 核对 `payload/02_code`、PGSR 补丁、三份第三方许可和两个 provenance 清单均存在。
5. 确认 README 和 `docs/VALIDATION_STATUS.md` 仍明确记录当前 0/4 的端到端失败状态。
6. 在当前候选四视频冷启动和跨机器运行均未完成前，不创建稳定标签或正式 Release。

## 以后公开时

公开之前先完成 `docs/PUBLICATION_BOUNDARY.md` 中的门槛。不要通过删掉完整源码来消除路径告警；应将机器路径改造成配置，并用可审计差异说明算法逻辑没有变化。每次算法修复使用独立提交，并在提交信息或变更记录中关联对应 RA 编号和验证范围。
