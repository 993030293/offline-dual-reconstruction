"""Keep failed attempts, deterministic recovery, manual intervention, and measured costs distinct."""

import argparse
from pathlib import Path
import time

from common import atomic_json, read_json
from metric_recovery_accounting import metric_geometry_recoveries


def review(root):
    config = read_json(root / "CONFIG.json")
    side = root / config["case"] / "optimized"
    failed = []
    paths = [*root.glob("stages/*/PROCESS.json"), *root.glob("failures/*/**/PROCESS.json"), *side.glob("*_process.json")]
    for path in sorted(set(paths)):
        value = read_json(path)
        if value.get("status", "").startswith("failed"):
            metric = path.name == "metric_process.json" and path.parent == side
            failed.append({"process_receipt": str(path), "status": value["status"],
                "exit_code": value.get("returncode"), "elapsed_seconds": value.get("elapsed_seconds"),
                "category": "algorithm_metric_BA_failure" if metric else "unexpected_stage_failure",
                "command": value.get("command")})
    manual = [{"path": str(path), **read_json(path)} for path in sorted(root.glob("MANUAL_INTERVENTION_*.json"))]
    controllers = [{"path": str(path), "status": read_json(path).get("status")}
                   for path in sorted(root.glob("FAILURE_*.json"))]
    attempts = [read_json(path) for path in sorted(root.glob("CONTROLLER_ATTEMPT_*.json"))]
    diagnostic_failures = [{"path": str(path), **read_json(path)}
        for path in sorted(root.glob("diagnostics/*FAILURE*.json"))]
    failure_details = [row for row in diagnostic_failures if row['original_exception'].startswith(('F01:', 'F02:', 'F03:', 'F04:'))]
    diagnostic_failures = [row for row in diagnostic_failures if row not in failure_details]
    recoveries = []
    recovery_path = side / "metric_sim3_renderer_recovery/RESULT.json"
    if recovery_path.is_file():
        value = read_json(recovery_path)
        recoveries.append({"path": str(recovery_path), "status": value["status"],
            "preexisting_deterministic_recovery": True, "original_BA_guard_passed": value["original_BA_guard_passed"],
            "independent_accuracy_claim": value["independent_accuracy_claim"]})
    stages = []
    numeric_recoveries = metric_geometry_recoveries(side)
    for path in sorted(root.glob("stages/*/COMPLETED.json")):
        value = read_json(path)
        stages.append({"stage": value["label"], "seconds": value["process"]["elapsed_seconds"],
            "receipt": str(path), "peak_process_tree_RSS_bytes": value["process"]["peak_tree_RSS_bytes"]})
    active = read_json(root / "STATE.json")
    current_elapsed = time.time() - active["first_started_unix"]
    result_path = root / "DELIVERY_RESULT.json"
    complete = result_path.is_file() and not active["status"].startswith("failed")
    if complete:
        actual = read_json(result_path)
        browser = read_json(root / "browser_review/RESULT.json")
        complete = (actual["status"] == "both_delivered_binaries_and_actual_viewers_verified"
            and browser["status"] == "both_viewers_actual_interactions_verified"
            and len(browser["rows"]) == 2 and not browser["pageErrors"])
    ledger = {"status": "dual_outputs_complete_with_recorded_exceptions_and_intervention" if complete else "incomplete_delivery_review",
        "failed_native_processes": failed, "failed_controller_attempts": controllers,
        "deterministic_recoveries": recoveries, "manual_interventions": manual,
        "offline_metric_geometry_recoveries": numeric_recoveries,
        "known_original_metric_guard_failure_count": len(numeric_recoveries),
        "diagnostic_failures": diagnostic_failures, "formal_failure_cause_details": failure_details,
        "zero_failure_claim_allowed": not failed and not controllers and not numeric_recoveries,
        "unassisted_single_uninterrupted_raw_to_delivery_claim_allowed": complete and not manual and not controllers
            and len(attempts) == 1 and not attempts[0]["resume"]
            and all(row['recorded_original_metric_recovery_guards_passed'] for row in numeric_recoveries),
        "controller_attempts": attempts, "successful_stage_costs": stages,
        "successful_stage_seconds": sum(row["seconds"] for row in stages),
        "failed_extra_stage_seconds": sum(row["elapsed_seconds"] or 0 for row in failed if row["category"] != "algorithm_metric_BA_failure"),
        "algorithm_failure_cost_already_in_front_stage_total": True,
        "wall_seconds_including_development_and_controller_pauses": current_elapsed,
        "intervention_time_included_in_wall_not_algorithm_benchmark": True,
        "independent_metric_GT_verified": False, "review_kind": "separate_self_review_not_independent_agent_review"}
    atomic_json(root / "EXCEPTION_LEDGER.json", ledger)
    lines = ["# " + config['case'] + " 双输出运行报告", "", "本报告把算法失败、预先定义的恢复和开发助手干预分开记录。", "",
        "当前状态：`"+ledger["status"]+"`。从原始录制重新计算，成功阶段的复用只发生在同一新运行中。", "",
        "不能把助手修复后重新跑通表述为原始流程从未失败或无需人工干预。", "", "## 输入与输出", "",
        "原始输入：`"+config["raw"]+"`。粗位姿生产者以本轮 metric/CONTRACT.json 和原始传感器后端回执为准。没有使用独立 GT，固定先验权重不等于真实传感器协方差。", ""]
    if complete:
        output = read_json(root / "DELIVERY_RESULT.json")
        lines.extend(["已交付 "+str(output["shared_camera_count"])+" 个共同相机、"+str(output["gaussian_count"])
            +" 个 3D 高斯、"+str(output["surface_points"])+" 个 PGSR 网格表面采样点。", "",
            "表面网格含 "+str(output["mesh_vertices"])+" 个顶点、"+str(output["mesh_triangles"])+" 个三角形。", "",
            "另保留完整 "+str(output["full_surface_points"])+" 顶点表面点云；500,000 点为预览采样，不代表完整几何已丢弃。", "",
            "预览："+output["viewer"]["url"], "", "交付文件在 `delivery/`；浏览器行为回执为 `browser_review/RESULT.json`。", ""])
    lines.extend(["## 异常与恢复", "", "|类别|状态|证据|", "|---|---|---|"])
    for row in failed:
        lines.append("|"+row["category"]+"|"+row["status"]+" / exit "+str(row["exit_code"])+"|`"+row["process_receipt"]+"`|")
    for row in recoveries:
        lines.append("|代码中已有的恢复|"+row["status"]+"|`"+row["path"]+"`|")
    for row in numeric_recoveries:
        lines.append('|原始米制候选失败后离线恢复|' + row['recovery_status']
            + '；位置先验 ' + str(row['actual_position_priors']) + '；另计物理残差 '
            + str(row['separate_physical_depth_residuals']) + '|`' + row['original_failure_receipt'] + '`|')
    for row in manual:
        lines.append("|助手/操作人员干预|"+row["note"].replace("|", "/")+"|`"+row["path"]+"`|")
    for row in diagnostic_failures:
        lines.append("|候选诊断失败（未停止已运行训练）|"+row["original_exception"]+"|`"+row["path"]+"`|")
    for row in failure_details:
        lines.append("|上述实际停机的原因补充（不重复计数）|"+row["original_exception"]+"|`"+row["path"]+"`|")
    lines.extend(["", "原 BA 模型未通过投影检查，不得用于训练或称为已修复。Sim(3) 恢复只保证保留观测的投影等价与米制接口，不证明物体尺寸的独立准确性。", "",
        "## 成本", "", "|成功阶段|实际进程秒数|", "|---|---:|"])
    for row in stages:
        lines.append("|"+row["stage"]+"|"+format(row["seconds"], ".3f")+"|")
    lines.extend(["", "阶段总成本（含前端内部失败与恢复）："+format(ledger["successful_stage_seconds"], ".3f")
        +" 秒。额外失败阶段成本："+format(ledger["failed_extra_stage_seconds"], ".3f")+" 秒。", "",
        "从首个启动到复核的墙钟时间："+format(current_elapsed, ".3f")+" 秒，包含开发、人工修正和停顿，不能作为算法公平速度指标。", "",
        "## 离线边界与质量", "",
        "运行时不依赖大模型或云 API。Python 子进程使用 loopback 网络守卫；浏览器只允许本机请求，依赖与渲染资源来自本地安装。首轮前端在守卫加入前已启动，保留此时间边界，不追溯宣称当时守卫已经生效。", "",
        "六个采集视角是训练视角诊断，不是盲测。TSDF 体素 4 mm 和采样 500,000 点是处理参数，不代表 4 mm 的测量准确性。自由移动不保证没有拍摄覆盖的表面也能正确重建。", "",
        "当前验收针对已有机器和环境；没有验证独立移机安装包、其他物体、全部旧优化项或独立尺度/几何 GT。文本复核直接阅读原文；二进制哈希只用于源代码与产物身份。", "",
        "复核方式：单对话的分离自审，没有使用独立子智能体。", "",
        "最终整合版本没有在另一个新目录再次进行从原始数据到 15,000 迭代训练与双输出的完整无人干预冷启动；本轮原始数据运行包含分段修复和重试。", "",
        "训练计数修复以独立真实两迭代 PGSR 测试复核；打包读回以跨分块损坏输入测试及真实完整产物验证。短测试不能代替完整训练或论文级验证。", ""])
    surface_path = root / "pgsr/SURFACE_RESULT.json"
    if surface_path.is_file():
        surface = read_json(surface_path)
        lines.extend(["## 表面融合修复边界", "",
            "本轮完整表面使用 `"+surface.get("backend", "legacy_global_TSDF")+"`。首轮旧式融合的异常退出和源代码快照保留在失败目录。", "",
            "修复版保留全部相机、4 mm 体素、16 mm 截断和 20 m 深度上限，以空间分块限制内存；该实现只通过了已记录的已知平面/颜色/块边界测试，没有证明与旧后端在真实场景上完全等价。", "",
            "块边界共享世界网格，三角形按半开区间归属；共享顶点可能重复，没有执行全局焊接，也没有宣称网格闭合。", "",
            "表面结果回执：`"+str(surface_path)+"`。", ""])
    quality_path = root / 'QUALITY_REVIEW.json'
    if quality_path.is_file():
        quality = read_json(quality_path)
        lines.extend(['## 显示质量自审', '', quality['conclusion'], '',
            '全景预览 '+str(quality['surface_preview_total_points'])+' 点中，自动聚焦球内为 '
            +str(quality['surface_focus_preview_points'])+' 点。这个范围不是物体 GT 分割。', '',
            '完整网格点数包含块边界重复顶点及背景几何；点数多不能替代物体质量或独立精度验收。', '',
            '证据：`'+str(quality_path)+'`。', ''])
    (root / "RUN_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    print(review(args.run_root.resolve())["status"])


if __name__ == "__main__":
    main()
