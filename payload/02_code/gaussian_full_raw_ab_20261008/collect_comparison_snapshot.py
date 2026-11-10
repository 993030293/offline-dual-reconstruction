"""Build a versioned comparison from current receipts, retaining costs and missing outcomes."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from atomic_json_windows import atomic_json, read_json


def optional(path):
    return read_json(path) if path.is_file() else None


def selected(value, fields):
    return {field: value[field] for field in fields if field in value} if value is not None else None


def stage_costs(side):
    rows = []
    for path in sorted(side.glob("*_process.json")):
        value = read_json(path)
        rows.append({"receipt": str(path), "stage": path.name.removesuffix("_process.json"),
            **selected(value, ("status", "returncode", "elapsed_seconds", "peak_tree_RSS_bytes")),
            "archived_failed_attempt": False})
    for path in sorted(side.glob("FAILED_STAGE_ATTEMPT_*.json")):
        value = read_json(path)
        rows.append({"receipt": str(path), "stage": value["stage"], "reason": value["reason"],
            **selected(value["receipt"], ("status", "returncode", "elapsed_seconds", "peak_tree_RSS_bytes")),
            "archived_failed_attempt": True, "archive": value["archive"]})
    interrupted = [{"receipt": str(path), "record": read_json(path)}
                   for path in sorted(side.glob("INTERRUPTED_ATTEMPT_*.json"))]
    completed = sum(row.get("elapsed_seconds", 0) for row in rows if row["status"] == "completed")
    failed = sum(row.get("elapsed_seconds", 0) for row in rows if row["status"].startswith("failed"))
    return {"rows": rows, "completed_guarded_stage_wall_sum_seconds": completed,
        "failed_guarded_stage_wall_sum_seconds": failed, "known_guarded_stage_wall_sum_seconds": completed + failed,
        "interrupted_attempts_retained": interrupted,
        "sum_is_not_parallel_campaign_makespan_or_isolated_video_latency": True,
        "GPU_wait_is_included_if_inside_the_stage_guard": True,
        "incomplete_or_unreceipted_cost_not_invented": True}


def side_result(side):
    state = optional(side / "PIPELINE_STATE.json")
    initial = optional(side / "initial/SELECTION_RECEIPT.json")
    sfm = optional(side / "sfm/RESULT.json")
    sfm_restore = optional(side / "sfm/SAME_CAMPAIGN_RESTORE.json")
    alignment = optional(side / "metric/ALIGNMENT.json")
    metric = optional(side / "metric/RESULT.json")
    admission = optional(side / "export/TRAINING_ADMISSION.json")
    train = optional(side / "training/STATE.json")
    evaluation = optional(side / "evaluation/RESULT.json")
    localization = optional(side / "localization/LOCALIZATION.json")
    scale_paths = [side / "scale_consistency/RESULT.json", side / "sensor_scale/RESULT.json"]
    present_scale_paths = [path for path in scale_paths if path.is_file()]
    if len(present_scale_paths) > 1:
        raise ValueError("Multiple scale diagnostics require an explicit source choice")
    scale_path = present_scale_paths[0] if present_scale_paths else None
    consistency = optional(scale_path) if scale_path else None
    training_process = optional(side / "training_process.json")
    if train and train["status"] == "completed":
        if not training_process or training_process["status"] != "completed" or training_process.get("returncode") != 0:
            raise ValueError("A full training state needs its successful actual process receipt")
        if train["iteration"] != train["target_iterations"] or not admission or train["image_count"] != admission["training_count"]:
            raise ValueError("The complete training iteration or admitted image denominator differs")
    if evaluation:
        reference = read_json(side / "initial/EVALUATION_FRAMES.json")["rows"]
        if len(evaluation["rows"]) != 24 or [row["image_name"] for row in evaluation["rows"]] != [row["image_name"] for row in reference]:
            raise ValueError("A comparison removed or reordered a frozen evaluation failure")
        if evaluation["summary"]["full_image"]["rendered_count"] != sum(row["localized"] for row in evaluation["rows"]):
            raise ValueError("The full rendered image summary and actual complete rows disagree")
    values = {"pipeline": selected(state, ("status", "pid", "process_birth", "current_stage", "controller_elapsed_seconds", "error")),
        "coarse": optional(side / "coarse/COARSE_RECEIPT.json"),
        "initial": selected(initial, ("status", "raw_frame_count", "selected_count", "quality_and_coarse_pose_eligible_count",
            "matching_candidate_pair_count", "decision_counts", "elapsed_seconds", "timings", "module_trace")),
        "SfM": selected(sfm, ("status", "initial_count", "best_registered_count", "registration_fraction", "all_initial_registered",
            "matching_seconds", "mapping_seconds", "elapsed_seconds", "models", "shard_partition", "shard_merge")),
        "SfM_same_campaign_restore": sfm_restore,
        "metric_BA": selected(metric, ("status", "selected", "results", "elapsed_seconds")),
        "alignment": alignment,
        "training_admission": selected(admission, ("status", "full_registered_denominator", "initial_count", "original_registered_count",
            "initial_not_registered", "added_registered_names", "training_count", "excluded_names", "source_model", "training_opencv_model")),
        "training_process": selected(training_process, ("status", "returncode", "elapsed_seconds", "peak_tree_RSS_bytes")),
        "training": selected(train, ("status", "iteration", "target_iterations", "executed_iterations", "image_count", "gaussian_count",
            "training_call_seconds", "elapsed_seconds", "maximum_cuda_reserved_bytes", "maximum_rss_bytes", "final_model",
            "mean_samples", "scene_timing", "resident_image_cache", "host_image_storage", "ply_serialization")),
        "localization": selected(localization, ("status", "summary")),
        "evaluation": selected(evaluation, ("status", "summary", "elapsed_seconds", "gaussian_count")),
        "scale_consistency": selected(consistency, ("status", "selected_scan_denominator", "compared_sparse_points",
            "sparse_to_raw_LiDAR", "raw_LiDAR_to_sparse_map", "GT_or_absolute_accuracy_claim")),
        "scale_consistency_source": str(scale_path) if scale_path else None,
        "costs": stage_costs(side), "source_side": str(side)}
    if evaluation:
        values["evaluation_failed_rows"] = [row for row in evaluation["rows"] if not row["localized"]]
    return values


def comparison(sides, metric_contract):
    base, opt = (sides[mode] for mode in ("baseline", "optimized"))
    row = {}
    for key, field, label in (("initial", "elapsed_seconds", "initial"),
                              ("SfM", "elapsed_seconds", "SfM"),
                              ("training", "training_call_seconds", "training_call")):
        if base[key] and opt[key] and base[key].get(field) and opt[key].get(field):
            if key == "SfM" and (base["SfM_same_campaign_restore"] or opt["SfM_same_campaign_restore"]):
                row["SfM_cold_speed_comparison_not_available_after_explicit_failure_restore"] = True
            else:
                row[label + "_observed_reduction_fraction"] = 1 - opt[key][field] / base[key][field]
    if base["evaluation"] and opt["evaluation"]:
        b, o = [side["evaluation"]["summary"]["full_image"] for side in (base, opt)]
        all24 = all(item["rendered_count"] == item["complete_denominator"] == 24 for item in (b, o))
        psnr, ssim = o["mean_PSNR_dB"] - b["mean_PSNR_dB"], o["mean_SSIM"] - b["mean_SSIM"]
        row["image_quality"] = {"PSNR_delta_dB": psnr, "SSIM_delta": ssim, "both_complete24": all24,
            "quality_guard_met_on_rendered_subset": psnr >= metric_contract["quality"]["PSNR_delta_min_db"] and
                ssim >= metric_contract["quality"]["SSIM_delta_min"],
            "complete24_quality_guard_passed": all24 and psnr >= metric_contract["quality"]["PSNR_delta_min_db"] and
                ssim >= metric_contract["quality"]["SSIM_delta_min"],
            "map_conditioned_development_diagnostic_not_GT_or_blind_test": True}
    row.update(single_pair_measurements=True, different_training_iteration_policy=True,
        absolute_scale_accuracy_not_measured=True, all75_positive_claim_allowed=False)
    return row


def fmt(value, missing="待完成"):
    return missing if value is None else f"{value:.3f}"


def count(value, missing="待完成"):
    return missing if value is None else str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(args.campaign.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    contract = read_json(args.campaign / "METRIC_CONTRACT.json")
    rows, flat = [], []
    for case in read_json(args.campaign / "CASES.json")["cases"]:
        name = case["case"]
        sides = {mode: side_result(args.campaign / name / mode) for mode in ("baseline", "optimized")}
        rows.append({"case": name, "raw": case["raw"], "coarse_route": case["coarse_route"], "sides": sides,
            "initial_equivalence": optional(args.campaign / name / "INITIAL_COMPARISON.json"),
            "comparison": comparison(sides, contract)})
        for mode, side in sides.items():
            train, initial = side["training"] or {}, side["initial"] or {}
            evaluation = (side["evaluation"] or {}).get("summary", {}).get("full_image", {})
            alignment = side["alignment"] or {}
            flat.append({"case": name, "mode": mode, "pipeline_status": (side["pipeline"] or {}).get("status", "not_started"),
                "raw_frames": initial.get("raw_frame_count"), "initial_frames": initial.get("selected_count"),
                "initial_seconds": initial.get("elapsed_seconds"), "SfM_seconds": (side["SfM"] or {}).get("elapsed_seconds"),
                "SfM_contains_explicit_same_campaign_restore": bool(side["SfM_same_campaign_restore"]),
                "registered_frames": (side["training_admission"] or {}).get("full_registered_denominator"),
                "training_frames": train.get("image_count"), "training_status": train.get("status"), "iterations": train.get("iteration"),
                "training_call_seconds": train.get("training_call_seconds"), "Gaussians": train.get("gaussian_count"),
                "training_CUDA_reserved_bytes": train.get("maximum_cuda_reserved_bytes"),
                "training_Linux_RSS_bytes": train.get("maximum_rss_bytes"), "evaluation_rendered_count": evaluation.get("rendered_count"),
                "evaluation_complete_denominator": 24, "PSNR_dB": evaluation.get("mean_PSNR_dB"), "SSIM": evaluation.get("mean_SSIM"),
                "Sim3_scale_from_visual_units": alignment.get("new_Sim3", {}).get("scale"),
                "rough_position_residual_p50_m": alignment.get("residuals_m", {}).get("p50"),
                "rough_position_residual_p95_m": alignment.get("residuals_m", {}).get("p95"),
                "completed_stage_wall_sum_seconds": side["costs"]["completed_guarded_stage_wall_sum_seconds"],
                "failed_stage_wall_sum_seconds": side["costs"]["failed_guarded_stage_wall_sum_seconds"]})
    complete_models = sum(row["training_status"] == "completed" and row["iterations"] ==
        (10000 if row["mode"] == "baseline" else 12000) for row in flat)
    terminal = all(row["pipeline_status"].startswith(("fresh_training_completed", "failed")) for row in flat)
    state = {"status": "comparison_snapshot_not_final_acceptance", "created_UTC": datetime.now(timezone.utc).isoformat(),
        "completed_models": complete_models, "all_eight_models_complete": complete_models == 8, "cases": rows,
        "all_eight_pipeline_outcomes_terminal": terminal,
        "metric_contract": contract, "primary_campaign_state": optional(args.campaign / "CAMPAIGN_STATE.json"),
        "evaluation_campaign_state": optional(args.campaign / "EVALUATION_CAMPAIGN_STATE.json"),
        "four_around_box_inputs_are_not_evidence_for_all_object_classes": True}
    atomic_json(args.output / "COMPARISON.json", state)
    with (args.output / "COMPARISON.csv").open("x", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    outcome = "八条流程终态均已记录；四录制完整质量验收未全部通过。" if terminal else "仍有流程未收尾；快照不能作为最终四录制验收。"
    text = ["# 本轮原始录制对照快照", "", f"完整训练模型：{complete_models}/8。{outcome}", "",
        "| 录制 | 路线 | 流程状态 | 初选帧 | 初选秒 | SfM秒 | 训练秒 | 高斯 | 评价帧/24 | PSNR | SSIM |", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in flat:
        failed = row["pipeline_status"].startswith("failed")
        missing = "不可用（流程失败）" if failed else "待完成"
        status = "流程失败；保留原证据" if failed else "完整训练；评价另见分母" if row["training_status"] == "completed" else "运行中或待运行"
        text.append(f"| {row['case']} | {row['mode']} | {status} | {row['initial_frames']} | {fmt(row['initial_seconds'], missing)} | "
            f"{fmt(row['SfM_seconds'], missing)} | {fmt(row['training_call_seconds'], missing)} | {count(row['Gaussians'], missing)} | "
            f"{count(row['evaluation_rendered_count'], missing)}/24 | {fmt(row['PSNR_dB'], missing)} | {fmt(row['SSIM'], missing)} |")
    text += ["", "所有速度值都是本轮单次策略对比；基准 10000 次、优化 12000 次，不能称同迭代后端速度。", "",
        "尺度来自重新解码的粗里程计或新 LiDAR/IMU ICP，经新 Sim3 对齐得到。没有独立尺寸或 GT，不报告真实尺寸误差、ATE/RPE。", "",
        "阶段累计耗时包括位于该阶段保护器内的排队；失败成本单列，不能将并行阶段累计和当作单视频实测延迟。未完成和未知成本不编造。", "",
        "C1基准在本轮监控失败后完整恢复本轮已完成特征与匹配，再重新建图；表中该侧SfM秒数是成功恢复尝试，不是另一轮从空缓存的冷特征加建图耗时。原失败及其全部成本另列，不能用这两个SfM单值宣称冷启动提速。", "",
        "两侧定位来自各自的新地图，全图 PSNR/SSIM 属于开发诊断；中心区域不作为目标物体真值。所有定位失败保留完整 24 帧分母。", "",
        "四项数据均为绕箱录制，不能据此宣称泛化到所有未来物体。逐项优化有效性由另行完整模块核对决定。"]
    (args.output / "COMPARISON.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "completed_models": complete_models, "cases": len(rows)}))


if __name__ == "__main__":
    main()
