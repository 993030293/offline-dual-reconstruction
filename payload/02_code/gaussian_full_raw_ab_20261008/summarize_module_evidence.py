"""Reconcile every frozen mechanism with current evidence without inheriting old acceptance."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from atomic_json_windows import atomic_json, read_json


LABELS = {
    "mixed_current_outcomes_see_each_complete_receipt": "结果有不同边界，须看各份完整证据",
    "complete_raw_equivalence_verified_bundle_benefit": "四录制等价；组合有收益，未单项归因",
    "complete_current_input_cache_oracle_cold_and_warm_recorded": "缓存完整核对；冷构建与热命中分列",
    "common_projection_contract_exercised_no_counterfactual_ablation": "共同投影契约；无单独收益消融",
    "registration_augmentation_outcome_recorded": "增帧注册已记录，全部原始失败保留",
    "complete_saved_state_and_sampler_verified_final_numeric_equivalence_failed": "状态与采样一致；恢复最终数值失败",
    "actual_idle_frame_work_stops_verified_no_FPS_or_power_claim": "静止绘制停止；FPS和功耗未测",
    "complete_current_viewer_assets_verified_interaction_separately_required": "完整打包核对；交互与画质另验",
    "complete_training_policy_outcome_recorded_quality_separately_required": "新训练策略有成本收益；画质分母另验",
    "complete_full_model_PLY_oracle_recorded": "完整模型属性核对；组件收益另看耗时",
    "complete_current_localization_oracle_and_timing_outcomes_recorded": "原定位组合完整核对；收益看独立耗时",
    "complete_adapted_localization_oracle_and_timing_outcomes_recorded": "适配后定位完整核对；收益看独立耗时",
    "complete_process_receipt_repaired_localization_outcomes_recorded": "进程记录修复后完整核对；失败成本保留",
    "complete_native_float_pixels_and_SSIM_oracle_verified_partial_quality_denominator": "完整像素与指标函数一致；仅22/24诊断",
    "actual_training_invocation_verified_individual_benefit_not_isolated": "完整训练实际调用；未单独归因收益",
    "complete_current_map_component_oracle_and_timing_recorded": "完整地图组件等价；局部耗时已测",
    "conditional_or_scope_inapplicable_current_benefit_unverified": "条件未触发或入口不适用；收益未确认",
    "actual_motion_resolution_and_full_restoration_verified_no_FPS_claim": "移动分辨率与恢复通过；净速度未测",
    "actual_full_model_immutable_warm_cache_verified_single_repeat": "完整模型热命中实测；只有单次重复",
    "complete_payload_and_RNG_verified_storage_benefit_not_speed": "完整状态等价；存储受益，净加速未证实",
    "failed_current_composition_interface_preserved": "原组合接口失败；短退出不算提速",
    "failed_adapted_localization_composition_preserved": "适配组合仍有失败；短退出不算提速",
    "complete_current_binary_identity_oracle_verified_Windows_scope": "完整二进制身份等价；仅Windows共享语义",
    "complete_export_artifact_and_scientific_JSON_oracles_verified": "完整导出等价；组件耗时受益",
    "complete_current_browser_native_oracles_conditionally_verified": "真实浏览器原生输出等价；条件范围成立",
    "actual_global_NVML_samples_recorded_not_per_job_peak": "全设备资源采样；不能作为单任务峰值",
    "complete_native_Scene_camera_and_pixel_oracles_verified": "完整相机与像素等价；组件耗时受益",
    "complete_current_RGB_and_camera_oracle_verified_stage_timing": "完整RGB与相机等价；仅采样组件耗时",
    "complete_camera_state_oracle_verified_no_current_net_speed_acceptance": "最终相机状态一致；当前净提速未证实",
    "current_audit_still_running": "完整复测仍在执行；尚无最终收益结论",
    "isolated_complete_native_matching_outcomes_recorded": "独立完整匹配等价；包含准备合并成本",
    "complete_native_column_oracle_timing_needs_isolated_repeat": "完整匹配等价；原竞争耗时另保留",
    "failed_or_unfinished_current_evidence_recorded": "本轮失败或未准入；不能按成功接受",
    "not_independently_verified_this_campaign": "没有本轮独立有效性验证",
}


def localization_comparison(value, pair):
    baseline, candidate = (value["outcomes"][choice] for choice in pair)
    complete = all(item["successful_trials"] == 3 and
        all(item["complete24_semantic_rows_exact_per_trial"]) and
        all(item.get("complete_query_feature_arrays_exact_per_trial", [True]))
        for item in (baseline, candidate))
    return {"baseline": baseline, "candidate": candidate, "complete_pair_oracle_passed": complete,
        "observed_reduction_fraction": 1-candidate["median_whole_stage_seconds"] /
            baseline["median_whole_stage_seconds"] if complete else None,
        "failed_short_runtime_not_speed_benefit": not complete, "comparison_choices": list(pair),
        "all24_localization_failures_retained": complete and value["no_localization_failures_removed"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.campaign = args.campaign.resolve()
    args.output = args.output.resolve()
    args.output.resolve().relative_to(args.campaign.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    registry = read_json(args.campaign / "MODULE_REGISTRY.json")["modules"]
    if len(registry) != 75 or len({row["id"] for row in registry}) != 75:
        raise ValueError("The frozen unique mechanism denominator changed")
    rows = {row["id"]: {"id": row["id"], "round": row["round"],
        "prior_evidence_reference_only": row["prior_evidence"], "status": "not_independently_verified_this_campaign",
        "evidence": [], "scope": [], "notes": [], "current_outcome_statuses": [], "old_acceptance_is_not_current_acceptance": True,
        "whole_pipeline_gain_or_generalization_not_inferred": True} for row in registry}

    def keys(prefixes):
        result = []
        for prefix in prefixes:
            names = [name for name in rows if name == prefix or name.startswith(prefix + "_")]
            if len(names) != 1:
                raise ValueError("A unique frozen mechanism must be bound: " + prefix)
            result += names
        return result

    def bind(prefixes, path, status, scope, note, native=None):
        path = args.campaign / path
        if not path.is_file():
            return
        value = read_json(path)
        active = value.get("status") in ("running", "waiting_for_primary_if_requested")
        failed = str(value.get("status", "")).startswith("failed") or value.get("status") == "subset_checks_pending"
        if active:
            status = "current_audit_still_running"
        elif failed:
            status = "failed_or_unfinished_current_evidence_recorded"
        for name in keys(prefixes):
            row = rows[name]
            row["status"] = status
            row["current_outcome_statuses"].append(status)
            row["evidence"].append({"path": str(path), "native_status": value.get("status"),
                "conclusion": status,
                "outcome_fields": native(value) if native and not active and not failed else
                    {"native_status": value.get("status"), "reported_error": value.get("error")} if active or failed else None})
            row["scope"].append(scope)
            row["notes"].append(note)

    initial = ["P1_QUALITY_THROUGHPUT", "P1_COMPACT_DERIVATIVE", "P1_BATCHED_COVERAGE", "P18", "Q50", "Q53", "Q78",
        "R99", "R104", "R107", "R141", "R147", "R149", "R144"]
    for case in read_json(args.campaign / "CASES.json")["cases"]:
        name = case["case"]
        bind(initial, name + "/INITIAL_COMPARISON.json", "complete_raw_equivalence_verified_bundle_benefit",
            name + ": all raw decisions, selected JPEG bytes, held evaluation images and pair lists",
            "四项初选均来自本侧原始录制。组合阶段收益可见；不能把组合耗时分别归给每个机制，也不能据此报告整链提速。",
            lambda v: {field: v[field] for field in ("checks", "raw_frames", "selected_count", "max_quality_abs_error",
                "max_pose_abs_error", "baseline_initial_seconds", "optimized_initial_seconds", "initial_observed_reduction_fraction")})
        for mode in ("baseline", "optimized"):
            side = name + "/" + mode
            bind(["P0"], side + "/export/TRAINING_ADMISSION.json", "common_projection_contract_exercised_no_counterfactual_ablation",
                side + ": admitted new map and full calibrated projection", "完整 K 是两侧共同修正后的契约；不把共同正确行为称作本次单项速度收益。")
            bind(["E2"], side + "/temporal/RESULT.json", "registration_augmentation_outcome_recorded",
                side + ": complete attempted temporal additions", "保留全部原始注册失败与后加帧失败；增帧流程已执行不代表全部原始帧恢复。",
                lambda v: {field: v.get(field) for field in ("added_count", "original_count", "original_registration_guard_passed", "elapsed_seconds")})
            bind(["E2"], side + "/pose_gap/RESULT.json", "registration_augmentation_outcome_recorded",
                side + ": complete attempted pose-gap additions", "时间覆盖缺口增帧没有外推里程计；其局部注册门不能替代完整原始初选注册门。",
                lambda v: {field: v.get(field) for field in ("added_count", "original_count", "pose_gap_registration_guard_passed", "original_registration_guard_passed", "elapsed_seconds")})
            if mode == "optimized":
                bind(["Q31", "Q58", "Q59", "Q60", "R157"], side + "/training/EXECUTION_MODULES.json",
                    "actual_training_invocation_verified_individual_benefit_not_isolated", side + ": all training cameras and native initializer",
                    "实际调用及完整图像数量由 trace 证明；本轮未用这份整链耗时单独归因相机预取、opaque alpha、clamp、投影构建或 device zero。",
                    lambda v: v["traces"])
                bind(["P6"], side + "/training/STATE.json", "complete_training_policy_outcome_recorded_quality_separately_required",
                    side + ": native 10000 versus gsplat 12000 policy", "迭代数不同，只能作原已接受训练策略的成本/质量对比。画质分母完整性单独判断。",
                    lambda v: {field: v.get(field) for field in ("iteration", "target_iterations", "image_count", "training_call_seconds", "gaussian_count", "maximum_cuda_reserved_bytes", "maximum_rss_bytes")})

    rows[keys(["R104"])[0]]["notes"].append("当前完整质量组合的 R144 供应相同灰度统计；R104 的旧独立实现没有另计一次收益。")
    for prefix in ("Q49", "R85", "R112", "R132", "R113", "R94"):
        bind([prefix], "box2/map_modules/RESULT.json", "complete_current_map_component_oracle_and_timing_recorded",
            "Box2 optimized full current map, three complete native pairs", "完整输出相等；时间处于本轮并行环境，只能支持该组件的观测结果。R113 的小幅收益可能在噪声范围内。",
            lambda v, prefix=prefix: v["modules"][prefix])
    for prefix in ("P1_PROVENANCE_CACHE", "Q33", "Q47", "Q48"):
        bind([prefix], "box2/cache_modules/RESULT.json", "complete_current_input_cache_oracle_cold_and_warm_recorded",
            "Box2 complete new raw quality / selected JPEG / all24 PNG / all24 SIFT according to module",
            "冷构建费用与重复命中费用分列；首次从原始数据处理没有历史缓存。不能把热缓存收益当作首次整链收益。",
            lambda v, prefix=prefix: {key: child for key, child in v["modules"][prefix].items() if key != "trials"})
    rows[keys(["Q48"])[0]]["notes"].append("Q48缓存诊断采用原固定提取选项及OpenCV 8线程，全部24查询数组与本轮4线程主定位已保存数组相等。缓存键仍绑定完整选项/线程，4线程主入口应有独立命名空间，不能静默复用8线程缓存身份。")
    bind(["R165"], "box2/matching_modules/RESULT.json", "complete_native_column_oracle_timing_needs_isolated_repeat",
        "Box2 identical current features and complete native pair graph, three pairs",
        "原三次完整列/值/blob 相等；存在与 8 线程建图重叠造成 CPU 预算超额的可能，独立复跑结果优先于该速度值。",
        lambda v: {field: v.get(field) for field in ("status", "all_original_pairs", "median_seconds", "reduction_fraction", "complete_oracles")})
    bind(["R165"], "box2/matching_modules_isolated/RESULT.json", "isolated_complete_native_matching_outcomes_recorded",
        "Box2 isolated same-current-feature full graph, three pairs", "保留先前有竞争试验，独立核对完整列/值/blob 与准备和合并成本。",
        lambda v: {field: v.get(field) for field in ("median_seconds", "reduction_fraction", "complete_oracles")})
    bind(["P3", "R96", "R168"], "box2/checkpoint_modules/RESULT.json",
        "complete_saved_state_and_sampler_verified_final_numeric_equivalence_failed",
        "Box2 all1003 cameras, continuous12000 versus checkpoint6000 and full native/combined resume12000",
        "保存/加载的全部状态和 RNG 及 12000 个采样名称完全相等；两种恢复最终高斯数量和全部属性未等价。原生恢复对照也失败，原因尚未证实。不能报告最终数值复现通过。",
        lambda v: {field: v[field] for field in ("checks", "complete_saved_state_and_sampler_passed", "final_training_numeric_diagnostic_passed", "checkpoint")})
    bind(["Q76", "R84"], "box2/checkpoint_transport_modules/RESULT.json", "complete_payload_and_RNG_verified_storage_benefit_not_speed",
        "Box2 complete new checkpoint6000, native/gzip/zstd three repetitions", "完整负载/RNG 保留；gzip 压缩加恢复显著更慢，zstd 没有明显净速度收益，仅可有条件节省归档存储。",
        lambda v: {field: v[field] for field in ("rows", "full_state_and_rng_preserved", "source_checkpoint_unchanged")})
    bind(["Q61", "R127", "P22"], "box2/gpu_evaluation_modules/RESULT.json", "complete_full_model_PLY_oracle_recorded",
        "Box2 both complete new Gaussian PLY models, all62 fields, three pairs", "完整 PLY 字段和原生输出核对；不能从序列化组件时间推断训练速度。",
        lambda v: v["families"])
    bind(["R100", "Q30", "R135", "R134", "R172"], "box2/float_image_modules/RESULT.json",
        "complete_native_float_pixels_and_SSIM_oracle_verified_partial_quality_denominator",
        "Box2 both new full models, all22 successful native float render views; two failures retained in24",
        "使用真实原生 float 渲染输入；量化逐像素完全相等，SSIM 使用冻结 1e-6 容差。完整评价分母为24，不能把22帧函数一致性说成完整画质验收。",
        lambda v: {field: v[field] for field in ("families", "all_float_pixels_exact", "all_float_SSIM_values_within_frozen_tolerance", "common_complete_images", "failures")})
    bind(["R123"], "box2/RGB_sampling_modules/RESULT.json", "complete_current_RGB_and_camera_oracle_verified_stage_timing",
        "Box2 all1028 fresh selected/added/evaluation JPEG images, three pairs",
        "完整 RGB 和相机数据相等；包含采样映射冷构建，JPEG 解码位于被测采样 API 外。约束作用域不能扩大成整条定位速度。",
        lambda v: {field: v[field] for field in ("rows", "input_count", "images_unchanged")})
    bind(["Q68", "R92"], "box2/input_identity_modules/RESULT.json", "complete_current_binary_identity_oracle_verified_Windows_scope",
        "Box2 all1013 current binary input files, three complete identity passes per trial, three trials",
        "完整非文档二进制身份相等；持有文件的写入拒绝只验证 Windows 共享语义，不扩大到 WSL，也不以哈希验收文档内容。",
        lambda v: {field: v[field] for field in ("rows", "input_count", "isolated_Windows_write_blocked_while_held", "isolated_copy_mutable_after_close")})
    bind(["R153", "R156"], "box2/scene_image_modules_retry_002/RESULT.json", "complete_native_Scene_camera_and_pixel_oracles_verified",
        "Box2 all1003 cameras, full native Scene and original Gaussian initializer; three pairs",
        "真实文件读取从2006降到1003，完整像素、矩阵、元数据和初始化属性相等；没有执行训练迭代，因此组件时间不冒充训练收益。",
        lambda v: {field: v[field] for field in ("complete_current_image_count", "handoff_rows", "digest_rows", "all_current_images_unchanged")})
    bind(["R140"], "box2/export_dag_modules_retry_001/RESULT.json", "complete_export_artifact_and_scientific_JSON_oracles_verified",
        "Box2 current full map and roles, all1033 binary/image outputs and four scientific JSON, three pairs",
        "输出所有权适配独立登记；原 worker 算法不改。中位整阶段收益约12.61%，存在另一建图并行，不是独立四场景速度证据。",
        lambda v: {field: v[field] for field in ("rows", "median_whole_process_reduction", "all_export_files_and_scientific_values_equal")})

    bind(["R138", "R164"], "viewer/browser_module_audit_003/RESULT.json", "complete_current_browser_native_oracles_conditionally_verified",
        "Box1/Box2 both full new models; all SH3; three pairs; actual Spark worker and complete RGBA",
        "R138 仅全模型位于相机背后可省 draw；R164 减少临时数组新分配，异步工作调用数不同，不能报告同工作量速度/RSS/显存收益。",
        lambda v: {"branches": [row["branch"] for row in v["rows"]], "errors": v["errors"]})
    bind(["R162"], "viewer/browser_module_audit_003/RESULT.json", "complete_camera_state_oracle_verified_no_current_net_speed_acceptance",
        "Box1/Box2 all registered camera rows, three repetitions", "所有最终或原生 fallback 状态相等。Box2基准1004视角都不适用；其他模型的CPU更新数增加。GPU帧数和净时间收益未测，不能默认启用为提速。",
        lambda v: {row["branch"]: row["orbit_rows"] for row in v["rows"]})
    bind(["P8"], "box1/visual_review/OPTIMIZED_IDLE_CHECK.json", "actual_idle_frame_work_stops_verified_no_FPS_or_power_claim",
        "Box1 new optimized full native model, actual UI", "两个静止观察期间绘制计数相等，无 pending 和播放；没有测功耗或FPS。", lambda v: v["checks"])
    bind(["Q69"], "box1/visual_review/OPTIMIZED_MOTION_RESOLUTION_CHECK.json", "actual_motion_resolution_and_full_restoration_verified_no_FPS_claim",
        "Box1 actual path playback and stop", "移动 ratio1.125、静止1.5，停止后恢复原物理像素网格；完整静止RGBA对照尚未执行。", lambda v: v["checks"])
    bind(["Q71"], "box1/visual_review/OPTIMIZED_CACHE_CHECK.json", "actual_full_model_immutable_warm_cache_verified_single_repeat",
        "Box1 same complete new model reopened in native browser", "完整新模型和同构建Spark资源均真实命中，四个 payload 检查保留；单次热载入不代表稳定或首次收益。", lambda v: v["checks"])
    for case in ("box2", "box1", "c1", "c0_july"):
        for mode in ("baseline", "optimized"):
            branch = case + "-" + mode
            bind(["P10", "P14", "Q77", "P25"], "viewer/" + branch + "/ASSET_MANIFEST.json",
                "complete_current_viewer_assets_verified_interaction_separately_required", branch + ": complete scene and display focus assets",
                "完整 population、SH3和packed encode/decode 原生数组核对。自动聚焦只是通用显示裁剪；打包不是原浮点PLY无损几何表示。旧构建共享不等于所有未来视角质量。")
    resource_path = "resource_monitor_002/RESULT.json" if (args.campaign / "resource_monitor_002/RESULT.json").is_file() else "resource_monitor_002/STATE.json"
    bind(["R122"], resource_path, "actual_global_NVML_samples_recorded_not_per_job_peak",
        "Active campaign device-global NVML cadence2seconds", "Windows桌面和其他程序同时占用GPU，必须区分全设备采样与训练Torch的自有最大reserved；采样不是连续峰值。",
        lambda v: {"samples": v.get("samples"), "maximum_global_samples": v.get("maximum_global_samples"), "last": v.get("last")})
    localization_pairs = {"P27": ("eager_metadata", "lazy_metadata"), "Q28": ("lazy_original", "lazy_metadata"),
        "Q51": ("lazy_metadata", "bytes"), "R94": ("bytes", "indexed"), "R90": ("indexed", "integer"),
        "R120": ("integer", "prepared"), "R124": ("prepared", "process"), "R171": ("prepared", "strict"),
        "R102": ("prepared", "ordered")}
    localization_result = args.campaign / "box2/localization_modules/RESULT.json"
    for prefix, pair in localization_pairs.items():
        if localization_result.is_file():
            current = localization_comparison(read_json(localization_result), pair)
            status = "complete_current_localization_oracle_and_timing_outcomes_recorded" if current["complete_pair_oracle_passed"] else "failed_current_composition_interface_preserved"
            bind([prefix], "box2/localization_modules/RESULT.json", status,
                "Box2 all24 current own-map queries, cold private variants, three repetitions",
                "功能逐帧核对包括两项失败；三次耗时包含本轮完整准备。原始与候选定位行可能不同，必须查看完整 oracle，不能只凭进程成功宣称结果相等或提速。",
                lambda v, pair=pair: localization_comparison(v, pair))
        else:
            bind([prefix], "box2/localization_modules/STATE.json", "current_audit_still_running",
                "Box2 complete24 current own-map queries, fresh private variants", "完整三次定位组件核对正在等待主任务；尚未生成RESULT时不能宣布当前收益。")
        compatible_result = args.campaign / "box2/localization_modules_compatibility_001/RESULT.json"
        if compatible_result.is_file():
            current = localization_comparison(read_json(compatible_result), pair)
            status = "complete_adapted_localization_oracle_and_timing_outcomes_recorded" if current["complete_pair_oracle_passed"] else "failed_adapted_localization_composition_preserved"
            bind([prefix], "box2/localization_modules_compatibility_001/RESULT.json", status,
                "Box2 complete24 current queries, three new independent adapted trials and all query arrays",
                "原18次组合接口失败保留。新适配只在组合入口把原生字节存储还原为同一float32/512数值，并把完整原调用SIFT选项传给两名工作进程；不修改距离、几何、PnP或质量阈值。三次全定位行和完整查询特征数组分别核对。",
                lambda v, pair=pair: localization_comparison(v, pair))
        else:
            bind([prefix], "box2/localization_modules_compatibility_001/STATE.json", "current_audit_still_running",
                "Box2 complete24 current queries and feature arrays, three new independent adapted trials",
                "旧组合失败保留；兼容适配仍在复测，尚未完成的重复不能报告最终一致性或收益。")
        if prefix == "R124":
            retry_result = args.campaign / "box2/localization_process_receipt_retry_001/RESULT.json"
            if retry_result.is_file():
                current = localization_comparison(read_json(retry_result), pair)
                status = "complete_process_receipt_repaired_localization_outcomes_recorded" if current["complete_pair_oracle_passed"] else "failed_adapted_localization_composition_preserved"
                bind([prefix], "box2/localization_process_receipt_retry_001/RESULT.json", status,
                    "Three complete current process trials after JSON-only enum receipt repair",
                    "首次适配进程完成24查询后因配置枚举写记录失败，三次原失败与费用保留。额外适配仅规范记录字段，实际工作进程选项和所有算法阈值保持原样；完整定位行、查询特征和关闭后的进程所有权单独核对。",
                    lambda v, pair=pair: localization_comparison(v, pair))
            else:
                bind([prefix], "box2/localization_process_receipt_retry_001/STATE.json", "current_audit_still_running",
                    "Queued complete process trials; original option receipt failures preserved",
                    "配置枚举记录修复后，等待原完整复测终态才重跑三次，不覆盖旧失败。")
    for prefix, note in (("Q55", "有界MCAP解压是内存失败恢复分支；本轮四组初选未需要该分支，尚无本轮完整成本比较。"),
        ("R91", "原后训练DAG要求同输入的两套12000模型和完整24帧质量门，本轮主对照为10000/12000且Box2仅22/24可渲染；未作为主整链执行。"),
        ("R133", "原host worker将输出限制在旧hc001隔离目录，本轮新输出不满足其所有权入口，未报告此分支加速。"),
        ("R142", "原恢复诊断还绑定1016相机；本轮当前1003/1197相机，尚未执行相应完整恢复成本比较。")):
        row = rows[keys([prefix])[0]]
        row["status"] = "conditional_or_scope_inapplicable_current_benefit_unverified"
        row["notes"].append(note)
    for prefix in ("P22",):
        row = rows[keys([prefix])[0]]
        row["notes"].append("完整新模型已写出并进入原生渲染；向量PLY完整输出核对见R127，同家族不能把共享序列化速度重复归因。")
    for row in rows.values():
        outcomes = sorted(set(row["current_outcome_statuses"]))
        row["current_outcome_statuses"] = outcomes
        if len(outcomes) > 1:
            row["status"] = "mixed_current_outcomes_see_each_complete_receipt"
    counts = Counter(row["status"] for row in rows.values())
    result = {"status": "versioned_current_evidence_ledger_not_final_all75_acceptance", "created_UTC": datetime.now(timezone.utc).isoformat(),
        "frozen_unique_mechanisms": 75, "all75_effective": False, "status_counts": dict(counts), "modules": list(rows.values()),
        "subcomponent_and_bundle_benefits_not_added_together": True, "failure_unexecuted_and_scope_limits_retained": True}
    atomic_json(args.output / "MODULE_OUTCOMES.json", result)
    text = ["# 本轮 75 项优化证据账本", "", "这是本轮状态与完整证据的逐项记录，不能继承旧接受数量或宣称全部有效。", "",
        "| ID | 当前结论 | 本轮证据 | 适用边界 |", "|---|---|---|---|"]
    for row in rows.values():
        paths = "、".join(f"[{Path(item['path']).parent.name}]({Path(item['path']).as_posix()})" for item in row["evidence"])
        label = LABELS[row["status"]]
        text.append(f"| {row['id']} | {label} | {paths or '未有本轮独立结果'} | {'；'.join(dict.fromkeys(row['notes']))} |")
    (args.output / "MODULE_OUTCOMES.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "count": 75, "status_counts": dict(counts)}))


if __name__ == "__main__":
    main()
