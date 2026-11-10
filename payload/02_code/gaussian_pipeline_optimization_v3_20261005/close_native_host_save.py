"""Close the finalizer owner and publish an accurately scoped cost decision."""

import time

STARTED = time.perf_counter()

import ast
import json
from pathlib import Path
import statistics
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
BASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
OUT = BASE / "hc001"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def gone(row):
    try:
        return psutil.Process(row["pid"]).create_time() != row["creation_time"]
    except psutil.NoSuchProcess:
        return True


def main():
    if (OUT / "ACCEPTANCE.json").exists():
        raise FileExistsError("Never overwrite a mechanism decision")
    result = read(OUT / "FINAL_COST_INITIAL_001.json")
    review = read(OUT / "a2/REVIEW.json")
    campaign = read(OUT / "c0/RESULT.json")
    lease = read(OUT / "FINALIZE_LEASE_001.json")
    guard = read(OUT / "FINALIZE_GUARD_001.json")
    rows = review["Windows"] + guard["owned_process_identities"]
    rows += [{"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]},
        {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}]
    for index in (1, 2, 3):
        prior = read(OUT / f"REVIEW_LEASE_{index:03d}.json")
        rows += [{"pid": prior["guard_pid"], "creation_time": prior["guard_creation_time"]},
            {"pid": prior["child_pid"], "creation_time": prior["child_creation_time"]}]
        rows += read(OUT / f"REVIEW_GUARD_{index:03d}.json")["owned_process_identities"]
    unique = {(row["pid"], row["creation_time"]): {"pid": row["pid"], "creation_time": row["creation_time"]} for row in rows}
    win_rows = [{**row, "gone": gone(row)} for row in unique.values()]
    source_names = ("profile_current_gaussian_save.py", "native_host_file_worker.py", "native_host_gaussian_files.py",
        "profile_host_gaussian_save.py", "run_native_host_save_campaign.py", "check_native_host_file_recovery.py",
        "review_native_host_save.py", "finalize_native_host_save.py", "close_native_host_save.py")
    for name in source_names:
        ast.parse((CODE / name).read_text(encoding="utf-8"), filename=name)
    task = (DOC / "R133_NATIVE_HOST_FILE_TASK.md").read_text(encoding="utf-8")
    checks = {
        "initial_complete_cost_and_all_required_checks_passed": result["status"] == "accepted_scoped_engineering" and all(result["checks"].values()),
        "actual_finalizer_guard_and_lease_closed_successfully": lease["returncode"] == 0 and bool(lease.get("lease_released_utc")) and guard["status"] == "completed",
        "all_actual_saved_experiment_review_and_finalizer_Windows_instances_gone": all(row["gone"] for row in win_rows),
        "both_review_failures_preserved_and_charged": "FileNotFoundError" in (OUT / "REVIEW_001.log").read_text(encoding="utf-8")
            and "JSONDecodeError" in (OUT / "REVIEW_002.log").read_text(encoding="utf-8")
            and all(read(OUT / f"REVIEW_LEASE_{index:03d}.json")["returncode"] == 1 for index in (1, 2)),
        "preregistered5percent_and_original_protection_scope_read_directly": "V2速度至少5%" in task and "全部bytes/floatbits/参数/grad/layout/RNG" in task,
        "actual_R126_profile_is_diagnostic_no_device_implementation_claim": read(BASE / "dp001/SOURCE_COST_SCREEN_001.json")["new_optimization_count"] == 0}
    fees = result["first_and_required_review_seconds"] + result["campaign_envelope_extra_seconds"] + lease["child_elapsed_seconds"]
    close_seconds = time.perf_counter() - STARTED
    fees += close_seconds
    surcharge = fees / 6
    medians = {mode: {key: statistics.median(row[key] for row in campaign["rows"] if row["mode"] == mode) + surcharge
        for key in ("complete_phase_seconds", "complete_whole_seconds")} for mode in ("cpu", "host")}
    phase = 1 - medians["host"]["complete_phase_seconds"] / medians["cpu"]["complete_phase_seconds"]
    whole = 1 - medians["host"]["complete_whole_seconds"] / medians["cpu"]["complete_whole_seconds"]
    ratios = [{"pair": index, "complete_whole_ratio_with_common_fees":
        (next(row for row in campaign["rows"] if row["pair"] == index and row["mode"] == "host")["complete_whole_seconds"] + surcharge) /
        (next(row for row in campaign["rows"] if row["pair"] == index and row["mode"] == "cpu")["complete_whole_seconds"] + surcharge)} for index in range(3)]
    checks["final_all_measured_fees_speed_whole_and_each_pair_gates"] = phase >= .05 and whole >= 0 and all(row["complete_whole_ratio_with_common_fees"] <= 1.1 for row in ratios)
    result.update(status="accepted_scoped_engineering" if all(checks.values()) else "failed_final_close_verification",
        new_optimization_count=1 if all(checks.values()) else 0, final_close_checks=checks,
        medians_with_all_measured_fees=medians, complete_phase_reduction=phase, complete_whole_reduction=whole,
        common_fee_seconds=fees, per_side_surcharge_seconds=surcharge, pair_ratios_with_fees=ratios,
        finalizer_full_exited_lease_seconds=lease["child_elapsed_seconds"], close_before_publication_seconds=close_seconds,
        final_Windows_processes=win_rows, all8_prior_experiment_and_review_leases_released=True,
        small_final_metadata_publication_and_outer_process_close_accounting="all_first_diagnostics_failed_successful_reviews_and_exited_finalizer_charged_plus_this_closure_before_publication_timer")
    atomic_json(OUT / "FINAL_COST_001.json", result)
    closure = {"status": "closed_scoped_acceptance_ready" if all(checks.values()) else "failed_closure",
        "checks": checks, "actual_Windows_instances_closed": len(win_rows),
        "actual_Linux_instances_closed_in_saved_review": len(review["Linux"]),
        "source_copy_count": len(review["source_copies"]), "new_optimization_count": result["new_optimization_count"],
        "measured_complete_phase_reduction": phase, "measured_complete_whole_reduction": whole,
        "root_single_chat_self_review": True, "no_subagent": True}
    atomic_json(OUT / "FINAL_CLOSURE_001.json", closure)
    if not all(checks.values()):
        raise AssertionError("A final owner, policy or cost closure failed")
    atomic_json(OUT / "ACCEPTANCE.json", result)
    r126 = read(BASE / "dp001/SOURCE_COST_SCREEN_001.json")
    (DOC / "R126_DEVICE_PLY_PACKING_SOURCE_RESULT.md").write_text(
        "# R126：当前保存成本的实际核对\n\n"
        "2026-10-06，单chat有界诊断及自审，候选暂缓0；没有实施GPU组装或宣称该算法实测失败。\n\n"
        "直接对当前1016帧分支的526951高斯调用未修改R127/P25：全部参数、grad/layout、RNG、源输入、原loader读回、全部字节及另外一次未修改GaussianModel完整原生文件校对通过。\n\n"
        f"cProfile诊断：save2.43134秒，包含保护及读回的诊断phase4.29736秒；writer全部非文件工作最多{r126['maximum_profiled_writer_non_file_work_seconds']:.8f}秒。只加入首次诊断12.9951397秒六侧费用后，理想消除该部分的诊断收益空间约{100*r126['diagnostic_speed_envelope_after_only_first_fee']:.4f}%，小于5%。这不是候选三对测量或正式性能上界证明。\n\n"
        "当前保存原生tuples之前观察到进程峰值1059045376 bytes；逻辑属性矩阵130683848 bytes，占约12.34%，不能把逻辑bytes当候选实测RSS，更不能借后来原生tuples的2497671168 bytes峰值。当前没有满足15%内存收益的实证。因而按实际瓶颈暂缓R126，保留其独立ID和0计数。\n\n"
        "更大的可测开销是C盘文件复制和两个完整身份读，随后独立执行R133；R133收益不归R126。未执行R126 device实现、三对、恢复、其他物体或新训练。\n\n"
        "原件：[完整成本诊断](C:/lidar-camera-fusion-mask-robust-downsampling/04_outputs/gaussian_pipeline_optimization_v3_20261005/dp001/p0/RESULT.json)、[源范围筛选](C:/lidar-camera-fusion-mask-robust-downsampling/04_outputs/gaussian_pipeline_optimization_v3_20261005/dp001/SOURCE_COST_SCREEN_001.json)。\n",
        encoding="utf-8")
    (DOC / "R133_NATIVE_HOST_FILE_RESULTS.md").write_text(
        "# R133：原文件内容保持的Windows主机执行\n\n"
        "2026-10-06。confirmed：完成真实模型三对、必要保护、完整成本和root单chat自审，接纳一项限定工程优化；父新增由18升为19/50，父50目标尚未完成。原模型、训练、生产入口与当前预览未改动。\n\n"
        "模型与高斯计算继续在WSL；仅把C盘文件的实体复制和原两次完整读取校验交给现有Windows Python。双方共同当前R127、P25、Q61。保留原完整expected-byte独立验证、物理副本、全部点/字段/精度和原SHA读取；没有链接、身份免读、并行SHA或GPU组装。\n\n"
        "| 比较范围 | 原CPU/WSL路径 | 主机文件路径 | 中位减少 |\n|---|---:|---:|---:|\n"
        f"| 全部实测费用后的完整保存阶段 | {medians['cpu']['complete_phase_seconds']:.8f}s | {medians['host']['complete_phase_seconds']:.8f}s | {100*phase:.5f}% |\n"
        f"| 全部实测费用后的完整入口 | {medians['cpu']['complete_whole_seconds']:.8f}s | {medians['host']['complete_whole_seconds']:.8f}s | {100*whole:.5f}% |\n\n"
        "原三对阶段中位5.837737139→4.889494867秒，完整入口9.898137000→8.822022800秒，不把这两个未计补充费值直接用于验收。\n\n"
        f"费用保留：R126用于选择本机制的首次profile、48generic、12回退权限/取消/恢复及4真实host中断、两次失败审计与成功审计、campaign完整包络的额外部分、最终退出和关闭计时，共计{fees:.8f}秒，各六侧分摊{ surcharge:.8f}秒。campaign的六侧whole只扣一次，不重复加父comparison，也不删除第一对冷启动。最后极小JSON/报告发布自身在上述关闭计时截止之后，不声称包含未发生的后续工作或人工研究时间。\n\n"
        "三对各自完整526951高斯point/oracle均逐字节精确，和另外一次未修改GaussianModel完整原生oracle一致；全部六组参数、grad/layout/version、RNG、原loader roundtrip及源输入保持。原48通用SH0/1/3、空/非连续/特殊bits、坏header/body/EOF及不支持格式保护通过。12组Linux原取消/实际权限错误走事先选定回退且保留prefix/event；恢复真实使用host。4组在host运行时向调用者发KeyboardInterrupt/SystemExit，实际子进程排空后传播，完整已复制文件保留而不提交event，原同目录重试拒绝及新目录完整恢复通过。\n\n"
        f"root自审14项及最终关闭检查通过，19当前身份、86源副本、{len(win_rows)}保存Win实例与9保存Linux实例已退出，8个先前lease已释放。正在执行本次关闭脚本自身的退出由调用端另外核对，不把它混入已退出计数。\n\n"
        "failed_verification历史：第一次审计把/opt下plyfile.py误当Windows路径，第二次跨系统JSON携带Windows反斜杠报Invalid escape。两次原日志/退出与耗时保留；以Linux直接文本/完整源码读取和无Windows反斜杠的映射解决。只修改审计器，算法三对源码/结果未变。失败审计器完整旧源码没有单独归档，不宣称有其完整不可变快照。\n\n"
        "unconfirmed：其它物体完整训练、其它主机/文件系统、已派发期间Windows ACL被改变、新GT/新训练/新浏览器和生产部署未执行。组件不依赖箱体形状或ROI，但这里只有固定Windows/WSL/C盘及私有输出准入，不能宣称跨主机或所有未来物体端到端有效。\n\n"
        "官方依据：[Microsoft WSL文件系统](https://learn.microsoft.com/en-us/windows/wsl/filesystems)、[Python3.11 shutil](https://docs.python.org/3.11/library/shutil.html)。官方建议工具与文件系统一致、支持互操作且copyfile后端分平台；本项速度结论来自本地实际对照。\n\n"
        "原件：[正式接纳与费用](C:/lidar-camera-fusion-mask-robust-downsampling/04_outputs/gaussian_pipeline_optimization_v3_20261005/hc001/ACCEPTANCE.json)、[三对完整保存](C:/lidar-camera-fusion-mask-robust-downsampling/04_outputs/gaussian_pipeline_optimization_v3_20261005/hc001/c0/RESULT.json)、[恢复](C:/lidar-camera-fusion-mask-robust-downsampling/04_outputs/gaussian_pipeline_optimization_v3_20261005/hc001/r0/RESULT.json)、[单chat自审](C:/lidar-camera-fusion-mask-robust-downsampling/04_outputs/gaussian_pipeline_optimization_v3_20261005/hc001/a2/REVIEW.json)、[最终关闭](C:/lidar-camera-fusion-mask-robust-downsampling/04_outputs/gaussian_pipeline_optimization_v3_20261005/hc001/FINAL_CLOSURE_001.json)。\n",
        encoding="utf-8")
    docs = [DOC / name for name in ("R126_DEVICE_PLY_PACKING_SOURCE_RESULT.md", "R133_NATIVE_HOST_FILE_RESULTS.md")]
    texts = [path.read_text(encoding="utf-8") for path in docs]
    text_checks = {"direct_text_both_reports_nonempty": all(len(text) > 500 for text in texts),
        "R126_diagnostic_no_effect_claim": "不是候选三对测量" in texts[0] and "暂缓0" in texts[0],
        "R133_scope_failures_and_no50_completion_claim": "19/50" in texts[1] and "Invalid escape" in texts[1]
            and "其它物体" in texts[1] and "尚未完成" in texts[1]}
    atomic_json(OUT / "DOCUMENT_TEXT_REVIEW_001.json", {"status": "passed" if all(text_checks.values()) else "failed",
        "checks": text_checks, "files": [str(path) for path in docs], "no_document_hash_or_screenshot": True})
    if not all(text_checks.values()):
        raise AssertionError("A direct result text check failed")
    print(json.dumps({"status": result["status"], "checks": checks, "phase_reduction": phase,
        "whole_reduction": whole, "fee_seconds": fees, "medians": medians,
        "Win_closed": len(win_rows), "text_checks": text_checks}), flush=True)


if __name__ == "__main__":
    main()
