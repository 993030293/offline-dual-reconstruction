"""Record final publisher exits and incorporate known standalone envelope differences."""

import json
import msvcrt
from pathlib import Path
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/native_quality_workspace_001"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def gone(row):
    try:
        return abs(psutil.Process(row["pid"]).create_time() - row["creation_time"]) >= .001
    except psutil.NoSuchProcess:
        return True


def main():
    started = time.perf_counter()
    destination = OWN / "FINAL_POST_EXIT_001.json"
    if destination.exists():
        raise FileExistsError("Preserve the final post-exit and fee receipt")
    publication = load(OWN / "FINAL_PUBLICATION_001.json")
    acceptance = load(OWN / "ACCEPTANCE.json")
    formal = load(OWN / "c1/RESULT.json")
    parent = load(DOC / "CURRENT_STATE.json")
    instances = [load(OWN / (stage + "_EXECUTION_001.json")) for stage in ("COST", "PUBLICATION")]
    checks = {"all_original9_publication_checks": len(publication["checks"]) == 9 and all(publication["checks"].values()),
              "both_actual_cost_and_text_publisher_instances_exited": all(gone(row) for row in instances),
              "parent22_unique_receipts_and_no_active_candidate_or_production_change": parent["additional_accepted_count"] == 22
                  and len(set(parent["accepted_mechanisms"])) == len(set(parent["acceptance_receipts"])) == 22
                  and parent["active_candidates"] == [] and not parent["production_modified"]}
    with (ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/coordination_001/exclusive_benchmark.lock").open("r+b", buffering=0) as handle:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            checks["actual_shared_OS_lock_still_available"] = True
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    native_text = []
    for name in ("R141_NATIVE_QUALITY_WORKSPACE_TASK.md", "R141_NATIVE_QUALITY_WORKSPACE_RESULTS.md", "CANDIDATE_BOARD.md", "PLAN.md", "OPTIMIZE_CHECKLIST.md"):
        text = (DOC / name).read_text(encoding="utf-8")
        if not text.strip() or "\ufffd" in text:
            raise AssertionError("A native final document is empty or corrupt")
        native_text.append({"path": str(DOC / name), "direct_native_text_characters": len(text)})
    checks["updated_native_result_cost_rounding_and_scope"] = all(word in (DOC / "R141_NATIVE_QUALITY_WORKSPACE_RESULTS.md").read_text(encoding="utf-8")
        for word in ("8.7%", "8.1%", "22/50", "未部署", "未见物体", "OS", "后置关闭"))
    nodes = load(OWN / "POST_PUBLICATION_TOOL_COSTS_001.json")["nodes"]
    difference = sum(row["actual_envelope_seconds"] - row["already_charged_seconds"] for row in nodes)
    checks["all_known_standalone_envelope_differences_nonnegative"] = all(row["actual_envelope_seconds"] >= row["already_charged_seconds"] for row in nodes)
    elapsed = time.perf_counter() - started
    shared = acceptance["all_measured_shared_fees_through_cost_publication_seconds"] + difference + elapsed
    medians = {name: {mode: seconds + shared / 6 for mode, seconds in values.items()} for name, values in formal["medians"].items()}
    reductions = {name: 1 - values["workspace"] / values["native"] for name, values in medians.items()}
    checks["all_known_fees_including_post_exit_phase5percent_and_whole_protection"] = reductions["phase_seconds"] >= .05 and reductions[
        "whole_seconds"] >= 0 and all(pair["whole_seconds"]["workspace"] + shared / 6 <= 1.1 * (
            pair["whole_seconds"]["native"] + shared / 6) for pair in formal["pairs"])
    if not all(checks.values()):
        raise AssertionError(json.dumps(checks))
    payload = {"status": "actual_publishers_exited_updated_native_text_all_known_post_exit_cost_and_parent22_verified",
               "checks": checks, "actual_publisher_instances_exited": instances, "native_text_read": native_text,
               "all_measured_shared_fees_seconds": shared, "complete_medians_seconds": medians, "complete_reductions": reductions,
               "known_standalone_envelope_differences_seconds": difference, "current_close_before_final_write_seconds": elapsed,
               "last_receipt_serialization_stdout_process_shutdown_not_separately_timed": True,
               "close_pid": psutil.Process().pid, "close_creation_time": psutil.Process().create_time(),
               "parent_goal_active_not_complete": True, "counted": False}
    (OWN / "POST_EXIT_CHECKER_SOURCE_001.py").write_bytes(Path(__file__).read_bytes())
    atomic_json(destination, payload)
    acceptance.update(all_measured_shared_fees_through_cost_publication_seconds=shared,
                      complete_phase_and_whole_medians_seconds=medians, complete_reductions=reductions,
                      final_post_exit=str(destination), actual_known_standalone_envelope_differences_seconds=difference)
    atomic_json(OWN / "ACCEPTANCE.json", acceptance)
    atomic_json(OWN / "FINAL_COST_001.json", payload)
    print({"status": payload["status"], "checks": len(checks), "shared_fees_seconds": shared,
           "complete_medians": medians, "complete_reductions": reductions, "accepted_count_contribution": 1})


if __name__ == "__main__":
    main()
