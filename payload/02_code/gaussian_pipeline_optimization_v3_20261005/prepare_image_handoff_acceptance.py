"""Prepare scoped work acceptance after real closer exit and native text review."""

from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
import time

import psutil

from run_open_image_handoff_campaign import OUT
from review_scene_batch_negative import read, windows_closed
from scene_transform_cost_node import ROOT
from runtime import atomic_json, data_identity


def main():
    started = time.perf_counter()
    cost = read(OUT / "COST_CLOSURE_PENDING_EXIT_001.json")
    formal = read(OUT / "c0/RESULT.json")
    review = read(OUT / "r1/RESULT.json")
    if not all(cost["checks"].values()) or not windows_closed(cost["own_process"]) or not windows_closed(review["own_process"]):
        raise ValueError("The actual cost closer or root review did not exit successfully")
    if formal["sources"] != [data_identity(row["path"]) for row in formal["sources"]]:
        raise ValueError("An actual complete candidate source changed after review")
    docs = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
    report, next_action = [docs / name for name in ("R156_OPEN_IMAGE_HANDOFF_RESULTS.md", "NEXT_SCENE_CONSUMER_BRIEF.md")]
    texts = [path.read_text(encoding="utf-8") for path in (report, next_action)]
    required = ("4064", "3048", "10.889214", "0.130851", "110.8", "15342", "16256", "v1", "v2", "r0", "21", "不是", "未")
    if not all(token in texts[0] for token in required) or "尚未" in texts[0]:
        raise ValueError("The native result report lost a checked value or claim boundary")
    local_links = re.findall(r"\]\((C:/[^)]+)\)", texts[0])
    if not local_links or not all(Path(link).is_file() for link in local_links):
        raise ValueError("A native report reference does not resolve to its actual result")
    parent = read(docs / "CURRENT_STATE.json")
    if parent["additional_accepted_count"] != 29 or parent["R156_new_optimization_count"] != 0 or len(parent["accepted_mechanisms"]) != 29:
        raise ValueError("The parent must remain unpromoted until actual publisher exit")
    process = psutil.Process()
    source_copy = OUT / "PUBLISHER_SOURCE_002.py"
    shutil.copyfile(Path(__file__), source_copy)
    fees = cost["known_fee_plus_registered10_reserve_seconds"] + .4085664 + .2598131 + .268376
    control, handoff = (formal["raw_whole_medians"][mode] for mode in ("control", "handoff"))
    adjusted = {"control": control + fees / 6, "handoff": handoff + fees / 6}
    expected_links = [OUT / "PILLOW_LIFECYCLE_001.json", docs / "R156_OPEN_IMAGE_HANDOFF_TASK.md",
                      OUT / "c0/RESULT.json", OUT / "COST_CLOSURE_PENDING_EXIT_001.json",
                      OUT / "v3/RESULT.json", OUT / "g1/RESULT.json", OUT / "r1/RESULT.json",
                      ROOT / "02_code/gaussian_pipeline_optimization_v3_20261005/open_image_handoff.py"]
    checks = {"all_original21_root_and5_cost_checks": all(review["checks"].values()) and all(cost["checks"].values()),
              "root_and_cost_closer_own_actual_instances_exited": windows_closed(review["own_process"]) and windows_closed(cost["own_process"]),
              "native_report_values_boundaries_and_all_expected_actual_reference_paths": {Path(link) for link in local_links} == set(expected_links) and all(Path(link).is_file() for link in local_links),
              "net_work10percent_and_whole1point01_gates_still_pass": cost["net_source_read_work_under_registered_six_side_model"] >= .10 and adjusted["handoff"] / adjusted["control"] <= 1.01 and max(formal["whole_ratios"]) <= 1.1,
              "one_distinct_unpromoted_family_and_parent29": parent["R156_new_optimization_count"] == 0 and not any(name.startswith("R156") for name in parent["accepted_mechanisms"])}
    receipt = {"status": "scoped_work_acceptance_prepared_actual_publisher_exit_pending0", "checks": checks,
        "new_optimization_count": 0, "parent_count": 29, "own_process": {"pid": process.pid, "creation_time": process.create_time()},
        "raw_source_read_work_reduction": .25, "net_source_read_work_under_registered_six_side_model": cost["net_source_read_work_under_registered_six_side_model"],
        "raw_whole_medians": formal["raw_whole_medians"], "raw_whole_reduction": formal["raw_whole_reduction"],
        "known_fee_plus_registered10_reserve_before_publisher_exit_seconds": fees,
        "actual_cost_closer_tool_exit": {"chunk_id": "981448", "returncode": 0, "seconds": .4085664},
        "actual_native_text_review_tool_exit": {"chunk_id": "07f6f2", "returncode": 0, "seconds": .2598131},
        "prior_failed_publisher_expected_link_count_tool_exit": {"chunk_id": "2caf9e", "returncode": 1, "seconds": .268376},
        "speed3percent_gate_passed": False, "RSS_increase_bytes": cost["RSS_increase_bytes"],
        "scope": "optional_loaded_native_RGB_image_handoff_complete_original_camera_preparation_source_file_work_only",
        "all_original_quality_input_and_recovery_gates_kept": True, "production_default_deployed": False,
        "training_GT_FPS_energy_or_unknown_object_whole_pipeline_improvement": False,
        "report": str(report), "next_action": str(next_action), "local_text_references_verified": local_links,
        "actual_source": data_identity(Path(__file__)), "source_copy": data_identity(source_copy),
        "root_review_is_independent_agent": False, "goal_remains_active": True,
        "before_result_seconds": time.perf_counter() - started, "created_utc": datetime.now(timezone.utc).isoformat()}
    atomic_json(OUT / "ACCEPTANCE_PREPARED_002.json", receipt)
    if not all(checks.values()):
        raise AssertionError("A prepared work receipt failed its original complete gates")
    print({"checks": checks, "new": 0, "publisher_actual_exit_pending": True})


if __name__ == "__main__":
    main()
