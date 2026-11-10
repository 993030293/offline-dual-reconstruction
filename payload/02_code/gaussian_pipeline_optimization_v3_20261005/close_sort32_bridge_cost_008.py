"""Close the reviewed diagnostic without counting an optimization or changing the goal."""
import datetime
import json
from pathlib import Path

import psutil

from review_orbit_convergence_dag import closed, matches

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup008"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    review = read(OUT / "SOURCE_COST_REVIEW_001.json")
    guard, lease = read(OUT / "review_GUARD.json"), read(OUT / "review_lease.json")
    births = review["ended_process_birth_records"] + guard["owned_process_identities"] + [review["reviewer_process_identity"]]
    births.extend({"pid": lease[k + "_pid"], "creation_time": lease[k + "_creation_time"]} for k in ("guard", "child"))
    master = read(DOC / "CURRENT_STATE.json")
    report = (DOC / "VIEWER_SORT32_BRIDGE_COST_008_RESULTS.md").read_text(encoding="utf-8")
    controls = [(DOC / name).read_text(encoding="utf-8") for name in ("PLAN.md", "CANDIDATE_BOARD.md", "OPTIMIZE_CHECKLIST.md")]
    brief = (DOC / "NEXT_SFM_MATCHING_COST_009_BRIEF.md").read_text(encoding="utf-8")
    checks = {
        "review_actual_d7f1b9_exit0_guard_lease_and_all16_checks": guard["returncode"] == lease["returncode"] == 0 and len(review["checks"]) == 16 and all(review["checks"].values()),
        "reviewer_guard_browser_node_and_python_births_actually_ended": all(closed(row) for row in births),
        "reviewed_current_code_and_frozen_inputs_still_same": all(matches(row) for row in review["source_identities"]),
        "parent35_mechanisms_and_receipts_unique_no_diagnostic_increment": master["additional_accepted_count"] == 35 and len(master["accepted_mechanisms"]) == len(set(master["accepted_mechanisms"])) == len(master["acceptance_receipts"]) == len(set(master["acceptance_receipts"])) == 35 and master["viewer_sort32_bridge_cost008_new_optimization_count"] == 0,
        "normal_complete_quality_and_internal_WASM_and_lifecycle_boundaries_explicit": all(value in report for value in ("13项", "内部复制", "没有另做真实context", "本轮计0", "不支持", "p0", "p1", "p2")),
        "009_next_source_cost_only_original_fractions_and_limits_kept": master["next_candidate_brief"].endswith("NEXT_SFM_MATCHING_COST_009_BRIEF.md") and all(value in brief for value in ("11583", "1002", "原whole", "没有009实现", "不自动重跑整个mapper", "single chat/no subagents")),
        "all_three_controls_read_as_text_and_scope_current": all("35/50 active" in text and "009" in text[:1600] and "008" in text[:1600] for text in controls),
        "single_chat_no_running_candidate_no_new_training_or_deployment_claim": master["execution_mode"] == "single_chat_no_subagents_or_side_chat_dispatch" and master["active_candidates"] == [] and "未启用R164" in report and "本轮计0" in report,
    }
    payload = {"status": "review_and_prior_jobs_actually_closed_final_writer_requires_exit_check0" if all(checks.values()) else "failed_diagnostic_post_closure0",
               "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "checks": checks,
               "actual_review_tool_chunk": "d7f1b9", "actual_review_tool_exit_code": 0,
               "ended_process_birth_records": births,
               "last_writer_process_identity": {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()},
               "known_guard_and_strict_JSON_failure_seconds_before_this_writer": review["known_guard_seconds_no_nested_double_charge"] + guard["elapsed_seconds"] + review["source_reader_failure"]["seconds"],
               "metadata_other_source_read_and_final_writer_cost_not_a_production_speed_measurement": True,
               "parent_count": 35, "new_optimization_count": 0, "goal_status": "active", "next_diagnostic": "009_complete_SfM_matching_source_cost_preregistered_only"}
    with (OUT / "ACTUAL_SOURCE_COST_POST_EXIT_001.json").open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": payload["status"], "checks": checks, "ended_birth_records": len(births), "count": 35, "new_count": 0}))
    if not all(checks.values()):
        raise AssertionError("The diagnostic post closure failed")


if __name__ == "__main__":
    main()
