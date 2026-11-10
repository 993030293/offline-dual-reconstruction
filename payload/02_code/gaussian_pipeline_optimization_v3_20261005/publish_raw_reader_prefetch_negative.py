"""Observe the completed RAW review and validate its native text publication."""

import json
import msvcrt
from pathlib import Path
import re
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/raw_reader_scorer_dag_001"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from gpu_query_node import remaining


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    began = time.perf_counter()
    output = OWN / "FINAL_SCREEN_PUBLICATION_001.json"
    if output.exists():
        raise FileExistsError("The final negative publication observation must be fresh")
    review = read(OWN / "r0/RESULT.json")
    leases = [read(OWN / name) for name in ("GENERIC_LEASE_001.json", "PILOT_LEASE_001.json", "REVIEW_LEASE_001.json")]
    identities = [*review["observed_Windows_instances"], review["own_process"]]
    for lease in leases:
        for prefix in ("guard", "child"):
            identities.append({"pid": lease[prefix + "_pid"], "creation_time": lease[prefix + "_creation_time"]})
    unique = {(row["pid"], row["creation_time"]): row for row in identities}
    live = remaining(list(unique.values()))
    state = read(DOC / "CURRENT_STATE.json")
    texts = {name: (DOC / name).read_text(encoding="utf-8") for name in
        ("R143_RAW_READER_SCORER_DAG_TASK.md", "R143_RAW_READER_SCORER_DAG_RESULTS.md", "PLAN.md", "CANDIDATE_BOARD.md", "OPTIMIZE_CHECKLIST.md")}
    links = [{"path": match.group(1), "exists_or_this_pending_publication": Path(match.group(1)).exists() or Path(match.group(1)) == output}
        for match in re.finditer(r"\]\((C:/[^)]+)\)", texts["R143_RAW_READER_SCORER_DAG_RESULTS.md"])]
    coordination = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/coordination_001"
    with (coordination / "exclusive_benchmark.lock").open("r+b", buffering=0) as stream:
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    owner = read(coordination / "LOCK_OWNER.json")
    fee = sum(lease["child_elapsed_seconds"] for lease in leases)
    checks = {"review_all11_original_source_scope_guards": len(review["checks"]) == 11 and all(review["checks"].values()),
        "all16_current_executed_source_inputs_still_exact": review["executed_source_inputs"] == [data_identity(Path(row["path"])) for row in review["executed_source_inputs"]],
        "review_and_all_saved_actual_Windows_instances_exited": not live,
        "three_actual_leases_released_and_OS_lock_free": all(lease["returncode"] == 0 and lease.get("lease_released_utc") for lease in leases)
            and owner["status"] == "released" and bool(owner.get("lease_released_utc")),
        "parent22_unique_accepted_mechanisms_receipts_and_no_live_candidate": state["additional_accepted_count"] == 22
            and len(state["accepted_mechanisms"]) == len(set(state["accepted_mechanisms"])) == 22
            and len(state["acceptance_receipts"]) == len(set(state["acceptance_receipts"])) == 22 and not state["active_candidates"],
        "negative_component0_no_full_entry_claim": state["R143_new_optimization_count"] == 0 and state["R143_component_reduction"] < .05
            and "full7338" in state["R143_not_run"],
        "all_measured_generic_pilot_review_computational_envelopes_kept": abs(fee - state["R143_measured_computational_envelopes_seconds"]) < 1e-9,
        "native_report_all_links_and_scope_agree": bool(links) and all(row["exists_or_this_pending_publication"] for row in links)
            and all("R143" in value for value in texts.values()) and "未运行" in texts["R143_RAW_READER_SCORER_DAG_RESULTS.md"],
        "B007_V2_cost_bound_saved_and_below5percent": read(ROOT / state["B007_V2_cost_lower_bound_receipt"])["maximum_possible_complete_phase_reduction_after_this_lower_bound"] < .05}
    if not all(checks.values()):
        raise AssertionError("A RAW post-exit, cost or native text publication guard failed")
    atomic_json(output, {"status": "negative_RAW_component_review_post_exit_and_native_text_verified", "checks": checks,
        "observed_actual_Windows_instances": list(unique.values()), "actual_live_instances": live,
        "native_text_characters": {name: len(value) for name, value in texts.items()}, "report_links": links,
        "known_generic_pilot_review_computational_envelopes_seconds": fee,
        "publisher_owner": {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()},
        "publisher_own_actual_shutdown_observation_pending_separate": True,
        "last_text_edit_publication_and_serialization_not_full_research_wall_time": True,
        "new_optimization_count": 0, "parent_count": 22, "goal_still_active": True,
        "publication_before_last_serialization_seconds": time.perf_counter() - began})
    print(json.dumps({"status": "negative_RAW_component_review_post_exit_and_native_text_verified", "checks": checks, "instances": len(unique)}), flush=True)


if __name__ == "__main__":
    main()
