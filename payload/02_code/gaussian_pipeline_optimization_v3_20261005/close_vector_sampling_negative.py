"""Close measured negative sampling costs without treating missing checks as passed."""

import argparse
from datetime import datetime, timezone
import json
import msvcrt
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity

PARENT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
BASE = PARENT / "ud001/v5_simd_001"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"
DESTINATION = BASE / "FINAL_NEGATIVE_CLOSURE_001.json"


def unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("A receipt contains a duplicate JSON key")
        result[key] = value
    return result


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=unique_object)


def closed(row):
    try:
        return psutil.Process(row["pid"]).create_time() != row["creation_time"]
    except psutil.NoSuchProcess:
        return True


def verify():
    output = BASE / "final_audit_001"
    output.mkdir(exist_ok=False)
    state = read(DOC / "CURRENT_STATE.json")
    review = read(BASE / "r0/RESULT.json")
    formal = read(BASE / "c0/RESULT.json")
    leases = [read(BASE / (name + "_lease.json")) for name in (
        "b0", "g0", "g1", "s0", "m0", "d0", "c0_first", "c0_remaining")]
    reports = {}
    links = []
    for name in ("V5_EXISTING_COST_RANK_RESULTS.md", "R123_VECTOR_SAMPLING_RESULTS.md",
                 "NEXT_SCENE_TRANSFORM_COST_BRIEF.md"):
        text = (DOC / name).read_text(encoding="utf-8-sig")
        reports[name] = {"characters": len(text), "lines": len(text.splitlines()),
                         "has_explicit_active28_scope": "28/50" in text and "active" in text}
        for target in re.findall(r"\]\((C:/[^)]+)\)", text):
            path = Path(target)
            links.append({"path": target, "exists_or_this_pending_receipt": path.is_file() or path == DESTINATION})
    heads = {}
    for name in ("PLAN.md", "OPTIMIZE_CHECKLIST.md", "CANDIDATE_BOARD.md", "GENERALIZATION_PROTOCOL.md"):
        first = (DOC / name).read_text(encoding="utf-8-sig").split("\n\n")[1]
        heads[name] = "28/50" in first and "R123" in first
    source_ids = []
    for mode in ("native", "vector"):
        contract = read(BASE / "c0/t0" / mode / "CONTRACT.json")
        source_ids.extend(contract["sources"])
    current_sources = all(row == data_identity(row["path"]) for row in source_ids)
    code = Path(__file__).resolve().parent
    prior_review_source = code / "review_vector_sampling_cost_negative.py"
    own = psutil.Process()
    checks = {
        "saved17_negative_review_checks_passed": len(review["checks"]) == 17 and all(review["checks"].values()),
        "current28_unique_mechanisms_and28_parseable_receipts": state["additional_accepted_count"] == 28
            and len(set(state["accepted_mechanisms"])) == 28
            and len(set(state["acceptance_receipts"])) == 28
            and all(isinstance(read(ROOT / path), dict) for path in state["acceptance_receipts"]),
        "R110_and_R123_negative_no_new_count_or_production": state["R110_V5_new_optimization_count"] == 0
            and state["R123_vector_new_optimization_count"] == 0
            and not any(name.startswith("R123") or name.startswith("R110") for name in state["accepted_mechanisms"])
            and not state["R123_vector_production_default_deployed"],
        "uniform3percent_and_whole_gates_unchanged": state["V5_minimum_complete_stage_net_reduction"] == .03
            and state["V5_maximum_whole_median_ratio"] == 1.01 and state["V5_maximum_each_whole_ratio"] == 1.1,
        "same_three_complete1050_formal_outputs_not_replaced": len(formal["pairs"]) == 3
            and all(row["comparison"]["complete_byte_pairs"] == 1046
                    and row["comparison"]["complete_JSON_pairs"] == 4 for row in formal["pairs"]),
        "frozen_formal_sources_and_prior_review_snapshot_unchanged": current_sources
            and prior_review_source.read_bytes() == (BASE / "r0/source_snapshot" / prior_review_source.name).read_bytes(),
        "all99_saved_actual_instances_still_closed": len(review["closed_instances"]) == 99
            and all(closed(row) for row in review["closed_instances"]),
        "all8_saved_leases_actual_exit_release_preserved": len(leases) == 8
            and all(row["status"] == "child_exited" and row.get("lease_released_utc") for row in leases),
        "native_text_scope_and_current_heads_consistent_no_document_hash": all(row["has_explicit_active28_scope"] for row in reports.values())
            and all(heads.values()) and bool(links) and all(row["exists_or_this_pending_receipt"] for row in links),
        "recovery_still_not_run_and_unknown_object_not_proven": not (BASE / "v0").exists()
            and not state["R123_vector_actual_partial_recovery_executed"]
            and not state["R123_vector_unknown_object_full_pipeline_or_GT_proven"],
    }
    if not all(checks.values()):
        atomic_json(output / "FAILED.json", {"checks": checks, "new_optimization_count": 0})
        raise AssertionError("A negative sampling closure consistency check failed")
    with (PARENT / "coordination_001/exclusive_benchmark.lock").open("r+b", buffering=0) as stream:
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    snapshots = output / "source_snapshot"
    snapshots.mkdir()
    shutil.copyfile(Path(__file__), snapshots / Path(__file__).name)
    atomic_json(output / "RESULT.json", {"checks": checks, "reports_native_text_structure": reports,
        "changed_heads_consistent": heads, "artifact_links": links, "actual_OS_lock_acquired_released": True,
        "own_process": {"pid": own.pid, "creation_time": own.create_time()},
        "root_separate_self_review_not_independent_agent": True,
        "document_structural_checks_do_not_replace_root_native_content_review": True,
        "source_identity": data_identity(Path(__file__)), "new_optimization_count": 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        verify()
        return
    if DESTINATION.exists() or (BASE / "final_audit_001").exists():
        raise FileExistsError("A final closure requires a new private path")
    begin = time.perf_counter()
    command = [sys.executable, "-X", "utf8", str(Path(__file__)), "--verify-only"]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=30)
    elapsed = time.perf_counter() - begin
    (BASE / "final_audit_001.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError("The final negative closure verifier failed: " + completed.stderr)
    audit = read(BASE / "final_audit_001/RESULT.json")
    if not closed(audit["own_process"]):
        raise AssertionError("The actual closure child instance has not exited")
    review = read(BASE / "r0/RESULT.json")
    closed_commands = {"prior_negative_review_actual_command_exit": 1.6263549,
                       "prior_post_exit_source_and_metrics_actual_command_exit": .3008248,
                       "this_actual_verifier_spawn_to_exit": elapsed}
    known = review["known_fee_lower_bound_before_this_review_exit_seconds"] + sum(closed_commands.values())
    phase = {mode: values["phase_seconds"] + known / 6
             for mode, values in review["medians_before_fees"].items()}
    upper = 1 - phase["vector"] / phase["native"]
    if upper >= .03:
        raise AssertionError("The fixed3percent negative cost upper bound was not established")
    owner = psutil.Process()
    atomic_json(DESTINATION, {"status": "closed_negative_R123_vector_net_stage_upper_below_uniform3percent0",
        "published_utc": datetime.now(timezone.utc).isoformat(), "checks": audit["checks"],
        "root_native_text_content_review_separate_from_structural_checks": True,
        "root_self_review_not_independent_agent": True, "actual_verifier_command": command,
        "actual_verifier_returncode": completed.returncode, "actual_verifier_instance_closed": True,
        "actual_closed_additional_command_seconds": closed_commands,
        "known_fee_lower_bound_seconds": known, "complete_phase_medians_lower_bound_seconds": phase,
        "complete_stage_reduction_upper_bound": upper,
        "raw_stage_reduction": review["raw_stage_reduction"], "raw_whole_reduction": review["raw_whole_reduction"],
        "actual_saved_instance_count_closed": len(review["closed_instances"]), "saved_leases_released": 8,
        "actual_OS_lock_acquired_and_released": True, "prepared_recovery_not_executed": True,
        "unknown_historical_text_research_and_parent_publisher_tail_not_assumed_zero": True,
        "no_total_RandD_ROI_or_full_engineering_acceptance_claim": True,
        "publisher": {"pid": owner.pid, "creation_time": owner.create_time()},
        "source_identity": data_identity(Path(__file__)), "new_optimization_count": 0,
        "parent_count": 28, "remaining_count": 22, "goal_remains_active": True})
    print(json.dumps({"status": "closed_negative0", "known_fee_lower_bound_seconds": known,
        "stage_upper": upper, "checks": len(audit["checks"]), "parent_count": 28}))


if __name__ == "__main__":
    main()
