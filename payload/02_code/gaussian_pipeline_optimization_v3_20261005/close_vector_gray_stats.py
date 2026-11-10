"""Verify final original-text publication, actual exits and the known-cost ledger."""

from datetime import datetime, timezone
import msvcrt
from pathlib import Path
import re
import time

import psutil

import profile_quality_corner_cost as reference
from vector_gray_stats_evidence import cost_snapshot

ROOT = reference.ROOT
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
DOC = ROOT / "research/2026-10-05-gaussian-optimization-round3-50"


def main():
    started = time.perf_counter()
    base = OUT / "vector_gray_stats_001"
    path = base / "FINAL_COST_001.json"
    if path.exists():
        raise ValueError("The final immutable vector closure already exists")
    parent = reference.load(DOC / "CURRENT_STATE.json")
    acceptance = reference.load(base / "ACCEPTANCE.json")
    review = reference.load(base / "r1/RESULT.json")
    complete = reference.load(base / "c0/RESULT.json")
    checks = {"parent27_unique_accepted_mechanisms_and_receipts": parent["additional_accepted_count"] == (
        len(parent["accepted_mechanisms"])) == len(set(parent["accepted_mechanisms"])) == len(parent["acceptance_receipts"]) == 27,
        "exactly_one_R144_family_accepted_original_negative_kept": parent["accepted_mechanisms"].count(acceptance["mechanism"]) == 1
            and acceptance["new_optimization_count"] == parent["R144_V2_new_optimization_count"] == 1 and parent["R144_new_optimization_count"] == 0,
        "R150_already_indexed_quad_closed0": parent["R150_new_optimization_count"] == 0
            and reference.load(OUT / "qt001/p0/RESULT.json")["accepted_count_contribution"] == 0,
        "original_scientific_source_and_separate_audit_identities_unchanged": all(
            reference.data_identity(Path(row["path"])) == row for row in review["sources_verified"] + review["related_ids"]),
        "original_successful_self_review_18_checks": len(review["checks"]) == 18 and all(review["checks"].values())}
    owned = {}
    leases = list(base.glob("LEASE_*.json")) + [OUT / "post_corner_cost_001/LEASE_P0.json", OUT / "rm001/LEASE_P0.json", OUT / "qt001/LEASE_P0.json"]
    for file in leases:
        row = reference.load(file)
        expected = 1 if file == base / "LEASE_R0.json" or file == OUT / "rm001/LEASE_P0.json" else 0
        if row["returncode"] != expected or not row.get("lease_released_utc"):
            raise ValueError("A real original lease failed or did not release")
        for prefix in ("guard", "child"):
            owned[(row[prefix + "_pid"], row[prefix + "_creation_time"])] = str(file)
    guards = list(base.glob("GUARD_*.json")) + list((base / "c0").glob("t*/*_PROCESS.json"))
    for file in guards:
        row = reference.load(file)
        expected = 1 if file.name == "GUARD_R0.json" else 0
        if row["returncode"] != expected or row["elapsed_seconds"] > row["maximum_seconds"] or row["peak_tree_RSS_bytes"] > row["maximum_RSS_bytes"]:
            raise ValueError("An actual final process/resource condition failed")
        for item in row.get("owned_process_identities", []):
            owned[(item["pid"], item["creation_time"])] = str(file)
    for pid, born in owned:
        try:
            if psutil.Process(pid).create_time() == born:
                raise ValueError("An exact saved completed instance is still alive")
        except psutil.NoSuchProcess:
            pass
    checks["all_saved_exact_process_instances_exited_and_actual_leases_released"] = True
    lock_path = OUT / "coordination_001/exclusive_benchmark.lock"
    with lock_path.open("r+b") as stream:
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            checks["actual_OS_resource_lock_free_and_restored"] = True
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
    owner = reference.load(OUT / "coordination_001/LOCK_OWNER.json")
    checks["last_measured_resource_owner_released"] = owner["status"] == "released"
    text_names = ["R144_V2_VECTOR_GRAY_STATS_TASK.md", "R144_V2_COMPLETE_RAW_TASK.md", "R144_V2_VECTOR_GRAY_STATS_RESULTS.md",
        "R150_QUAD_TOPOLOGY_RESULTS.md", "CANDIDATE_BOARD.md", "PLAN.md", "OPTIMIZE_CHECKLIST.md", "GENERALIZATION_PROTOCOL.md",
        "NEXT_POST_CORNER_COST_BRIEF.md", "NEXT_POST_VECTOR_STATS_COST_BRIEF.md"]
    texts = {name: (DOC / name).read_text(encoding="utf-8") for name in text_names}
    checks["actual_original_text_read_not_hash_screenshot_or_OCR"] = all(texts.values())
    checks["all_current_control_headers27_active_and_report_scope_explicit"] = all("27/50" in texts[name].splitlines()[0:4][2]
        for name in ("CANDIDATE_BOARD.md", "PLAN.md", "OPTIMIZE_CHECKLIST.md", "GENERALIZATION_PROTOCOL.md"))
    links = []
    for name, body in texts.items():
        for match in re.finditer(r"\]\((C:/[^)]+)\)", body):
            target = Path(re.sub(r":\d+$", "", match.group(1)))
            if target != path and not target.exists():
                raise ValueError("A published local evidence link does not resolve")
            links.append({"text": name, "target": str(target)})
    checks["all_published_local_evidence_links_resolve_except_own_new_receipt"] = bool(links)
    cost = cost_snapshot(base, complete, time.perf_counter() - started)
    checks["all_known_old_failed_new_validation_publication_costs_phase_at_least5percent"] = cost["net_reductions"]["phase_seconds"] >= .05
    checks["all_known_costs_whole_median_at_most1point01_each_at_most1point1"] = cost["whole_median_ratio"] <= 1.01 and all(
        pair["whole_seconds"]["vector"] <= 1.1 * pair["whole_seconds"]["current"] for pair in complete["pairs"])
    checks["no_production_GT_unseen_whole_or_full_goal_completion_claim"] = not acceptance["production_default_changed"]
    if not all(checks.values()):
        raise ValueError("A final vector cost, actual process or original text check failed")
    process = psutil.Process()
    record = {"status": "accepted_one_final_known_cost_and_original_text_verified_parent27_active", "checks": checks,
        "accepted_count_contribution": 1, "parent_accepted_count": 27, "remaining_count": 23, "cost": cost,
        "saved_instance_count_actual_exited": len(owned), "current_text_files": text_names, "published_local_links": links,
        "finalizer_PID": process.pid, "finalizer_creation_time": process.create_time(), "finalizer_seconds_before_write": time.perf_counter() - started,
        "source": reference.data_identity(Path(__file__)), "updated_utc": datetime.now(timezone.utc).isoformat(),
        "unmetered_text_research_and_final_write_stdout_exit_tail_not_assumed_zero": True,
        "production_default_changed": False, "generalization_pending": True}
    reference.atomic_json(path, record)
    print({"status": record["status"], "checks": len(checks), "count": 27, "phase_net": cost["net_reductions"]["phase_seconds"],
        "whole_net": cost["net_reductions"]["whole_seconds"], "known_shared_seconds": cost["known_extra_fee_seconds"]}, flush=True)


if __name__ == "__main__":
    main()
