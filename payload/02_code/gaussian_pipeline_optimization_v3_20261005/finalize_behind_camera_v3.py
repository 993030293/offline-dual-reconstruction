"""Publish one scoped work optimization after all measured costs and exits."""

from datetime import datetime, timezone
import json
import msvcrt
from pathlib import Path
import shutil
import statistics
import time

import psutil

from review_frustum_negative import ROOT, load, save, exited

OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/behind_camera_draw_001"


def main():
    started = time.perf_counter()
    process = psutil.Process()
    publisher = {"pid": process.pid, "creation_time": process.create_time()}
    review = load(OWN / "v3_a0/RESULT.json")
    checks = {"all_root_review_checks_passed": review["status"] == "root_self_review_passed_complete_cost_pending" and all(review["checks"].values())}
    identities, leases = [], []
    for name in ["g0", "t0", "t1", "p0", "c0", "r0", "a0"]:
        lease = load(OWN / ("V3_LEASE_" + name.upper() + ".json"))
        guard = load(OWN / ("V3_" + name.upper() + "_PROCESS.json"))
        leases.append(lease)
        for entry in guard["owned_process_identities"] + [{"pid": lease[k + "_pid"], "creation_time": lease[k + "_creation_time"]} for k in ["guard", "child"]]:
            identities.append({"node": name, **entry, "actual_owned_instance_exited": exited(entry["pid"], entry["creation_time"])})
    checks["all7_actual_V3_job_instances_and_leases_closed"] = all(x["actual_owned_instance_exited"] for x in identities) and all(x["status"] == "child_exited" and x["returncode"] == 0 and x.get("lease_released_utc") for x in leases)
    old_shader = load(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/r145_001/FINAL_NEGATIVE_PUBLICATION_002.json")
    checks["all87_old_shader_family_process_records_remain_exited"] = all(exited(x["pid"], x["creation_time"]) for x in old_shader["actual_Windows_instances"])
    coordination = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/coordination_001"
    owner = load(coordination / "LOCK_OWNER.json")
    checks["last_resource_lease_actual_released"] = owner["status"] == "released" and owner["owner"] == "root_R138_V3_review_001" and exited(owner["guard_pid"], owner["guard_creation_time"])
    with (coordination / "exclusive_benchmark.lock").open("r+b", buffering=0) as handle:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            checks["actual_OS_benchmark_lock_available"] = True
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    checks["private_family_storage_and_disk_budgets_preserved"] = sum(p.stat().st_size for p in OWN.rglob("*") if p.is_file()) < 30 * 1024 ** 3 and shutil.disk_usage(ROOT).free >= 20 * 1024 ** 3
    formal = load(OWN / "v3_c0/RESULT.json")
    review_cost = next(x for x in leases if x["owner"] == "root_R138_V3_review_001")["child_elapsed_seconds"]
    required = review["old_R138_R145_and_known_metadata_fees_seconds"] + review["new_V3_required_fees_excluding_formal_episodes_and_current_review"] + review_cost
    common = required / 6
    pairs = [{**p, "complete_native_seconds": p["nativeSeconds"] + common,
        "complete_candidate_seconds": p["candidateSeconds"] + common} for p in formal["pairs"]]
    checks["fresh_three_pairs_work10_and_complete_whole_median_and_each_guards_passed"] = len(pairs) == 3 and all(p["reduction"] >= .1 and p["complete_candidate_seconds"] <= 1.1 * p["complete_native_seconds"] for p in pairs) and statistics.median(p["complete_candidate_seconds"] for p in pairs) <= statistics.median(p["complete_native_seconds"] for p in pairs)
    state = load(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    checks["same_visibility_family_not_previously_counted"] = state["additional_accepted_count"] == 22 and "R138_V3" not in state["accepted_mechanisms"] and state["R145_new_optimization_count"] == 0 and not (OWN / "V3_ACCEPTANCE_001.json").exists()
    result = {"status": "accepted_scoped_engineering_work_optimization" if all(checks.values()) else "failed_verification",
        "created_utc": datetime.now(timezone.utc).isoformat(), "mechanism": "R138_V3",
        "scope": "unchanged_static_packed_identity_transform_models_supported_camera_complete_fixed_native_viewer_route",
        "main_metric": "actual_WebGL_triangles_over_complete_124_position_route", "work_gate": .1,
        "user_authorized_gate_revision": "20_percent_to10_percent_new_preregistered_fresh_V3_not_old_receipt_relabel",
        "work_reduction_median": statistics.median(p["reduction"] for p in pairs), "pairs": pairs,
        "raw_whole_medians_seconds": {"native": statistics.median(p["nativeSeconds"] for p in pairs), "candidate": statistics.median(p["candidateSeconds"] for p in pairs)},
        "complete_known_cost_whole_medians_seconds": {"native": statistics.median(p["complete_native_seconds"] for p in pairs), "candidate": statistics.median(p["complete_candidate_seconds"] for p in pairs)},
        "all_known_required_measured_fees_seconds_before_metadata_write": required,
        "formal_episode_spans_not_counted_twice": True,
        "review": str(OWN / "v3_a0/RESULT.json"), "generic51": str(OWN / "v3_g0/RESULT.json"),
        "validated_transient18": str(OWN / "v3_t1/RESULT.json"), "actual_recovery": str(OWN / "v3_r0/RESULT.json"),
        "checks": checks, "actual_V3_Windows_instances": identities,
        "unique_V3_instances": len(set((x["pid"], x["creation_time"]) for x in identities)),
        "all7_V3_measured_lease_child_seconds": sum(x["child_elapsed_seconds"] for x in leases),
        "publisher_identity": publisher, "publisher_actual_exit_and_metadata_cost_require_external_post_check": True,
        "metadata_seconds_before_write": time.perf_counter() - started,
        "new_optimization_count": 1 if all(checks.values()) else 0,
        "one_visibility_family_count_including_R138_and_R145": True,
        "not_claimed": ["FPS", "GPU_time", "energy", "blind_unseen_object_chain", "GT_accuracy", "new_training", "production_deployment"],
        "not_metered_as_runtime_cost": ["research_text_authoring", "some_readonly_intake_and_tool_queries"]}
    save(OWN / "V3_ACCEPTANCE_001.json", result)
    print(json.dumps({"status": result["status"], "checks": len(checks), "instances": result["unique_V3_instances"],
        "work_median": result["work_reduction_median"], "raw_medians": result["raw_whole_medians_seconds"],
        "complete_medians": result["complete_known_cost_whole_medians_seconds"], "known_fees": required}), flush=True)
    if not all(checks.values()):
        raise AssertionError(json.dumps({k: v for k, v in checks.items() if not v}))


if __name__ == "__main__":
    main()
