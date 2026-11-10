"""Review complete fresh viewer work comparisons before cost publication."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import psutil

from review_frustum_negative import ROOT, load, save, exited

OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/behind_camera_draw_001"
JOBS = ["g0", "t0", "t1", "p0", "c0", "r0"]


def signature(path):
    content = path.read_bytes()
    return len(content), hashlib.sha256(content).hexdigest()


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output.resolve()
    output.relative_to(OWN)
    output.mkdir(parents=True, exist_ok=False)
    process = psutil.Process()
    save(output / "PROCESS_IDENTITY.json", {"pid": process.pid, "creation_time": process.create_time()})
    data = {name: load(OWN / ("v3_" + name) / "RESULT.json") for name in JOBS}
    checks = {}
    checks["fresh51_generic_and_original_buttons_passed"] = data["g0"]["status"] == "generic_passed" and len(data["g0"]["generic"]["checks"]) == 51 and all(c["passed"] for c in data["g0"]["generic"]["checks"])
    rows = data["t1"]["rows"]
    samples = [r for mode in ["native", "candidate"] for step in data["t1"][mode]["steps"] for r in step["samples"]]
    checks["all18_actual_transition_comparisons_and36_canvas_reads_valid"] = data["t1"]["status"] == "actual18_transition_frames_exact" and len(rows) == 18 and len(samples) == 36 and all(r["camera_population_exact"] and r["rgba_bytes_exact"] for r in rows) and all(r["displayedFramebufferValidated"] for r in samples)
    checks["actual_original_PBO_fault_reproduced_and_not_called_quality_failure"] = sum(r["originalTypedReadError"] == 1282 for r in samples) == 3 and all(r["originalTypedReadError"] == (1282 if r["originalPackBufferBound"] else None) for r in samples) and data["t0"]["newOptimizationCount"] == 0
    checks["all_back_frames_really_skipped_renderer_and_front_frames_restored"] = all(r["actual_candidate_reject"] and r["candidate_active"] == 0 if r["sign"] == 1 else r["candidate_visible"] and r["candidate_active"] == r["native_active"] for r in rows)
    camera_keys = ["name", "width", "height", "rgba_sha256", "projection", "position", "direction", "count", "mode", "world", "fullProjection", "quaternion"]
    full_pairs = []
    for name in ["p0", "c0", "r0"]:
        result = data[name]
        pair_rows = []
        for pair in result["pairs"]:
            trial = pair["trial"]
            episodes = [e for e in result["episodes"] if e["trial"] == trial]
            a = next(e for e in episodes if e["enabled"] is False)
            b = next(e for e in episodes if e["enabled"] is True)
            protected = len(a["captures"]) == len(b["captures"]) == 124 and all(all(x[k] == y[k] for k in camera_keys) for x, y in zip(a["captures"], b["captures"]))
            actual = 1 - b["final"]["draw"]["triangles"] / a["final"]["draw"]["triangles"]
            pair_rows.append({"trial": trial, "all124_protected": protected, "reduction_exact": actual == pair["reduction"],
                "native_seconds": a["elapsedSeconds"], "candidate_seconds": b["elapsedSeconds"], "actual_work_reduction": actual})
        full_pairs.append({"node": name, "pairs": pair_rows})
    checks["all_screen_formal_and_recovery124_denominators_rechecked"] = all(p["all124_protected"] and p["reduction_exact"] for row in full_pairs for p in row["pairs"])
    formal = data["c0"]
    checks["three_new_formal_interleaved_pairs_work10_and_whole_guards_passed"] = formal["status"] == "measured_protection_passed_acceptance_pending" and len(formal["pairs"]) == 3 and [e["enabled"] for e in formal["episodes"]] == [False, True, True, False, False, True] and all(p["reduction"] >= .1 and p["candidateSeconds"] <= p["nativeSeconds"] and p["candidateSeconds"] <= 1.1 * p["nativeSeconds"] for p in formal["pairs"])
    injected = load(OWN / "v3_r0/EPISODE_FAILURE_injected_true.json")
    checks["actual_consumer_error_and_new_context_full_recovery_passed"] = data["r0"]["status"] == "actual_new_context_recovery_passed" and "registered_after_actual_back_view_injection" in injected["error"] and len(injected["captures"]) == 28 and all(c["passed"] for c in data["r0"]["checks"])
    snapshots, inputs, instances, leases = [], [], [], []
    current = {}
    for name, result in data.items():
        for row in result["sources"]:
            source = OWN / ("v3_" + name) / "sources" / Path(row["path"]).name
            snapshots.append({"node": name, "path": str(source), "executed_source_preserved": signature(source) == (row["bytes"], row["sha256"])})
        for row in result["inputIdentities"]:
            path = row["path"]
            if path not in current:
                current[path] = signature(Path(path))
            inputs.append({"node": name, "path": path, "current_original_unchanged": current[path] == (row["bytes"], row["sha256"])})
        lease = load(OWN / ("V3_LEASE_" + name.upper() + ".json"))
        guard = load(OWN / ("V3_" + name.upper() + "_PROCESS.json"))
        leases.append(lease)
        for entry in guard["owned_process_identities"] + [{"pid": lease[k + "_pid"], "creation_time": lease[k + "_creation_time"]} for k in ["guard", "child"]]:
            instances.append({"node": name, **entry, "actual_owned_instance_exited": exited(entry["pid"], entry["creation_time"])})
    checks["all_executed_source_snapshots_and_original_assets_unchanged"] = all(r["executed_source_preserved"] for r in snapshots) and all(r["current_original_unchanged"] for r in inputs)
    checks["all_six_actual_measured_job_processes_and_leases_closed"] = all(r["actual_owned_instance_exited"] for r in instances) and all(r["status"] == "child_exited" and r["returncode"] == 0 and r.get("lease_released_utc") for r in leases)
    checks["all_measured_process_tree_RSS_and_time_budgets_preserved"] = all(load(OWN / ("V3_" + name.upper() + "_PROCESS.json"))["peak_tree_RSS_bytes"] <= 8 * 1024 ** 3 and load(OWN / ("V3_" + name.upper() + "_PROCESS.json"))["elapsed_seconds"] <= 900 for name in JOBS)
    old = load(OWN / "NEGATIVE_CLOSURE_001.json")
    shader = load(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/r145_001/FINAL_NEGATIVE_PUBLICATION_002.json")
    formal_span = next(x for x in leases if x["owner"] == "root_R138_V3_formal_001")["child_elapsed_seconds"]
    formal_episodes = sum(e["elapsedSeconds"] for e in formal["episodes"])
    checks["formal_overhead_excludes_already_counted_episode_spans"] = formal_span >= formal_episodes
    first_fees = old["all_known_actual_nonoverlapping_fees_seconds"] + shader["all11_measured_lease_child_seconds"] + .3592663 + .3928535
    v3_fees = sum(x["child_elapsed_seconds"] for x in leases if x["owner"] != "root_R138_V3_formal_001") + formal_span - formal_episodes
    state = load(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    accepted = []
    for path in state["acceptance_receipts"]:
        row = load(ROOT / path)
        values = [row[k] for k in ["new_optimization_count", "additional_accepted_count", "accepted_count_contribution"] if k in row]
        accepted.append(bool(values) and all(x == 1 for x in values))
    checks["all22_existing_acceptances_preserved_and_family_not_counted"] = state["additional_accepted_count"] == len(state["accepted_mechanisms"]) == len(set(state["acceptance_receipts"])) == 22 and all(accepted) and "R138_V3" not in state["accepted_mechanisms"]
    checks["all_nonzero_quality_and_identity_checks_passed"] = all(all(c["passed"] for c in data[n]["checks"]) for n in ["g0", "t1", "p0", "c0", "r0"])
    result = {"status": "root_self_review_passed_complete_cost_pending" if all(checks.values()) else "failed_verification",
        "review_type": "separate_root_self_review_no_independent_agent", "checks": checks, "full_protected_pairs": full_pairs,
        "source_snapshots": snapshots, "original_inputs": inputs, "actual_Windows_instances": instances,
        "old_R138_R145_and_known_metadata_fees_seconds": first_fees,
        "new_V3_required_fees_excluding_formal_episodes_and_current_review": v3_fees,
        "formal_episode_spans_seconds": formal_episodes, "formal_overhead_seconds": formal_span - formal_episodes,
        "own_review_lease_cost_and_exit_must_be_added_after_exit": True,
        "accepted_count_contribution": 0, "parent_count": 22,
        "not_run": ["external_whole_object_chain", "GT", "new_training", "production_deployment"],
        "review_seconds_before_publication": time.perf_counter() - started}
    save(output / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "checks": len(checks), "snapshots": len(snapshots), "inputs": len(inputs),
        "instances": len(instances), "known_required_fees": first_fees + v3_fees}), flush=True)
    if not all(checks.values()):
        raise AssertionError(json.dumps({k: v for k, v in checks.items() if not v}))


if __name__ == "__main__":
    main()
