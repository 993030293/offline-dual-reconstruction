"""Close the actual negative speed branch without rerunning its algorithm."""

import math
from pathlib import Path
import statistics
import time

STARTED = time.perf_counter()
from array_match_node import linux_identity_check, remaining
from array_match_node_config import OUT, read, atomic_json, data_identity


def main():
    result, generic = read(OUT/"c0/RESULT.json"), read(OUT/"g0/RESULT.json")
    checks = {"original_cost_failure": result["status"] == "failed_verification" and not result["checks"]["complete_phase_reduces10percent"],
        "all_three_full_pairs_scientific_outputs_exact": len(result["pairs"]) == 3
            and all(all(p["checks"].values()) for p in result["pairs"]),
        "all180_generic_and16_guards_passed": generic["status"] == "passed" and len(generic["rows"]) == 180
            and len(generic["checks"]) == 16 and all(generic["checks"].values()),
        "original_frozen_sources_unchanged": all(r == data_identity(r["path"]) for r in result["sources"])}
    initial = read(OUT/"GENERIC_LEASE_001.json")["child_elapsed_seconds"]
    fee = initial+sum(p["complete_compare_seconds"] for p in result["pairs"])
    checks["all_known_first_fees_once_recomputed"] = fee == result["complete_first_fees"]
    phase = {m: statistics.median(p["raw_CPU_phase_seconds"][m]+fee/6 for p in result["pairs"]) for m in ("incumbent", "array")}
    original_reduction = 1-phase["array"]/phase["incumbent"]
    checks["phase_gate_recomputed_without_relaxing"] = math.isclose(original_reduction,
        result["reductions"]["complete_phase_seconds"], rel_tol=0, abs_tol=1e-14) and original_reduction < .1
    identities, linux = [], []
    for trial in range(3):
        for label in ("b", "c"):
            node = read(OUT/f"c0/t{trial}/n{label}/NODE.json")
            side = read(OUT/f"c0/t{trial}/{label}/RESULT.json")
            bindings = read(OUT/f"c0/t{trial}/{label}/BINDING_RESTORATION.json")
            checks[f"t{trial}_{label}_normal_owned_close"] = (node["status"] == "completed" and not node["live_Windows_after_wait"]
                and all(not p["alive"] for p in node["actual_Linux_after_wait"]) and node["final_source_unchanged"])
            checks[f"t{trial}_{label}_actual24_48_full_map_and_binding"] = all(side["checks"].values()) and bindings["worker_restored"] and bindings["argv_restored"]
            identities += node["guard"].get("owned_process_identities", [])+[node["owner"]]
            for receipt in node["Linux_resource_receipts"]:
                linux.append(linux_identity_check(receipt))
    leases = [read(OUT/name) for name in ("GENERIC_LEASE_001.json", "CAMPAIGN_LEASE_001.json")]
    checks["both_complete_leases_released"] = all(r.get("lease_released_utc") and r["returncode"] == 0 for r in leases)
    for lease in leases:
        identities += [{"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]},
            {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}]
    unique = {(r["pid"], r["creation_time"]): r for r in identities}
    live = remaining(list(unique.values()))
    checks["all_saved_Windows_instances_gone_now"] = not live
    checks["all_six_native_Linux_instances_gone_now"] = len(linux) == 6 and all(not r["alive"] for r in linux)
    elapsed = time.perf_counter()-STARTED
    fee_closed = fee+elapsed+0.2368964
    closed_phase = {m: statistics.median(p["raw_CPU_phase_seconds"][m]+fee_closed/6 for p in result["pairs"])
        for m in ("incumbent", "array")}
    closure = {"status": "closed_negative_complete_speed" if all(checks.values()) else "failed_closure_check",
        "review_kind": "root_self_review_no_subagent", "checks": checks, "new_optimization_count": 0,
        "original_phase_reduction": original_reduction, "whole_reduction": result["reductions"]["complete_whole_seconds"],
        "mandatory_final_review_seconds": elapsed, "first_review_syntax_failure_tool_wall_seconds": 0.2368964, "all_first_fees_with_final_close_seconds": fee_closed,
        "closed_phase_seconds": closed_phase, "closed_phase_reduction": 1-closed_phase["array"]/closed_phase["incumbent"],
        "saved_Windows_instances_checked": len(unique), "current_alive_Windows": live, "current_Linux_instances": linux,
        "sources_preserved": True, "external_recovery_new_training_preview": "not_run_after_required_cost_failure",
        "full_scientific_outputs_changed": False, "complete_semantic_acceptance": False}
    atomic_json(OUT/"NEGATIVE_CLOSURE_001.json", closure)
    print({k: closure[k] for k in ("status", "checks", "original_phase_reduction", "closed_phase_reduction", "saved_Windows_instances_checked")}, flush=True)
    if not all(checks.values()):
        raise AssertionError("The preserved negative branch did not close")


if __name__ == "__main__":
    main()
