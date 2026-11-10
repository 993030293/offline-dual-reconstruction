"""Close the whole-cost failure while preserving the measured monitor benefit."""

import math
from pathlib import Path
import statistics
import time

STARTED = time.perf_counter()
from nvml_owned_node import linux_identity_check, remaining
from nvml_node_config import OUT, read, atomic_json, data_identity


def main():
    result, generic = read(OUT/"c1/RESULT.json"), read(OUT/"g1/RESULT.json")
    checks = {"original_whole_median_cost_failure": result["status"] == "failed_verification"
            and not result["checks"]["complete_whole_median_not_regressed"],
        "monitor_component_gain_preserved": result["checks"]["complete_phase_reduces10percent"],
        "all3_full24_48_scientific_pairs_exact": len(result["pairs"]) == 3 and all(all(p["checks"].values()) for p in result["pairs"]),
        "all4_stable_actual_GPU_buffers_and7_guards": generic["status"] == "passed" and len(generic["rows"]) == 4
            and len(generic["checks"]) == 7 and all(generic["checks"].values()),
        "all_frozen_campaign_sources_unchanged": all(row == data_identity(row["path"]) for row in result["sources"])}
    fee_names = ("PROBE_LEASE_001.json", "PROBE_LEASE_002.json", "GENERIC_LEASE_001.json", "GENERIC_LEASE_002.json", "CAMPAIGN_LEASE_001.json")
    fee = sum(read(OUT/name)["child_elapsed_seconds"] for name in fee_names)+sum(p["complete_compare_seconds"] for p in result["pairs"])
    checks["all_known_first_fees_once"] = fee == result["complete_first_fees"]
    whole = {m: statistics.median(p["raw_whole_seconds"][m]+fee/6 for p in result["pairs"]) for m in ("cli", "nvml")}
    whole_reduction = 1-whole["nvml"]/whole["cli"]
    checks["original_median_rule_recomputed_not_switched_to_paired_median"] = math.isclose(whole_reduction,
        result["reductions"]["complete_whole_seconds"], rel_tol=0, abs_tol=1e-14) and whole_reduction < 0
    identities, linux = [], []
    for trial in range(3):
        for label in ("b", "c"):
            node = read(OUT/f"c1/t{trial}/n{label}/NODE.json")
            side = read(OUT/f"c1/t{trial}/{label}/RESULT.json")
            bindings = read(OUT/f"c1/t{trial}/{label}/BINDING_RESTORATION.json")
            checks[f"t{trial}_{label}_actual_guards_and_close"] = (node["status"] == "completed"
                and not node["live_Windows_after_wait"] and all(not r["alive"] for r in node["actual_Linux_after_wait"])
                and node["monitoring"]["source_unchanged"] and node["monitoring"]["library_closed"]
                and all(side["checks"].values()) and bindings["worker_restored"] and bindings["argv_restored"])
            identities += node["guard"].get("owned_process_identities", [])+[node["owner"]]
            linux += [linux_identity_check(r) for r in node["Linux_resource_receipts"]]
    for folder in ("g0", "g1"):
        linux.append(linux_identity_check(read(OUT/folder/"WSL_RESOURCE.json")))
    leases = []
    for name in (*fee_names, "CAMPAIGN_LEASE_002.json"):
        value = read(OUT/name)
        checks[name+"_released"] = bool(value.get("lease_released_utc"))
        identities += [{"pid": value["guard_pid"], "creation_time": value["guard_creation_time"]},
            {"pid": value["child_pid"], "creation_time": value["child_creation_time"]}]
        leases.append({"path": name, "exit": value["returncode"], "whole_child_seconds": value["child_elapsed_seconds"]})
    for name in ("GENERIC_PROCESS_001.json", "GENERIC_PROCESS_002.json"):
        identities += read(OUT/name).get("owned_process_identities", [])
    unique = {(r["pid"], r["creation_time"]): r for r in identities}
    live = remaining(list(unique.values()))
    checks["all_saved_Windows_instances_gone_now"] = not live
    checks["all8_actual_Linux_instances_gone_now"] = len(linux) == 8 and all(not row["alive"] for row in linux)
    elapsed = time.perf_counter()-STARTED
    final_fee = fee+elapsed
    phase = {m: statistics.median(p["raw_CPU_phase_seconds"][m]+final_fee/6 for p in result["pairs"]) for m in ("cli", "nvml")}
    whole_closed = {m: statistics.median(p["raw_whole_seconds"][m]+final_fee/6 for p in result["pairs"]) for m in ("cli", "nvml")}
    closure = {"status": "closed_negative_whole_cost" if all(checks.values()) else "failed_closure_check",
        "review_kind": "root_self_review_no_subagent", "checks": checks, "new_optimization_count": 0,
        "original_full_monitor_reduction": result["reductions"]["complete_phase_seconds"], "original_whole_reduction": whole_reduction,
        "raw_whole_median_regression_seconds": whole["nvml"]-whole["cli"],
        "final_review_seconds": elapsed, "complete_first_fees_with_final_review": final_fee,
        "closed_phase_seconds": phase, "closed_phase_reduction": 1-phase["nvml"]/phase["cli"],
        "closed_whole_seconds": whole_closed, "closed_whole_reduction": 1-whole_closed["nvml"]/whole_closed["cli"],
        "saved_Windows_instances_checked": len(unique), "current_live_Windows": live, "current_Linux": linux,
        "first_receipts_once": leases, "full_scientific_outputs_changed": False,
        "why_whole_changed": "unconfirmed_short_difference_not_used_to_relax_gate_or_claim_model_regression",
        "later_new_training_external_object_preview": "not_run_after_required_whole_cost_failure",
        "complete_semantic_acceptance": False}
    atomic_json(OUT/"NEGATIVE_CLOSURE_001.json", closure)
    print({k: closure[k] for k in ("status", "checks", "closed_phase_reduction", "closed_whole_reduction", "saved_Windows_instances_checked")}, flush=True)
    if not all(checks.values()):
        raise AssertionError("A complete negative resource branch did not close")


if __name__ == "__main__":
    main()
