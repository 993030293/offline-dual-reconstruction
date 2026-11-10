"""Preserve actual integration failures and verify owned resource closure."""

from pathlib import Path
import sys

CODE = Path(__file__).parent
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from fused_match_node_002 import linux_identity_check, remaining
from fused_match_node_config import read, OUT


def main():
    generic = read(OUT/"g0/RESULT.json")
    checks = {"actual9_GPU_oracle_and8_guards_passed": generic["status"] == "passed" and len(generic["rows"]) == 9
        and len(generic["checks"]) == 8 and all(generic["checks"].values()),
        "mathematical_kernel_unchanged_after_oracle": generic["sources"][1]["sha256"] == data_identity(CODE/"fused_integer_reduction.py")["sha256"],
        "no_false_completed_formal_pairs": not read(OUT/"c0/STATE.json")["pairs"] and not (OUT/"c0/RESULT.json").exists(),
        "diagnostic_not_full24_accepted": not (OUT/"d0/RESULT.json").exists()}
    identities, linux, nodes = [], {}, []
    expected = (("ng0", "completed", 0), ("c0/t0/nb", "completed", 0), ("c0/t0/nc", "failed", 1), ("nd0", "failed", 1))
    for label, status, exitcode in expected:
        node = read(OUT/label/"NODE.json")
        checks[label+"_actual_expected_status"] = node["status"] == status and node["returncode"] == exitcode
        checks[label+"_normal_owned_drain"] = not node["live_Windows_after_wait"] and all(not p["alive"] for p in node["actual_Linux_after_wait"])
        identities += node["guard"].get("owned_process_identities", [])+[node["owner"]]
        for row in node["Linux_resource_receipts"]:
            linux[(row["pid"], row["creation_time"])] = row
        nodes.append({"path": label, "status": node["status"], "returncode": node["returncode"],
            "complete_node_seconds": node["complete_node_seconds"], "resources": node["resource_peaks"]})
    leases = []
    for name in ("GENERIC_LEASE_001.json", "CAMPAIGN_LEASE_001.json", "DIAGNOSTIC_LEASE_002.json"):
        lease = read(OUT/name)
        checks[name+"_released"] = bool(lease.get("lease_released_utc"))
        identities += [{"pid": lease["guard_pid"], "creation_time": lease["guard_creation_time"]},
            {"pid": lease["child_pid"], "creation_time": lease["child_creation_time"]}]
        leases.append({"path": name, "returncode": lease["returncode"], "complete_child_seconds": lease["child_elapsed_seconds"]})
    unique = {(r["pid"], r["creation_time"]): r for r in identities}
    live = remaining(list(unique.values()))
    checks["all_saved_Windows_instances_gone_now"] = not live
    linux_now = [linux_identity_check(row) for row in linux.values()]
    checks["all_saved_native_Linux_instances_gone_now"] = all(not row["alive"] for row in linux_now)
    first = (OUT/"c0/t0/nc/child.log").read_text(encoding="utf-8")
    second = (OUT/"nd0/child.log").read_text(encoding="utf-8")
    checks["original_startup_EOF_preserved"] = "EOFError" in first
    checks["original_atomic_PermissionError_preserved"] = "PermissionError: [WinError 5]" in second and "PARTIAL_RESULTS.json" in second
    result = {"status": "closed_unaccepted_integration_failures" if all(checks.values()) else "failed_closure_check",
        "review_kind": "root_self_review_no_subagent", "checks": checks, "nodes": nodes, "first_cost_receipts_once": leases,
        "first_saved_child_seconds": sum(r["complete_child_seconds"] for r in leases),
        "recorded_Windows_instances_checked": len(unique), "live_Windows_now": live, "actual_Linux_now": linux_now,
        "kernel_contract": "actual_GPU_passed", "full24_scientific_equivalence": "unverified_after_partial_atomic_write_failure",
        "formal_speedup": "not_established", "atomic_write_root_cause": "unconfirmed",
        "new_optimization_count": 0, "complete_semantic_acceptance": False,
        "later_formal_pairs_external_new_training_preview": "not_run_after_integration_failures"}
    atomic_json(OUT/"UNACCEPTED_CLOSURE_001.json", result)
    print({k: result[k] for k in ("status", "checks", "first_saved_child_seconds", "recorded_Windows_instances_checked")}, flush=True)
    if not all(checks.values()):
        raise AssertionError("An unaccepted branch did not close")


if __name__ == "__main__":
    main()
