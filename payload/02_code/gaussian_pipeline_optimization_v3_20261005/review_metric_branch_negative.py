"""Close a saved R119 quality failure without rerunning experiments."""

import json
from pathlib import Path
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/metric_branch_dag_001"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    started = time.perf_counter()
    destination = OUT/"NEGATIVE_CLOSURE_001.json"
    if destination.exists():
        raise FileExistsError(destination)
    result = read(OUT/"q0/RESULT.json")
    deltas = result["per_view_deltas"]
    names = {row["image_name"] for row in deltas}
    failed = [row for row in deltas if row["PSNR_delta_dB"] < -.1 or row["SSIM_delta"] < -.002]
    checks = {"actual_quality_failure": result["status"] == "required_own_full24_quality_failed",
        "complete24_full_target": len(deltas) == 48 and len(names) == 24 and all(
            {row["region"] for row in deltas if row["image_name"] == name} == {"full_image", "target"} for name in names),
        "original_thresholds_recomputed": len(failed) == 17 and all(row["passed"] == (
            row["PSNR_delta_dB"] >= -.1 and row["SSIM_delta"] >= -.002) for row in deltas),
        "mean_guard_not_used_to_hide_per_view_failure": result["checks"]["original_mean_guard"]
            and not result["checks"]["all_original_per_view_quality"],
        "both_actual_full12000_trainings": all(read(OUT/"t0"/mode/"RESULT.json")["checks"]["full12000_completed"]
            and read(OUT/"t0"/mode/"training/STATE.json")["iteration"] == 12000 for mode in ("serial", "dag")),
        "both_actual_complete_localizations": all(all(read(OUT/folder/"RESULT.json")["checks"].values()) for folder in ("h0", "h1")),
        "later_seeds_not_run": not (OUT/"t1").exists() and not (OUT/"t2").exists(),
        "formal_cost_not_run": not (OUT/"campaign_001").exists()}
    identities = {}
    node_closes = []
    for name in ("n0s", "n0d", "nq0"):
        node = read(OUT/name/"NODE.json")
        checks[name+"_owned_close"] = not node["live_Windows_after_wait"] and not any(
            row["alive"] for row in node["actual_Linux_after_wait"]) and node["final_source_unchanged"]
        checks[name+"_expected_exit"] = node["returncode"] == (1 if name == "nq0" else 0)
        for row in node["guard"].get("owned_process_identities", []):
            identities[(row["pid"], row["creation_time"])] = row
        node_closes.append({"node": name, "complete_node_seconds": node["complete_node_seconds"],
            "actual_Linux_after_wait": node["actual_Linux_after_wait"]})
    fees = []
    for path in sorted(OUT.glob("*.json")):
        value = read(path)
        if not isinstance(value, dict) or "guard_pid" not in value or "child_elapsed_seconds" not in value:
            continue
        checks[path.stem+"_lease_released"] = value.get("lease_released_utc") is not None
        identities[(value["guard_pid"], value["guard_creation_time"])] = {
            "pid": value["guard_pid"], "creation_time": value["guard_creation_time"]}
        identities[(value["child_pid"], value["child_creation_time"])] = {
            "pid": value["child_pid"], "creation_time": value["child_creation_time"]}
        fees.append({"receipt": str(path), "complete_child_seconds": value["child_elapsed_seconds"],
                     "returncode": value["returncode"], "lock_wait_seconds_separate": value["wait_seconds"]})
    alive = []
    for row in identities.values():
        try:
            if psutil.Process(row["pid"]).create_time() == row["creation_time"]:
                alive.append(row)
        except psutil.NoSuchProcess:
            pass
    checks["all_recorded_Windows_instances_gone"] = not alive
    receipt = {"status": "closed_negative_quality" if all(checks.values()) else "negative_closure_review_failed",
        "review_kind": "root_self_review_no_subagent", "checks": checks,
        "failed_full_target_rows": failed, "failed_count": len(failed),
        "minimum_PSNR_delta_dB": min(row["PSNR_delta_dB"] for row in deltas),
        "minimum_SSIM_delta": min(row["SSIM_delta"] for row in deltas),
        "formal_speedup_claim": False, "new_optimization_count": 0,
        "first_node_cost_receipts_once_each": fees, "total_saved_child_seconds": sum(row["complete_child_seconds"] for row in fees),
        "recorded_Windows_instances_checked": len(identities), "actual_alive_recorded_Windows": alive,
        "node_close_receipts": node_closes, "later_quality_cost_recovery_external_preview": "not_run_after_required_quality_failure",
        "complete_semantic_acceptance": False, "elapsed_seconds": time.perf_counter()-started}
    destination.write_text(json.dumps(receipt, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "checks": len(checks), "failed_checks": [k for k, v in checks.items() if not v],
                      "failed_quality_rows": len(failed), "new_optimization_count": 0}))
    if not all(checks.values()):
        raise AssertionError("Negative scope self-review failed")


if __name__ == "__main__":
    main()
