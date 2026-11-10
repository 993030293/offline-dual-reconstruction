"""Close an early cost-negative remap family without claiming formal speed."""

import json
from pathlib import Path
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    started = time.perf_counter()
    out = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/rm001"
    result_path = out/"NEGATIVE_CLOSURE_001.json"
    if result_path.exists():
        raise FileExistsError("The root self-review must have an unused output")
    first = [read(out/name) for name in ("PROBE_LEASE_001.json", "PROBE_LEASE_002.json", "PIXEL_LEASE_001.json", "PIXEL_LEASE_002.json")]
    guards = [read(out/name) for name in ("p0_guard.json", "p1_guard.json", "g0_guard.json", "g1_guard.json")]
    results = [read(out/name/"RESULT.json") for name in ("g0", "g1")]
    checks, instances = {}, {}
    for version, result in enumerate(results):
        checks[f"v{version}_all43_complete_pixels_cameras_PNG"] = len(result["rows"]) == 43 and all(result["checks"].values())
        checks[f"v{version}_sources_unchanged"] = all(record == data_identity(Path(record["path"])) for record in result["sources"])
        checks[f"v{version}_both_reused_maps_slower_in_diagnostic"] = all(r["candidate_seconds"] > r["original_seconds"] for r in result["rows"][1:3])
        checks[f"v{version}_maps_closed"] = result["trace"]["closed"] and result["trace"]["map_bytes"] == 0
    for i, (lease, guard) in enumerate(zip(first, guards)):
        checks[f"receipt{i}_released_and_child_done"] = bool(lease.get("lease_released_utc")) and lease["returncode"] == guard["returncode"]
        for row in [dict(pid=lease["guard_pid"], creation_time=lease["guard_creation_time"]),
            dict(pid=lease["child_pid"], creation_time=lease["child_creation_time"]), *guard.get("owned_process_identities", [])]:
            instances[row["pid"], row["creation_time"]] = row
    live = []
    for (pid, creation_time), row in instances.items():
        try:
            process = psutil.Process(pid)
            if process.create_time() == creation_time and process.status() != psutil.STATUS_ZOMBIE:
                live.append(row)
        except psutil.NoSuchProcess:
            pass
    checks["all_actual_Windows_instances_gone"] = not live
    checks["first_failed_serialization_preserved"] = guards[0]["returncode"] == 1 and "CameraModelId is not JSON serializable" in (out/"p0.log").read_text(encoding="utf-8")
    checks["no_formal_campaign_or_new_training"] = not (out/"c0").exists()
    if not all(checks.values()):
        raise AssertionError(json.dumps(checks))
    atomic_json(result_path, {"status": "closed_negative_diagnostic_speed", "review_kind": "root_self_review_no_subagent",
        "checks": checks, "new_optimization_count": 0, "saved_Windows_instances_checked": len(instances), "current_live_Windows": live,
        "all_four_first_child_seconds_once": sum(row["child_elapsed_seconds"] for row in first),
        "whole_root_review_seconds": time.perf_counter()-started,
        "reason": "Both exact pixel implementations are slower even on reused-map calls; no formal three-pair claim",
        "not_run": ["remaining_failure_and_unsupported_guards", "formal_full24_48_cost_pairs", "external_objects", "new_training", "new_preview"]})
    print(json.dumps({"status": "closed_negative_diagnostic_speed", "checks": len(checks), "counted": False}))


if __name__ == "__main__":
    main()
