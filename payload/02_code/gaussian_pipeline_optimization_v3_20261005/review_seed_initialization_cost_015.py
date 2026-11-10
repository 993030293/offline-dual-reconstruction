"""Close the unchanged seed diagnostic against the existing full reference."""

import json
from pathlib import Path
import shutil
import subprocess

import psutil

from run_cpu_dependency_restore import ROOT, compare, read


def main():
    base = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup015"
    p = base / "p0"
    ref = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup014/R168/r1"
    result, trace, guard, lease = [read(path) for path in (
        p / "RESULT.json", p / "SEED_INITIALIZATION_TRACE.json", base / "p0_GUARD.json", base / "p0_LEASE.json")]
    fields = trace["calls"][0]["fields"]
    code = '''import json,psutil
from pathlib import Path
p=Path("/mnt/c/lidar-camera-fusion-mask-robust-downsampling/04_outputs/gaussian_pipeline_optimization_v3_20261005/oup015/p0")
rows=[json.loads((p/n).read_text()) for n in ("PROCESS_IDENTITY.json","WSL_RESOURCE.json","CPU_BOOTSTRAP_RESOURCE.json")]
out=[]
for r in rows:
 try:
  q=psutil.Process(r["pid"]);ended=q.create_time()!=r["creation_time"] or not q.is_running()
 except psutil.NoSuchProcess:
  ended=True
 out.append({"pid":r["pid"],"creation_time":r["creation_time"],"saved_instance_ended":ended})
print(json.dumps(out))
assert all(r["saved_instance_ended"] for r in out)
'''
    status = subprocess.run(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--", "/opt/e26_thin_surface_v1/venv/bin/python",
        "-B", "-c", code], check=True, timeout=15, text=True, encoding="utf-8", stdout=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW)
    native = json.loads(status.stdout)
    births = [(lease[key + "_pid"], lease[key + "_creation_time"]) for key in ("guard", "child")]
    births += [(row["pid"], row["creation_time"]) for row in guard["owned_process_identities"]]
    closed = []
    for pid, born in set(births):
        try:
            q = psutil.Process(pid)
            ended = q.create_time() != born or not q.is_running()
        except psutil.NoSuchProcess:
            ended = True
        closed.append({"pid": pid, "creation_time": born, "saved_instance_ended": ended})
    compare_checks = compare(ref, p)
    checks = {
        "all13_original_complete_restore_and_observer_checks": len(result["checks"]) == 13 and all(result["checks"].values()),
        "all6_complete_reference_results_pixels_geometry_and_files_same": len(compare_checks) == 6 and all(compare_checks.values()),
        "all_original1038_inputs_same_as_frozen_reference": read(p / "CONTRACT.json")["inputs"] == read(ref / "CONTRACT.json")["inputs"],
        "observer_exactly_one_original_call_and_binding_restored": len(trace["calls"]) == 1 and trace["original_create_from_pcd_binding_restored"],
        "actual_seed75767_and_full_cameras1016_kept": trace["calls"][0]["seed_points"] == 75767 and trace["calls"][0]["cam_infos"] == 1016,
        "actual8_fields_and_logical18232848_bytes": len(fields) == 8 and sum(r["logical_bytes"] for r in fields.values()) == 18232848,
        "all_seed_fields_float32_cuda0_and_expected_shapes": all(row["dtype"] == "torch.float32" and row["device"] == "cuda:0" for row in fields.values())
            and fields["_xyz"]["shape"] == [75767, 3] and fields["_features_rest"]["shape"] == [75767, 15, 3]
            and fields["_exposure"]["shape"] == [1016, 3, 4],
        "actual_original_call_elapsed_and_after_synchronize_measured": trace["calls"][0]["original_API_and_after_sync_seconds"] > 0,
        "actual02d024_guard_exit0_and_lease_released": guard["returncode"] == lease["returncode"] == 0 and guard["status"] == "completed" and bool(lease.get("lease_released_utc")),
        "all_saved_Windows_and_native_births_closed": all(row["saved_instance_ended"] for row in native + closed),
        "family2GiB_disk20GiB_and_original_resource_budgets": sum(q.stat().st_size for q in base.rglob("*") if q.is_file()) < 2 * 1024**3
            and shutil.disk_usage(ROOT).free >= 20 * 1024**3 and guard["peak_tree_RSS_bytes"] <= guard["maximum_RSS_bytes"]
            and all(read(p / name)["peak_tree_RSS_bytes"] <= read(p / name)["maximum_RSS_bytes"] for name in ("CPU_BOOTSTRAP_RESOURCE.json", "WSL_RESOURCE.json")),
        "parent37_and_no_new_candidate_count": read(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")["additional_accepted_count"] == 37
            and result["new_optimization_count"] == 0,
    }
    report = {"checks": checks, "reference_checks": compare_checks, "native_saved_instances": native,
        "Windows_saved_instances": closed, "known_complete_diagnostic_lease_seconds": lease["child_elapsed_seconds"],
        "complete_guard_seconds": guard["elapsed_seconds"], "seed_original_API_and_after_sync_seconds": trace["calls"][0]["original_API_and_after_sync_seconds"],
        "seed_raw_optimistic_fraction_of_guard_complete": trace["calls"][0]["original_API_and_after_sync_seconds"] / guard["elapsed_seconds"],
        "logical_retained_seed_bytes": 18232848, "logical_bytes_not_unique_allocation_or_already_saved_work": True,
        "no_initialization_skipped_no_new_algorithm_or_training": True, "parent_count": 37, "new_optimization_count": 0,
        "writer_identity": {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()}}
    (base / "REVIEW_001.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"checks": checks, "seed_seconds": report["seed_original_API_and_after_sync_seconds"],
        "seed_fraction": report["seed_raw_optimistic_fraction_of_guard_complete"], "new_count": 0}), flush=True)
    if not all(checks.values()):
        raise AssertionError("The original seed diagnostic closure failed")


if __name__ == "__main__":
    main()
