"""Compare every original output after one bounded complete cost diagnostic."""

from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

from run_fused_camera_digest_campaign import compare
from review_scene_batch_negative import read
from scene_transform_cost_node import ROOT, CODE
from runtime import atomic_json, data_identity

OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/sclc001"


def main():
    started = time.perf_counter()
    if not all(read(OUT / "PREFLIGHT_001.json")["checks"].values()):
        raise ValueError("The current full-input environment preflight did not pass")
    leaf, node = OUT / "p0", OUT / "p0_node"
    paths = [Path(__file__), *[CODE / name for name in ("timed_camera_preparation.py", "profile_composed_scene_cost.py",
        "composed_scene_cost_node.py", "composed_scene_cost_preflight.py")]]
    paths += [Path(row["path"]) for row in read(OUT.parent / "scfq001/c0/RESULT.json")["sources"]]
    paths = list(dict.fromkeys(paths))
    identities = [data_identity(path) for path in paths]
    archive = OUT / "run_sources"
    archive.mkdir(exist_ok=False)
    for index, path in enumerate(paths):
        shutil.copyfile(path, archive / (f"{index:03d}_" + path.name))
    command = [sys.executable, "-X", "utf8", str(CODE / "composed_scene_cost_node.py"),
        "--kind", "camera_oracle", "--mode", "diagnostic", "--node", str(node), "--output", str(leaf), "--",
        "wsl", "-d", "E26-C2-LIV-20260907-134802", "--", "/usr/bin/env", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
        "/opt/e26_thin_surface_v1/venv/bin/python", "-B",
        "/mnt/c/lidar-camera-fusion-mask-robust-downsampling/02_code/gaussian_pipeline_optimization_v3_20261005/profile_composed_scene_cost.py",
        "--mode", "baseline", "--fused-enabled", "1", "--identity-enabled", "1", "--library",
        "/mnt/c/lidar-camera-fusion-mask-robust-downsampling/04_outputs/gaussian_pipeline_optimization_v3_20261005/scf001/g0/build/fused_camera_digest.so",
        "--output", "/mnt/c/lidar-camera-fusion-mask-robust-downsampling/04_outputs/gaussian_pipeline_optimization_v3_20261005/sclc001/p0"]
    child_started = time.perf_counter()
    with (OUT / "p0.log").open("x", encoding="utf-8") as stream:
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        owner = {"pid": child.pid, "creation_time": psutil.Process(child.pid).create_time()}
        atomic_json(OUT / "STATE.json", {"status": "running_complete_preparation_cost_diagnostic0", "owner": owner,
            "command": command, "sources": identities, "new_optimization_count": 0})
        returncode = child.wait()
    elapsed = time.perf_counter() - child_started
    if returncode:
        atomic_json(OUT / "STATE.json", {"status": "failed_actual_complete_cost_diagnostic0", "returncode": returncode,
            "actual_spawn_to_exit_seconds": elapsed, "owner": owner, "sources": identities, "new_optimization_count": 0})
        raise RuntimeError("The real preparation diagnostic failed; preserve its original evidence")
    result, trace, identity, controller = (read(path) for path in (leaf / "RESULT.json", leaf / "PREPARATION_COST.json",
        leaf / "IDENTITY_TRACE.json", node / "NODE.json"))
    reference = OUT.parent / "scfq001/c0/t0/combo"
    compared = compare(reference, leaf)
    old_identity = read(reference / "IDENTITY_TRACE.json")
    checks = {
        "complete_reference_all15_fields_and2_full_files_exact": all(compared["checks"].values()),
        "all5_actual_preparation_timing_bindings_and_function_restoration": len(trace["checks"]) == 5 and all(trace["checks"].values()),
        "all_original4_identity_lists_and_full_checks_preserved": all(identity["checks"].values()) and [row["paths"] for row in identity["phases"]] == [row["paths"] for row in old_identity["phases"]],
        "actual_controller_and_all_Windows_Linux_workers_closed": controller["status"] == "completed" and not controller["live_Windows_after_wait"] and all(not row["alive"] for row in controller["actual_Linux_after_wait"]),
        "all_actual_current_sources_preserved": identities == [data_identity(path) for path in paths],
    }
    if not all(checks.values()):
        raise AssertionError("The complete timing diagnostic changed an original protected output")
    summary = {"status": "complete_composed_preparation_diagnostic_exact_reference_verified0", "checks": checks,
        "comparison": compared, "sources": identities, "owner": owner, "actual_spawn_to_exit_seconds": elapsed,
        "actual_internal_to_result_seconds": time.perf_counter() - started,
        "Scene_load_seconds": result["Scene_load_seconds"], "prefetch": result["prefetch"],
        "worker_cumulative_seconds": trace["worker_cumulative_seconds"],
        "worker_occupied_union_seconds": trace["worker_occupied_union_seconds"],
        "all_prepare_occupied_union_seconds": trace["all_prepare_occupied_union_seconds"],
        "host_call_cumulative_seconds": trace["host_call_cumulative_seconds"], "host_call_counts": trace["host_call_counts"],
        "full_process_tree_peak_RSS_bytes": result["full_process_tree_peak_RSS_bytes"],
        "reference_is_existing_output_not_fresh_cost_control": True, "timing_instrumentation_overhead_not_an_optimization": True,
        "new_optimization_count": 0, "parent_count": 29, "remaining": 21, "whole_training_new_objects_GT_not_verified": True}
    atomic_json(OUT / "RESULT.json", summary)
    atomic_json(OUT / "STATE.json", summary)
    print({key: summary[key] for key in ("checks", "Scene_load_seconds", "prefetch", "worker_cumulative_seconds", "worker_occupied_union_seconds", "host_call_cumulative_seconds")})


if __name__ == "__main__":
    main()
