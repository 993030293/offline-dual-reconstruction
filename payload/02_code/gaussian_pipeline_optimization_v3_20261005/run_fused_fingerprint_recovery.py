"""Run two real image17 errors, then recover the original complete Scene contract."""

from pathlib import Path
import subprocess
import sys
import time

import psutil

from scene_transform_cost_node import ROOT, CODE
from run_fused_camera_digest_campaign import compare
from review_scene_batch_negative import read
from runtime import atomic_json, data_identity

OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/scf001"


def main():
    output = OUT / "v1"
    output.mkdir(exist_ok=False)
    reference = read(OUT / "c0/t0/original/RESULT.json")
    fields = ("original_image", "alpha_mask", "world_view_transform", "projection_matrix", "full_proj_transform", "camera_center")
    expected = [{name: row[name] for name in fields} for row in reference["scientific"]["rows"][:16]]
    rows = []
    started = time.perf_counter()
    for mode, enabled, failure in (("fault_original", False, True), ("fault_fused", True, True), ("recovery_fused", True, False)):
        leaf, node = output / mode, output / (mode+"_node")
        script = "profile_faulted_camera_fingerprint.py" if failure else "profile_fused_camera_fingerprint.py"
        wrapper = "fused_fingerprint_fault_node.py" if failure else "fused_fingerprint_scene_node.py"
        mounted = "/mnt/c/" + str(leaf.relative_to(Path("C:/"))).replace("\\", "/")
        command = [sys.executable, "-X", "utf8", str(CODE / wrapper), "--kind", "camera_oracle", "--mode", "diagnostic", "--node", str(node), "--output", str(leaf),
            "--", "wsl", "-d", "E26-C2-LIV-20260907-134802", "--", "/usr/bin/env", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
            "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "/mnt/c/lidar-camera-fusion-mask-robust-downsampling/02_code/gaussian_pipeline_optimization_v3_20261005/"+script,
            "--mode", "baseline", "--fused-enabled", "1" if enabled else "0", "--library",
            "/mnt/c/lidar-camera-fusion-mask-robust-downsampling/04_outputs/gaussian_pipeline_optimization_v3_20261005/scf001/g0/build/fused_camera_digest.so", "--output", mounted]
        begin = time.perf_counter()
        with (output / (mode+".log")).open("x", encoding="utf-8") as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            owner = {"pid": child.pid, "creation_time": psutil.Process(child.pid).create_time()}
            returncode = child.wait()
        elapsed = time.perf_counter()-begin
        controller = read(node / "NODE.json")
        checks = {"actual_expected_exit": returncode == (1 if failure else 0),
                  "owned_Windows_and_Linux_workers_closed": not controller["live_Windows_after_wait"] and all(not row["alive"] for row in controller["actual_Linux_after_wait"]),
                  "actual_original_controller_terminal_status": controller["status"] == ("failed" if failure else "completed")}
        if failure:
            prefix = read(leaf / "FAULT_PREFIX.json")
            aborted = read(leaf / "FAILED_FINGERPRINT_TRACE.json")
            checks.update(actual_image17_after16_original_complete_fields=prefix["attempted_image"] == 17
                and prefix["first16_complete_camera_fields"] == expected,
                original_error_propagated="Forced original image consumption failure at17" in aborted["error"],
                no_false_full_RESULT=not (leaf / "RESULT.json").exists(),
                fused_all1016_prepared_partial16_aborted_weak_bindings_closed=not enabled
                    or aborted["trace"]["prepared"] == 1016 and aborted["trace"]["consumed"] == 16
                    and aborted["trace"]["aborted"] and aborted["trace"]["active_entries"] == 0)
        else:
            equality = compare(OUT / "c0/t0/original", leaf)
            checks.update(recovered_all1016_15_fields_inputs_sources_runtime_and2_files_exact=all(equality["checks"].values()))
        row = {"mode": mode, "checks": checks, "actual_spawn_to_exit_seconds": elapsed, "owner": owner,
               "controller": controller, "returncode": returncode}
        rows.append(row)
        atomic_json(output / "STATE.json", {"status": "running_actual_image17_errors_and_complete_recovery0", "rows": rows, "new_optimization_count": 0})
        if not all(checks.values()):
            raise AssertionError("A real complete pipeline error or recovery check failed: " + repr(checks))
    checks = {"both_real_prefix_errors_and_all_owned_exits": len(rows) == 3 and all(all(row["checks"].values()) for row in rows),
              "exact_unmodified_complete_consumer_recovery": rows[-1]["mode"] == "recovery_fused" and rows[-1]["returncode"] == 0,
              "formal_three_pairs_and_original_inputs_untouched": read(OUT / "c0/RESULT.json")["source_identities_unchanged"]
                and len(read(OUT / "c0/RESULT.json")["pairs"]) == 3}
    atomic_json(output / "RESULT.json", {"status": "two_actual_image17_errors_and_complete1016_recovery_exact0",
        "checks": checks, "rows": rows, "before_result_seconds": time.perf_counter()-started,
        "source": data_identity(Path(__file__)), "full_reconstruction_training_checkpoint_or_cross_process_cache_resume": "not_claimed",
        "new_optimization_count": 0})
    print("Actual full image17 errors and fresh full1016 recovery passed; new 0, final acceptance pending")


if __name__ == "__main__":
    main()
