"""Observe two actual complete identity failures and a fresh complete recovery."""

from pathlib import Path
import subprocess
import sys
import time

import psutil

from run_q68_fused_campaign import compare
from review_scene_batch_negative import read
from scene_transform_cost_node import ROOT, CODE
from runtime import atomic_json, data_identity

OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/scfq001"


def main():
    output = OUT / "v1"
    output.mkdir(exist_ok=False)
    reference = read(OUT / "c0/t0/control/RESULT.json")
    expected = reference["inputs"][:16]
    rows = []
    started = time.perf_counter()
    for mode, enabled, failure in (("fault_control", False, True), ("fault_combo", True, True), ("recovery_combo", True, False)):
        leaf, node = output / mode, output / (mode + "_node")
        script = "profile_faulted_q68_fingerprint.py" if failure else "profile_q68_fused_fingerprint.py"
        wrapper = "q68_fingerprint_fault_node.py" if failure else "q68_fused_fingerprint_node.py"
        mounted = "/mnt/c/" + leaf.relative_to(Path("C:/")).as_posix()
        command = [sys.executable, "-X", "utf8", str(CODE / wrapper), "--kind", "camera_oracle", "--mode", "diagnostic", "--node", str(node), "--output", str(leaf),
            "--", "wsl", "-d", "E26-C2-LIV-20260907-134802", "--", "/usr/bin/env", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
            "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "/mnt/c/lidar-camera-fusion-mask-robust-downsampling/02_code/gaussian_pipeline_optimization_v3_20261005/" + script,
            "--mode", "baseline", "--fused-enabled", "1", "--identity-enabled", "1" if enabled else "0", "--library",
            "/mnt/c/lidar-camera-fusion-mask-robust-downsampling/04_outputs/gaussian_pipeline_optimization_v3_20261005/scf001/g0/build/fused_camera_digest.so", "--output", mounted]
        begin = time.perf_counter()
        with (output / (mode + ".log")).open("x", encoding="utf-8") as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            owner = {"pid": child.pid, "creation_time": psutil.Process(child.pid).create_time()}
            returncode = child.wait()
        elapsed = time.perf_counter() - begin
        controller = read(node / "NODE.json")
        checks = {"actual_expected_exit": returncode == (1 if failure else 0),
            "owned_Windows_and_Linux_workers_closed": not controller["live_Windows_after_wait"] and all(not row["alive"] for row in controller["actual_Linux_after_wait"]),
            "actual_original_controller_terminal_status": controller["status"] == ("failed" if failure else "completed")}
        if failure:
            prefix = read(leaf / "FAULT_PREFIX.json")
            identity = read(leaf / "FAILED_IDENTITY_TRACE.json")
            aborted = read(leaf / "FAILED_FINGERPRINT_TRACE.json")
            phases = identity["phases"]
            checks.update(
                actual_identity17_after16_complete_original_identities=prefix["attempted_identity"] == 17 and prefix["first16_complete_identities"] == expected,
                original_error_propagated="Forced original complete input identity failure at17" in identity["error"] and "Forced original complete input identity failure at17" in aborted["error"],
                no_false_complete_RESULT=not (leaf / "RESULT.json").exists(),
                actual_full1016_images_consumed_and_all_weak_bindings_closed=aborted["trace"]["prepared"] == aborted["trace"]["consumed"] == 1016 and aborted["trace"]["cache_closed"] and aborted["trace"]["active_entries"] == 0,
                partial_input_phase_error_all_actual_identity_workers_joined=len(phases) == 3 and phases[-1]["phase"] == "inputs_after"
                    and phases[-1]["status"] == "failed" and phases[-1]["owned_identity_threads_closed"] and (not enabled or
                        phases[-1]["queue"]["consumed"] == 16 and phases[-1]["queue"]["closed"] and phases[-1]["queue"]["all_started_joined"] and bool(phases[-1]["queue"]["worker_errors"])))
        else:
            equality = compare(OUT / "c0/t0/control", leaf)
            checks.update(recovered_all1016_all_identity_phases_15_fields_inputs_sources_runtime_and2_files_exact=all(equality["checks"].values()))
        row = {"mode": mode, "checks": checks, "actual_spawn_to_exit_seconds": elapsed, "owner": owner,
            "controller": controller, "returncode": returncode}
        rows.append(row)
        atomic_json(output / "STATE.json", {"status": "running_actual_complete_identity17_errors_and_recovery0", "rows": rows, "new_optimization_count": 0})
        if not all(checks.values()):
            raise AssertionError("An actual complete identity failure or recovery contract failed: " + repr(checks))
    checks = {"two_actual_FIFO_errors_prefixes_and_all_owners_closed": len(rows) == 3 and all(all(row["checks"].values()) for row in rows),
        "fresh_unmodified_complete_consumer_recovery": rows[-1]["mode"] == "recovery_combo" and rows[-1]["returncode"] == 0,
        "formal_three_pairs_and_source_inputs_protected": len(read(OUT / "c0/RESULT.json")["pairs"]) == 3 and read(OUT / "c0/RESULT.json")["source_identities_unchanged"]}
    result = {"status": "two_actual_complete_identity17_errors_and_fresh_full1016_recovery_exact0", "checks": checks,
        "rows": rows, "before_result_seconds": time.perf_counter() - started, "source": data_identity(Path(__file__)),
        "no_training_checkpoint_or_cross_process_cache_resume_claim": True, "new_optimization_count": 0}
    atomic_json(output / "RESULT.json", result)
    atomic_json(output / "STATE.json", result)
    print("Two actual complete native identity17 failures and fresh full1016 recovery passed; combination new 0")


if __name__ == "__main__":
    main()
