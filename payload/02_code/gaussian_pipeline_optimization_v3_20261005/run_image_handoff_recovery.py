"""Verify real native preparation errors, private partial deletion and recovery."""

from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

from run_open_image_handoff_campaign import OUT, compare
from review_scene_batch_negative import read
from scene_transform_cost_node import ROOT, CODE
from runtime import atomic_json, data_identity


def main():
    output = OUT / "v3"
    output.mkdir(exist_ok=False)
    formal = read(OUT / "c0/RESULT.json")
    if len(formal["pairs"]) != 3 or formal["status"] != "three_complete_handoff_work_pairs_final_validation_and_cost_pending0":
        raise ValueError("Three positive work pairs must precede actual failure recovery")
    rows = []
    for mode, enabled, failure in (("fault_control", False, True), ("fault_handoff", True, True), ("recovery_handoff", True, False)):
        leaf, node = output / mode, output / (mode + "_node")
        if not failure:
            target = leaf.resolve()
            target.relative_to((OUT / "v3").resolve())
            if target.exists() or target.name != "recovery_handoff":
                raise ValueError("Only a fresh own recovery directory may be seeded and cleared")
            target.mkdir()
            scene = target / "scene"
            scene.mkdir()
            deleted = []
            for name in ("input.ply", "cameras.json"):
                source = output / "fault_handoff/scene" / name
                copied = scene / name
                shutil.copyfile(source, copied)
                if data_identity(source)["sha256"] != data_identity(copied)["sha256"]:
                    raise ValueError("A private partial-output recovery copy changed")
                deleted.append(data_identity(copied))
                copied.unlink()
            scene.rmdir()
            target.rmdir()
            atomic_json(output / "FORCED_PARTIAL_DELETE.json", {"private_copies_deleted": deleted, "original_failure_outputs_retained": True,
                "recovery_target_absent_before_fresh_entry": not leaf.exists(), "persistent_cache_not_present": True, "new_optimization_count": 0})
        script = "profile_faulted_image_handoff.py" if failure else "profile_open_image_handoff.py"
        wrapper = "image_handoff_fault_node.py" if failure else "open_image_handoff_node.py"
        mounted = "/mnt/c/" + leaf.relative_to(Path("C:/")).as_posix()
        command = [sys.executable, "-X", "utf8", str(CODE / wrapper), "--kind", "camera_oracle", "--mode", "diagnostic", "--node", str(node), "--output", str(leaf), "--",
            "wsl", "-d", "E26-C2-LIV-20260907-134802", "--", "/usr/bin/env", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
            "/opt/e26_thin_surface_v1/venv/bin/python", "-B", "/mnt/c/lidar-camera-fusion-mask-robust-downsampling/02_code/gaussian_pipeline_optimization_v3_20261005/" + script,
            "--mode", "baseline", "--fused-enabled", "1", "--identity-enabled", "1", "--handoff-enabled", "1" if enabled else "0", "--library",
            "/mnt/c/lidar-camera-fusion-mask-robust-downsampling/04_outputs/gaussian_pipeline_optimization_v3_20261005/scf001/g0/build/fused_camera_digest.so", "--output", mounted]
        started = time.perf_counter()
        with (output / (mode + ".log")).open("x", encoding="utf-8") as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            owner = {"pid": child.pid, "creation_time": psutil.Process(child.pid).create_time()}
            returncode = child.wait()
        controller = read(node / "NODE.json")
        trace = read(leaf / "IMAGE_HANDOFF_TRACE.json")
        work = read(leaf / "SOURCE_IMAGE_OPEN_WORK.json")
        checks = {"actual_expected_exit_and_controller": returncode == (1 if failure else 0) and controller["status"] == ("failed" if failure else "completed"),
            "owned_Windows_and_Linux_workers_closed": not controller["live_Windows_after_wait"] and all(not row["alive"] for row in controller["actual_Linux_after_wait"]),
            "native_bindings_images_threads_and_file_hooks_closed": trace["original_functions_restored"] and trace["owned_decode_threads_closed"] and trace["recorded"] == trace["closed_images"] and trace["remaining_images"] == 0 and trace["maximum_owned_images"] <= 8 and work["native_functions_restored"]}
        if failure:
            prefix = read(leaf / "FAULT_PREFIX.json")
            checks.update(actual_prepare17_error_and16_complete_native_cameras=prefix["attempted_camera"] == 17 and len(prefix["first16_complete_camera_fields"]) == 16 and prefix["error_message"] == "Forced original native image preparation failure at17",
                          failed_future_original_header_fallback=not enabled or trace["failed_future_native_header_fallbacks"] == 1,
                          no_false_complete_RESULT=not (leaf / "RESULT.json").exists())
        else:
            equality = compare(OUT / "c0/t0/control", leaf)
            checks.update(recovered_full1016_all15_scientific_fields_and_two_actual_files_exact=all(equality["checks"].values()))
        row = {"mode": mode, "checks": checks, "owner": owner, "controller": controller, "returncode": returncode,
               "actual_spawn_exit_and_compare_seconds": time.perf_counter() - started, "source_native_read_opens": work["successful_native_read_opens"]}
        rows.append(row)
        atomic_json(output / "STATE.json", {"status": "running_actual_prepare17_failure_and_fresh_recovery0", "rows": rows, "new_optimization_count": 0})
        if not all(checks.values()):
            raise AssertionError("A real native preparation failure or recovery contract failed: " + repr(checks))
    prefixes = [read(output / mode / "FAULT_PREFIX.json") for mode in ("fault_control", "fault_handoff")]
    checks = {"both_actual_prepare_errors_prefixes_and_owners_closed": all(all(row["checks"].values()) for row in rows) and prefixes[0]["first16_complete_camera_fields"] == prefixes[1]["first16_complete_camera_fields"],
              "actual_private_partial_delete_then_full_fresh_recovery": read(output / "FORCED_PARTIAL_DELETE.json")["recovery_target_absent_before_fresh_entry"] and rows[-1]["returncode"] == 0,
              "formal_and_failure_original_outputs_retained": len(formal["pairs"]) == 3 and all((output / mode / "FAULT_PREFIX.json").is_file() for mode in ("fault_control", "fault_handoff"))}
    atomic_json(output / "RESULT.json", {"status": "actual_prepare17_errors_and_fresh_full1016_recovery_exact0", "checks": checks,
        "rows": rows, "source": data_identity(Path(__file__)), "new_optimization_count": 0, "training_or_checkpoint_resume": "not_claimed"})
    if not all(checks.values()):
        raise AssertionError("A completed recovery summary contradicted its real entry checks")
    print({"status": "actual_errors_and_fresh_full1016_recovery_exact0", "source_work": sum(row["source_native_read_opens"] for row in rows), "new": 0})


if __name__ == "__main__":
    main()
