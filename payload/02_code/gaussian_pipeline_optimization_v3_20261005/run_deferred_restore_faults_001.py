"""Verify real comparison failures and SIGINT in four isolated complete entries."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

from run_cpu_dependency_restore import ROOT, CODE, linux, read, data_identity
from run_deferred_restore_equal_pair_001 import gpu, save, BASE


def inputs_unchanged(expected):
    prefix = linux(ROOT).rstrip("/") + "/"
    for row in expected:
        if not row["path"].startswith(prefix):
            raise ValueError("The original native input is outside the exact workspace")
        path = (ROOT / row["path"][len(prefix):]).resolve()
        path.relative_to(ROOT)
        current = data_identity(path)
        if any(current[key] != row[key] for key in ("bytes", "sha256")):
            return False
    return True


def close_failed(process, born, folder):
    if process.poll() is not None:
        process.wait()
        return
    result = subprocess.run(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
        "/opt/e26_thin_surface_v1/venv/bin/python", "-B",
        linux(CODE / "stop_owned_restore_fault_linux_001.py"), linux(folder)],
        timeout=25, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise RuntimeError("The exact native fault instance needs investigation before lease release")
    try:
        current = psutil.Process(process.pid)
        if current.create_time() == born and process.poll() is None:
            process.terminate()
        process.wait(timeout=10)
    except psutil.NoSuchProcess:
        pass


def main():
    output = Path(sys.argv[1]).resolve()
    output.relative_to(BASE)
    if output.exists() or shutil.disk_usage(ROOT).free < 20 * 1024**3:
        raise ValueError("Actual restore faults require a fresh private directory and disk reserve")
    output.mkdir()
    sources = [data_identity(CODE / name) for name in (Path(__file__).name,
        "profile_fault_deferred_restore_equal_001.py", "fault_deferred_restore_hook_001.py",
        "profile_deferred_restore_equal_001.py", "deferred_checkpoint_equal_001.py",
        "profile_native_restore_cost.py", "profile_cpu_dependency_restore.py",
        "stop_owned_restore_fault_linux_001.py", "stop_owned_restore_linux_001.py")]
    save(output / "PROTOCOL.json", {"sources": sources, "actual_mismatch_and_SIGINT": True,
        "complete1016_initial_scene_each_case_no_training_step": True, "parent_count": 36,
        "new_optimization_count": 0})
    (output / "sources").mkdir()
    for row in sources:
        shutil.copyfile(row["path"], output / "sources" / Path(row["path"]).name)
    rows, samples = [], [gpu()]
    original_inputs = read(BASE / "c3/native/CONTRACT.json")["inputs"]
    try:
        for kind in ("mismatch", "cancel"):
            for mode in ("native", "deferred"):
                name = kind + "_" + mode
                folder = output / name
                command = [sys.executable, "-X", "utf8", "-B", str(CODE / "run_bounded_journal_task.py"),
                    "--receipt", str(output / (name + "_GUARD.json")), "--log", str(output / (name + "_CONTROLLER.log")),
                    "--maximum-seconds", "190", "--maximum-rss-gib", "4", "--", "wsl", "-d",
                    "E26-C2-LIV-20260907-134802", "--", "/opt/e26_thin_surface_v1/venv/bin/python", "-B",
                    linux(CODE / "profile_fault_deferred_restore_equal_001.py"), "--output", linux(folder),
                    "--mode", "bootstrap", "--equality-mode", mode, "--fault", kind]
                started = time.perf_counter()
                with (output / (name + "_GUARD_STDOUT.log")).open("x", encoding="utf-8") as log:
                    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW)
                    born = psutil.Process(process.pid).create_time()
                    try:
                        stamp = 0
                        while process.poll() is None:
                            if time.perf_counter() - started > 205:
                                raise TimeoutError("The original actual fault entry did not drain")
                            if time.perf_counter() - stamp > 2:
                                samples.append(gpu())
                                stamp = time.perf_counter()
                            time.sleep(.1)
                        code = process.wait()
                    except BaseException:
                        close_failed(process, born, folder)
                        raise
                save(output / (name + "_ACTUAL_EXIT.json"), {"returncode": code,
                    "elapsed_seconds": time.perf_counter() - started,
                    "process_identity": {"pid": process.pid, "creation_time": born}})
                guard = read(output / (name + "_GUARD.json"))
                failure = read(folder / "FAILURE.json")
                fault = read(folder / "ACTUAL_FAULT_TRACE.json")
                equality = read(folder / "EQUALITY_EXIT_TRACE.json")
                expected_type = "AssertionError" if kind == "mismatch" else "KeyboardInterrupt"
                checks = {
                    "actual_child_failure_not_guard_budget_failure": code == 1 and guard["status"] == "failed_child" and guard["returncode"] != 0,
                    "actual_original_exception_type_kept": failure["type"] == fault["actual_exception_type"] == expected_type,
                    "actual_original_failure_message_kept": failure["error"] == (
                        "Restored tensors or optimizer state differ from the saved state" if kind == "mismatch" else ""),
                    "actual_CUDA_comparison_scope_entered": fault["comparison_scope_live"] and fault["actual_CUDA_compare_API_calls"] > 0,
                    "actual_registered_fault_delivered": fault["private_second_state_mutated"] if kind == "mismatch" else fault["actual_SIGINT_sent"],
                    "failure_did_not_publish_restore_or_success": not (folder / "RESULT.json").exists() and not (folder / "RESTORE_CHECK.json").exists(),
                    "original_equality_bindings_and_owned_stream_closed": equality["original_comparator_and_tensor_equal_bindings_restored"] and equality["exception_owned_current_stream_drained"],
                    "original_prepare_bindings_and_owned_signal_thread_closed": fault["original_prepare_and_tensor_API_bindings_restored"] and fault["owned_signal_thread_closed"],
                    "complete_original1038_inputs_kept": read(folder / "CONTRACT.json")["inputs"] == original_inputs,
                    "sources_inputs_and_owned_payload_unchanged": sources == [data_identity(row["path"]) for row in sources]
                        and inputs_unchanged(original_inputs)
                        and data_identity(folder / "checkpoint_001000.pth")["sha256"] == original_inputs[0]["sha256"],
                    "private_family_output2GiB_and_disk20GiB": sum(path.stat().st_size for path in BASE.rglob("*") if path.is_file()) < 2 * 1024**3
                        and shutil.disk_usage(ROOT).free >= 20 * 1024**3,
                }
                row = {"kind": kind, "mode": mode, "checks": checks, "fault": fault,
                    "failure": failure, "equality_exit": equality, "folder": str(folder)}
                rows.append(row)
                save(output / (name + "_CHECK.json"), row)
                print(json.dumps({"kind": kind, "mode": mode, "checks": checks}), flush=True)
                if not all(checks.values()):
                    raise AssertionError("Actual complete restore failure closure did not pass")
        save(output / "RESULT.json", {"checks": {"all_four_actual_fault_paths": len(rows) == 4 and all(all(row["checks"].values()) for row in rows)},
            "rows": rows, "sources": sources, "GPU_samples": samples, "new_optimization_count": 0,
            "fresh_complete_recovery_and_fee_acceptance_still_required": True})
    finally:
        save(output / "GPU_SAMPLES.json", samples)


if __name__ == "__main__":
    main()
