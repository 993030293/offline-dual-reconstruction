"""Run a fresh complete restore pair and retain all original quality checks."""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil

from run_cpu_dependency_restore import ROOT, CODE, compare, linux, read, data_identity

BASE = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup014/R168"


def save(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def gpu():
    line = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"], text=True).strip()
    used, free = map(int, line.split(","))
    if used > 12288 or free < 2048:
        raise MemoryError("The whole GPU reserve or usage limit was exceeded")
    return {"used_MiB": used, "free_MiB": free}


def cuda_rows(trace):
    return [{key: row[key] for key in ("shape", "dtype", "device")} for row in trace["rows"] if row.get("CUDA_left")]


def main():
    output = Path(sys.argv[1]).resolve()
    output.relative_to(BASE)
    order = sys.argv[2]
    if order not in ("AB", "BA") or output.exists() or shutil.disk_usage(ROOT).free < 20 * 1024**3:
        raise ValueError("The registered complete pair needs a fresh directory and order")
    output.mkdir(parents=True)
    sources = [data_identity(CODE / name) for name in (Path(__file__).name, "deferred_checkpoint_equal_001.py",
        "profile_deferred_restore_equal_001.py", "profile_cpu_dependency_restore.py", "profile_native_restore_cost.py", "run_cpu_dependency_restore.py")]
    save(output / "PROTOCOL.json", {"sources": sources, "order": order, "full_original1016_and1038_inputs": True,
        "original_two_private_loads_nine_restore_fields_no_new_training_step": True,
        "work_gate_net10percent_not_full_speed_claim": True, "whole_raw_and_fee_median_max1_01_each1_1": True,
        "parent_count": 36, "new_optimization_count": 0})
    (output / "sources").mkdir()
    for row in sources:
        shutil.copyfile(row["path"], output / "sources" / Path(row["path"]).name)
    rows, GPU_samples, processes = [], [], []
    GPU_samples.append(gpu())
    try:
        for mode in (("native", "deferred") if order == "AB" else ("deferred", "native")):
            folder = output / mode
            command = [sys.executable, "-X", "utf8", "-B", str(CODE / "run_bounded_journal_task.py"),
                "--receipt", str(output / (mode + "_GUARD.json")), "--log", str(output / (mode + "_CONTROLLER.log")),
                "--maximum-seconds", "190", "--maximum-rss-gib", "4", "--", "wsl", "-d",
                "E26-C2-LIV-20260907-134802", "--", "/opt/e26_thin_surface_v1/venv/bin/python", "-B",
                linux(CODE / "profile_deferred_restore_equal_001.py"), "--output", linux(folder),
                "--mode", "bootstrap", "--equality-mode", mode]
            started = time.perf_counter()
            with (output / (mode + "_GUARD_STDOUT.log")).open("x", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
                identity = {"pid": process.pid, "creation_time": psutil.Process(process.pid).create_time(), "mode": mode}
                processes.append(identity)
                stamp = 0
                while process.poll() is None:
                    if time.perf_counter() - started > 205:
                        raise TimeoutError("The registered complete side did not drain")
                    if time.perf_counter() - stamp > 2:
                        GPU_samples.append(gpu())
                        stamp = time.perf_counter()
                    time.sleep(.1)
                code = process.wait()
            elapsed = time.perf_counter() - started
            save(output / (mode + "_ACTUAL_EXIT.json"), {"returncode": code, "whole_complete_side_seconds": elapsed, "process_identity": identity})
            if code != 0:
                raise RuntimeError("The original complete restore or registered guard failed")
            result = read(folder / "RESULT.json")
            trace = read(folder / "EQUALITY_WORK_TRACE.json")
            if not all(result["checks"].values()) or read(folder / "WSL_RESOURCE.json")["status"] != "completed":
                raise AssertionError("The original full restore or resource contract failed")
            row = {"mode": mode, "whole_complete_side_seconds": elapsed,
                "GPU_boolean_materializations": trace["CUDA_boolean_materializations"],
                "complete_CUDA_comparison_leaves": trace["CUDA_comparison_leaves"], "folder": str(folder)}
            rows.append(row)
            save(output / (mode + "_ENTRY.json"), row)
            print(json.dumps(row), flush=True)
        before = next(row for row in rows if row["mode"] == "native")
        after = next(row for row in rows if row["mode"] == "deferred")
        checks = compare(output / "native", output / "deferred")
        btrace, ctrace = [read(output / mode / "EQUALITY_WORK_TRACE.json") for mode in ("native", "deferred")]
        checks.update({
            "all_complete_CUDA_comparison_shapes_dtypes_devices_and_order_kept": cuda_rows(btrace) == cuda_rows(ctrace) and len(cuda_rows(btrace)) == 22,
            "all22_current_native_materializations_and_comparisons_kept_before_reduction": before["GPU_boolean_materializations"] == before["complete_CUDA_comparison_leaves"] == after["complete_CUDA_comparison_leaves"] == 22,
            "actual_full_numeric_tree_work_reduction_at_least10percent": after["GPU_boolean_materializations"] <= .9 * before["GPU_boolean_materializations"],
            "each_raw_complete_entry_ratio_at_most1_1": after["whole_complete_side_seconds"] / before["whole_complete_side_seconds"] <= 1.1,
            "all_original_and_new_equality_bindings_closed": btrace["original_comparator_and_tensor_equal_bindings_restored"] and ctrace["original_comparator_and_tensor_equal_bindings_restored"],
            "same_frozen_sources_unchanged": sources == [data_identity(row["path"]) for row in sources],
            "all_original_inputs1038_identical_between_sides": read(output / "native/CONTRACT.json")["inputs"] == read(output / "deferred/CONTRACT.json")["inputs"],
            "private_all_family_output2GiB_and_disk20GiB": sum(path.stat().st_size for path in BASE.rglob("*") if path.is_file()) < 2 * 1024**3 and shutil.disk_usage(ROOT).free >= 20 * 1024**3,
        })
        save(output / "RESULT.json", {"checks": checks, "entries": rows,
            "raw_work_reduction": 1 - after["GPU_boolean_materializations"] / before["GPU_boolean_materializations"],
            "raw_complete_whole_ratio": after["whole_complete_side_seconds"] / before["whole_complete_side_seconds"],
            "GPU_samples": GPU_samples, "processes": processes, "sources": sources,
            "parent_count": 36, "new_optimization_count": 0,
            "not_acceptance_without_three_NEW_cost_generic_fault_cancel_fresh_recovery_and_review": True})
        print(json.dumps({"checks": checks, "new_count": 0}), flush=True)
        if not all(checks.values()):
            raise AssertionError("The preregistered complete pair failed its first-screen contract")
    finally:
        save(output / "GPU_SAMPLES.json", GPU_samples)


if __name__ == "__main__":
    main()
