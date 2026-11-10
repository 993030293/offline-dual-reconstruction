"""Retain three fresh contiguous restore pairs and stop on the first failure."""

import time

STARTED = time.perf_counter()

import json
from pathlib import Path
import statistics
import subprocess
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity

OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/nrc001/c0"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def linux(path):
    return "/mnt/c/" + str(path).replace("\\", "/")[3:]


def compare(left, right):
    first, second = (read(path / "RESULT.json") for path in (left, right))
    checks = {
        "original_all_nine_restore_fields_and_complete_sampling_exact":
            first["complete_original_restore_receipt"]["checks"] == second["complete_original_restore_receipt"]["checks"]
            and first["actual_next32_names"] == second["actual_next32_names"],
        "all1016_complete_pixels_matrices_calibration_and_order_exact":
            first["complete_original_camera_rows"] == second["complete_original_camera_rows"],
        "full_native_state_device_alias_layout_exact": first["complete_saved_device_alias_layout"] == second["complete_saved_device_alias_layout"],
        "native_Torch_defaults_exact": first["Torch_defaults"] == second["Torch_defaults"],
        "legacy_native_sampler_oracle_exact": read(left / "LEGACY_SAMPLER_ORACLE.json") == read(right / "LEGACY_SAMPLER_ORACLE.json"),
        "original_complete_scene_files_exact": all((left / "scene" / name).read_bytes() == (right / "scene" / name).read_bytes()
            for name in ("input.ply", "cameras.json")),
    }
    return checks


def main():
    OWN.mkdir(parents=True, exist_ok=False)
    atomic_json(OWN / "PROCESS_IDENTITY.json", {"pid": psutil.Process().pid, "creation_time": psutil.Process().create_time()})
    rows, pairs = [], []
    atomic_json(OWN / "CONTRACT.json", {"sources": [data_identity(CODE / name) for name in
        ("run_cpu_dependency_restore.py", "profile_cpu_dependency_restore.py", "profile_native_restore_cost.py", "native_cpu_dependency_bootstrap.py")],
        "original_two_loads_and_complete1016_protected": True, "first_raw_whole_screen_minimum3percent": .03,
        "raw_median_ratio_maximum1_01": 1.01, "raw_each_ratio_maximum1_1": 1.1,
        "all_old_new_known_fees_and_actual_error_cancel_recovery_required_before_acceptance": True,
        "new_optimization_count": 0})
    try:
        for trial in range(3):
            for mode in (("incumbent", "bootstrap") if trial % 2 == 0 else ("bootstrap", "incumbent")):
                started = time.perf_counter()
                folder = OWN / f"t{trial}" / mode
                folder.parent.mkdir(exist_ok=True)
                guard = folder.with_name(mode + "_GUARD.json")
                command = [sys.executable, "-B", str(CODE / "run_bounded_journal_task.py"),
                    "--receipt", str(guard), "--log", str(folder.with_name(mode + ".log")),
                    "--maximum-seconds", "600", "--maximum-rss-gib", "4", "--", "wsl", "-d",
                    "E26-C2-LIV-20260907-134802", "--", "/opt/e26_thin_surface_v1/venv/bin/python", "-B",
                    linux(CODE / "profile_cpu_dependency_restore.py"), "--output", linux(folder), "--mode", mode]
                code = subprocess.run(command).returncode
                whole = time.perf_counter() - started
                atomic_json(folder.with_name(mode + "_ACTUAL_GUARD_EXIT.json"), {"returncode": code})
                if code != 0:
                    raise RuntimeError("The original complete restore or outer guard failed")
                result, resource = read(folder / "RESULT.json"), read(folder / "WSL_RESOURCE.json")
                if not all(result["checks"].values()) or resource["status"] != "completed":
                    raise AssertionError("An original complete restore or resource protection failed")
                row = {"trial": trial, "mode": mode, "whole_complete_entry_seconds": whole,
                    "folder": str(folder), "peak_CUDA_reserved_bytes": result["peak_CUDA_reserved_bytes"]}
                rows.append(row)
                atomic_json(OWN / "STATE.json", {"status": "running", "rows": rows, "new_optimization_count": 0})
                print(json.dumps(row), flush=True)
            left, right = (OWN / f"t{trial}" / mode for mode in ("incumbent", "bootstrap"))
            checks = compare(left, right)
            values = {row["mode"]: row["whole_complete_entry_seconds"] for row in rows if row["trial"] == trial}
            ratio = values["bootstrap"] / values["incumbent"]
            checks["each_raw_whole_ratio_at_most1_1"] = ratio <= 1.1
            if trial == 0:
                checks["first_raw_whole_screen_at_least3percent"] = ratio <= .97
            pairs.append({"trial": trial, "checks": checks, "whole_ratio": ratio})
            if not all(checks.values()):
                raise AssertionError("A registered first-screen or original full scientific protection failed")
        medians = {mode: statistics.median(row["whole_complete_entry_seconds"] for row in rows if row["mode"] == mode)
            for mode in ("incumbent", "bootstrap")}
        checks = {"all_three_pairs_original_complete_outputs_exact": len(pairs) == 3 and all(all(pair["checks"].values()) for pair in pairs),
            "raw_complete_entry_median_reduction_at_least3percent": 1 - medians["bootstrap"] / medians["incumbent"] >= .03}
        result = {"status": "three_raw_complete_pairs_passed_acceptance_pending" if all(checks.values()) else "closed_negative_raw_median",
            "checks": checks, "rows": rows, "pairs": pairs, "raw_median_seconds": medians,
            "raw_complete_entry_reduction": 1 - medians["bootstrap"] / medians["incumbent"],
            "actual_controller_seconds": time.perf_counter() - STARTED, "new_optimization_count": 0,
            "full_net_cost_recovery_public_component_and_root_acceptance_executed": False}
        atomic_json(OWN / "RESULT.json", result)
        print(json.dumps({key: result[key] for key in ("status", "checks", "raw_median_seconds", "raw_complete_entry_reduction")}), flush=True)
        if not all(checks.values()):
            raise SystemExit(17)
    except BaseException as error:
        atomic_json(OWN / "FAILURE.json", {"type": type(error).__name__, "error": str(error), "rows": rows, "pairs": pairs,
            "actual_controller_seconds": time.perf_counter() - STARTED, "new_optimization_count": 0})
        raise


if __name__ == "__main__":
    main()
