"""Run fresh alternating complete saves without replacing a failed first pair."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/hc001"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity

MODEL = ROOT / "04_outputs/gaussian_pipeline_optimization_v2_20261005/camera_quality_training_001/seed0/combined/model/point_cloud/iteration_12000/point_cloud.ply"
NATIVE = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/dp001/p0/independent_unmodified_native.ply"
RUNTIME = ROOT / "04_outputs/box2_3dgs_display_v1_20260930"


def mounted(path):
    path = Path(path).resolve()
    return "/mnt/c/" + str(path.relative_to(Path("C:/"))).replace("\\", "/")


def file_exact(left, right):
    with Path(left).open("rb") as a, Path(right).open("rb") as b:
        while True:
            x, y = a.read(1024**2), b.read(1024**2)
            if x != y:
                return False
            if not x:
                return True


def process_gone(row):
    try:
        return psutil.Process(row["pid"]).create_time() != row["creation_time"]
    except psutil.NoSuchProcess:
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(OUT.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    generic = json.loads((OUT / "g0/HOST_GENERIC_RESULT.json").read_text(encoding="utf-8"))
    if generic["status"] != "host_generic_passed" or not all(generic["checks"].values()):
        raise RuntimeError("The unchanged required generic must pass before real complete pairs")
    paths = [Path(__file__), CODE / "native_host_gaussian_files.py", CODE / "native_host_file_worker.py",
        CODE / "profile_host_gaussian_save.py", CODE / "streamed_gaussian_ply_oracle_002.py",
        CODE / "profile_streamed_gaussian_ply_002.py", CODE / "mapped_image_linux_guard.py",
        CODE / "run_bounded_journal_task.py", CODE / "run_with_resource_lease.py", MODEL, NATIVE]
    identities = [data_identity(path) for path in paths]
    atomic_json(args.output / "CONTRACT.json", {
        "sources_inputs": identities, "orders": [["cpu", "host"], ["host", "cpu"], ["cpu", "host"]],
        "stop_if_first_complete_pair_both_phase_and_whole_are_worse": True,
        "whole_each_pair_maximum_ratio": 1.1, "speed_gate": .05, "counted": False})
    rows, pairs = [], []
    for index, order in enumerate((("cpu", "host"), ("host", "cpu"), ("cpu", "host"))):
        for mode in order:
            output = args.output / str(index) / mode
            receipt = args.output / f"{index}_{mode}_GUARD.json"
            command = [sys.executable, "-B", str(CODE / "run_bounded_journal_task.py"),
                "--receipt", str(receipt), "--log", str(args.output / f"{index}_{mode}.log"),
                "--maximum-seconds", "1750", "--maximum-rss-gib", "8", "--", "wsl", "-d",
                "E26-C2-LIV-20260907-134802", "/opt/e26_thin_surface_v1/venv/bin/python", "-B",
                mounted(CODE / "profile_host_gaussian_save.py"), "--mode", mode,
                "--runtime", mounted(RUNTIME), "--model", mounted(MODEL),
                "--native-proof", mounted(NATIVE), "--output", mounted(output)]
            before = time.perf_counter()
            if identities != [data_identity(path) for path in paths]:
                raise RuntimeError("A frozen complete save input changed before execution")
            subprocess.run(command, cwd=ROOT, check=True)
            result = json.loads((output / "RESULT.json").read_text(encoding="utf-8"))
            guard = json.loads(receipt.read_text(encoding="utf-8"))
            linux = json.loads((output / "WSL_RESOURCE.json").read_text(encoding="utf-8"))
            if result["status"] != "complete_host_save_verified" or not all(result["checks"].values()):
                raise AssertionError("A complete save contract failed")
            if guard["status"] != "completed" or linux["status"] != "completed":
                raise AssertionError("An actual resource guard failed")
            if not all(process_gone(row) for row in result["host_events"]):
                raise AssertionError("An actual Windows file worker remains alive")
            if identities != [data_identity(path) for path in paths]:
                raise RuntimeError("A complete input changed after execution")
            row = {"pair": index, "mode": mode, "complete_phase_seconds": result["complete_phase_seconds"],
                "complete_whole_seconds": time.perf_counter() - before, "result": str(output / "RESULT.json"),
                "guard": str(receipt), "actual_host_instances_gone": True}
            rows.append(row)
            atomic_json(args.output / "PARTIAL.json", {"rows": rows, "counted": False})
        compare_started = time.perf_counter()
        equal = all(file_exact(args.output / str(index) / "cpu" / name,
            args.output / str(index) / "host" / name) for name in ("point_cloud.ply", "native_oracle.ply"))
        comparison = time.perf_counter() - compare_started
        if not equal:
            raise AssertionError("Both complete independent physical PLY artifacts must be byte identical")
        pair_rows = {row["mode"]: row for row in rows if row["pair"] == index}
        ratio = pair_rows["host"]["complete_whole_seconds"] / pair_rows["cpu"]["complete_whole_seconds"]
        pair = {"pair": index, "complete_two_file_bytes_equal": equal, "comparison_seconds": comparison,
            "whole_ratio": ratio, "within_original_whole_cap": ratio <= 1.1}
        pairs.append(pair)
        if not pair["within_original_whole_cap"]:
            break
        if index == 0 and all(pair_rows["host"][key] > pair_rows["cpu"][key]
                for key in ("complete_phase_seconds", "complete_whole_seconds")):
            break
    first = sum(json.loads(path.read_text(encoding="utf-8"))["child_elapsed_seconds"] for path in (
        ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/dp001/PROFILE_LEASE_001.json",
        OUT / "GENERIC_LEASE_001.json"))
    raw = {mode: {key: statistics.median(row[key] for row in rows if row["mode"] == mode)
        for key in ("complete_phase_seconds", "complete_whole_seconds")} for mode in ("cpu", "host")}
    result = {"status": "three_complete_pairs_outputs_verified_pending_required_recovery_and_review"
        if len(pairs) == 3 and all(pair["within_original_whole_cap"] for pair in pairs)
        else "first_cost_negative_stop_no_acceptance", "rows": rows, "pairs": pairs,
        "raw_medians_not_acceptance": raw, "required_first_diagnostic_seconds": first,
        "measured_parent_before_final_publish_seconds": time.perf_counter() - started,
        "complete_source_input_unchanged": identities == [data_identity(path) for path in paths],
        "new_optimization_count": 0, "counted": False}
    atomic_json(args.output / "RESULT.json", result)
    print(json.dumps({key: result[key] for key in ("status", "raw_medians_not_acceptance", "required_first_diagnostic_seconds")}), flush=True)


if __name__ == "__main__":
    main()
