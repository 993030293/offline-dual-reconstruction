"""Keep the cold first full pair and stop on any registered graph-screen failure."""

import time
STARTED = time.perf_counter()

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

import psutil

from run_interior_conv_campaign import prepare, identity_set, read, linux, windows
from runtime import atomic_json, data_identity

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/eg001"
LOCALIZATION = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/p27_newmap_localization_lazy_serial_001/H_LOCALIZATION.json"


def compare(first, second):
    left, right = (read(folder / "pair/evaluation/EVALUATION.json") for folder in (first, second))
    error, exact = 0.0, True
    for label in ("baseline", "combined"):
        a, b = left["results"][label]["rows"], right["results"][label]["rows"]
        if len(a) != 24 or len(b) != 24:
            raise ValueError("The original 24-view denominator changed")
        for original, actual in zip(a, b):
            one, two = dict(original), dict(actual)
            for region in ("full_image", "target"):
                x, y = dict(one.pop(region)), dict(two.pop(region))
                error = max(error, abs(x.pop("SSIM") - y.pop("SSIM")))
                exact = exact and x == y
            exact = exact and one == two
    return {"all48_PSNR_MSE_and_metadata_exact": exact, "all96_SSIM_original_tolerance": error <= 1e-7,
            "maximum_absolute_SSIM_error": error,
            "original_full_camera_and_photo_contract_exact": read(first / "pair/evaluation/EVALUATION_CONTRACT.json")
                == read(second / "pair/evaluation/EVALUATION_CONTRACT.json"),
            "complete_JPEG_bytes_exact": (first / "pair/evaluation/SIX_VIEW_COMPARISON.jpg").read_bytes()
                == (second / "pair/evaluation/SIX_VIEW_COMPARISON.jpg").read_bytes()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-name", choices=("s0", "s1"), default="s0")
    args = parser.parse_args()
    output = OWN / args.output_name
    output.mkdir(parents=True, exist_ok=False)
    process = psutil.Process()
    atomic_json(output / "PROCESS_IDENTITY.json", {"pid": process.pid, "creation_time": process.create_time()})
    generic = read(OWN / "g0/RESULT.json")
    if not all(generic["checks"].values()) or generic["status"] != "generic_passed":
        raise ValueError("The actual graph generic failed")
    for row in generic["sources"]:
        if row["path"].startswith("/mnt/c/"):
            if data_identity(windows(row["path"]))["sha256"] != row["sha256"]:
                raise ValueError("A graph generic-tested workspace source changed")
        elif row["path"] != "/opt/e26_thin_surface_v1/venv/lib/python3.11/site-packages/torch/cuda/graphs.py":
            raise ValueError("An unexpected native dependency requires its own runtime verifier")
    paths, inputs = identity_set()
    paths.extend([Path(__file__), CODE / "evaluate_graph_reference_ssim.py", CODE / "graph_reference_ssim.py"])
    paths = list(dict.fromkeys(paths))
    sources, identities = [data_identity(path) for path in paths], [data_identity(path) for path in inputs]
    copies = output / "source_copies"
    copies.mkdir()
    for index, path in enumerate(paths):
        shutil.copyfile(path, copies / (f"{index:02d}_" + path.name))
    atomic_json(output / "CONTRACT.json", {"sources": sources, "inputs": identities,
        "common_Q30_Q61_R100_R135_R134": True, "minimum_raw_screen_stage_reduction": .03,
        "each_whole_ratio_limit": 1.1, "new_optimization_count": 0, "old_Q65_same_unaccepted_family": True})
    rows = []
    try:
        for mode in ("incumbent", "graph"):
            before = time.perf_counter()
            if shutil.disk_usage(OWN).free < 20 * 1024**3 or time.perf_counter() - STARTED > 300:
                raise RuntimeError("The original disk or time reserve is unavailable")
            if sources != [data_identity(path) for path in paths] or identities != [data_identity(path) for path in inputs]:
                raise ValueError("A current source or complete original input changed before consumption")
            folder = output / mode
            pair = prepare(folder)
            command = [sys.executable, "-B", str(CODE / "run_bounded_journal_task.py"),
                "--receipt", str(folder / "WINDOWS_GUARD.json"), "--log", str(folder / "EVALUATION.log"),
                "--maximum-seconds", "300", "--maximum-rss-gib", "8", "--", "wsl", "-d",
                "E26-C2-LIV-20260907-134802", "--", "/opt/e26_thin_surface_v1/venv/bin/python", "-B",
                linux(CODE / "evaluate_graph_reference_ssim.py"), "--mode", mode, "--pair", linux(pair),
                "--localization", linux(LOCALIZATION)]
            code = subprocess.run(command).returncode
            atomic_json(folder / "ACTUAL_GUARD_EXIT.json", {"returncode": code})
            if code != 0:
                raise RuntimeError("The actual complete graph evaluator or guard failed")
            guard, measured, graph = (read(folder / name) for name in
                ("WINDOWS_GUARD.json", "measurement/RESULT.json", "GRAPH_RESULT.json"))
            resource = read(folder / "measurement/WSL_RESOURCE.json")
            if (guard["returncode"] != 0 or resource["status"] != "completed"
                    or not all(measured["checks"].values()) or not all(graph["checks"].values())):
                raise AssertionError("An original quality, input, resource or lifetime guard failed")
            if sources != [data_identity(path) for path in paths] or identities != [data_identity(path) for path in inputs]:
                raise ValueError("A graph full evaluation changed a complete original input or source")
            row = {"mode": mode, "whole_seconds": time.perf_counter() - before,
                "complete_SSIM_stage_seconds": measured["complete_SSIM_seconds_including_input_protection"]
                    + measured["composition_source_copy_identity_seconds"] + graph["close_seconds_all_actual_cache_calls"]
                    + graph["actual_native_dependency_validation"]["seconds"],
                "peak_CUDA_reserved_bytes": measured["peak_CUDA_reserved_bytes"],
                "cache_state": measured["cache_before_clear"], "folder": str(folder)}
            rows.append(row)
            atomic_json(output / "STATE.json", {"status": "running", "rows": rows, "new_optimization_count": 0})
        comparison = compare(output / "incumbent", output / "graph")
        flags = {key: value for key, value in comparison.items() if key != "maximum_absolute_SSIM_error"}
        reduction = 1 - rows[1]["complete_SSIM_stage_seconds"] / rows[0]["complete_SSIM_stage_seconds"]
        ratio = rows[1]["whole_seconds"] / rows[0]["whole_seconds"]
        flags.update(raw_stage_reduction_at_least3percent=reduction >= .03, whole_ratio_at_most1_1=ratio <= 1.1)
        result = {"status": "first_pair_passed_acceptance_pending" if all(flags.values()) else "closed_negative_first_raw_screen",
            "checks": flags, "comparison": comparison, "rows": rows, "raw_stage_reduction": reduction,
            "whole_ratio": ratio, "actual_controller_seconds": time.perf_counter() - STARTED,
            "new_optimization_count": 0, "formal_remaining_pairs_and_full_recovery_executed": False,
            "old_Q65_and_new_shared_fees_not_used_to_rescue_raw_failure": True}
        atomic_json(output / "RESULT.json", result)
        print(json.dumps({key: result[key] for key in ("status", "checks", "raw_stage_reduction", "whole_ratio")}), flush=True)
        if not all(flags.values()):
            raise SystemExit(17)
    except BaseException as error:
        atomic_json(output / "FAILURE.json", {"type": type(error).__name__, "message": str(error),
            "rows": rows, "actual_controller_seconds": time.perf_counter() - STARTED, "new_optimization_count": 0})
        raise


if __name__ == "__main__":
    main()
