"""Compare one complete cold dataset pair before any full-export promotion."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import psutil
import pycolmap

from dataset_sampling_carrier import cached_dataset, closed, ROOT, OWN
from probe_dataset_sampling_route import EXPORT, IMAGES, LIBRARY
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def leaf(mode, output):
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    process = psutil.Process()
    atomic_json(output / "PROCESS_IDENTITY.json", {"pid": process.pid, "creation_time": process.create_time()})
    original = read(EXPORT / "TRAINING_ADMISSION.json")
    model = Path(original["training_opencv_model"])
    reconstruction = pycolmap.Reconstruction(model)
    names = sorted(image.name for image in reconstruction.images.values() if image.has_pose)
    if (len(names), reconstruction.num_points3D(), reconstruction.compute_num_observations()) != (1016, 75767, 959120):
        raise ValueError("The complete original dataset denominator changed")
    sources = [Path(__file__), Path(__file__).with_name("dataset_sampling_carrier.py"),
               Path(__file__).with_name("complete_carrier_configuration.py"), Path(__file__).with_name("native_camera_sampling.py"),
               Path(__file__).with_name("camera_pixel_remap.py"), Path(__file__).with_name("probe_dataset_sampling_route.py"),
               Path(__file__).with_name("native_camera_sampling.cpp"),
               ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005/runtime.py", LIBRARY]
    inputs = [EXPORT / "TRAINING_ADMISSION.json", *sorted(model.glob("*.bin")), *[IMAGES / name for name in names]]
    source_ids, input_ids = [data_identity(path) for path in sources], [data_identity(path) for path in inputs]
    atomic_json(output / "CONTRACT.json", {"mode": mode, "sources": source_ids, "inputs": input_ids,
                "all_complete1016_images": True, "cold_owned_output": True, "new_optimization_count": 0})
    snapshots = output / "source_snapshot"
    snapshots.mkdir()
    for index, path in enumerate(sources):
        if path.suffix in (".py", ".cpp"): shutil.copyfile(path, snapshots / (f"{index:02d}_" + path.name))
    before = time.perf_counter()
    if mode == "native":
        pycolmap.undistort_images(output / "dataset", model, IMAGES, image_names=names,
                output_type="COLMAP", num_patch_match_src_images=0, jpeg_quality=97, num_threads=8)
        trace = {"route": "original_complete_native_dataset", "closed": True}
    else:
        trace = cached_dataset(output / "dataset", model, IMAGES, names, LIBRARY, output / "sampling_proof")
    operation_seconds = time.perf_counter() - before
    reread = pycolmap.Reconstruction(output / "dataset/sparse")
    checks = {"all1016_75767_959120_native_output": (reread.num_reg_images(), reread.num_points3D(), reread.compute_num_observations()) == (1016, 75767, 959120),
              "all1016_expected_images": sorted(path.name for path in (output / "dataset/images").iterdir() if path.is_file()) == names,
              "all_input_and_source_identities_unchanged": source_ids == [data_identity(path) for path in sources]
              and input_ids == [data_identity(path) for path in inputs], "all_owned_workers_closed": trace["closed"],
              "actual_cached_route_selected": mode == "native" or trace["route"] == "native_full_model_carrier_cached_four_process_RGB_sampling",
              "disk_output_budget": sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) < 2 * 1024**3 and shutil.disk_usage(ROOT).free >= 20 * 1024**3}
    if not all(checks.values()): raise AssertionError("An original complete dataset protection failed")
    atomic_json(output / "RESULT.json", {"status": "one_complete_dataset_side_checked_not_acceptance", "mode": mode,
                "checks": checks, "operation_seconds": operation_seconds, "complete_side_before_receipt_seconds": time.perf_counter() - started,
                "trace": trace, "new_optimization_count": 0})


def campaign(output):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for mode in ("native", "cached"):
        side = output / mode
        command = [sys.executable, "-X", "utf8", str(Path(__file__)), "--mode", mode, "--output", str(side)]
        started = time.perf_counter()
        with (output / (mode + ".log")).open("xb") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            owner = {"pid": child.pid, "creation_time": psutil.Process(child.pid).create_time()}
            atomic_json(output / (mode + "_OWNER.json"), owner)
            code = child.wait()
        elapsed = time.perf_counter() - started
        if code: raise RuntimeError("A complete dataset side failed; its log and partial outputs are preserved")
        result = read(side / "RESULT.json")
        rows.append({"mode": mode, "actual_process_whole_seconds": elapsed, "owner": owner, "result": result})
    before = time.perf_counter()
    sides = [output / row["mode"] / "dataset" for row in rows]
    members = [{p.relative_to(side).as_posix(): p for p in side.rglob("*") if p.is_file()} for side in sides]
    if members[0].keys() != members[1].keys(): raise AssertionError("Complete native dataset file membership differs")
    total = 0
    for name in sorted(members[0]):
        if members[0][name].read_bytes() != members[1][name].read_bytes():
            raise AssertionError("Complete native dataset bytes differ: " + name)
        total += members[0][name].stat().st_size
    compare_seconds = time.perf_counter() - before
    checks = {"all_complete_native_dataset_bytes_exact": True,
              "all_two_actual_side_instances_closed": all(closed(row["owner"]) for row in rows),
              "all_original_side_checks_passed": all(all(row["result"]["checks"].values()) for row in rows),
              "entire_owned_dataset_diagnostic_disk_budget": sum(path.stat().st_size for path in OWN.rglob("*") if path.is_file()) <= 30 * 1024**3}
    if not all(checks.values()): raise AssertionError("A complete dataset campaign protection failed")
    whole_ratio = rows[1]["actual_process_whole_seconds"] / rows[0]["actual_process_whole_seconds"]
    stage_ratio = rows[1]["result"]["operation_seconds"] / rows[0]["result"]["operation_seconds"]
    result = {"status": "complete_first_dataset_pair_cost_guarded_not_formal_or_full1050_export", "checks": checks, "rows": rows,
              "complete_file_pairs": len(members[0]), "complete_byte_pairs_bytes": total,
              "full_dataset_comparison_seconds": compare_seconds, "stage_ratio_before_family_fees": stage_ratio,
              "whole_ratio_before_family_fees": whole_ratio,
              "continue_to_full1050_only_if_stage_at_least5percent_and_whole_at_most1point1": stage_ratio <= .95 and whole_ratio <= 1.1,
              "new_optimization_count": 0}
    atomic_json(output / "RESULT.json", result)
    print(json.dumps({k:result[k] for k in ("status", "complete_file_pairs", "stage_ratio_before_family_fees", "whole_ratio_before_family_fees", "continue_to_full1050_only_if_stage_at_least5percent_and_whole_at_most1point1", "new_optimization_count")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("campaign", "native", "cached"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.relative_to(OWN.resolve())
    started = time.perf_counter()
    try:
        if args.mode == "campaign": campaign(args.output)
        else: leaf(args.mode, args.output)
    except BaseException as error:
        if args.output.is_dir() and not (args.output / "FAILURE.json").exists():
            atomic_json(args.output / "FAILURE.json", {"status": "failed_diagnostic_partial_outputs_preserved",
                "mode": args.mode, "error_type": type(error).__name__, "error": str(error),
                "traceback": traceback.format_exc(), "elapsed_seconds_before_failure_receipt": time.perf_counter() - started,
                "new_optimization_count": 0})
        raise


if __name__ == "__main__":
    main()
