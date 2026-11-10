"""Measure unchanged grid support on the entire existing Oxford model."""

import time

IMPORT_STARTED = time.perf_counter()

import argparse
import gc
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

import pycolmap

from grid_support import grid_support_execution
from machine_json import exact_json_values
from profile_grid_support import ROOT, V1, native, original_metric_module
from runtime import atomic_json, data_identity


MODEL = ROOT/"04_outputs/public_full_e26_scene_v3_20260929T141800Z/attempt02/01_visual_workspace/sparse/0"
OWNED = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/grid_support_001"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_leaf(args):
    output = args.output.resolve()
    output.relative_to(OWNED)
    output.mkdir(exist_ok=False)
    contract = read(args.contract)
    identities = contract["sources_and_inputs"]
    paths = [Path(row["path"]) for row in identities]
    module, common = original_metric_module()
    if [data_identity(path) for path in paths] != identities:
        raise RuntimeError("Frozen external sources or complete model inputs changed")
    policy = read(common.PACKAGE/"config/METRIC_ALIGNMENT_POLICY.json")["camera_geometry_admission"]
    model = pycolmap.Reconstruction(MODEL)
    before = (model.num_images(), model.num_reg_images(), model.num_points3D())
    if before != (360, 360, 133020):
        raise AssertionError("The complete original Oxford denominator changed")
    with grid_support_execution(module, args.leaf_mode == "grid") as trace:
        results = [module.camera_geometry_metrics(model, image, policy)
                   for image in model.images.values()]
    after = (model.num_images(), model.num_reg_images(), model.num_points3D())
    observations = sum(row["triangulated_points"] for row in results)
    if (before != after or observations != 848219 or len(results) != 360
            or not trace["original_binding_restored"] or len(trace["calls"]) != 360
            or any(row["status"] != "passed_original_full_camera_support_function"
                   for row in trace["calls"])):
        raise AssertionError("Full original support membership or restoration failed")
    scientific = {"registered_images": before[1], "points": before[2],
                  "observations": observations, "full_image_support": results}
    atomic_json(output/"SUPPORT.json", scientific)
    exact_json_values(scientific, read(output/"SUPPORT.json"))
    del model
    gc.collect()
    if [data_identity(path) for path in paths] != identities:
        raise RuntimeError("Frozen sources or external model changed during the component")
    if (sum(path.stat().st_size for path in output.parent.rglob("*") if path.is_file()) > 30*1024**3
            or shutil.disk_usage(output).free < 20*1024**3):
        raise MemoryError("External output or disk headroom budget failed")
    result = {"status": "complete_external_original_support_and_guards_passed",
              "trace": trace, "registered_images": before[1], "points": before[2],
              "observations": observations,
              "complete_leaf_before_receipt_seconds": time.perf_counter()-IMPORT_STARTED,
              "counted": False}
    atomic_json(output.parent/f"{output.name}_leaf_RESULT.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--leaf-mode", choices=("native", "grid"))
    args = parser.parse_args()
    if args.leaf_mode:
        if args.contract is None:
            raise ValueError("A fresh external leaf requires the frozen contract")
        run_leaf(args)
        return
    output = args.output.resolve()
    output.relative_to(OWNED)
    output.mkdir(exist_ok=False)
    module, common = original_metric_module()
    source_names = ("grid_support.py", "profile_grid_support.py", "machine_json.py",
                    "profile_sparse_ply_oracle.py", "sparse_ply_oracle.py")
    sources = [Path(__file__).resolve(), *(Path(__file__).with_name(name).resolve() for name in source_names),
               Path(native.__file__).resolve(), V1/"runtime.py", V1/"run_metric_refinement.py",
               ROOT/"02_code/box2_lio_guided_sfm_v1_20261002/common.py", Path(module.__file__),
               common.PACKAGE/"config/METRIC_ALIGNMENT_POLICY.json"]
    inputs = sorted(MODEL.glob("*.bin"))
    if [path.name for path in inputs] != ["cameras.bin", "frames.bin", "images.bin", "points3D.bin", "rigs.bin"]:
        raise AssertionError("All five real native external input files are required")
    identities = [data_identity(path) for path in sources+inputs]
    snapshot = output/"source_snapshot"
    snapshot.mkdir()
    for index, path in enumerate(sources):
        shutil.copyfile(path, snapshot/f"{index:02d}_{path.name}")
    contract = output/"CONTRACT.json"
    atomic_json(contract, {"mechanism": "R112_unchanged_grid_support_external_component",
        "sources_and_inputs": identities, "source_count": len(sources), "pairs": 3,
        "complete_registered_images": 360, "complete_points": 133020, "complete_observations": 848219,
        "minimum_complete_component_reduction": .1, "maximum_each_whole_ratio": 1.1,
        "whole_median_nonregression_required": True, "first_preparation_divisor": 6,
        "cold_fresh_process_every_side": True, "all_source_and_input_bytes_checked_twice_every_leaf": True,
        "scientific_JSON_no_omissions_or_path_normalization": True,
        "cost_scope": "complete_model_read_all_original_support_full_guards_publication_restoration_process_close",
        "external_is_existing_development_component_not_GT_or_whole_pipeline_accuracy": True, "counted": False})
    first_prepare = time.perf_counter()-IMPORT_STARTED
    pairs = []
    try:
        for trial in range(3):
            rows = {}
            order = ("native", "grid") if trial%2 == 0 else ("grid", "native")
            for label in order:
                folder = output/f"pair{trial}_{label.ljust(6, '_')}"
                started = time.perf_counter()
                with (output/f"{folder.name}.log").open("xb") as log:
                    process = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                        "--output", str(folder), "--leaf-mode", label, "--contract", str(contract)],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW)
                if process.returncode:
                    raise RuntimeError(f"A complete external fresh process failed: {label}, {process.returncode}")
                leaf = read(output/f"{folder.name}_leaf_RESULT.json")
                if leaf["status"] != "complete_external_original_support_and_guards_passed":
                    raise AssertionError("The external source or scientific guards failed")
                whole = time.perf_counter()-started
                external = whole-leaf["complete_leaf_before_receipt_seconds"]
                if external < 0:
                    raise AssertionError("The parent envelope omitted child execution")
                rows[label] = {"whole_seconds": whole+first_prepare/6,
                    "phase_seconds": whole+first_prepare/6, "folder": str(folder),
                    "first_child_start_receipt_and_close_seconds": external,
                    "source_bound_leaf_result": str(output/f"{folder.name}_leaf_RESULT.json"),
                    "child_returncode": process.returncode}
            started = time.perf_counter()
            left, right = [Path(rows[label]["folder"])/"SUPPORT.json" for label in ("native", "grid")]
            exact_json_values(read(left), read(right))
            left_bytes, right_bytes = left.read_bytes(), right.read_bytes()
            if left_bytes != right_bytes:
                raise AssertionError("An entire original external support scientific output changed")
            comparison_fee = time.perf_counter()-started
            for row in rows.values():
                row["phase_seconds"] += comparison_fee
                row["whole_seconds"] += comparison_fee
            pairs.append({"trial": trial, "rows": rows, "complete_support_bytes": len(left_bytes),
                "all_fields_types_order_and_float_bits_exact": True,
                "complete_comparison_seconds_charged_each_side": comparison_fee})
            atomic_json(output/"STATE.json", {"status": "measuring", "pairs": pairs, "counted": False})
            if rows["grid"]["whole_seconds"] > 1.1*rows["native"]["whole_seconds"]:
                raise RuntimeError("The first required external whole per-pair cost gate failed")
        medians = {label: {field: statistics.median(pair["rows"][label][field] for pair in pairs)
                         for field in ("phase_seconds", "whole_seconds")} for label in ("native", "grid")}
        reduction = 1-medians["grid"]["phase_seconds"]/medians["native"]["phase_seconds"]
        checks = {"three_complete_fresh_process_pairs": len(pairs) == 3,
            "complete_component_reduction_at_least10pct": reduction >= .1,
            "each_whole_at_most1_1": all(pair["rows"]["grid"]["whole_seconds"] <=
                1.1*pair["rows"]["native"]["whole_seconds"] for pair in pairs),
            "whole_median_nonregressed": medians["grid"]["whole_seconds"] <= medians["native"]["whole_seconds"],
            "all_complete_original_support_bytes_exact": True,
            "all_frozen_source_and_full_external_inputs_unchanged": identities == [data_identity(path) for path in sources+inputs]}
        result = {"status": "external_complete_grid_component_guards_passed_pending_review" if all(checks.values())
            else "external_complete_grid_component_guard_failed", "checks": checks, "medians": medians,
            "complete_component_reduction": reduction, "pairs": pairs, "first_prepare_seconds": first_prepare,
            "actual_complete_campaign_seconds": time.perf_counter()-IMPORT_STARTED,
            "registered_images": 360, "points": 133020, "observations": 848219,
            "GT_or_TLS_inputs_used": False, "whole_pipeline_generalization_proven": False, "counted": False}
        atomic_json(output/"RESULT.json", result)
        print(json.dumps({"status": result["status"], "medians": medians, "reduction": reduction}))
    except BaseException as error:
        atomic_json(output/"FAILURE.json", {"error": repr(error), "pairs": pairs,
            "actual_complete_campaign_seconds": time.perf_counter()-IMPORT_STARTED,
            "failed_outputs_preserved": True, "counted": False})
        raise


if __name__ == "__main__":
    main()
