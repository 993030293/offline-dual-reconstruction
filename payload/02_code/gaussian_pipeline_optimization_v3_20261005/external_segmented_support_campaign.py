"""Compare fixed support execution on all existing Oxford cameras in fresh processes."""

import time
STARTED = time.perf_counter()

import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess

import pycolmap

from profile_grid_support import ROOT, original_metric_module
from grid_support import grid_support_execution
from native_observation_index import snapshot
from segmented_camera_support import native_support_many
from machine_json import exact_json_values
from runtime import atomic_json, data_identity

CODE = Path(__file__).parent
OWNED = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/sg001"
MODEL = ROOT/"04_outputs/public_full_e26_scene_v3_20260929T141800Z/attempt02/01_visual_workspace/sparse/0"
PYTHON = ROOT/"05_envs/sfm_py311_v1/Scripts/python.exe"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def leaf(args):
    args.output.mkdir(parents=True, exist_ok=False)
    module, common = original_metric_module()
    metric = module.camera_geometry_metrics
    policy_path = common.PACKAGE/"config/METRIC_ALIGNMENT_POLICY.json"
    sources = [Path(__file__), CODE/"segmented_camera_support.py", CODE/"native_observation_index.py",
               CODE/"binary_map_qa.py", CODE/"grid_support.py", Path(module.__file__), policy_path]
    source_ids = [data_identity(path) for path in sources]
    inputs = [data_identity(path) for path in sorted(MODEL.glob("*.bin"))]
    source_copy = args.output/"source_snapshot"
    source_copy.mkdir()
    for ordinal, path in enumerate(sources):
        shutil.copyfile(path, source_copy/(f"{ordinal:03d}_"+path.name))
    policy = read(policy_path)["camera_geometry_admission"]
    model = pycolmap.Reconstruction(MODEL)
    images = [image for image in model.images.values() if image.has_pose]
    if len(images) != 360 or model.num_points3D() != 133020 or model.compute_num_observations() != 848219:
        raise ValueError("The complete public SfM component denominator changed")
    before = snapshot(model, args.output/"owner_before")
    setup_seconds = time.perf_counter()-STARTED
    with grid_support_execution(module, True) as grid:
        started = time.perf_counter()
        if args.leaf_mode == "native":
            rows = [module.camera_geometry_metrics(model, image, policy) for image in images]
            trace = {"route": "original_scalar_with_R112", "closed": True}
        else:
            rows, trace = native_support_many(model, images, policy, args.output/"batch_state", module.camera_geometry_metrics)
        working_seconds = time.perf_counter()-started
    closing = time.perf_counter()
    after = snapshot(model, args.output/"owner_after")
    checks = {
        "all360_complete_original_fields_and_order": len(rows) == 360
            and [row["image_name"] for row in rows] == [image.name for image in images],
        "complete_native_model_unchanged": before == after,
        "all_source_and_public_model_files_unchanged": source_ids == [data_identity(path) for path in sources]
            and inputs == [data_identity(path) for path in sorted(MODEL.glob("*.bin"))],
        "actual_supported_or_preselected_original_route": trace["route"] in ("whole_map_segmented_support", "native_before_execution")
            if args.leaf_mode == "segmented" else len(grid["calls"]) == 360,
        "private_original_binding_restored": module.camera_geometry_metrics is metric and grid["original_binding_restored"],
        "helper_closed_and_serialized_state_verified": trace["closed"]
            and (trace.get("index") is None or trace["index"]["complete_native_state_unchanged"]),
        "owned_output_and_disk_headroom": sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file()) <= 30*1024**3
            and shutil.disk_usage(args.output).free >= 20*1024**3,
    }
    closing_seconds = time.perf_counter()-closing
    result = {"status": "complete_Oxford360_support_component_guarded" if all(checks.values()) else "failed_external_component_guard",
        "checks": checks, "mode": args.leaf_mode, "rows": rows, "sources": source_ids, "inputs": inputs,
        "before": before, "after": after, "trace": trace, "working_seconds": working_seconds,
        "complete_phase_seconds": setup_seconds+grid["setup_seconds"]+working_seconds+grid["restore_seconds"]+closing_seconds,
        "whole_before_result_seconds": time.perf_counter()-STARTED,
        "independent_GT_metric_pose_or_new_object_training_claim": False, "new_optimization_count": 0}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps({"status": result["status"], "checks": checks, "route": trace["route"]}))
    if not all(checks.values()):
        raise AssertionError("A required fixed-code external component protection failed")


def campaign(args):
    args.output.mkdir(parents=True, exist_ok=False)
    state = {"status": "three_complete360_component_pairs_registered", "sides": 0, "new_optimization_count": 0}
    atomic_json(args.output/"STATE.json", state)
    pairs = []
    try:
        for ordinal, order in enumerate((("native", "segmented"), ("segmented", "native"), ("native", "segmented"))):
            pair = args.output/f"t{ordinal}"
            pair.mkdir()
            nodes = {}
            for mode in order:
                command = [str(PYTHON), "-B", str(CODE/"run_bounded_journal_task.py"), "--receipt", str(pair/f"{mode}_guard.json"),
                    "--log", str(pair/f"{mode}.log"), "--maximum-seconds", "1750", "--maximum-rss-gib", "8", "--",
                    str(PYTHON), "-B", str(Path(__file__)), "--leaf-mode", mode, "--output", str(pair/mode)]
                started = time.perf_counter()
                process = subprocess.run(command, cwd=ROOT, check=False)
                seconds = time.perf_counter()-started
                if process.returncode != 0:
                    raise RuntimeError("An external component failed; later sides were not started")
                result = read(pair/mode/"RESULT.json")
                if not all(result["checks"].values()):
                    raise AssertionError("An external component original guard failed")
                tail = seconds-result["whole_before_result_seconds"]
                if tail < 0:
                    raise ValueError("The complete external process clock is inconsistent")
                nodes[mode] = {"whole_seconds": seconds, "phase_seconds": result["complete_phase_seconds"]+tail,
                               "route": result["trace"]["route"]}
                state.update(status="running_full_external_pairs", sides=state["sides"]+1)
                atomic_json(args.output/"STATE.json", state)
            comparison_started = time.perf_counter()
            results = {mode: read(pair/mode/"RESULT.json") for mode in ("native", "segmented")}
            for key in ("rows", "sources", "inputs", "before", "after"):
                exact_json_values(results["native"][key], results["segmented"][key])
            row = {"pair": ordinal, "order": list(order), "nodes": nodes,
                   "all360_complete_fields_types_order_and_float_bits_exact": True,
                   "comparison_seconds": time.perf_counter()-comparison_started}
            pairs.append(row)
            atomic_json(pair/"PAIR.json", row)
        median = {scope: {mode: statistics.median(pair["nodes"][mode][scope] for pair in pairs)
                          for mode in ("native", "segmented")} for scope in ("whole_seconds", "phase_seconds")}
        result = {"status": "three_full360_external_component_pairs_exact", "pairs": pairs, "medians": median,
                  "complete_campaign_before_result_seconds": time.perf_counter()-STARTED,
                  "external_fees_charged_to_local_adoption": True,
                  "GT_metric_or_complete_new_object_quality_claim": False, "new_optimization_count": 0}
        atomic_json(args.output/"RESULT.json", result)
        state.update(status=result["status"], completed_pairs=3)
        atomic_json(args.output/"STATE.json", state)
        print(json.dumps({"status": result["status"], "medians": median}))
    except BaseException as error:
        state.update(status="failed_complete_external_component", error=repr(error))
        atomic_json(args.output/"FAILURE.json", {"state": state, "pairs_before_failure": pairs,
            "complete_seconds": time.perf_counter()-STARTED, "new_optimization_count": 0})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--leaf-mode", choices=("native", "segmented"))
    args = parser.parse_args()
    args.output.resolve().relative_to(OWNED)
    leaf(args) if args.leaf_mode else campaign(args)


if __name__ == "__main__":
    main()
