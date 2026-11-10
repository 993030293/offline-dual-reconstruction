"""Run the original complete export with one whole-map support execution change."""

import time
STARTED = time.perf_counter()

import argparse
import json
from pathlib import Path
import shutil
import sys

from profile_grid_support import original_metric_module
from profile_sparse_ply_oracle import ROOT, V1, V1OUT, native
from grid_support import grid_support_execution
from segmented_export_execution import segmented_export_execution
from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "segmented"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/sg001")
    args.output.mkdir(parents=True, exist_ok=False)
    module, common = original_metric_module()
    code = Path(__file__).parent
    ba = V1OUT / "new_map_metric_BA_001"
    roles = V1OUT / "new_map_roles_postBA_001/TRAINING_ROLES.json"
    images = V1OUT / "e2_geometry_support_001/images"
    sources = [Path(__file__), code/"segmented_export_execution.py", code/"segmented_camera_support.py",
        code/"grid_support.py", code/"native_observation_index.py", code/"binary_map_qa.py",
        code/"profile_grid_support.py", code/"profile_sparse_ply_oracle.py", code/"sparse_ply_oracle.py",
        code/"machine_json.py", Path(native.__file__), V1/"run_metric_refinement.py", V1/"runtime.py",
        Path(module.__file__), Path(common.__file__), common.PACKAGE/"config/METRIC_ALIGNMENT_POLICY.json"]
    inputs = [ba/"RESULT.json", roles, *sorted((ba/"weak_position_prior/model").glob("*.bin")),
              *sorted(path for path in images.iterdir() if path.is_file())]
    source_ids, input_ids = [data_identity(path) for path in sources], [data_identity(path) for path in inputs]
    saved = args.output / "source_snapshot"
    saved.mkdir()
    for ordinal, path in enumerate(sources):
        shutil.copyfile(path, saved / (f"{ordinal:03d}_" + path.name))
    atomic_json(args.output/"CONTRACT.json", {"mode": args.mode, "sources": source_ids, "inputs": input_ids,
        "common_R112": True, "all_original_camera_image_point_denominators": True,
        "all_snapshot_build_verify_and_input_source_costs_in_phase": True, "new_optimization_count": 0})
    setup_seconds = time.perf_counter()-STARTED
    previous, main_function = sys.argv, native.main
    writer, json_writer = native.write_initial_ply, native.atomic_json
    metric = module.camera_geometry_metrics
    try:
        with grid_support_execution(module, True) as grid:
            with segmented_export_execution(native, args.output/"support_states", args.mode == "segmented") as trace:
                sys.argv = [str(Path(native.__file__)), "--ba", str(ba), "--roles", str(roles),
                            "--images", str(images), "--output", str(args.output/"export")]
                before = time.perf_counter()
                native.main()
                main_seconds = time.perf_counter()-before
        sys.argv = previous
        closing = time.perf_counter()
        admission = json.loads((args.output/"export/TRAINING_ADMISSION.json").read_text(encoding="utf-8"))
        files = [path for path in (args.output/"export").rglob("*") if path.is_file()]
        expected_stages = ["before", "after", "final"]
        counts = [admission["full_registered_denominator"], admission["training_count"], admission["training_count"]]
        checks = {
            "original_export_and_admission_verified": admission["status"] == "export_and_admission_verified_training_pending",
            "all1021_1016_75767_959120_denominators": admission["full_registered_denominator"] == 1021
                and admission["training_count"] == 1016 and admission["training_points"] == 75767
                and admission["training_observations"] == 959120,
            "all3053_supports_three_separate_states": [stage["stage"] for stage in trace["stages"]] == expected_stages
                and [stage["images"] for stage in trace["stages"]] == counts and sum(counts) == 3053,
            "three_support_stages_completed": all(stage["status"] == "complete_original_camera_support_fields" for stage in trace["stages"]),
            "actual_routes_and_state_guards": all(stage["details"]["route"] == "whole_map_segmented_support"
                and stage["details"]["index"]["complete_native_state_unchanged"] for stage in trace["stages"])
                if args.mode == "segmented" else len(grid["calls"]) == 3053,
            "all_original1050_outputs_four_scientific_JSON": len(files) == 1050
                and sum(path.suffix == ".json" for path in files) == 4,
            "original_writers_hooks_argv_restored": native.main is main_function and trace["original_main_restored"]
                and native.write_initial_ply is writer and native.atomic_json is json_writer
                and module.camera_geometry_metrics is metric and grid["original_binding_restored"] and sys.argv is previous,
            "sources_all_original_inputs_unchanged": source_ids == [data_identity(path) for path in sources]
                and input_ids == [data_identity(path) for path in inputs],
            "all_owned_output_and_free_space_bounds": sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file()) <= 30*1024**3
                and shutil.disk_usage(args.output).free >= 20*1024**3,
        }
        closing_seconds = time.perf_counter()-closing
        phase = setup_seconds + sum(stage["complete_seconds"] for stage in trace["stages"])
        phase += grid["setup_seconds"] + grid["restore_seconds"] + trace["setup_seconds"] + trace["restore_seconds"] + closing_seconds
        result = {"status": "complete_original_export_segmented_support_guarded" if all(checks.values()) else "failed_original_export_guard",
            "mode": args.mode, "checks": checks, "sources": source_ids, "inputs": input_ids, "admission": admission,
            "trace": trace, "source_input_setup_seconds": setup_seconds, "close_verify_seconds": closing_seconds,
            "complete_affected_phase_seconds": phase, "original_main_seconds": main_seconds,
            "whole_before_result_seconds": time.perf_counter()-STARTED, "new_optimization_count": 0}
        atomic_json(args.output/"RESULT.json", result)
        print(json.dumps({"status": result["status"], "checks": checks, "phase_seconds": phase,
                          "whole_before_result_seconds": result["whole_before_result_seconds"]}))
        if not all(checks.values()):
            raise AssertionError("A required complete original export protection failed")
    except BaseException as error:
        atomic_json(args.output/"FAILURE.json", {"status": "failed_complete_segmented_export_no_acceptance",
            "error": repr(error), "sources": source_ids, "inputs": input_ids,
            "complete_seconds": time.perf_counter()-STARTED, "new_optimization_count": 0})
        raise
    finally:
        sys.argv = previous
        atomic_json(args.output/"BINDING_RESTORATION.json", {"native_main_restored": native.main is main_function,
            "argv_restored": sys.argv is previous, "metric_restored": module.camera_geometry_metrics is metric,
            "original_writers_restored": native.write_initial_ply is writer and native.atomic_json is json_writer,
            "new_optimization_count": 0})


if __name__ == "__main__":
    main()
