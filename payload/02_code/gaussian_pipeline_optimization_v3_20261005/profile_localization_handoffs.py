"""Time original full localization handoffs without changing numerical work."""

import time

STARTED = time.perf_counter()

import argparse
import ast
import json
from pathlib import Path
import shutil
import sys
import threading

import integer_sift_matching
import array_match_geometry
import localize_geometry_sift_fusion as entry
from run_sift_process_campaign_005 import compare_pair

import feature_localization
import fusion_owned_json_write
from runtime import atomic_json, data_identity


ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
OWN = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/olp001"
BASELINE = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/fusion_geometry_sift_001/c1/t0/c"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def scoped_writer(output):
    """Change only the existing private writer's allowed diagnostic root."""
    original = fusion_owned_json_write.COMPOSED_SOURCE
    before_path = "04_outputs/gaussian_pipeline_optimization_v3_20261005/fusion_geometry_sift_001"
    after_path = "04_outputs/gaussian_pipeline_optimization_v3_20261005/olp001"
    if original.count(before_path) != 1:
        raise AssertionError("The original private writer scope anchor changed")
    source = original.replace(before_path, after_path)
    before, after = ast.parse(original), ast.parse(source)
    fields = [node for node in ast.walk(before) if isinstance(node, ast.Constant) and node.value == before_path]
    if len(fields) != 1:
        raise AssertionError("The private writer ownership constant is ambiguous")
    fields[0].value = after_path
    if ast.dump(before) != ast.dump(after):
        raise AssertionError("The diagnostic changed writer behavior beyond its output scope")
    executed = output / "PRIVATE_SCOPED_WRITER.py"
    executed.write_text(source, encoding="utf-8")
    namespace = {"__file__": str(fusion_owned_json_write.SOURCE_PATH), "__name__": __name__ + ".private_writer"}
    exec(compile(source, str(fusion_owned_json_write.SOURCE_PATH), "exec"), namespace)
    atomic_json(output / "PRIVATE_WRITER_BINDING.json", {"actual_executed_source": data_identity(executed),
                "only_allowed_root_constant_changed_AST_exact": True, "new_optimization_count": 0})
    return namespace["owned_atomic_json"], source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(OWN)
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), CODE / "localization_handoff_node.py", Path(entry.__file__),
               CODE / "integer_sift_matching.py", CODE / "array_match_geometry.py",
               Path(feature_localization.__file__), CODE / "query_preparation_pipeline.py",
               CODE / "native_sift_process.py", CODE / "sift_process_node_003.py"]
    identities = [data_identity(path) for path in sources]
    copies = output / "executed_sources"
    copies.mkdir()
    for index, path in enumerate(sources):
        shutil.copyfile(path, copies / (str(index) + "_" + path.name))
    atomic_json(output / "CONTRACT.json", {"sources": identities, "input_scope": "unchanged_current_F001_process_full24_48",
                "only_observational_timing_wrappers": True, "baseline": str(BASELINE), "new_optimization_count": 0,
                "candidate_scheduler_implemented": False, "only_fixed_offline_queries_no_online_claim": True})
    rows = {"GPU_request": [], "CPU_geometry": [], "CPU_pose": []}
    original_batch = integer_sift_matching.DistanceWorker.batch
    original_geometry = array_match_geometry.ArrayMatchGeometry.__call__
    original_pose = feature_localization.estimate_pose
    original_writer, original_write_source = entry.owned_atomic_json, entry.COMPOSED_SOURCE
    original_scope, previous_argv = entry.OWNED, sys.argv

    def measured(label, operation, args, kwargs):
        start = time.perf_counter()
        ok = False
        try:
            result = operation(*args, **kwargs)
            ok = True
            return result
        finally:
            rows[label].append({"ordinal": len(rows[label]), "start": start, "end": time.perf_counter(),
                                "thread": threading.get_ident(), "original_call_completed": ok})

    def batch(self, query, trains):
        return measured("GPU_request", original_batch, (self, query, trains), {})

    def geometry(self, *args, **kwargs):
        return measured("CPU_geometry", original_geometry, (self, *args), kwargs)

    def pose(*args, **kwargs):
        return measured("CPU_pose", original_pose, args, kwargs)

    try:
        entry.owned_atomic_json, entry.COMPOSED_SOURCE = scoped_writer(output)
        integer_sift_matching.DistanceWorker.batch = batch
        array_match_geometry.ArrayMatchGeometry.__call__ = geometry
        feature_localization.estimate_pose = pose
        entry.OWNED = OWN
        sys.argv = [str(Path(entry.__file__)), "--mode", "process", "--output", str(output / "c")]
        entry.main()
        original = read(output / "c/RESULT.json")
        protected = compare_pair(BASELINE, output / "c")
        prep = read(output / "c/e/PREPARATION.json")
        totals = {label: sum(row["end"] - row["start"] for row in values) for label, values in rows.items()}
        query_phase = prep["complete_query_loop_and_final_map_verification_seconds"]
        optimistic_overlap = totals["CPU_geometry"] + totals["CPU_pose"]
        checks = {"all_original_complete_24_48_guards_passed": all(original["checks"].values()),
                  "all24_scientific_outputs_and_original_reference_images_exact": protected["all_protected"],
                  "all48_actual_GPU_requests_timed": len(rows["GPU_request"]) == 48,
                  "all48_actual_current_array_geometries_timed": len(rows["CPU_geometry"]) == 48,
                  "all24_actual_original_pose_estimators_timed": len(rows["CPU_pose"]) == 24,
                  "all_original_measured_calls_completed": all(row["original_call_completed"] for values in rows.values() for row in values),
                  "all_timing_hooks_stayed_on_original_caller": len({row["thread"] for values in rows.values() for row in values}) == 1,
                  "current_sources_unchanged": identities == [data_identity(path) for path in sources]}
        if not all(checks.values()):
            raise AssertionError("The actual original-entry diagnostic failed a required scientific or hook guard")
        result = {"status": "complete_current_localization_handoff_diagnostic_passed", "checks": checks,
                  "original_result": str(output / "c/RESULT.json"), "actual_complete_comparison": protected,
                  "actual_intervals": rows, "actual_component_seconds": totals,
                  "complete_query_loop_and_map_verify_seconds": query_phase,
                  "optimistic_cross_query_overlap_upper_bound_seconds": optimistic_overlap,
                  "optimistic_query_phase_speed_reduction_upper_bound_before_all_fees": optimistic_overlap / query_phase,
                  "CPU_totals_not_added_to_GPU_as_whole_wall": True, "measured_whole_not_a_performance_pair": True,
                  "complete_before_publication_seconds": time.perf_counter() - STARTED,
                  "baseline_same_current_R120_R124_scope": True, "new_optimization_count": 0,
                  "scheduler_implementation_and_formal_pairs": "not_run_diagnostic_first",
                  "external_unseen_object_training_metric_GT_production": "not_verified"}
        atomic_json(output / "RESULT.json", result)
        print(json.dumps({"status": result["status"], "components": totals, "query_phase": query_phase,
                          "optimistic_query_phase_reduction_upper_bound": optimistic_overlap / query_phase,
                          "new_optimization_count": 0}))
    except BaseException as error:
        atomic_json(output / "FAILURE.json", {"error_type": type(error).__name__, "error": str(error),
                                               "actual_interval_counts": {label: len(values) for label, values in rows.items()},
                                               "new_optimization_count": 0})
        raise
    finally:
        integer_sift_matching.DistanceWorker.batch = original_batch
        array_match_geometry.ArrayMatchGeometry.__call__ = original_geometry
        feature_localization.estimate_pose = original_pose
        entry.owned_atomic_json, entry.COMPOSED_SOURCE = original_writer, original_write_source
        entry.OWNED = original_scope
        sys.argv = previous_argv
        atomic_json(output / "BINDING_RESTORATION.json", {
            "batch_restored": integer_sift_matching.DistanceWorker.batch is original_batch,
            "geometry_restored": array_match_geometry.ArrayMatchGeometry.__call__ is original_geometry,
            "pose_restored": feature_localization.estimate_pose is original_pose,
            "writer_context_restored": entry.owned_atomic_json is original_writer,
            "writer_source_restored": entry.COMPOSED_SOURCE is original_write_source,
            "scope_restored": entry.OWNED is original_scope, "argv_restored": sys.argv is previous_argv,
            "new_optimization_count": 0})


if __name__ == "__main__":
    main()
