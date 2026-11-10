"""Measure full fresh-map QA and admission mechanisms with complete native oracles."""

import argparse
from contextlib import ExitStack
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"
V3 = ROOT / "02_code/gaussian_pipeline_optimization_v3_20261005"
PACKAGE = ROOT / "05_packages/e26_video_to_target_m3_nollm_v1r23_20260927_installed_candidate"
sys.path[:0] = [str(V1), str(V2), str(V3), str(PACKAGE / "code/validated_fresh_runtime")]
from runtime import atomic_json, data_identity
from run_metric_refinement import projection_validity, observation_identity
from native_observation_checks import projection_validity_native, observation_identity_native
from binary_map_qa import projection_validity_binary
from grid_support import grid_support_execution
from segmented_camera_support import native_support_many
from point_filter_predicate import filter_points_columns
from native_observation_index import NativeObservationIndex, admission as index_admission
import export_admitted_map
import metric_admission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    policy = json.loads((PACKAGE / "config/METRIC_ALIGNMENT_POLICY.json").read_text())
    identities = [data_identity(path) for path in sorted(args.model.glob("*.bin"))]
    model = pycolmap.Reconstruction(args.model)
    registered = sorted((image for image in model.images.values() if image.has_pose), key=lambda image: image.image_id)
    record = {"status": "full_fresh_map_module_audit_running", "model": str(args.model), "sources": identities,
        "source": data_identity(Path(__file__)), "registered_count": len(registered),
        "observations": model.compute_num_observations(), "modules": {}, "repeat_count": 3,
        "OS_page_cache_cleared": False, "no_independent_geometry_accuracy_claim": True}
    atomic_json(args.output / "CONTRACT.json", record)

    def timed(function):
        before = time.perf_counter()
        result = function()
        return result, time.perf_counter()-before

    def pairs(module_id, baseline, optimized):
        trials = []
        for index in range(3):
            results = {}
            for name in (("baseline", "optimized") if index != 1 else ("optimized", "baseline")):
                gc.collect()
                value, seconds = timed(baseline if name == "baseline" else optimized)
                results[name] = value
                trials.append({"pair": index, "side": name, "seconds": seconds})
            if results["baseline"] != results["optimized"]:
                record["modules"][module_id] = {"status": "failed_exact_complete_output_oracle", "trials": trials}
                atomic_json(args.output / "RESULT.json", record)
                raise ValueError("Complete output differs for " + module_id)
        baseline_median = float(np.median([row["seconds"] for row in trials if row["side"] == "baseline"]))
        optimized_median = float(np.median([row["seconds"] for row in trials if row["side"] == "optimized"]))
        result = {"status": "exact_output_with_observed_stage_benefit" if optimized_median < baseline_median else "exact_output_no_observed_stage_speed_benefit",
                  "trials": trials, "baseline_median_seconds": baseline_median,
                  "optimized_median_seconds": optimized_median, "reduction_fraction": 1-optimized_median/baseline_median,
                  "all_complete_outputs_equal": True, "whole_pipeline_speed_claim": False}
        record["modules"][module_id] = result
        atomic_json(args.output / "RESULT.json", record)
        print(json.dumps({"module": module_id, **{key: result[key] for key in ("status", "reduction_fraction")}}), flush=True)

    def original_QA():
        return projection_validity(model), observation_identity(model)

    def native_QA():
        return projection_validity_native(model), observation_identity_native(model)

    pairs("Q49", original_QA, native_QA)
    pairs("R85", lambda: projection_validity(pycolmap.Reconstruction(args.model)),
          lambda: projection_validity_binary(args.model))
    support_policy = policy["camera_geometry_admission"]

    def original_support():
        return [metric_admission.camera_geometry_metrics(model, image, support_policy) for image in registered]

    def grid_support():
        with grid_support_execution(metric_admission, True):
            return original_support()

    pairs("R112", original_support, grid_support)
    counter = [0]
    support_traces = []

    def segmented_support():
        counter[0] += 1
        value, trace = native_support_many(model, registered, support_policy,
            args.output / ("segmented_" + str(counter[0])), metric_admission.camera_geometry_metrics)
        support_traces.append(trace)
        return value

    pairs("R132", original_support, segmented_support)
    record["modules"]["R132"]["actual_execution_trace"] = support_traces
    centers = np.asarray([image.projection_center() for image in registered])
    lo = centers.min(0)-policy["point_filter"]["camera_aabb_margin_m"]
    hi = centers.max(0)+policy["point_filter"]["camera_aabb_margin_m"]
    filter_traces = []

    def filter_points(optimized):
        local = pycolmap.Reconstruction(model)
        detail = {}
        removed = (filter_points_columns(local, policy["point_filter"], lo, hi, detail=detail) if optimized
                   else export_admitted_map.filter_points(local, policy["point_filter"], lo, hi))
        if optimized:
            filter_traces.append(detail)
        # Complete remaining points, poses and observation membership are compared.
        return removed, sorted((int(identity), point.xyz.tolist(), point.color.tolist(), float(point.error),
                    [(int(e.image_id), int(e.point2D_idx)) for e in point.track.elements])
                    for identity, point in local.points3D.items()), observation_identity(local)

    pairs("R113", lambda: filter_points(False), lambda: filter_points(True))
    record["modules"]["R113"]["actual_execution_trace"] = filter_traces
    supported, reason = index_admission(model)
    if supported and len(model.images) == len(registered):
        indices = []

        def original_observations():
            output = []
            for repeat in range(2):
                for image in registered:
                    points = list(image.points2D)
                    xy = np.asarray([point.xy for point in points], np.float64).reshape(-1, 2)
                    ids = np.asarray([int(point.point3D_id) if point.has_point3D() else -1 for point in points], np.int64)
                    output.append((image.image_id, xy.tobytes(), ids.tobytes(), (ids >= 0).tobytes()))
            return output

        def indexed_observations():
            index = NativeObservationIndex(model, args.output / ("index_" + str(len(indices))))
            output = []
            for repeat in range(2):
                for image in registered:
                    xy, ids, valid = index.get(model, image.image_id)
                    output.append((image.image_id, xy.tobytes(), ids.tobytes(), valid.tobytes()))
            index.verify_unchanged()
            indices.append(index.receipt())
            return output

        pairs("R94", original_observations, indexed_observations)
        record["modules"]["R94"]["actual_execution_trace"] = indices
    else:
        record["modules"]["R94"] = {"status": "not_triggered_native_admission", "reason": reason,
                                     "unregistered_rows_require_the_existing_localization_wrapper": True}
    if identities != [data_identity(path) for path in sorted(args.model.glob("*.bin"))]:
        raise ValueError("Module diagnostics changed the preserved fresh source map")
    record.update(status="full_fresh_map_module_audit_completed", elapsed_seconds=time.monotonic()-started,
                  source_unchanged=True)
    atomic_json(args.output / "RESULT.json", record)


if __name__ == "__main__":
    main()
