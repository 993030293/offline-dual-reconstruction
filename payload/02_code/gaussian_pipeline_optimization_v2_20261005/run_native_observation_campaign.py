"""Compare complete projection QA on the actual map with invalid-depth oracles."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity
from run_metric_refinement import projection_validity, observation_identity
from native_observation_checks import projection_validity_native, observation_identity_native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(__file__).with_name("native_observation_checks.py"), V1/"run_metric_refinement.py"]
    inputs = sorted(args.model.glob("*.bin"))
    identities = [data_identity(p) for p in sources+inputs]
    atomic_json(args.output/"PROTOCOL.json", {"sources_and_inputs": identities,
        "three_actual_full_map_pairs": True, "minimum_reduction": .1, "maximum_seconds": 600,
        "maximum_RSS_bytes": 8*1024**3, "complete_invalid_rows_and_denominator_exact": True,
        "fresh_coordinate_table_each_call": True, "GT_used": False, "counted": False})
    model = pycolmap.Reconstruction(args.model)
    pairs = []
    for trial in range(3):
        rows, values = {}, {}
        order = ["baseline", "native"] if trial % 2 == 0 else ["native", "baseline"]
        for mode in order:
            started = time.perf_counter()
            values[mode] = (projection_validity if mode == "baseline" else projection_validity_native)(model)
            rows[mode] = time.perf_counter()-started
        item = {"trial": trial, "seconds": rows, "result_exact": values["baseline"] == values["native"],
            "observation_count": values["native"]["observation_count"], "invalid_count": values["native"]["invalid_count"]}
        pairs.append(item)
        atomic_json(args.output/"STATE.json", {"status": "running", "pairs": pairs})
        print(json.dumps(item), flush=True)
        if not item["result_exact"]:
            raise ValueError("Native observation projection QA changed its complete result")
    checks = {"all_three_actual_results_exact": all(p["result_exact"] for p in pairs)}
    identity_reference = observation_identity(model)
    checks["complete_observation_identity_order_exact"] = identity_reference == observation_identity_native(model)
    first = next(im for im in model.images.values() if im.has_pose and im.num_points3D)
    point_id = int(first.get_observation_points2D()[0].point3D_id)
    for label, location in (("nonfinite_coordinate", np.array([np.nan, 0, 0.])),
                            ("negative_depth", first.cam_from_world().inverse()*np.array([0., 0., -1.]))):
        diagnostic = pycolmap.Reconstruction(model)
        diagnostic.points3D[point_id].xyz = location
        left, right = projection_validity(diagnostic), projection_validity_native(diagnostic)
        checks[label+"_failure_preserved_exact"] = left == right and not right["valid"] and right["invalid_count"] > 0
        atomic_json(args.output/(label+".json"), {"role": "mutated_clone_failure_fixture_only", "result": right})
    median = {mode: statistics.median(p["seconds"][mode] for p in pairs) for mode in ("baseline", "native")}
    reduction = 1-median["native"]/median["baseline"]
    checks["actual_QA_stage_reduces10percent"] = reduction >= .1
    checks["sources_and_frozen_inputs_unchanged"] = identities == [data_identity(p) for p in sources+inputs]
    result = {"status": "stage_accepted_combination_pending" if all(checks.values()) else "failed",
        "checks": checks, "pairs": pairs, "median_seconds": median, "stage_reduction": reduction,
        "model_registered": model.num_reg_images(), "model_points": model.num_points3D(),
        "invalid_fixtures_are_not_quality_evidence": True, "stage_counted": all(checks.values()),
        "full_route_combination_accepted": False, "absolute_accuracy_claim": False}
    atomic_json(args.output/"ACCEPTANCE.json", result)
    print(json.dumps(result), flush=True)
    if not all(checks.values()):
        raise AssertionError("Native observation checks failed an original guard")


if __name__ == "__main__":
    main()
