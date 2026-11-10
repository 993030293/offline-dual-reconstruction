"""Measure complete real-map QA algorithms against the accepted native incumbent."""

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
V2 = ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import atomic_json, data_identity
from native_observation_checks import projection_validity_native, observation_identity_native
from qa_algorithms import projection_validity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("qa_algorithms.py"), V2/"native_observation_checks.py"]
    inputs = sorted(args.model.glob("*.bin"))
    identities = [data_identity(path) for path in paths+inputs]
    modes = {"incumbent": projection_validity_native,
        "R79_dense": lambda model: projection_validity(model, dense=True),
        "R80_grouped": lambda model: projection_validity(model, grouped=True),
        "R81_threads": lambda model: projection_validity(model, workers=4),
        "dense_grouped": lambda model: projection_validity(model, dense=True, grouped=True),
        "dense_threads": lambda model: projection_validity(model, dense=True, workers=4)}
    atomic_json(args.output/"PROTOCOL.json", {"status": "frozen_before_measurement", "sources_inputs": identities,
        "incumbent": "Q49_complete_native_observations", "trials": 3, "minimum_speed_reduction": .1,
        "all1016_images_959120_observations_required": True, "fresh_coordinate_table_in_every_call": True,
        "model_and_point_coordinates_can_change_between_calls": True,
        "invalid_result_identity_and_order_exact": True, "NaN_and_negative_depth_failure_oracles": True,
        "map_load_cost_reported_separately_and_not_claimed_as_improved": True,
        "parallelism_only_inside_registered_QA_algorithm": True, "counted": False})
    start = time.perf_counter()
    model = pycolmap.Reconstruction(args.model)
    load_seconds = time.perf_counter()-start
    if model.num_reg_images() != 1016 or model.num_points3D() != 75767:
        raise ValueError("The actual fixed map denominator changed")
    reference_identity = observation_identity_native(model)
    if len(reference_identity) != 959120:
        raise ValueError("The full real observation denominator changed")
    trials = []
    for trial in range(3):
        values, elapsed = {}, {}
        order = list(modes) if trial % 2 == 0 else list(reversed(modes))
        for mode in order:
            before = time.perf_counter()
            values[mode] = modes[mode](model)
            elapsed[mode] = time.perf_counter()-before
        exact = {name: values[name] == values["incumbent"] for name in modes}
        row = {"trial": trial, "seconds": elapsed, "result_exact": exact}
        trials.append(row)
        atomic_json(args.output/"STATE.json", {"status": "running", "trials": trials})
        print(json.dumps(row), flush=True)
        if not all(exact.values()) or values["incumbent"]["observation_count"] != 959120:
            raise ValueError("A full-map candidate changed its original QA output")
    first = next(image for image in model.images.values() if image.has_pose and image.num_points3D)
    point_id = int(first.get_observation_points2D()[0].point3D_id)
    fixture_checks = {}
    fixtures = (("nonfinite", np.array([np.nan, 0., 0.])),
        ("negative_depth", first.cam_from_world().inverse()*np.array([0., 0., -1.])),
        ("positive_restoration", model.points3D[point_id].xyz.copy()))
    diagnostic = pycolmap.Reconstruction(model)
    for label, coordinate in fixtures:
        diagnostic.points3D[point_id].xyz = coordinate
        baseline = projection_validity_native(diagnostic)
        checks = {mode: fn(diagnostic) == baseline for mode, fn in modes.items()}
        checks["original_failure_or_restoration_observed"] = baseline["valid"] == (label == "positive_restoration")
        fixture_checks[label] = checks
        atomic_json(args.output/(label+".json"), {"role": "mutated_clone_only", "checks": checks, "result": baseline})
    unchanged = identities == [data_identity(path) for path in paths+inputs]
    median = {mode: statistics.median(row["seconds"][mode] for row in trials) for mode in modes}
    results = {}
    for mode in modes:
        if mode == "incumbent":
            continue
        reduction = 1-median[mode]/median["incumbent"]
        checks = {"three_full_results_exact": all(row["result_exact"][mode] for row in trials),
            "fresh_mutated_and_restored_full_results_exact": all(all(values.values()) for values in fixture_checks.values()),
            "sources_and_inputs_unchanged": unchanged, "speed_reduction_at_least10percent": reduction >= .1}
        results[mode] = {"status": "stage_benefit_passed_related_entry_pending" if all(checks.values()) else "rejected",
            "checks": checks, "reduction": reduction, "saved_seconds": median["incumbent"]-median[mode],
            "counted": False}
    result = {"status": "campaign_completed_candidates_not_yet_accepted", "trials": trials,
        "median_seconds": median, "map_load_seconds_common": load_seconds, "results": results,
        "fixture_checks": fixture_checks, "registered": 1016, "points": 75767, "observations": 959120,
        "independent_accuracy_claim": False, "additional_accepted_count": 0}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps(result), flush=True)
    if not unchanged or not all(all(values.values()) for values in fixture_checks.values()):
        raise ValueError("An original source or bad-coordinate QA guard failed")


if __name__ == "__main__":
    main()
