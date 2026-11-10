"""Compare complete selector decisions across counts, scales, boundaries and errors."""

import argparse
import copy
import importlib
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path[:0] = [str(V1), str(ROOT/"02_code/box2_initial_pose_quality_v2_20261002")]
from runtime import atomic_json, data_identity
import selection_engine
from spatial_novelty import NoveltyIndex, spatial_novelty_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    baseline = importlib.import_module("initial_selection")
    paths = [Path(__file__), Path(__file__).with_name("spatial_novelty.py"), V1/"selection_engine.py", Path(baseline.__file__)]
    sources = [data_identity(path) for path in paths]
    atomic_json(args.output/"PROTOCOL.json", {"sources": sources, "exact_complete_decisions_not_just_selected_ids": True,
        "scales": [1e-9, 1., 1e9], "counts": [1, 17, 211], "near_boundary_and_large_offset_cases": True,
        "no_GT_geometry_or_object_category_input": True, "full_three_actual_pairs_pending": True})
    original = selection_engine.choose_batched
    cases = []
    for scale in (1e-9, 1., 1e9):
        for count in (1, 17, 211):
            rng = np.random.default_rng(123+count)
            rotations = Rotation.random(count, random_state=rng)
            centers = rng.normal(size=(count, 3))*.08*scale
            rows = []
            for i in range(count):
                rows.append({"camera_frame": i, "camera_header_ns": i*30000000,
                    "coarse_pose_valid": i % 23 != 11, "excluded_role": "H" if i % 31 == 15 else None,
                    "C_W_m": centers[i].tolist(), "quaternion_WC_xyzw": rotations[i].as_quat().tolist(),
                    "R_WC": rotations[i].as_matrix().tolist(), "quality": {"sharpness": 100.+i,
                        "clipped_fraction": .01, "mean_brightness": 100., "corner_count_half_resolution": 200,
                        "occupied_grid_cells_4x4": 16, "quality_score": float(rng.uniform(1, 5))}})
            policy = copy.deepcopy(baseline.POLICY)
            policy["translation_min_m"] *= scale
            policy["position_cell_m"] *= scale
            policy["maximum_exported_frames"] = 80
            left, right = copy.deepcopy(rows), copy.deepcopy(rows)
            a = original(left, policy, 10., baseline)
            with spatial_novelty_execution(True) as trace:
                b = selection_engine.choose_batched(right, policy, 10., baseline)
            cases.append({"scale": scale, "count": count, "complete_selected_eligible_and_all_rows_exact": a == b and left == right,
                "trace": trace})
    boundary = []
    for origin in (0., 2.**38*.035, -2.**38*.035):
        radius = .035
        values = [origin, origin+radius, np.nextafter(origin+radius, origin), np.nextafter(origin+radius, np.inf), origin-radius]
        positions = np.asarray([[x, 0., 0.] for x in values])
        rotations = Rotation.from_euler("z", np.asarray([0., 0., .2, 180., .2]).reshape(-1, 1), degrees=True)
        trace = {key: 0 for key in ("hash_instances", "native_instances", "hash_queries", "native_queries", "selected_rows_considered", "actual_angular_rows")}
        index = NoveltyIndex(positions, rotations, radius, 1.2, trace)
        selected = []
        for i in range(len(values)):
            if selected:
                boundary.append(index.novel(i, selected) == index.native(i, selected))
            index.add(i)
            selected.append(i)
    unsupported = []
    for radius in (0., 1e-200, 1e200):
        trace = {key: 0 for key in ("hash_instances", "native_instances", "hash_queries", "native_queries", "selected_rows_considered", "actual_angular_rows")}
        index = NoveltyIndex(np.zeros((2, 3)), Rotation.identity(2), radius, 1.2, trace)
        index.add(0)
        unsupported.append(not index.enabled and index.novel(1, [0]) == index.native(1, [0]))
    empty_errors = []
    for enabled in (False, True):
        try:
            with spatial_novelty_execution(enabled):
                selection_engine.choose_batched([], baseline.POLICY, 10., baseline)
        except ValueError as error:
            empty_errors.append(str(error))
    try:
        with spatial_novelty_execution(True) as failed:
            raise RuntimeError("injected_selector_consumer_failure")
    except RuntimeError:
        pass
    checks = {"all9_count_scale_complete_selector_decisions_exact": len(cases) == 9 and all(row["complete_selected_eligible_and_all_rows_exact"] for row in cases),
        "all12_threshold_neighbors_and_large_signed_offsets_exact": len(boundary) == 12 and all(boundary),
        "all3_unsupported_radius_routes_use_original_before_hash_queries": all(unsupported),
        "empty_input_original_exception_exact": len(empty_errors) == 2 and empty_errors[0] == empty_errors[1],
        "all_original_aliases_restored_after_consumer_error": failed["aliases_restored"] and selection_engine.choose_batched is original,
        "sources_unchanged": sources == [data_identity(path) for path in paths]}
    result = {"status": "passed" if all(checks.values()) else "failed_verification", "sources": sources,
        "checks": checks, "cases": cases, "counted": False, "new_object_whole_pipeline_pending": True}
    atomic_json(args.output/"RESULT.json", result)
    print({"status": result["status"], "checks": checks}, flush=True)
    if not all(checks.values()):
        raise ValueError("A complete original selector invariant failed")


if __name__ == "__main__":
    main()
