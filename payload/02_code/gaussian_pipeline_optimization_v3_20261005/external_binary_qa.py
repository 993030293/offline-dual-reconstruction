"""Check unchanged binary QA on a complete real external native reconstruction."""

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
from binary_map_qa import projection_validity_binary, image_arrays, points_array, camera_array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("binary_map_qa.py"), V2/"native_observation_checks.py",
        *sorted(args.model.glob("*.bin"))]
    identity = [data_identity(p) for p in paths]
    protocol = {"source_inputs": identity, "control": "previously_accepted_Q49_native_read_and_QA",
        "candidate": "unchanged_accepted_R85_binary_array_read_and_QA", "three_real_full_pairs": True,
        "original_invalid_output_order_and_values_exact": True, "external": "Oxford_Observatory_real_native_SfM",
        "GT_or_TLS_inputs": False, "new_optimization_count": 0,
        "component_test_not_external_SfM_accuracy_or_Gaussian_quality": True}
    atomic_json(args.output/"PROTOCOL.json", protocol)
    original = pycolmap.Reconstruction(args.model)
    coordinates, tracks = points_array(args.model/"points3D.bin")
    images, cameras = image_arrays(args.model/"images.bin"), camera_array(args.model/"cameras.bin")
    native_identity = observation_identity_native(original)
    actual_identity = [(row["id"], int(slot), int(row["data"]["point_id"][slot]))
        for row in sorted(images, key=lambda r: r["id"]) for slot in np.flatnonzero(row["data"]["point_id"] >= 0)]
    checks = {"all_observed_point_and_keypoint_identity_exact": native_identity == actual_identity,
        "all_real_point_coordinates_and_tracks_exact": all(np.array_equal(coordinates[i], point.xyz)
            and int(tracks[i]) == point.track.length() for i, point in original.points3D.items()),
        "all_real_image_metadata_pixels_and_camera_pose_exact": all(row["name"] == original.images[row["id"]].name
            and np.array_equal(row["pose"].matrix(), original.images[row["id"]].cam_from_world().matrix())
            and np.array_equal(row["data"]["xy"], np.asarray([p.xy for p in original.images[row["id"]].points2D])) for row in images),
        "all_camera_parameters_exact": all(np.array_equal(camera.params, original.cameras[i].params) for i, camera in cameras.items())}
    timing = []
    for trial in range(3):
        values, results = {}, {}
        for mode in (["native", "binary"] if trial % 2 == 0 else ["binary", "native"]):
            before = time.perf_counter()
            if mode == "native":
                model = pycolmap.Reconstruction(args.model)
                result = projection_validity_native(model)
                del model
            else:
                result = projection_validity_binary(args.model)
            values[mode] = time.perf_counter()-before
            results[mode] = result
        timing.append({"trial": trial, "seconds": values, "full_results_exact": results["native"] == results["binary"],
            "all_observations": results["native"]["observation_count"], "valid": results["native"]["valid"]})
    medians = {mode: statistics.median(r["seconds"][mode] for r in timing) for mode in ("native", "binary")}
    reduction = 1-medians["binary"]/medians["native"]
    checks.update(all_three_complete_projection_results_exact=all(r["full_results_exact"] for r in timing),
        external_complete_stage_reduction_at_least10percent=reduction >= .1,
        sources_and_external_inputs_unchanged=identity == [data_identity(p) for p in paths])
    receipt = {"status": "external_native_QA_component_guard_passed" if all(checks.values()) else "failed_verification",
        "checks": checks, "pairs": timing, "median_seconds": medians, "reduction": reduction,
        "registered_images": original.num_reg_images(), "points": original.num_points3D(),
        "observations": len(native_identity), "independent_geometry_accuracy_claim": False,
        "new_optimization_count": 0, "source_inputs": identity}
    atomic_json(args.output/"RESULT.json", receipt)
    print({k: receipt[k] for k in ("status", "checks", "median_seconds", "reduction", "registered_images", "observations")})
    if not all(checks.values()):
        raise ValueError("An external whole native QA contract or cost guard failed")


if __name__ == "__main__":
    main()
