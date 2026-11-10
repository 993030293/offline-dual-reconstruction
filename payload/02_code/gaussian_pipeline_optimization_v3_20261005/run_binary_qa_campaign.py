"""Verify full binary observation identity and measure complete disk-to-QA cost."""

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
from binary_map_qa import projection_validity_binary, image_arrays, camera_array, points_array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("binary_map_qa.py"), V2/"native_observation_checks.py"]
    inputs = sorted(args.model.glob("*.bin"))
    identities = [data_identity(path) for path in paths+inputs]
    atomic_json(args.output/"PROTOCOL.json", {"incumbent": "native_pycolmap_read_and_Q49_QA",
        "source_inputs": identities, "trials": 3, "minimum_reduction": .1,
        "includes_all_native_disk_reads_and_fresh_point_and_camera_arrays": True,
        "all1016_full_pose_pixel_point_id_and_all959120_validity_guard": True,
        "negative_and_NaN_clones_written_and_read_afresh": True, "malformed_binary_rejection_required": True,
        "immutable_model_files_only_not_an_in_memory_BA_QA_replacement": True, "counted": False})
    model = pycolmap.Reconstruction(args.model)
    images, cameras = image_arrays(args.model/"images.bin"), camera_array(args.model/"cameras.bin")
    coordinates, tracks = points_array(args.model/"points3D.bin")
    actual_identity = []
    checks = {"all1016_images": len(images) == model.num_reg_images() == 1016,
        "all75767_points": int((tracks >= 0).sum()) == model.num_points3D() == 75767,
        "all_point_coordinates_and_track_lengths_exact": all(np.array_equal(coordinates[int(identity)], point.xyz)
            and tracks[int(identity)] == point.track.length() for identity, point in model.points3D.items()),
        "all_camera_models_and_params_exact": all(cameras[identity].model == camera.model
            and np.array_equal(cameras[identity].params, camera.params) for identity, camera in model.cameras.items())}
    for row in sorted(images, key=lambda row: row["id"]):
        image = model.images[row["id"]]
        points = image.points2D
        pixel_exact = np.array_equal(row["data"]["xy"], np.asarray([point.xy for point in points]))
        id_exact = np.array_equal(row["data"]["point_id"], np.asarray([int(point.point3D_id) if point.has_point3D() else -1 for point in points]))
        pose_exact = np.array_equal(row["pose"].matrix(), image.cam_from_world().matrix())
        checks["image_"+str(row["id"])] = pixel_exact and id_exact and pose_exact and row["name"] == image.name and row["camera_id"] == image.camera_id
        actual_identity.extend((row["id"], int(index), int(row["data"]["point_id"][index]))
            for index in np.flatnonzero(row["data"]["point_id"] >= 0))
    checks["all959120_observation_identity_order_exact"] = actual_identity == observation_identity_native(model) and len(actual_identity) == 959120
    if not all(checks.values()):
        atomic_json(args.output/"FAILED_IDENTITY.json", {"checks": checks})
        raise ValueError("The native complete binary QA identity changed")
    trials = []
    for trial in range(3):
        values, elapsed = {}, {}
        for mode in (["incumbent", "binary"] if trial % 2 == 0 else ["binary", "incumbent"]):
            started = time.perf_counter()
            values[mode] = projection_validity_native(pycolmap.Reconstruction(args.model)) if mode == "incumbent" else projection_validity_binary(args.model)
            elapsed[mode] = time.perf_counter()-started
        row = {"trial": trial, "seconds": elapsed, "result_exact": values["incumbent"] == values["binary"]}
        trials.append(row)
        atomic_json(args.output/"STATE.json", {"status": "running", "trials": trials})
        print(json.dumps(row), flush=True)
        checks["trial_"+str(trial)+"_all_result_exact"] = row["result_exact"]
    first = next(image for image in model.images.values() if image.has_pose and image.num_points3D)
    point_id = int(first.get_observation_points2D()[0].point3D_id)
    clone = pycolmap.Reconstruction(model)
    old = model.points3D[point_id].xyz.copy()
    for label, xyz in (("nan", np.array([np.nan, 0., 0.])), ("negative", first.cam_from_world().inverse()*np.array([0., 0., -1.])), ("restored", old)):
        clone.points3D[point_id].xyz = xyz
        folder = args.output/("fixture_"+label)
        folder.mkdir()
        clone.write_binary(folder)
        left, right = projection_validity_native(clone), projection_validity_binary(folder)
        # Binary image order is explicitly compared to its native file readback.
        readback = projection_validity_native(pycolmap.Reconstruction(folder))
        checks[label+"_native_file_and_memory_failure_identity_exact"] = left == readback and right == readback and right["valid"] == (label == "restored")
        atomic_json(args.output/(label+".json"), {"result": right})
    malformed = args.output/"malformed"
    malformed.mkdir()
    for name in ("images.bin", "cameras.bin", "points3D.bin"):
        payload = (args.model/name).read_bytes()
        for label, raw in (("truncated", payload[:-1]), ("trailing", payload+b"x")):
            path = malformed/(name+"."+label)
            path.write_bytes(raw)
            fn = image_arrays if name == "images.bin" else camera_array if name == "cameras.bin" else points_array
            try:
                fn(path)
                checks[name+"_"+label+"_rejected"] = False
            except ValueError:
                checks[name+"_"+label+"_rejected"] = True
    median = {mode: statistics.median(row["seconds"][mode] for row in trials) for mode in ("incumbent", "binary")}
    reduction = 1-median["binary"]/median["incumbent"]
    checks["at_least10percent_complete_reduction"] = reduction >= .1
    checks["all_sources_and_inputs_unchanged"] = identities == [data_identity(path) for path in paths+inputs]
    result = {"status": "stage_passed_reusable_entry_pending" if all(checks.values()) else "failed_verification",
        "checks": checks, "median_seconds": median, "reduction": reduction, "saved_seconds": median["incumbent"]-median["binary"],
        "trials": trials, "images": 1016, "observations": 959120, "in_memory_BA_supported": False, "counted": False}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "checks"}), flush=True)
    if not all(checks.values()):
        raise ValueError("A full binary QA or original failure guard failed")


if __name__ == "__main__":
    main()
