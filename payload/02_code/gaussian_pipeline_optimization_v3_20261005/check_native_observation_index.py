"""Exercise exact pixels, sparse IDs, ownership and native mutation rejection."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from native_observation_index import NativeObservationIndex, admission
from binary_map_qa import image_arrays


def rejected(operation):
    try:
        operation()
    except (ValueError, MemoryError):
        return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("native_observation_index.py"), Path(__file__).with_name("binary_map_qa.py"),
        ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005/observation_index.py"]
    identity = [data_identity(p) for p in paths]
    folders = sorted(p for p in args.fixtures.iterdir() if p.is_dir() and (p/"images.bin").is_file())
    if len(folders) != 9:
        raise ValueError("The existing nine multi-camera arbitrary-geometry fixtures are missing")
    atomic_json(args.output/"PROTOCOL.json", {"sources": identity, "fixtures": list(map(str, folders)),
        "all_three_geometries_and_three_scales_mixed_models": True, "all_pixels_sparse_ids_sentinel_exact": True,
        "complete_map_mutation_reject_and_explicit_format_route": True, "synthetic_not_real_object_accuracy": True, "new_optimization_count": 0})
    rows = []
    for number, folder in enumerate(folders):
        model = pycolmap.Reconstruction(folder)
        index = NativeObservationIndex(model, args.output/f"case{number}")
        checks = []
        for image_id, image in model.images.items():
            xy, ids, valid = index.get(model, image_id)
            expected_xy = np.asarray([p.xy for p in image.points2D])
            expected_valid = np.asarray([p.has_point3D() for p in image.points2D])
            expected_ids = np.asarray([int(p.point3D_id) if p.has_point3D() else -1 for p in image.points2D], np.int64)
            checks.append(np.array_equal(xy, expected_xy) and np.array_equal(ids, expected_ids) and np.array_equal(valid, expected_valid)
                and not xy.flags.writeable and not ids.flags.writeable and not valid.flags.writeable)
            index.get(model, image_id)
        cross = rejected(lambda: index.get(pycolmap.Reconstruction(model), next(iter(model.images))))
        index.verify_unchanged()
        rows.append({"fixture": folder.name, "all_pixels_and_sentinels_exact_read_only": all(checks),
            "cross_model_rejected": cross, "native_unchanged_snapshot_exact": index.verified, "receipt": index.receipt()})
    failures = {}
    for mutation in ("pixels", "point_coordinates", "calibration", "frame_pose", "image_name", "point_assignment", "track", "point_color"):
        model = pycolmap.Reconstruction(folders[0])
        index = NativeObservationIndex(model, args.output/("mutated_"+mutation))
        image = next(iter(model.images.values()))
        point = next(iter(model.points3D.values()))
        if mutation == "pixels":
            image.points2D[0].xy = image.points2D[0].xy+np.array([1., 0.])
        elif mutation == "point_coordinates":
            point.xyz = point.xyz+np.array([.001, 0., 0.])
        elif mutation == "calibration":
            camera = next(iter(model.cameras.values()))
            values = camera.params.copy()
            values[0] += 1.
            camera.params = values
        elif mutation == "frame_pose":
            frame = model.frames[image.frame_id]
            pose = frame.rig_from_world
            frame.rig_from_world = pycolmap.Rigid3d(pose.rotation, pose.translation+np.array([.001, 0., 0.]))
        elif mutation == "image_name":
            image.name = "modified.png"
        elif mutation == "point_assignment":
            image.reset_point3D_for_point2D(0)
        elif mutation == "track":
            point.track.delete_element(image.image_id, 0)
        else:
            point.color = np.array([1, 2, 3], np.uint8)
        failures[mutation+"_rejected"] = rejected(index.verify_unchanged)
    model = pycolmap.Reconstruction(folders[0])
    next(iter(model.images.values())).name = "subfolder/generic.png"
    failures["unsupported_name_explicit_original_route"] = not admission(model)[0]
    raw = (folders[0]/"images.bin").read_bytes()
    for name, payload in (("truncated", raw[:-1]), ("trailing", raw+b"x")):
        path = args.output/(name+".bin")
        path.write_bytes(payload)
        failures[name+"_rejected"] = rejected(lambda: image_arrays(path))
    checks = {"all_nine_generic_exact_cases": all(r["all_pixels_and_sentinels_exact_read_only"] and r["cross_model_rejected"]
            and r["native_unchanged_snapshot_exact"] for r in rows),
        "all_eight_mutations_and_bad_payload_routes_rejected": all(failures.values()),
        "sources_unchanged": identity == [data_identity(p) for p in paths]}
    result = {"status": "native_index_contract_passed" if all(checks.values()) else "failed_verification", "checks": checks,
        "cases": rows, "failures": failures, "sources": identity, "new_optimization_count": 0,
        "new_object_quality_claim": False}
    atomic_json(args.output/"RESULT.json", result)
    print({k: result[k] for k in ("status", "checks", "failures")})
    if not all(checks.values()):
        raise ValueError("An immutable native observation index contract failed")


if __name__ == "__main__":
    main()
