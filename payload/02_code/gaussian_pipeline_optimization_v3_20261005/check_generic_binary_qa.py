"""Check unchanged binary QA on non-box geometry, mixed cameras and metric scales."""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import atomic_json, data_identity
from native_observation_checks import projection_validity_native, observation_identity_native
from binary_map_qa import projection_validity_binary, image_arrays


def write_case(folder, geometry, scale):
    folder.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20261005)
    angle = np.linspace(0, 2*np.pi, 160, endpoint=False)
    if geometry == "curved_surface":
        xyz = np.column_stack((.7*np.cos(angle), .6*np.sin(angle), 3+.2*np.sin(3*angle)))
    elif geometry == "thin_branching_surface":
        xyz = np.column_stack((rng.uniform(-.6, .6, 160), rng.uniform(-.1, .1, 160), rng.uniform(2, 4, 160)))
    else:
        xyz = rng.uniform([-.6, -.4, 2], [.6, .4, 4], (160, 3))
    xyz *= scale
    ids = np.arange(len(xyz))*17+3
    definitions = [
        (1, "PINHOLE", 800, 600, [410., 470., 400., 300.]),
        (2, "OPENCV", 720, 540, [340., 390., 350., 260., .01, -.005, .001, -.002]),
        (3, "OPENCV_FISHEYE", 640, 480, [320., 330., 315., 245., .01, -.004, .0002, 0.])]
    cameras = {i: pycolmap.Camera(camera_id=i, model=model, width=width, height=height, params=params)
        for i, model, width, height, params in definitions}
    (folder/"cameras.txt").write_text("\n".join(" ".join(map(str, (i, model, width, height, *params)))
        for i, model, width, height, params in definitions)+"\n")
    images, tracks = [], {int(i): [] for i in ids}
    for index in range(9):
        image_id, camera_id = index*7+2, index%3+1
        rotation, _ = cv2.Rodrigues(np.array([.01*index, -.02*index, .07*index]))
        translation = np.array([.04*index, -.02*index, .03*index])*scale
        q = pycolmap.Rotation3d(rotation).quat
        pixel = cameras[camera_id].img_from_cam(xyz@rotation.T+translation)
        images.append(" ".join(map(str, (image_id, q[3], *q[:3], *translation, camera_id, f"generic_{index:04d}.png"))))
        values = []
        for slot, (point_id, xy) in enumerate(zip(ids, pixel)):
            values += [str(xy[0]), str(xy[1]), str(int(point_id))]
            tracks[int(point_id)] += [str(image_id), str(slot)]
        values += ["10.0", "20.0", "-1"]
        images.append(" ".join(values))
    (folder/"images.txt").write_text("\n".join(images)+"\n")
    (folder/"points3D.txt").write_text("\n".join(" ".join(map(str, (int(i), *point, 100, 130, 160, .1, *tracks[int(i)])))
        for i, point in zip(ids, xyz))+"\n")
    model = pycolmap.Reconstruction()
    model.read_text(folder)
    model.write_binary(folder)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name("binary_map_qa.py")
    frozen = data_identity(source)
    atomic_json(args.output/"PROTOCOL.json", {"R85_source": frozen, "geometries": ["curved_surface", "thin_branching_surface", "irregular_cloud"],
        "metric_scales": [.01, 1., 100.], "calibrations": ["PINHOLE", "OPENCV", "OPENCV_FISHEYE"],
        "sparse_point_and_image_ids": True, "unobserved_sentinel_preserved": True,
        "all_text_read_directly": True, "synthetic_contract_not_real_object_quality": True, "new_optimization_count": 0})
    rows = []
    for geometry in ("curved_surface", "thin_branching_surface", "irregular_cloud"):
        for scale in (.01, 1., 100.):
            folder = args.output/f"{geometry}_{scale}"
            model = write_case(folder, geometry, scale)
            native, binary = projection_validity_native(model), projection_validity_binary(folder)
            image_data = image_arrays(folder/"images.bin")
            identity = [(r["id"], int(slot), int(r["data"]["point_id"][slot])) for r in sorted(image_data, key=lambda r: r["id"])
                for slot in np.flatnonzero(r["data"]["point_id"] >= 0)]
            checks = {"all_observation_identity_exact": observation_identity_native(model) == identity,
                "all_projection_results_exact": native == binary,
                "all_three_different_calibrations_and_nine_poses": len(model.cameras) == 3 and model.num_reg_images() == 9,
                "all1440_observations_and_non_observed_sentinels": native["observation_count"] == 1440
                    and all(r["data"]["point_id"][-1] == -1 for r in image_data)}
            model.points3D[min(model.points3D)].xyz = np.array([0., 0., -3.])*scale
            bad = folder/"negative_depth"
            bad.mkdir()
            model.write_binary(bad)
            negative_native, negative_binary = projection_validity_native(model), projection_validity_binary(bad)
            checks["negative_depth_original_failure_values_and_order_exact"] = negative_native == negative_binary and not negative_native["valid"]
            rows.append({"geometry": geometry, "scale": scale, "checks": checks})
    checks = {"all_nine_geometry_calibration_scale_cases_and_failure_orders": all(all(r["checks"].values()) for r in rows),
        "accepted_R85_source_unchanged": data_identity(source) == frozen}
    result = {"status": "generic_geometry_and_multicamera_contract_passed" if all(checks.values()) else "failed_verification",
        "checks": checks, "rows": rows, "new_optimization_count": 0,
        "independent_real_geometry_or_whole_pipeline_quality_claim": False}
    atomic_json(args.output/"RESULT.json", result)
    print({k: result[k] for k in ("status", "checks")})
    if not all(checks.values()):
        raise ValueError("An unchanged generic binary QA contract failed")


if __name__ == "__main__":
    main()
