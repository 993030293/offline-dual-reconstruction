"""Reproject a fixed sample of recorded registered LiDAR using the new cameras."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pycolmap

from common import MCAP, configure_stdout, load_mcap_tools, read_json, write_json
from run_prior_ba import quantiles


def recorded_xyz(payload, Cdr):
    cdr = Cdr(payload)
    sec, ns, frame = cdr.header()
    height, width, field_count = cdr.u32(), cdr.u32(), cdr.u32()
    fields = {name: (offset, datatype, count) for name, offset, datatype, count in
              [(cdr.string(), cdr.u32(), cdr.u8(), cdr.u32()) for _ in range(field_count)]}
    endian, point_step, row_step, data_length = cdr.u8(), cdr.u32(), cdr.u32(), cdr.u32()
    if endian or row_step != width * point_step or any(fields[name][1:] != (7, 1) for name in ("x", "y", "z")):
        raise ValueError("Unsupported registered cloud field, byte-order or row layout")
    if data_length != height * row_step:
        raise ValueError("Registered cloud byte count is inconsistent")
    raw = memoryview(payload)[cdr.offset:cdr.offset + data_length]
    xyz = np.column_stack([np.ndarray((height * width,), dtype="<f4", buffer=raw,
                                      offset=fields[name][0], strides=(point_step,)) for name in ("x", "y", "z")])
    return xyz[np.isfinite(xyz).all(1)], frame, int(sec) * 1_000_000_000 + int(ns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    run = parser.parse_args().run.resolve()
    output = run / "05_evaluation/fixed_recorded_lidar_projection"
    output.mkdir(exist_ok=False)
    McapIndex, Cdr, _, _, _ = load_mcap_tools()
    index = McapIndex(MCAP)
    topic = "/localization/lidar/points_registered_full"
    count = index.message_count(topic)
    selected = np.unique(np.linspace(round((count - 1) * .1), round((count - 1) * .68), 30, dtype=int))
    cloud_rows, all_points = [], []
    for frame_index, message in index.iter_selected_messages(topic, selected.tolist()):
        xyz, frame, timestamp = recorded_xyz(message["payload"], Cdr)
        if frame != "odom":
            raise ValueError("Registered points are not in the odometry world frame")
        sampled = xyz[np.linspace(0, len(xyz) - 1, min(len(xyz), 1500), dtype=int)]
        all_points.append(sampled)
        cloud_rows.append({"message_index": frame_index, "header_ns": timestamp, "frame_id": frame,
                           "input_finite_points": len(xyz), "sampled_points": len(sampled)})
    points = np.concatenate(all_points).astype(float)
    np.savez_compressed(output / "fixed_recorded_odometry_points.npz", xyz_m=points)
    a = pycolmap.Reconstruction(run / "02_same_180_ba/A1_visual/model")
    views = [sorted(a.images.values(), key=lambda im: im.name)[i] for i in [0, 36, 72, 108, 144, 179]]
    cases, rows = ["A1_visual", "B_w_sigma015", "C_augmented_212"], []
    reference_projections = {}
    for case in cases:
        model = a if case == cases[0] else pycolmap.Reconstruction(run / ("03_coverage_augmented/model" if case.startswith("C_") else f"02_same_180_ba/{case}/model"))
        by_name = {im.name: im for im in model.images.values()}
        for source in views:
            image = by_name[source.name]
            pose, camera = image.cam_from_world(), model.cameras[image.camera_id]
            xyz = points @ pose.rotation.matrix().T + pose.translation
            projected = camera.img_from_cam(xyz)
            valid = (xyz[:, 2] > 0) & np.isfinite(projected).all(1)
            inside = valid & (projected[:, 0] >= 0) & (projected[:, 0] < camera.width) & (projected[:, 1] >= 0) & (projected[:, 1] < camera.height)
            record = {"case": case, "image_name": image.name, "fixed_point_count": len(points),
                      "positive_finite_projection_count": int(valid.sum()), "in_image_count": int(inside.sum())}
            if case == cases[0]:
                reference_projections[image.name] = (projected, inside)
            else:
                previous, previous_inside = reference_projections[image.name]
                common = inside & previous_inside
                record["projection_delta_vs_A1_px"] = quantiles(np.linalg.norm(projected[common] - previous[common], axis=1))
            photo = cv2.imread(str(run / "00_contract/images" / image.name))
            indices = np.flatnonzero(inside)
            z = xyz[indices, 2]
            color = cv2.applyColorMap(np.clip((z / 8) * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO).reshape(-1, 3)
            for point_index, paint in zip(indices, color):
                px, py = np.rint(projected[point_index]).astype(int)
                if 0 <= px < camera.width and 0 <= py < camera.height:
                    cv2.circle(photo, (int(px), int(py)), 1, tuple(int(x) for x in paint), -1)
            target = output / f"{case}_{Path(image.name).stem}.png"
            if not cv2.imwrite(str(target), photo):
                raise RuntimeError("Projection overlay encoding failed")
            rows.append(record)
    write_json(output / "LIDAR_REPROJECTION_RECEIPT.json", {"status": "completed",
        "map_label": "recorded_odometry_map", "topic": topic, "source": MCAP,
        "cloud_rows": cloud_rows, "fixed_points": len(points), "rows": rows,
        "body_trajectory_or_LiDAR_world_points_optimized": False,
        "new_camera_projection_recomputed": True,
        "boundary": "projection_and_coordinate_contract_only_not_independent_surface_alignment_or_accuracy"})
    print(f"Reprojected {len(points)} fixed recorded LiDAR samples into {len(rows)} views")


if __name__ == "__main__":
    configure_stdout()
    main()
