"""Inspect only observed keypoints through COLMAP's native observation accessors."""

import numpy as np


def observed_entries(image):
    indices = image.get_observation_point2D_idxs()
    points = image.get_observation_points2D()
    if len(indices) != len(points) or len(points) != image.num_points3D:
        raise ValueError("Native observation accessors returned different denominators")
    if indices and (indices[0] < 0 or indices[-1] >= image.num_points2D()
                    or any(a >= b for a, b in zip(indices, indices[1:]))):
        raise ValueError("Native observed keypoint indices are not strictly ordered")
    return indices, points


def projection_validity_native(model):
    """Rebuild point coordinates for every QA call, including after mutable BA."""
    coordinates = {int(i): p.xyz.copy() for i, p in model.points3D.items()}
    invalid, observation_count = [], 0
    for image in model.images.values():
        if not image.has_pose:
            continue
        indices, points = observed_entries(image)
        observation_count += len(points)
        if not points:
            continue
        ids = [int(p.point3D_id) for p in points]
        xyz = np.asarray([coordinates[i] for i in ids])
        pose = image.cam_from_world()
        camera_xyz = xyz @ pose.rotation.matrix().T+pose.translation
        projected = model.cameras[image.camera_id].img_from_cam(camera_xyz)
        finite = np.isfinite(projected).all(axis=1)
        valid = finite & np.isfinite(camera_xyz).all(axis=1) & (camera_xyz[:, 2] > 0)
        for slot in np.flatnonzero(~valid):
            point_id = ids[slot]
            invalid.append({"image_name": image.name, "point2D_index": indices[slot], "point3D_id": point_id,
                "depth_m": float(camera_xyz[slot, 2]) if np.isfinite(camera_xyz[slot, 2]) else None,
                "finite_camera_coordinates": bool(np.isfinite(camera_xyz[slot]).all()),
                "finite_projection": bool(finite[slot]), "track_length": model.points3D[point_id].track.length()})
    return {"valid": not invalid, "observation_count": observation_count, "invalid_count": len(invalid), "invalid": invalid}


def observation_identity_native(model):
    result = []
    for image in sorted(model.images.values(), key=lambda im: im.image_id):
        indices, points = observed_entries(image)
        result.extend((int(image.image_id), index, int(point.point3D_id)) for index, point in zip(indices, points))
    return result
