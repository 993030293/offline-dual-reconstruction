"""Compare fresh point tables, grouped projection and bounded read-only QA."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from native_observation_checks import observed_entries


def coordinate_table(model, dense=False):
    if not dense:
        return {int(i): p.xyz.copy() for i, p in model.points3D.items()}
    ids = np.asarray(list(model.points3D), dtype=np.int64)
    if len(ids) == 0 or ids.min() < 0 or ids.max() >= 10_000_000:
        raise ValueError("Dense QA point IDs exceed the explicit allocation admission")
    result = np.full((int(ids.max())+1, 3), np.nan, dtype=np.float64)
    result[ids] = np.asarray([model.points3D[int(i)].xyz for i in ids])
    return result


def camera_rows(model, coordinates, image, dense=False):
    indices, points = observed_entries(image)
    ids = [int(p.point3D_id) for p in points]
    if not points:
        return image, indices, ids, np.empty((0, 3))
    if dense:
        xyz = coordinates[np.asarray(ids, dtype=np.int64)]
    else:
        xyz = np.asarray([coordinates[i] for i in ids])
    pose = image.cam_from_world()
    camera_xyz = xyz @ pose.rotation.matrix().T+pose.translation
    return image, indices, ids, camera_xyz


def invalid_rows(model, row, projected):
    image, indices, ids, camera_xyz = row
    finite = np.isfinite(projected).all(axis=1)
    valid = finite & np.isfinite(camera_xyz).all(axis=1) & (camera_xyz[:, 2] > 0)
    return [{"image_name": image.name, "point2D_index": indices[slot], "point3D_id": ids[slot],
        "depth_m": float(camera_xyz[slot, 2]) if np.isfinite(camera_xyz[slot, 2]) else None,
        "finite_camera_coordinates": bool(np.isfinite(camera_xyz[slot]).all()),
        "finite_projection": bool(finite[slot]), "track_length": model.points3D[ids[slot]].track.length()}
        for slot in np.flatnonzero(~valid)]


def projection_validity(model, *, dense=False, grouped=False, workers=1):
    if workers not in (1, 4) or grouped and workers != 1:
        raise ValueError("Only registered independent QA algorithms may be composed")
    coordinates = coordinate_table(model, dense)
    images = [image for image in model.images.values() if image.has_pose]
    if grouped:
        rows = [camera_rows(model, coordinates, image, dense) for image in images]
        groups = {}
        for index, row in enumerate(rows):
            groups.setdefault(row[0].camera_id, []).append(index)
        pixels = {}
        for camera_id, members in groups.items():
            counts = [len(rows[index][3]) for index in members]
            xyz = np.concatenate([rows[index][3] for index in members], axis=0)
            projected = model.cameras[camera_id].img_from_cam(xyz)
            offset = 0
            for index, count in zip(members, counts):
                pixels[index] = projected[offset:offset+count]
                offset += count
            if offset != len(projected):
                raise ValueError("Grouped native projection lost an observation")
        invalid = [value for index, row in enumerate(rows) for value in invalid_rows(model, row, pixels[index])]
        count = sum(len(row[2]) for row in rows)
    else:
        def check(image):
            row = camera_rows(model, coordinates, image, dense)
            projected = model.cameras[image.camera_id].img_from_cam(row[3])
            return len(row[2]), invalid_rows(model, row, projected)

        if workers == 1:
            results = [check(image) for image in images]
        else:
            # Each task reads one immutable image; executor shutdown propagates failures.
            with ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(check, images))
        invalid = [value for _, values in results for value in values]
        count = sum(value for value, _ in results)
    return {"valid": not invalid, "observation_count": count, "invalid_count": len(invalid), "invalid": invalid}
