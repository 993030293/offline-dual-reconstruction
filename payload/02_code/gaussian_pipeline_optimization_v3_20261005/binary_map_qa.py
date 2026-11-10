"""Read all native COLMAP observations as owned structured arrays for projection QA."""

from pathlib import Path
import struct

import numpy as np
import pycolmap

COUNT = struct.Struct("<Q")
IMAGE = struct.Struct("<i4d3di")
POINT = struct.Struct("<Q3d3BdQ")
OBSERVATION = np.dtype([("xy", "<f8", (2,)), ("point_id", "<i8")])


def bounded_unpack(layout, raw, offset):
    if offset+layout.size > len(raw):
        raise ValueError("A complete COLMAP record is truncated")
    return layout.unpack_from(raw, offset), offset+layout.size


def points_array(path):
    raw = Path(path).read_bytes()
    (count,), offset = bounded_unpack(COUNT, raw, 0)
    if not 0 < count <= 1_000_000:
        raise ValueError("The fixed QA point count exceeds its allocation admission")
    ids, xyz, tracks = np.empty(count, np.int64), np.empty((count, 3), np.float64), np.empty(count, np.int64)
    for index in range(count):
        values, offset = bounded_unpack(POINT, raw, offset)
        ids[index], xyz[index], tracks[index] = values[0], values[1:4], values[-1]
        offset += int(values[-1])*8
        if offset > len(raw):
            raise ValueError("A native point track is truncated")
    if offset != len(raw) or len(np.unique(ids)) != count or ids.min() < 0 or ids.max() >= 10_000_000:
        raise ValueError("Native point payload, identity or allocation bounds changed")
    coordinates = np.full((int(ids.max())+1, 3), np.nan, np.float64)
    track_count = np.full(int(ids.max())+1, -1, np.int64)
    coordinates[ids], track_count[ids] = xyz, tracks
    return coordinates, track_count


def image_arrays(path):
    raw = Path(path).read_bytes()
    (count,), offset = bounded_unpack(COUNT, raw, 0)
    if not 0 < count <= 100_000:
        raise ValueError("The fixed QA camera count exceeds its explicit allocation admission")
    result, seen = [], set()
    for _ in range(count):
        metadata, offset = bounded_unpack(IMAGE, raw, offset)
        end = raw.find(b"\0", offset)
        if end < offset or end-offset > 4096:
            raise ValueError("A native image name is missing or exceeds its bound")
        name = raw[offset:end].decode("utf-8")
        if not name or Path(name).name != name or metadata[0] <= 0 or metadata[0] in seen:
            raise ValueError("A native image name or identity is invalid")
        seen.add(metadata[0])
        (observations,), offset = bounded_unpack(COUNT, raw, end+1)
        size = int(observations)*OBSERVATION.itemsize
        if offset+size > len(raw):
            raise ValueError("The native observation array is truncated")
        data = np.frombuffer(raw, OBSERVATION, int(observations), offset)
        offset += size
        qvec, translation = metadata[1:5], np.asarray(metadata[5:8])
        pose = pycolmap.Rigid3d(pycolmap.Rotation3d(np.asarray([*qvec[1:], qvec[0]])), translation)
        result.append({"id": int(metadata[0]), "name": name, "camera_id": int(metadata[8]),
            "pose": pose, "data": data})
    if offset != len(raw):
        raise ValueError("Native image payload contains unclaimed bytes")
    return result


def camera_array(path):
    raw = Path(path).read_bytes()
    (count,), offset = bounded_unpack(COUNT, raw, 0)
    header = struct.Struct("<iiQQ")
    result = {}
    if not 0 < count <= 100_000:
        raise ValueError("The native calibration count is outside its bounded admission")
    for _ in range(count):
        (identity, model, width, height), offset = bounded_unpack(header, raw, offset)
        camera = pycolmap.Camera.create_from_model_id(identity, pycolmap.CameraModelId(model), 1., width, height)
        size = len(camera.params)*8
        if identity in result or offset+size > len(raw) or identity <= 0:
            raise ValueError("Native calibration identity or parameter payload is invalid")
        camera.params = np.frombuffer(raw, "<f8", len(camera.params), offset).copy()
        result[identity], offset = camera, offset+size
    if offset != len(raw):
        raise ValueError("Native calibration contains unclaimed bytes")
    return result


def projection_validity_binary(folder):
    folder = Path(folder)
    coordinates, tracks = points_array(folder/"points3D.bin")
    images, cameras = image_arrays(folder/"images.bin"), camera_array(folder/"cameras.bin")
    order = pycolmap.ImageMap()
    rows = {}
    for image in images:
        order[image["id"]] = pycolmap.Image(image_id=image["id"], name=image["name"], camera_id=image["camera_id"])
        rows[image["id"]] = image
    invalid, count = [], 0
    # Retain the native unordered-map enumeration without constructing point objects.
    for image_id in order:
        image = rows[image_id]
        ids_all = image["data"]["point_id"]
        indices = np.flatnonzero(ids_all >= 0)
        ids = ids_all[indices]
        if len(ids) and (ids.max() >= len(coordinates) or np.any(tracks[ids] < 0)):
            raise ValueError("An observed native point has no corresponding map point")
        if np.any(ids_all < -1):
            raise ValueError("The native unobserved point sentinel changed")
        count += len(ids)
        pose = image["pose"]
        camera_xyz = coordinates[ids] @ pose.rotation.matrix().T+pose.translation
        projected = cameras[image["camera_id"]].img_from_cam(camera_xyz)
        finite = np.isfinite(projected).all(axis=1)
        valid = finite & np.isfinite(camera_xyz).all(axis=1) & (camera_xyz[:, 2] > 0)
        for slot in np.flatnonzero(~valid):
            invalid.append({"image_name": image["name"], "point2D_index": int(indices[slot]), "point3D_id": int(ids[slot]),
                "depth_m": float(camera_xyz[slot, 2]) if np.isfinite(camera_xyz[slot, 2]) else None,
                "finite_camera_coordinates": bool(np.isfinite(camera_xyz[slot]).all()),
                "finite_projection": bool(finite[slot]), "track_length": int(tracks[ids[slot]])})
    return {"valid": not invalid, "observation_count": count, "invalid_count": len(invalid), "invalid": invalid}
