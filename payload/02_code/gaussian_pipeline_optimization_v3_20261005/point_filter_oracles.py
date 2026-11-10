"""Independent R113 exact values, native state, and complete output checks."""

import json
from pathlib import Path
import struct

import numpy as np


def exact_values(first, second):
    if type(first) is not type(second):
        raise AssertionError("A value type changed")
    if isinstance(first, dict):
        if list(first) != list(second):
            raise AssertionError("A dictionary order changed")
        for key in first:
            exact_values(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        if len(first) != len(second):
            raise AssertionError("A sequence length changed")
        for left, right in zip(first, second, strict=True):
            exact_values(left, right)
    elif isinstance(first, float):
        if struct.pack("!d", first) != struct.pack("!d", second):
            raise AssertionError("A floating-point bit changed")
    elif first != second:
        raise AssertionError("A scalar value changed")


def array_bits(value):
    value = np.asarray(value)
    return (value.dtype.str, tuple(value.shape), value.tobytes())


def native_state(model, output):
    """Capture ordered live associations plus every original native binary field."""
    points = [(int(point_id), array_bits(point.xyz), array_bits(point.color), float(point.error),
               int(point.track.length()), [(int(e.image_id), int(e.point2D_idx)) for e in point.track.elements])
              for point_id, point in model.points3D.items()]
    images = [(int(image_id), str(image.name), int(image.camera_id), int(image.frame_id), bool(image.has_pose),
               bool(image.has_camera_ptr() if callable(image.has_camera_ptr) else image.has_camera_ptr),
               bool(image.has_frame_ptr() if callable(image.has_frame_ptr) else image.has_frame_ptr),
               [(array_bits(point.xy), int(point.point3D_id)) for point in image.points2D])
              for image_id, image in model.images.items()]
    cameras = [(int(camera_id), str(camera.model), int(camera.width), int(camera.height),
                array_bits(camera.params), bool(camera.has_prior_focal_length))
               for camera_id, camera in model.cameras.items()]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        model.write_binary(output)
        writing = {"ok": True}
    except BaseException as error:
        writing = {"ok": False, "type": type(error).__name__, "message": str(error)}
    binaries = {path.name: path.read_bytes() for path in sorted(output.iterdir()) if path.is_file()}
    return {"points": points, "images": images, "cameras": cameras,
            "native_binary_write": writing, "complete_native_binary_fields": binaries}


def normalize_json(value, folder, top=True):
    if isinstance(value, dict):
        observations = {"elapsed_seconds", "peak_RSS_bytes"} if top else set()
        return {key: normalize_json(item, folder, False) for key, item in value.items()
                if key not in observations}
    if isinstance(value, list):
        return [normalize_json(item, folder, False) for item in value]
    if isinstance(value, str):
        return value.replace(str(folder), "<CURRENT_EXPORT>").replace(folder.as_posix(), "<CURRENT_EXPORT>")
    return value


def compare_outputs(left, right):
    left, right = Path(left), Path(right)
    paths = [{path.relative_to(folder).as_posix(): path for path in folder.rglob("*") if path.is_file()}
             for folder in (left, right)]
    if paths[0].keys() != paths[1].keys():
        raise AssertionError("Complete exported file membership differs")
    bytes_checked = files = json_files = 0
    for name in sorted(paths[0]):
        first, second = paths[0][name], paths[1][name]
        if first.suffix.lower() == ".json":
            exact_values(normalize_json(json.loads(first.read_text(encoding="utf-8")), left),
                         normalize_json(json.loads(second.read_text(encoding="utf-8")), right))
            json_files += 1
        else:
            if first.stat().st_size != second.stat().st_size:
                raise AssertionError("Complete output size differs: " + name)
            with first.open("rb") as a, second.open("rb") as b:
                while block := a.read(4 * 1024**2):
                    if block != b.read(len(block)):
                        raise AssertionError("Complete output bytes differ: " + name)
                    bytes_checked += len(block)
                if b.read(1):
                    raise AssertionError("Trailing output bytes differ: " + name)
            files += 1
    return {"complete_byte_pairs": files, "complete_JSON_pairs": json_files,
            "complete_compared_bytes": bytes_checked,
            "observed_JSON_top_level_omissions": ["elapsed_seconds", "peak_RSS_bytes"]}
