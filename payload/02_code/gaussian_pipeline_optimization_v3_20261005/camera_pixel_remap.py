"""Reuse calibration sampling geometry for owned array-origin RGB bitmaps."""

from contextlib import contextmanager
import json
import time

import numpy as np
import pycolmap

SUPPORTED = {"PINHOLE", "SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "OPENCV_FISHEYE"}


def camera_fields(camera):
    return {"camera_id": int(camera.camera_id), "model": camera.model_name,
        "width": int(camera.width), "height": int(camera.height), "params": camera.params.tolist(),
        "has_prior_focal_length": bool(camera.has_prior_focal_length)}


class CameraPixelRemap:
    def __init__(self, original, maximum_bytes=256*1024**2):
        if pycolmap.__version__ != "4.1.1" or np.__version__ != "2.3.5" or type(maximum_bytes) is not int or maximum_bytes <= 0:
            raise ValueError("The pinned native geometry, NumPy ABI and positive map budget are required")
        self.original, self.maximum_bytes = original, maximum_bytes
        self.key, self.maps, self.calibration = None, None, None
        self.closed = False
        self.trace = {"routes": [], "builds": 0, "build_seconds": 0., "close_seconds": 0., "map_bytes": 0,
            "closed": False, "maximum_bytes": maximum_bytes, "calls": []}

    def build(self, camera, calibration):
        started = time.perf_counter()
        scaled = pycolmap.Camera(**camera_fields(calibration))
        if (scaled.width, scaled.height) != (camera.width, camera.height):
            scaled.rescale(camera.width, camera.height)
        x0 = np.empty((camera.height, camera.width), np.int32)
        y0 = np.empty_like(x0)
        dx = np.empty(x0.shape, np.float64)
        dy = np.empty_like(dx)
        valid = np.empty(x0.shape, bool)
        columns = np.arange(camera.width, dtype=np.float64)+.5
        for first in range(0, camera.height, 64):
            stop = min(first+64, camera.height)
            points = np.column_stack((np.tile(columns, stop-first),
                np.repeat(np.arange(first, stop, dtype=np.float64)+.5, camera.width)))
            rays = scaled.cam_from_img(points)
            source = camera.img_from_cam(np.column_stack((rays, np.ones(len(rays), np.float64))))-.5
            finite = np.isfinite(source).all(axis=1)
            source[~finite] = 0
            floors = np.floor(source)
            good = finite & (floors[:, 0] >= 0) & (floors[:, 0] < camera.width-1)
            good &= (floors[:, 1] >= 0) & (floors[:, 1] < camera.height-1)
            floors[~good] = 0
            shape = (stop-first, camera.width)
            x0[first:stop], y0[first:stop] = floors[:, 0].astype(np.int32).reshape(shape), floors[:, 1].astype(np.int32).reshape(shape)
            dx[first:stop], dy[first:stop] = (source[:, 0]-floors[:, 0]).reshape(shape), (source[:, 1]-floors[:, 1]).reshape(shape)
            valid[first:stop] = good.reshape(shape)
        self.maps = x0, y0, dx, dy, valid
        for array in self.maps:
            array.setflags(write=False)
        self.calibration = camera_fields(calibration)
        self.trace["map_bytes"] = sum(array.nbytes for array in self.maps)
        if self.trace["map_bytes"] > self.maximum_bytes:
            raise MemoryError("The complete owned sampling map exceeds its declared budget")
        self.trace["builds"] += 1
        self.trace["build_seconds"] += time.perf_counter()-started

    def __call__(self, options, bitmap, camera):
        if self.closed:
            raise RuntimeError("An owned sampling map cannot be used after close")
        started = time.perf_counter()
        supported = (camera.model_name in SUPPORTED and bitmap.is_rgb
            and bitmap.width == camera.width and bitmap.height == camera.height
            and camera.width*camera.height*25 <= self.maximum_bytes)
        if not supported:
            self.trace["routes"].append("original_before_mapping")
            return self.original(options, bitmap, camera)
        key = json.dumps({"camera": camera_fields(camera), "options": options.todict()}, sort_keys=True, allow_nan=False)
        if self.key != key:
            calibration = pycolmap.undistort_camera(options, camera)
            self.build(camera, calibration)
            self.key = key
        raw = bitmap.to_array()
        if raw.dtype != np.uint8 or raw.shape != (camera.height, camera.width, 3):
            raise ValueError("An owned array-origin RGB bitmap changed its native layout")
        output = np.zeros_like(raw)
        x0, y0, dx, dy, valid = self.maps
        for first in range(0, camera.height, 64):
            stop = min(first+64, camera.height)
            xs, ys = x0[first:stop], y0[first:stop]
            wx, wy = dx[first:stop, :, None], dy[first:stop, :, None]
            upper = (1.-wx)*raw[ys, xs]+wx*raw[ys, xs+1]
            lower = (1.-wx)*raw[ys+1, xs]+wx*raw[ys+1, xs+1]
            colors = ((1.-wy)*upper+wy*lower).astype(np.float32)
            rounded = np.clip(np.floor(colors.astype(np.float64)+.5), 0, 255).astype(np.uint8)
            rounded[~valid[first:stop]] = 0
            output[first:stop] = rounded
        result = pycolmap.Bitmap.from_array(output)
        calibration = pycolmap.Camera(**self.calibration)
        if (calibration.width, calibration.height) != (camera.width, camera.height):
            result.rescale(calibration.width, calibration.height)
        self.trace["routes"].append("owned_calibration_map")
        self.trace["calls"].append({"seconds": time.perf_counter()-started, "camera": camera_fields(camera),
            "result_camera": camera_fields(calibration), "map_bytes": self.trace["map_bytes"]})
        return result, calibration

    def close(self):
        started = time.perf_counter()
        self.maps, self.key, self.calibration = None, None, None
        self.closed = True
        self.trace.update(closed=True, map_bytes=0)
        self.trace["close_seconds"] += time.perf_counter()-started


@contextmanager
def camera_pixel_remap(enabled):
    original = pycolmap.undistort_image
    operator = CameraPixelRemap(original) if enabled else None
    try:
        if operator is not None:
            pycolmap.undistort_image = operator
        yield operator.trace if operator is not None else None
    finally:
        pycolmap.undistort_image = original
        if operator is not None:
            operator.close()
