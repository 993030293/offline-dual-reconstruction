"""Replace only pinned array-origin reference sampling in a private process."""

from contextlib import contextmanager
import json
from pathlib import Path
import struct
import subprocess
import threading
import time

import numpy as np
import psutil
import pycolmap

from camera_pixel_remap import CameraPixelRemap, SUPPORTED, camera_fields
from integer_sift_matching import read_exact

HEADER = struct.Struct("<I")
ROOT = Path(__file__).resolve().parents[2]
OWNED = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/rm_gpu001"


def linux(path):
    value = Path(path).resolve().as_posix()
    if value[:3].lower() != "c:/":
        raise ValueError("The worker must use the existing C drive workspace")
    return "/mnt/c/" + value[3:]


def identity(path):
    from runtime import data_identity
    return data_identity(path)


class PixelWorker:
    def __init__(self, output):
        self.output = Path(output)
        self.command = ["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
                        "/opt/e26_thin_surface_v1/venv/bin/python", "-B",
                        linux(Path(__file__).with_name("camera_remap_stdio_worker.py")),
                        "--output", linux(self.output)]
        self.log = (self.output / "worker.log").open("xb")
        self.stopped, self.failed = False, False
        started = time.perf_counter()
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, bufsize=0)
        self.trace = {"command": self.command, "pid": self.process.pid,
                      "creation_time": psutil.Process(self.process.pid).create_time(), "calls": []}
        try:
            info = self.rpc({"command": "info"})
            info.pop("payload")
            if info["status"] != "ready":
                raise ValueError("The owned pixel worker did not confirm its CUDA environment")
            self.trace["info"] = info
            self.trace["startup_seconds"] = time.perf_counter() - started
        except BaseException:
            self.failed = True
            self.close()
            raise

    def rpc(self, record, payload=b""):
        if self.stopped or self.failed:
            raise RuntimeError("A closed or failed pixel worker cannot accept more requests")
        encoded = json.dumps(record, separators=(",", ":"), allow_nan=False).encode("utf-8")
        try:
            for block in (HEADER.pack(len(encoded)), encoded, payload):
                view = memoryview(block)
                while view:
                    count = self.process.stdin.write(view)
                    if not count:
                        raise EOFError("The owned pixel request pipe stopped accepting bytes")
                    view = view[count:]
            self.process.stdin.flush()
            size = HEADER.unpack(read_exact(self.process.stdout, HEADER.size))[0]
            if not 0 < size <= 65536:
                raise ValueError("The pixel response metadata exceeds its declared bound")
            response = json.loads(read_exact(self.process.stdout, size))
            if response["status"] == "failed":
                raise RuntimeError(response["error"])
            size = response["payload_bytes"]
            if type(size) is not int or not 0 <= size <= 256 * 1024**2:
                raise ValueError("The pixel response payload exceeds its declared bound")
            response["payload"] = read_exact(self.process.stdout, size)
            return response
        except BaseException:
            self.failed = True
            raise

    def close(self):
        if not self.stopped:
            started = time.perf_counter()
            try:
                if not self.failed:
                    row = self.rpc({"command": "stop"})
                    row.pop("payload")
                    self.trace["shutdown"] = row
            finally:
                self.process.stdin.close()
                self.process.stdout.close()
                self.trace["returncode"] = self.process.wait(timeout=30)
                self.log.close()
                self.stopped = True
                self.trace["closed"] = True
                self.trace["shutdown_seconds"] = time.perf_counter() - started
            if not self.failed and self.trace["returncode"] != 0:
                raise RuntimeError("The owned pixel worker did not shut down successfully")


class GpuCameraPixelRemap:
    def __init__(self, original, output, array_origin_only, maximum_bytes=256 * 1024**2):
        if array_origin_only is not True:
            raise ValueError("Only the pinned from_array reference caller can preserve metadata")
        self.output = Path(output).resolve()
        self.output.relative_to(OWNED)
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "maps").mkdir()
        (self.output / "batches").mkdir()
        self.geometry = CameraPixelRemap(original, maximum_bytes)
        self.original, self.maximum_bytes = original, maximum_bytes
        self.worker, self.group, self.map_ids = None, None, None
        self.closed = False
        self.lock = threading.Lock()
        self.trace = {"routes": [], "calls": [], "closed": False, "map_builds": 0,
                      "maximum_map_bytes": maximum_bytes, "array_origin_only": True,
                      "close_seconds": 0., "worker": None}

    def __call__(self, options, bitmap, camera):
        with self.lock:
            if self.closed:
                raise RuntimeError("A closed exact sampling map cannot be used")
            started = time.perf_counter()
            supported = (camera.model_name in SUPPORTED and bitmap.is_rgb
                and (bitmap.width, bitmap.height) == (camera.width, camera.height)
                and min(camera.width, camera.height) >= 2
                and camera.width * camera.height * 25 <= self.maximum_bytes)
            if not supported:
                self.trace["routes"].append("original_before_mapping")
                return self.original(options, bitmap, camera)
            key = json.dumps({"camera": camera_fields(camera), "options": options.todict()},
                             sort_keys=True, allow_nan=False)
            if self.geometry.key != key:
                calibration = pycolmap.undistort_camera(options, camera)
                self.geometry.build(camera, calibration)
                self.geometry.key = key
                self.group = f"maps/map_{self.trace['map_builds']:03d}"
                folder = self.output / self.group
                folder.mkdir(exist_ok=False)
                paths = []
                for name, value in zip(("x0", "y0", "dx", "dy", "valid"), self.geometry.maps):
                    path = folder / (name + ".npy")
                    np.save(path, value, allow_pickle=False)
                    paths.append(path)
                self.map_ids = [identity(path) for path in paths]
                self.trace["map_builds"] += 1
            raw = bitmap.to_array()
            if raw.dtype != np.uint8 or raw.shape != (camera.height, camera.width, 3):
                raise ValueError("The pinned array-origin RGB input changed its layout")
            if self.worker is None:
                self.worker = PixelWorker(self.output)
                self.trace["worker"] = self.worker.trace
            batch = f"batches/batch_{len(self.trace['calls']):03d}"
            folder = self.output / batch
            folder.mkdir(exist_ok=False)
            source_path, result_path = folder / "raw.npy", folder / "sampled.npy"
            np.save(source_path, raw, allow_pickle=False)
            raw_id = identity(source_path)
            response = self.worker.rpc({"command": "sample_file", "width": camera.width, "height": camera.height,
                "group": self.group, "map_identities": [{k: row[k] for k in ("bytes", "sha256")} for row in self.map_ids],
                "batch": batch, "raw_identity": {k: raw_id[k] for k in ("bytes", "sha256")}})
            payload = response.pop("payload")
            result_id = identity(result_path)
            if (response["status"] != "sampled" or response["width"] != camera.width or response["height"] != camera.height
                    or payload or {k: result_id[k] for k in ("bytes", "sha256")} != response["result_identity"]
                    or raw_id != identity(source_path)):
                raise ValueError("The full exact RGB reply changed identity or layout")
            pixels = np.load(result_path, allow_pickle=False)
            if pixels.shape != raw.shape or pixels.dtype != raw.dtype:
                raise ValueError("The complete exact sampled RGB file changed layout")
            result = pycolmap.Bitmap.from_array(pixels)
            calibration = pycolmap.Camera(**self.geometry.calibration)
            if (calibration.width, calibration.height) != (camera.width, camera.height):
                result.rescale(calibration.width, calibration.height)
            response.update(total_map_startup_IPC_sample_and_native_rescale_seconds=time.perf_counter() - started,
                            caller_thread=threading.get_ident(), camera=camera_fields(camera),
                            result_camera=camera_fields(calibration))
            self.trace["calls"].append(response)
            self.trace["routes"].append("owned_exact_CUDA_array_reference")
            self.worker.trace["calls"].append(response)
            return result, calibration

    def close(self):
        with self.lock:
            if not self.closed:
                started = time.perf_counter()
                try:
                    if self.worker is not None:
                        self.worker.close()
                finally:
                    self.geometry.close()
                    self.closed = True
                    self.trace.update(closed=True, close_seconds=time.perf_counter() - started,
                                      owned_CPU_map_bytes_after_close=self.geometry.trace["map_bytes"])


@contextmanager
def camera_pixel_gpu_remap(enabled, output, array_origin_only=True):
    started = time.perf_counter()
    original = pycolmap.undistort_image
    operator = GpuCameraPixelRemap(original, output, array_origin_only) if enabled else None
    trace = {"routes": [], "calls": [], "closed": True, "close_seconds": 0., "worker": None} if operator is None else operator.trace
    trace["setup_seconds"] = time.perf_counter() - started
    try:
        if operator is not None:
            pycolmap.undistort_image = operator
        yield trace
    finally:
        pycolmap.undistort_image = original
        if operator is not None:
            operator.close()
        trace["original_global_binding_restored"] = pycolmap.undistort_image is original
