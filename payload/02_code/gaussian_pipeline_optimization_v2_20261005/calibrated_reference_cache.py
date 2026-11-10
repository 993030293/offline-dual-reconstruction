"""Reuse exact undistorted reference pixels and PNG bytes with calibration keys."""

from contextlib import contextmanager
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import PIL
from PIL import Image
import pycolmap

from query_feature_cache import file_identity, option_identity


def camera_fields(camera):
    return {"camera_id": int(camera.camera_id), "model": camera.model_name,
        "width": int(camera.width), "height": int(camera.height), "params": camera.params.tolist(),
        "has_prior_focal_length": bool(camera.has_prior_focal_length)}


class CalibratedReferenceCache:
    def __init__(self, root, original_undistort, original_save, maximum_bytes=512*1024**2):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if maximum_bytes <= 0:
            raise ValueError("Reference cache needs a positive disk budget")
        self.maximum_bytes = int(maximum_bytes)
        self.original_undistort, self.original_save = original_undistort, original_save
        binaries = list(Path(pycolmap.__file__).parent.glob("_core*.pyd"))
        if len(binaries) != 1:
            raise ValueError("The pinned Windows undistortion binary is ambiguous")
        self.producer = {"pycolmap": pycolmap.__version__, "numpy": np.__version__, "Pillow": PIL.__version__,
            "core": file_identity(binaries[0]), "PIL_writer": file_identity(inspect.getsourcefile(original_save)),
            "adapter": file_identity(Path(__file__))}
        self.pending = None
        self.records = []
        self.owner = self.root/"OWNER.json"
        with self.owner.open("x", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "purpose": "exclusive calibrated reference writer"}, stream)

    def undistort(self, options, bitmap, camera):
        if self.pending is not None:
            raise RuntimeError("An undistorted reference has not been saved before the next query")
        started = time.perf_counter()
        raw = bitmap.to_array()
        if raw.dtype != np.uint8 or raw.shape != (camera.height, camera.width, 3):
            raise ValueError("Reference cache requires exact sensor-sized RGB8 pixels")
        key_input = {"schema": 1, "raw_rgb_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
            "raw_shape": list(raw.shape), "raw_dtype": raw.dtype.str, "camera": camera_fields(camera),
            "options": option_identity(options.todict()), "producer": self.producer}
        key = hashlib.sha256(json.dumps(key_input, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        folder = self.root/key
        metadata, pixels, png = folder/"ENTRY.json", folder/"pixels.npy", folder/"reference.png"
        hit = metadata.is_file()
        if hit:
            receipt = json.loads(metadata.read_text(encoding="utf-8"))
            if receipt["key_input"] != key_input or receipt["pixels"] != file_identity(pixels) or receipt["png"] != file_identity(png):
                raise ValueError("Calibrated reference identity or committed artifact integrity failed")
            if receipt["pixels"]["bytes"]+receipt["png"]["bytes"] > self.maximum_bytes:
                raise MemoryError("Reference cache artifact exceeds the current budget")
            array = np.load(pixels, allow_pickle=False)
            reference_camera = pycolmap.Camera(**receipt["camera"])
            result = pycolmap.Bitmap.from_array(array), reference_camera
        else:
            if folder.exists():
                raise RuntimeError("An interrupted reference entry needs explicit recovery")
            if self.disk_bytes() >= self.maximum_bytes:
                raise MemoryError("Reference cache disk budget exhausted")
            result = self.original_undistort(options, bitmap, camera)
            array = result[0].to_array()
            folder.mkdir(exist_ok=False)
            with pixels.open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
        if array.dtype != np.uint8 or array.shape != (result[1].height, result[1].width, 3):
            raise ValueError("Cached reference pixels and returned calibration differ")
        self.pending = {"key": key, "key_input": key_input, "folder": folder, "array": array,
            "camera": camera_fields(result[1]), "hit": hit, "undistort_seconds": time.perf_counter()-started}
        return result

    def save(self, image, target, format=None, **parameters):
        if self.pending is None or Path(target).suffix.lower() != ".png" or format is not None or parameters:
            raise ValueError("Reference cache supports only the preserved PNG save after undistortion")
        target = Path(target)
        if target.exists():
            raise FileExistsError("The fresh reference output already exists")
        started = time.perf_counter()
        pending = self.pending
        if image.mode != "RGB" or not np.array_equal(np.asarray(image), pending["array"]):
            raise ValueError("The caller changed pixels before reference PNG serialization")
        folder = pending["folder"]
        png, pixels = folder/"reference.png", folder/"pixels.npy"
        if pending["hit"]:
            shutil.copyfile(png, target)
        else:
            self.original_save(image, target)
            if self.disk_bytes()+target.stat().st_size > self.maximum_bytes:
                raise MemoryError("Complete calibrated artifacts would exceed the reference cache budget")
            shutil.copyfile(target, png)
            if file_identity(png) != file_identity(target):
                raise ValueError("Calibrated PNG artifact copy changed bytes")
            receipt = {"key_input": pending["key_input"], "pixels": file_identity(pixels),
                       "png": file_identity(png), "camera": pending["camera"]}
            temporary = folder/("ENTRY."+str(os.getpid())+".tmp")
            temporary.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
            temporary.replace(folder/"ENTRY.json")
        self.records.append({"key": pending["key"], "hit": pending["hit"],
            "undistort_seconds": pending["undistort_seconds"], "save_seconds": time.perf_counter()-started,
            "target": str(target), "PNG": file_identity(target), "camera": pending["camera"]})
        self.pending = None

    def disk_bytes(self):
        return sum(p.stat().st_size for p in self.root.glob("*/*") if p.name in ("pixels.npy", "reference.png"))

    def close(self):
        self.owner.unlink()

    def receipt(self):
        return {"hits": sum(r["hit"] for r in self.records), "misses": sum(not r["hit"] for r in self.records),
            "records": self.records, "cache_bytes": self.disk_bytes(), "maximum_bytes": self.maximum_bytes}


@contextmanager
def calibrated_reference_cache(root, enabled=True):
    original_undistort, original_save = pycolmap.undistort_image, Image.Image.save
    cache = CalibratedReferenceCache(root, original_undistort, original_save) if enabled else None
    records = []

    def undistort(options, bitmap, camera):
        started = time.perf_counter()
        value = original_undistort(options, bitmap, camera) if cache is None else cache.undistort(options, bitmap, camera)
        records.append({"operation": "undistort", "seconds": time.perf_counter()-started})
        return value

    def save(image, target, format=None, **parameters):
        started = time.perf_counter()
        if cache is None:
            value = original_save(image, target, format, **parameters)
        else:
            value = cache.save(image, target, format, **parameters)
        records.append({"operation": "PNG_save", "seconds": time.perf_counter()-started})
        return value

    pycolmap.undistort_image, Image.Image.save = undistort, save
    try:
        yield {"operations": records, "cache": cache}
    finally:
        pycolmap.undistort_image, Image.Image.save = original_undistort, original_save
        if cache is not None:
            cache.close()
