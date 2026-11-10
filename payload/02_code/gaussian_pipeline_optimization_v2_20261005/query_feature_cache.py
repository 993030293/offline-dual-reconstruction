"""Reuse exact map-independent query SIFT artifacts with bounded provenance."""

import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
import pycolmap


def file_identity(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024**2), b""):
            digest.update(block)
    return {"bytes": Path(path).stat().st_size, "sha256": digest.hexdigest()}


def option_identity(value):
    if isinstance(value, dict):
        return {k: option_identity(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [option_identity(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if type(value).__module__.startswith("pycolmap") and hasattr(value, "name") and hasattr(value, "value"):
        return {"enum": type(value).__qualname__, "name": value.name, "value": value.value}
    raise TypeError("Unregistered SIFT option type: "+type(value).__qualname__)


class QueryFeatureCache:
    """Keep features independent of the map, neighbors, PnP and evaluation ROI."""

    def __init__(self, root, original_extra, maximum_bytes=512*1024**2):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or maximum_bytes <= 0:
            raise ValueError("A query feature cache needs an owned directory and positive budget")
        self.maximum_bytes = int(maximum_bytes)
        self.original_extra = original_extra
        options = pycolmap.FeatureExtractionOptions()
        options.sift.max_num_features, options.sift.peak_threshold = 12000, .004
        binaries = list(Path(pycolmap.__file__).parent.glob("_core*.pyd"))
        if len(binaries) != 1:
            raise ValueError("The pinned CPU SIFT backend identity is ambiguous")
        self.extractor_identity = {
            "options": option_identity(options.todict()), "device": "CPU", "pycolmap": pycolmap.__version__,
            "cv2": cv2.__version__, "numpy": np.__version__, "cv2_threads": 8,
            "feature_bank_source": file_identity(inspect.getsourcefile(original_extra)),
            "pycolmap_binary": file_identity(binaries[0])}
        self.records = []
        self.hits = self.misses = 0

    def __call__(self, bank, path):
        path = Path(path)
        started = time.perf_counter()
        if bank.protocol["SIFT_max_features"] != 12000 or bank.protocol["SIFT_peak_threshold"] != .004:
            raise ValueError("The preserved query SIFT options changed")
        if (option_identity(bank.extractor.options.todict()) != self.extractor_identity["options"]
                or bank.extractor.device != pycolmap.Device.cpu or cv2.getNumThreads() != 8):
            raise ValueError("The actual extractor options, device or thread contract changed")
        key_input = {"schema": 1, "raw_image": file_identity(path), "extractor": self.extractor_identity}
        key = hashlib.sha256(json.dumps(key_input, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        target = bank.cache/(path.stem+".npz")
        if target.exists():
            raise FileExistsError("The fresh query output artifact already exists")
        folder = self.root/key
        metadata, payload = folder/"ENTRY.json", folder/"features.npz"
        hit = metadata.is_file()
        if hit:
            receipt = json.loads(metadata.read_text(encoding="utf-8"))
            if receipt["key_input"] != key_input or receipt["payload"] != file_identity(payload):
                raise ValueError("Query feature cache identity or payload integrity failed")
            if receipt["payload"]["bytes"] > self.maximum_bytes:
                raise MemoryError("A cached query artifact exceeds the current admission budget")
            shutil.copyfile(payload, target)
            self.hits += 1
        else:
            if folder.exists():
                raise RuntimeError("An incomplete query feature cache entry requires explicit recovery")
            current_bytes = sum(p.stat().st_size for p in self.root.glob("*/features.npz"))
            if current_bytes >= self.maximum_bytes:
                raise MemoryError("Query feature cache budget is exhausted")
            values = self.original_extra(bank, path)
            if not target.is_file():
                raise ValueError("The original SIFT path did not create its complete artifact")
            identity = file_identity(target)
            if current_bytes+identity["bytes"] > self.maximum_bytes:
                raise MemoryError("A complete query feature artifact would exceed the cache budget")
            folder.mkdir(exist_ok=False)
            shutil.copyfile(target, payload)
            if file_identity(payload) != identity:
                raise ValueError("Query feature cache copy changed the payload")
            temporary = folder/("ENTRY."+str(os.getpid())+".tmp")
            temporary.write_text(json.dumps({"key_input": key_input, "payload": identity}, sort_keys=True), encoding="utf-8")
            temporary.replace(metadata)
            self.misses += 1
        with np.load(target, allow_pickle=False) as stored:
            if set(stored.files) != {"keypoints", "descriptors"}:
                raise ValueError("The original query feature artifact schema changed")
            keypoints, descriptors = stored["keypoints"].copy(), stored["descriptors"].copy()
        if len(keypoints) != len(descriptors) or descriptors.ndim != 2 or descriptors.shape[1] != 128:
            raise ValueError("Query feature keypoint and descriptor shapes differ")
        self.records.append({"name": path.name, "key": key, "seconds": time.perf_counter()-started,
                             "hit": hit,
                             "feature_count": len(keypoints), "payload": file_identity(target)})
        return keypoints, descriptors

    def receipt(self):
        return {"hits": self.hits, "misses": self.misses, "records": self.records,
                "maximum_bytes": self.maximum_bytes,
                "cache_bytes": sum(p.stat().st_size for p in self.root.glob("*/features.npz"))}
