"""Index one owned reconstruction using native snapshots, with complete verification."""

from contextlib import contextmanager
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"),
    str(ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005")]
from runtime import data_identity
from binary_map_qa import image_arrays
from observation_index import indexed_match


def admission(model):
    images = list(model.images.values())
    if not 0 < len(images) <= 100_000:
        return False, "native_image_record_count"
    if any(not im.name or Path(im.name).name != im.name or len(im.name.encode()) > 4096 for im in images):
        return False, "explicit_binary_reader_name_contract"
    records = sum(im.num_points2D() for im in images)
    estimated = records*24 + len(model.points3D)*52 + model.compute_num_observations()*8 + len(images)*4200
    if records*25 > 128*1024**2 or estimated > 256*1024**2:
        return False, "complete_snapshot_and_index_budgets"
    return True, "one_owned_map_native_snapshot_budget_admitted"


def snapshot(model, target):
    target.mkdir(parents=True, exist_ok=False)
    model.write_binary(target)
    files = sorted(target.glob("*.bin"))
    if not {"cameras.bin", "images.bin", "points3D.bin", "frames.bin", "rigs.bin"}.issubset({p.name for p in files}):
        raise ValueError("The complete native map snapshot omitted a state component")
    if sum(p.stat().st_size for p in files) > 256*1024**2:
        raise MemoryError("The predeclared complete snapshot budget was exceeded")
    return {p.name: {k: v for k, v in data_identity(p).items() if k != "path"} for p in files}


class NativeObservationIndex:
    """Bind read-only arrays and final native verification to one private map."""

    def __init__(self, model, output):
        compatible, reason = admission(model)
        if not compatible:
            raise ValueError("An unadmitted map cannot allocate an index: "+reason)
        start = time.perf_counter()
        self.model, self.output = model, Path(output)
        self.before = snapshot(model, self.output/"before")
        images = image_arrays(self.output/"before/images.bin")
        self.rows = {row["id"]: row for row in images}
        if set(self.rows) != set(model.images):
            raise ValueError("The native snapshot image identities changed")
        self.cache, self.hits, self.misses = {}, 0, 0
        self.bytes = sum(row["data"].nbytes for row in images)
        self.build_seconds = time.perf_counter()-start
        self.verify_seconds, self.verified = 0., False

    def get(self, model, image_id):
        if model is not self.model:
            raise ValueError("A native observation index cannot cross reconstruction ownership")
        row, image = self.rows[image_id], model.images[image_id]
        if image.name != row["name"] or image.camera_id != row["camera_id"] or image.num_points2D() != len(row["data"]):
            raise ValueError("An indexed image metadata or observation count changed")
        if image_id not in self.cache:
            xy, ids = row["data"]["xy"], row["data"]["point_id"]
            if not np.isfinite(xy).all() or np.any(ids < -1):
                raise ValueError("An indexed native pixel or unobserved sentinel is invalid")
            valid = ids >= 0
            valid.setflags(write=False)
            self.bytes += valid.nbytes
            if self.bytes > 128*1024**2:
                raise MemoryError("The complete native observation index exceeded its budget")
            self.cache[image_id] = xy, ids, valid
            self.misses += 1
        else:
            self.hits += 1
        return self.cache[image_id]

    def verify_unchanged(self):
        start = time.perf_counter()
        after = snapshot(self.model, self.output/"after")
        self.verify_seconds = time.perf_counter()-start
        if self.before != after:
            raise ValueError("The complete owned reconstruction changed during indexed localization")
        self.verified = True

    def receipt(self):
        return {"native_images": len(self.rows), "indexed_images": len(self.cache), "owned_array_bytes": self.bytes,
            "hits": self.hits, "misses": self.misses, "cold_build_seconds": self.build_seconds,
            "complete_verify_seconds": self.verify_seconds, "complete_native_state_unchanged": self.verified,
            "all_native_map_components": list(self.before), "complete_snapshot_bytes": sum(v["bytes"] for v in self.before.values())}


@contextmanager
def native_observation_index(output, enabled=True):
    import feature_localization
    original = feature_localization.FeatureBank.match
    index = None
    durations, routes = [], []
    trace = {"enabled": enabled, "new_optimization_count": 0}

    def match(bank, model, keypoints, descriptors, neighbors, geometric=True):
        nonlocal index
        start = time.perf_counter()
        try:
            compatible, reason = admission(model) if index is None else (True, "existing_owned_index")
            routes.append({"route": "indexed" if enabled and compatible else "original_before_execution", "reason": reason})
            if enabled and compatible:
                if index is None:
                    index = NativeObservationIndex(model, Path(output)/"native_map")
                return indexed_match(bank, model, keypoints, descriptors, neighbors, index, geometric)
            return original(bank, model, keypoints, descriptors, neighbors, geometric)
        finally:
            durations.append(time.perf_counter()-start)

    feature_localization.FeatureBank.match = match
    try:
        yield trace
        if index is not None:
            index.verify_unchanged()
    finally:
        feature_localization.FeatureBank.match = original
        trace.update({"calls": len(durations), "CPU_geometry_seconds_including_all_snapshot_and_verify": sum(durations)
            +(index.verify_seconds if index else 0), "routes": routes,
            "index": index.receipt() if index else None})
