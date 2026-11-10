"""Read immutable SIFT rows eagerly or through a bounded per-process LRU."""

from collections import OrderedDict
from pathlib import Path
import sqlite3

import numpy as np


class SiftFeatureStore:
    def __init__(self, database, registered_names, mode="lazy", capacity=32):
        if mode not in ("eager", "lazy") or capacity <= 0:
            raise ValueError("Invalid feature storage mode or capacity")
        self.mode, self.capacity = mode, int(capacity)
        self.connection = sqlite3.connect(Path(database).resolve().as_uri()+"?mode=ro", uri=True)
        self.names = {int(i): name for i, name in self.connection.execute(
            "SELECT image_id,name FROM images") if name in registered_names}
        if len(self.names) != len(registered_names):
            self.close()
            raise ValueError("The database does not uniquely cover the registered image names")
        self.cache = OrderedDict()
        self.hits = self.misses = self.evictions = 0
        self.maximum_cached_images = self.maximum_cached_array_bytes = 0
        if mode == "eager":
            query = "SELECT images.image_id,images.name,keypoints.rows,keypoints.cols,keypoints.data,descriptors.rows,descriptors.cols,descriptors.data FROM images JOIN keypoints USING(image_id) JOIN descriptors USING(image_id)"
            for image_id, *record in self.connection.execute(query):
                if image_id in self.names:
                    self.cache[image_id] = self.decode(image_id, record)
            if set(self.cache) != set(self.names):
                self.close()
                raise ValueError("An eager feature record is missing")
            self.record_peak()

    def decode(self, image_id, record):
        name, kr, kc, kd, dr, dc, dd = record
        if name != self.names[image_id] or kr != dr or dc != 128:
            raise ValueError("SIFT database row identity or descriptor shape changed")
        keypoints = np.frombuffer(kd, np.float32).reshape(kr, kc).copy()
        descriptors = np.frombuffer(dd, np.uint8).reshape(dr, dc).astype(np.float32)/512.
        return name, keypoints, descriptors

    def record_peak(self):
        self.maximum_cached_images = max(self.maximum_cached_images, len(self.cache))
        self.maximum_cached_array_bytes = max(self.maximum_cached_array_bytes,
            sum(k.nbytes+d.nbytes for _, k, d in self.cache.values()))

    def __contains__(self, image_id):
        return image_id in self.names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, image_id):
        if image_id not in self.names:
            raise KeyError(image_id)
        if image_id in self.cache:
            self.hits += 1
            if self.mode == "lazy":
                self.cache.move_to_end(image_id)
            return self.cache[image_id]
        if self.mode == "eager":
            raise KeyError("Missing eager feature record")
        query = "SELECT images.name,keypoints.rows,keypoints.cols,keypoints.data,descriptors.rows,descriptors.cols,descriptors.data FROM images JOIN keypoints USING(image_id) JOIN descriptors USING(image_id) WHERE images.image_id=?"
        record = self.connection.execute(query, (int(image_id),)).fetchone()
        if record is None:
            raise ValueError("A lazy feature record is missing")
        value = self.decode(image_id, record)
        if len(self.cache) == self.capacity:
            self.cache.popitem(last=False)
            self.evictions += 1
        self.cache[image_id] = value
        self.misses += 1
        self.record_peak()
        return value

    def receipt(self):
        return {"mode": self.mode, "capacity": self.capacity, "metadata_images": len(self.names),
            "cached_images": len(self.cache), "cached_array_bytes": sum(k.nbytes+d.nbytes for _, k, d in self.cache.values()),
            "maximum_cached_images": self.maximum_cached_images,
            "maximum_cached_array_bytes": self.maximum_cached_array_bytes,
            "hits": self.hits, "misses": self.misses, "evictions": self.evictions}

    def close(self):
        self.connection.close()
        if hasattr(self, "cache"):
            self.cache.clear()


def metadata_neighbors(store, model, rough, timestamp):
    """Preserve the original ordering without fetching descriptor payloads for names."""
    available = [i for i in sorted(model.images) if i in store]
    center, forward = np.asarray(rough["C_W_m"]), np.asarray(rough["R_WC"])[:, 2]
    temporal = sorted(available, key=lambda i: abs(int(store.names[i].split("_")[2].split(".")[0])-timestamp))[:4]
    scores = {}
    for image_id in available:
        image = model.images[image_id]
        view = image.cam_from_world().rotation.matrix().T[:, 2]
        scores[image_id] = np.linalg.norm(image.projection_center()-center)+.5*(1-np.clip(view@forward, -1, 1))
    spatial = sorted(available, key=lambda i: scores[i])[:8]
    return list(dict.fromkeys(temporal+spatial))
