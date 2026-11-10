"""Select neighbors from a process-local immutable camera snapshot."""

import numbers

import numpy as np


class FrozenCameraNeighbors:
    """Build once after loading a fixed map; rebuild after every map change."""

    def __init__(self, store, model):
        self.ids = np.asarray([i for i in sorted(model.images) if i in store], dtype=np.int64)
        if not len(self.ids):
            raise ValueError("A camera snapshot requires registered feature-bearing images")
        self.centers = np.stack([model.images[int(i)].projection_center() for i in self.ids])
        self.forwards = np.stack([model.images[int(i)].cam_from_world().rotation.matrix().T[:, 2]
                                  for i in self.ids])
        self.timestamps = np.asarray([int(store.names[int(i)].split("_")[2].split(".")[0])
                                      for i in self.ids], dtype=np.int64)
        if (not np.isfinite(self.centers).all() or not np.isfinite(self.forwards).all()
                or np.any(self.timestamps < 0)):
            raise ValueError("Camera metadata is non-finite or has invalid time")
        for array in (self.ids, self.centers, self.forwards, self.timestamps):
            array.setflags(write=False)
        self.queries = self.near_tie_fallbacks = 0

    def __call__(self, rough, timestamp):
        center = np.asarray(rough["C_W_m"], dtype=np.float64)
        rotation = np.asarray(rough["R_WC"], dtype=np.float64)
        if (center.shape != (3,) or rotation.shape != (3, 3)
                or not np.isfinite(center).all() or not np.isfinite(rotation).all()):
            raise ValueError("A query must contain a finite camera center and 3x3 rotation")
        if (not isinstance(timestamp, numbers.Integral) or isinstance(timestamp, bool)
                or timestamp < 0 or timestamp > np.iinfo(np.int64).max):
            raise ValueError("Query time must be nonnegative integer nanoseconds in int64 range")
        forward = rotation[:, 2]
        temporal = np.argsort(np.abs(self.timestamps - int(timestamp)), kind="stable")[:4]
        scores = (np.linalg.norm(self.centers-center, axis=1)
                  + .5*(1-np.clip(np.sum(self.forwards*forward, axis=1), -1, 1)))
        if not np.isfinite(scores).all():
            raise ValueError("Camera scoring overflowed")
        order = np.argsort(scores, kind="stable")
        boundary = order[:min(9, len(order))]
        tolerance = 64*np.finfo(np.float64).eps*max(1., float(np.max(np.abs(scores))))
        if len(boundary) > 1 and np.any(np.diff(scores[boundary]) <= tolerance):
            # Match the scalar reference near ties instead of changing stable order.
            exact = [np.linalg.norm(c-center)+.5*(1-np.clip(v@forward, -1, 1))
                     for c, v in zip(self.centers, self.forwards)]
            order = np.asarray(sorted(range(len(self.ids)), key=lambda i: exact[i]))
            self.near_tie_fallbacks += 1
        self.queries += 1
        return list(dict.fromkeys(int(self.ids[i]) for i in [*temporal, *order[:8]]))

    def receipt(self):
        return {"camera_count": len(self.ids), "queries": self.queries,
                "near_tie_fallbacks": self.near_tie_fallbacks,
                "snapshot_array_bytes": sum(a.nbytes for a in
                    (self.ids, self.centers, self.forwards, self.timestamps)),
                "cache_scope": "this immutable map and this process only"}
