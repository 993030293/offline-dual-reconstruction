"""Cache immutable image observations for one owned reconstruction instance."""

import time

import cv2
import numpy as np


class FrozenObservationIndex:
    """Keep geometry indexing separate from descriptor payload and matching."""

    def __init__(self, model, maximum_bytes=128*1024**2):
        self.model = model
        self.maximum_bytes = int(maximum_bytes)
        if self.maximum_bytes <= 0:
            raise ValueError("A geometry index requires a positive memory budget")
        self.cache = {}
        self.bytes = self.hits = self.misses = 0
        self.seconds = 0.

    def get(self, model, image_id):
        started = time.perf_counter()
        if model is not self.model:
            raise ValueError("A frozen observation index belongs to one map instance")
        image = model.images[image_id]
        if image_id in self.cache:
            self.hits += 1
            xy, ids, valid = self.cache[image_id]
            if len(xy) != image.num_points2D():
                raise ValueError("The frozen map observation count changed")
        else:
            points = image.points2D
            xy = np.asarray([p.xy for p in points], dtype=np.float64).reshape(-1, 2)
            ids = np.asarray([int(p.point3D_id) for p in points], dtype=np.uint64)
            valid = np.asarray([p.has_point3D() for p in points], dtype=bool)
            size = xy.nbytes+ids.nbytes+valid.nbytes
            if self.bytes+size > self.maximum_bytes:
                raise MemoryError("The registered geometry index memory budget was exceeded")
            if not np.isfinite(xy).all():
                raise ValueError("Map observation pixels must be finite")
            for value in (xy, ids, valid):
                value.setflags(write=False)
            self.cache[image_id] = xy, ids, valid
            self.bytes += size
            self.misses += 1
        self.seconds += time.perf_counter()-started
        return xy, ids, valid

    def verify_unchanged(self):
        for image_id, (xy, ids, valid) in self.cache.items():
            points = self.model.images[image_id].points2D
            current = (np.asarray([p.xy for p in points], dtype=np.float64).reshape(-1, 2),
                       np.asarray([int(p.point3D_id) for p in points], dtype=np.uint64),
                       np.asarray([p.has_point3D() for p in points], dtype=bool))
            if not all(np.array_equal(a, b) for a, b in zip((xy, ids, valid), current)):
                raise ValueError("The privately owned frozen reconstruction was modified")

    def receipt(self):
        return {"images": len(self.cache), "bytes": self.bytes, "maximum_bytes": self.maximum_bytes,
                "hits": self.hits, "misses": self.misses, "lookup_build_seconds": self.seconds}


def indexed_match(bank, model, keypoints, descriptors, neighbors, index, geometric=True):
    """Preserve the existing BF/F/ordering logic and replace geometry reads only."""
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    correspondences, edges, all_pairs = [], [], []
    for image_id in neighbors:
        name, train_keypoints, train_descriptors = bank.original[image_id]
        pixels, point_ids, valid = index.get(model, image_id)
        if len(train_keypoints) != len(pixels):
            raise ValueError("Original keypoint and map observation identity mismatch")
        if not np.allclose(train_keypoints[:, :2], pixels, atol=1e-3):
            raise ValueError("Original SIFT pixels and map pixels disagree")
        nearest = matcher.knnMatch(descriptors, train_descriptors, k=2)
        reverse = matcher.match(train_descriptors, descriptors)
        reverse_map = {m.queryIdx: m.trainIdx for m in reverse}
        pairs = [m for pair in nearest if len(pair) == 2 for m, n in [pair]
                 if m.distance < .8*n.distance and m.distance <= .7
                 and reverse_map.get(m.trainIdx) == m.queryIdx]
        raw_count = len(pairs)
        matrix = None
        if geometric and len(pairs) >= 15:
            xy1 = np.asarray([keypoints[m.queryIdx, :2] for m in pairs])
            xy2 = np.asarray([train_keypoints[m.trainIdx, :2] for m in pairs])
            cv2.setRNGSeed(20261002)
            matrix, mask = cv2.findFundamentalMat(xy1, xy2, cv2.FM_RANSAC, 4., .999, 10000)
            pairs = [] if matrix is None or mask is None else [m for m, ok in zip(pairs, mask.ravel()) if ok]
        if geometric and len(pairs) < 15:
            pairs = []
        edges.append({"train_image_id": image_id, "train_image_name": name,
                      "mutual_ratio_matches": raw_count, "geometric_inliers": len(pairs),
                      "F_from_query_localization_features": None if matrix is None else matrix.tolist()})
        for match in pairs:
            all_pairs.append((image_id, match.queryIdx, match.trainIdx, match.distance))
            if valid[match.trainIdx]:
                correspondences.append((match.distance, match.queryIdx, int(point_ids[match.trainIdx]),
                                        image_id, match.trainIdx))
    unique, used_query, used_point = [], set(), set()
    for distance, query, point_id, image_id, train_index in sorted(correspondences):
        if query in used_query or point_id in used_point:
            continue
        used_query.add(query)
        used_point.add(point_id)
        unique.append({"query": query, "point_id": point_id, "train_image_id": image_id,
                       "train_index": train_index, "distance": float(distance)})
    return unique, edges, all_pairs
