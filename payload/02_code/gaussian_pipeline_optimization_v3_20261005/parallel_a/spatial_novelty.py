"""Restrict the original novelty predicate to provably nearby selected centers."""

from collections import defaultdict
from contextlib import contextmanager
import inspect
from itertools import product
import time

import numpy as np

OFFSETS = tuple(product(range(-2, 3), repeat=3))


class NoveltyIndex:
    def __init__(self, positions, rotations, radius, angle, trace):
        self.positions, self.rotations = positions, rotations
        self.radius, self.angle, self.trace = radius, angle, trace
        self.cells = defaultdict(list)
        self.rank = {}
        self.enabled = (positions.dtype == np.float64 and positions.ndim == 2 and positions.shape[1] == 3
            and np.isfinite(positions).all() and np.isfinite(radius) and 2.**-450 <= radius <= 2.**450
            and np.isfinite(rotations.as_quat()).all()
            and np.max(np.abs(positions), initial=0.)/radius <= 2.**40)
        self.keys = np.floor(positions/radius).astype(np.int64) if self.enabled else None
        trace["hash_instances" if self.enabled else "native_instances"] += 1

    def add(self, index):
        if self.enabled:
            self.rank[index] = len(self.rank)
            self.cells[tuple(self.keys[index])].append(index)

    def native(self, index, selected):
        chosen = np.asarray(selected, dtype=int)
        distances = np.linalg.norm(self.positions[index]-self.positions[chosen], axis=1)
        angles = np.rad2deg((self.rotations[index].inv()*self.rotations[chosen]).magnitude())
        return bool(np.all((distances >= self.radius) | (angles >= self.angle)))

    def novel(self, index, selected):
        if not self.enabled or not selected:
            self.trace["native_queries"] += 1
            return self.native(index, selected)
        key = self.keys[index]
        found = []
        for dx, dy, dz in OFFSETS:
            found.extend(self.cells.get((key[0]+dx, key[1]+dy, key[2]+dz), ()))
        found.sort(key=self.rank.__getitem__)
        chosen = np.asarray(found, dtype=int)
        distances = np.linalg.norm(self.positions[index]-self.positions[chosen], axis=1)
        near = chosen[distances < self.radius]
        self.trace["hash_queries"] += 1
        self.trace["selected_rows_considered"] += len(selected)
        self.trace["actual_angular_rows"] += len(near)
        if not len(near):
            return True
        angles = np.rad2deg((self.rotations[index].inv()*self.rotations[near]).magnitude())
        return bool(np.all(angles >= self.angle))


@contextmanager
def spatial_novelty_execution(enabled=True):
    import selection_engine as native
    import run_initial_pipeline as entry
    started = time.perf_counter()
    original, entry_original = native.choose_batched, entry.choose_batched
    if original is not entry_original:
        raise ValueError("The original selector aliases have different ownership")
    trace = {"enabled": enabled, "hash_instances": 0, "native_instances": 0, "hash_queries": 0,
        "native_queries": 0, "selected_rows_considered": 0, "actual_angular_rows": 0,
        "capacity_does_not_cut_frames_or_features": True}
    if enabled:
        source = inspect.getsource(original)
        before = '''        index = indices[row["camera_frame"]]
        chosen = np.asarray(chosen_indices, dtype=int)
        distances = np.linalg.norm(positions[index]-positions[chosen], axis=1)
        # SciPy performs the same normalized relative quaternion operation in one batch.
        angles = np.rad2deg((rotations[index].inv()*rotations[chosen]).magnitude())
        novel = bool(np.all((distances >= policy["translation_min_m"]) |
                            (angles >= policy["rotation_min_degrees"])))'''
        replacements = [("    chosen_indices = []", "    chosen_indices = []\n    novelty = _NoveltyIndex(positions, rotations, policy['translation_min_m'], policy['rotation_min_degrees'], _trace)"),
            ('            chosen_indices.append(indices[row["camera_frame"]])',
                '            chosen_indices.append(indices[row["camera_frame"]])\n            novelty.add(indices[row["camera_frame"]])'),
            (before, '        index = indices[row["camera_frame"]]\n        novel = novelty.novel(index, chosen_indices)')]
        for old, new in replacements:
            if source.count(old) != 1:
                raise ValueError("A complete original novelty selector source anchor changed")
            source = source.replace(old, new)
        namespace = dict(original.__globals__)
        namespace.update(_NoveltyIndex=NoveltyIndex, _trace=trace)
        exec(compile(source, "<original_selection_with_bounded_spatial_novelty>", "exec"), namespace)
        native.choose_batched = entry.choose_batched = namespace[original.__name__]
    trace["setup_seconds"] = time.perf_counter()-started
    try:
        yield trace
    finally:
        started = time.perf_counter()
        native.choose_batched, entry.choose_batched = original, entry_original
        trace["restore_seconds"] = time.perf_counter()-started
        trace["aliases_restored"] = native.choose_batched is original and entry.choose_batched is entry_original
