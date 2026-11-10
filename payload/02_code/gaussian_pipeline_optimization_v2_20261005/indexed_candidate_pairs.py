"""Generate the same coarse spatial candidate graph using an exact local KD tree."""

from contextlib import contextmanager
import inspect
import time

import numpy as np
from scipy.spatial import cKDTree


def candidate_pairs(selected, indexed=True):
    positions = np.asarray([r["C_W_m"] for r in selected], dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
        raise ValueError("Candidate pairing requires finite metric camera positions")
    count, fallback = len(positions), 0
    if count < 2:
        raise ValueError("Candidate pairing requires at least two distinct image identities")
    if len({r["image_name"] for r in selected}) != count:
        raise ValueError("Candidate pairing received duplicate image names")
    if np.abs(positions).max() > 1e150:
        raise ValueError("Camera coordinates exceed the safe squared-distance range")
    if indexed:
        tree = cKDTree(positions, copy_data=True)
        distances, nearest = tree.query(positions, k=min(18, count), workers=1)
        if not np.isfinite(distances).all() or (nearest < 0).any() or (nearest >= count).any():
            raise ValueError("The exact spatial query returned an invalid distance or index")
    pairs = set()
    for i in range(count):
        for j in range(i+1, min(count, i+9)):
            pairs.add((i, j))
        if indexed:
            candidates = nearest[i]
            exact = np.linalg.norm(positions[candidates]-positions[i], axis=1)
            order = np.argsort(exact)
            tolerance = 64*np.finfo(np.float64).eps*max(1., float(exact.max()))
            ambiguous = (count <= 17 or exact[order[17]]-exact[order[16]] <= tolerance)
            if ambiguous:
                neighbors = np.argsort(np.linalg.norm(positions-positions[i], axis=1))[:17]
                fallback += 1
            else:
                neighbors = candidates[order[:17]]
        else:
            neighbors = np.argsort(np.linalg.norm(positions-positions[i], axis=1))[:17]
        for j in neighbors:
            if abs(int(j)-i) > 8:
                pairs.add(tuple(sorted((i, int(j)))))
    return pairs, {"frames": count, "pairs": len(pairs), "indexed": indexed,
        "ambiguous_boundary_full_reference_calls": fallback, "tree_scope": "one immutable input and one call"}


@contextmanager
def indexed_pair_generation(module, enabled=True):
    original = module.main
    source = inspect.getsource(original)
    before = '''    positions = np.asarray([row["C_W_m"] for row in selected])
    pairs = set()
    for i, row in enumerate(selected):
        for j in range(i+1, min(len(selected), i+9)):
            pairs.add((i, j))
        for j in np.argsort(np.linalg.norm(positions-positions[i], axis=1))[:17]:
            if abs(int(j)-i) > 8:
                pairs.add(tuple(sorted((i, int(j)))))
'''
    if source.count(before) != 1:
        raise ValueError("The preserved initial-pairing block changed; refuse approximate replacement")
    records = []

    def generate(selected):
        started = time.perf_counter()
        pairs, receipt = candidate_pairs(selected, enabled)
        records.append({**receipt, "seconds": time.perf_counter()-started})
        return pairs

    namespace = dict(module.__dict__)
    namespace["_exact_candidate_pairs"] = generate
    exec(compile(source.replace(before, "    pairs = _exact_candidate_pairs(selected)\n"),
                 str(module.__file__), "exec"), namespace)
    module.main = namespace["main"]
    try:
        yield records
    finally:
        module.main = original
