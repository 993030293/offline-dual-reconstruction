"""Store proven image-only scores in one immutable numeric table with exact keyed misses."""

from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import threading
import time

import numpy as np

INTEGER_FIELDS = {"corner_count_half_resolution", "occupied_grid_cells_4x4"}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def source_contract(fields):
    return {"schema_version": 1, "fields": list(fields), "numpy": np.__version__,
        "adapter_source_sha256": digest(Path(__file__).read_bytes()),
        "representation": "original float64 and int64 values; exact original content keys"}


def dtype_for(fields):
    return np.dtype([(name, "<i8" if name in INTEGER_FIELDS else "<f8") for name in fields])


def checked_record(path, fields):
    path = Path(path)
    if path.stat().st_size > 32768:
        raise ValueError("An image-only score record exceeds its declared size")
    raw = path.read_bytes()
    record = json.loads(raw)
    key, scores = record["key"], record["quality"]
    if key != path.stem or len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
        raise ValueError("A source score key differs from its immutable filename")
    if set(scores) != set(fields):
        raise ValueError("The original score fields changed")
    for name, value in scores.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
            raise ValueError("An image-only score value is unsupported or nonfinite")
        if name in INTEGER_FIELDS and not isinstance(value, int):
            raise ValueError("An original integer score changed its exact representation")
    canonical = json.dumps(scores, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if record["quality_sha256"] != digest(canonical):
        raise ValueError("A source image-only score failed its original payload checksum")
    return key, scores, record["quality_sha256"]


def build_table(source, output, fields, expected_records):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    paths = sorted(source.glob("*.json"))
    if len(paths) != expected_records or not 0 < len(paths) <= 20000:
        raise ValueError("The source score record denominator changed")
    keys = np.empty(len(paths), dtype="S64")
    table = np.empty(len(paths), dtype=dtype_for(fields))
    lineage = hashlib.sha256()
    for index, path in enumerate(paths):
        key, scores, payload_digest = checked_record(path, fields)
        keys[index] = key.encode("ascii")
        for name in fields:
            table[name][index] = scores[name]
        reconstructed = {name: int(table[name][index]) if name in INTEGER_FIELDS else float(table[name][index]) for name in fields}
        if reconstructed != scores:
            raise ValueError("The numeric table did not preserve a complete original score")
        lineage.update((key+":"+payload_digest+"\n").encode())
    if any(a >= b for a, b in zip(keys, keys[1:])):
        raise ValueError("The numeric score keys are not unique and strictly sorted")
    payloads = {}
    for name, value in (("keys.npy", keys), ("scores.npy", table)):
        stream = io.BytesIO()
        np.save(stream, value, allow_pickle=False)
        payload = stream.getvalue()
        (output/name).write_bytes(payload)
        payloads[name] = {"bytes": len(payload), "sha256": digest(payload)}
    manifest = {"contract": source_contract(fields), "source_directory": str(source.resolve()),
        "source_record_count": len(paths), "source_key_and_score_digest": lineage.hexdigest(), "payloads": payloads,
        "original_score_source_modified": False, "temporary_source_files_ignored": len(list(source.glob("*.tmp")))}
    (output/"MANIFEST.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return {"status": "built_pending_exact_lifecycle_and_actual_pairs", "records": len(paths),
        "build_seconds": time.perf_counter()-started, "artifact_bytes": sum(p.stat().st_size for p in output.iterdir()),
        "original_score_files_bytes": sum(p.stat().st_size for p in paths), "manifest": manifest,
        "counted": False, "no_pose_or_selection_decisions_stored": True}


class ColumnarScoreCache:
    """Read exact old score keys and send new image-only misses to an independent cache."""

    def __init__(self, original_class, directory, contract, artifact, misses, fields):
        started = time.perf_counter()
        self.artifact = Path(artifact)
        self.misses_cache = original_class(misses, contract)
        manifest = json.loads((self.artifact/"MANIFEST.json").read_text(encoding="utf-8"))
        if manifest["contract"] != source_contract(fields) or Path(manifest["source_directory"]) != Path(directory).resolve():
            raise ValueError("The immutable score table does not belong to the current source or numeric schema")
        arrays = {}
        for name in ("keys.npy", "scores.npy"):
            payload = (self.artifact/name).read_bytes()
            expected = manifest["payloads"][name]
            if len(payload) != expected["bytes"] or digest(payload) != expected["sha256"]:
                raise ValueError("The immutable numeric score payload was modified")
            arrays[name] = np.load(io.BytesIO(payload), allow_pickle=False)
            arrays[name].setflags(write=False)
        self.keys, self.table = arrays["keys.npy"], arrays["scores.npy"]
        if (self.keys.dtype != np.dtype("S64") or self.keys.ndim != 1 or self.table.dtype != dtype_for(fields) or
                self.table.shape != self.keys.shape or len(self.keys) != manifest["source_record_count"]):
            raise ValueError("The immutable numeric score table changed its types or row identity")
        if any(a >= b for a, b in zip(self.keys, self.keys[1:])):
            raise ValueError("The immutable numeric score keys changed order or uniqueness")
        if not all(np.isfinite(self.table[name]).all() for name in fields):
            raise ValueError("The immutable numeric score table has nonfinite values")
        self.lookup = {bytes(key).decode("ascii"): index for index, key in enumerate(self.keys)}
        self.fields = tuple(fields)
        self.hits = self.misses = self.writes = 0
        self.lock = threading.Lock()
        self.initialization_seconds = time.perf_counter()-started
        self.manifest = manifest
        self.key = self.misses_cache.key

    def read(self, key):
        index = self.lookup.get(key)
        if index is None:
            with self.lock:
                self.misses += 1
            return self.misses_cache.read(key)
        row = self.table[index]
        values = {name: int(row[name]) if name in INTEGER_FIELDS else float(row[name]) for name in self.fields}
        with self.lock:
            self.hits += 1
        return values

    def write(self, key, scores):
        if key in self.lookup:
            raise ValueError("A committed immutable score cannot be silently overwritten")
        self.misses_cache.write(key, scores)
        with self.lock:
            self.writes += 1

    def receipt(self):
        return {"records": len(self.keys), "initialization_seconds": self.initialization_seconds,
            "hits": self.hits, "misses": self.misses, "writes_to_independent_miss_directory": self.writes,
            "numeric_payload_array_bytes": self.keys.nbytes+self.table.nbytes,
            "original_score_source_modified": False, "all_table_values_read_only": not self.table.flags.writeable}


@contextmanager
def columnar_score_cache(module, artifact, misses, fields):
    original = module.ScoreCache
    instances = []

    def factory(directory, contract):
        instance = ColumnarScoreCache(original, directory, contract, artifact, misses, fields)
        instances.append(instance)
        return instance

    module.ScoreCache = factory
    try:
        yield instances
    finally:
        module.ScoreCache = original
