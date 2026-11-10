"""Verify feature bytes, errors, LRU identity, buffer ownership and complete close."""

import argparse
from pathlib import Path
import sqlite3
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
V1, V2 = (ROOT/("02_code/gaussian_pipeline_optimization_"+version+"_20261005") for version in ("v1", "v2"))
sys.path[:0] = [str(V1), str(V2), str(ROOT/"02_code/box2_lio_guided_sfm_v1_20261002")]
from runtime import atomic_json, data_identity
import sift_feature_store as native
import byte_sift_matching as byte
from feature_blob_views import feature_blob_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("feature_blob_views.py"), V2/"byte_sift_matching.py", V1/"sift_feature_store.py"]
    sources = [data_identity(path) for path in paths]
    atomic_json(args.output/"PROTOCOL.json", {"sources": sources, "immutable_bytes_only_new_route": True,
        "mutable_buffers_original_copy_route_before_execution": True, "all_arrays_values_errors_and_LRU_preserved": True,
        "input_pose_GT_object_shape_and_SfM_not_used": True, "complete_actual_three_cost_pairs_pending": True})
    rng = np.random.default_rng(321)
    records = {}
    for image_id, (count, width) in enumerate(((0, 4), (17, 2), (12001, 6)), 1):
        records[image_id] = ["generic_"+str(image_id), count, width,
            rng.normal(size=(count, width)).astype(np.float32).tobytes(), count, 128,
            rng.integers(0, 256, (count, 128), dtype=np.uint8).tobytes()]
    cases, errors = [], []
    fixture = SimpleNamespace(names={i: row[0] for i, row in records.items()})
    with byte.byte_sift_matching(True):
        original_values = {i: native.SiftFeatureStore.decode(fixture, i, row) for i, row in records.items()}
    for enabled in (False, True):
        with feature_blob_execution(enabled) as trace:
            with byte.byte_sift_matching(True):
                for image_id, row in records.items():
                    result = native.SiftFeatureStore.decode(fixture, image_id, row)
                    previous = original_values[image_id]
                    cases.append(result[0] == previous[0] and all(np.array_equal(a, b) and a.dtype == b.dtype and a.strides == b.strides
                        for a, b in zip(result[1:], previous[1:])) and (not enabled or row[1] == 0 or not result[1].flags.writeable and not result[2].flags.writeable))
                for change in ("name", "rows", "columns", "truncated"):
                    bad = list(records[2])
                    if change == "name": bad[0] = "different_name"
                    elif change == "rows": bad[4] += 1
                    elif change == "columns": bad[5] = 127
                    else: bad[3] = bad[3][:-1]
                    try:
                        native.SiftFeatureStore.decode(fixture, 2, bad)
                    except (ValueError, TypeError) as error:
                        errors.append((enabled, change, type(error).__name__, str(error)))
                mutable = list(records[2])
                mutable[3], mutable[6] = bytearray(mutable[3]), bytearray(mutable[6])
                result = native.SiftFeatureStore.decode(fixture, 2, mutable)
                mutable[3][:] = b"\x00"*len(mutable[3])
                mutable[6][:] = b"\x00"*len(mutable[6])
                cases.append(all(np.array_equal(a, b) and a.flags.writeable for a, b in zip(result[1:], original_values[2][1:])))
        cases.append(trace["all_function_bindings_restored"])
    database = args.output/"generic.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT)")
        connection.execute("CREATE TABLE keypoints(image_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB)")
        connection.execute("CREATE TABLE descriptors(image_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB)")
        for i, row in records.items():
            connection.execute("INSERT INTO images VALUES(?,?)", (i, row[0]))
            connection.execute("INSERT INTO keypoints VALUES(?,?,?,?)", (i, *row[1:4]))
            connection.execute("INSERT INTO descriptors VALUES(?,?,?,?)", (i, *row[4:7]))
    loaded = []
    for enabled in (False, True):
        with feature_blob_execution(enabled) as trace:
            with byte.byte_sift_matching(True):
                store = native.SiftFeatureStore(database, set(fixture.names.values()), "lazy", 2)
                try:
                    result = [store[i] for i in (2, 3, 2, 1, 3)]
                    stats = store.receipt()
                finally:
                    store.close()
                loaded.append((result, stats, trace))
                try:
                    store[2]
                except sqlite3.ProgrammingError:
                    closed_rejects = True
                else:
                    closed_rejects = False
                cases.append(closed_rejects and not store.cache)
    lifetime = all(a[0] == b[0] and all(np.array_equal(x, y) for x, y in zip(a[1:], b[1:]))
        for a, b in zip(loaded[0][0], loaded[1][0]))
    original_factory = byte.byte_sift_matching
    try:
        with feature_blob_execution(True) as failed:
            with byte.byte_sift_matching(True):
                raise RuntimeError("injected_read_consumer_failure")
    except RuntimeError:
        pass
    checks = {"all_counts_layouts_dtypes_strides_and_mutable_copy_routes_exact": all(cases),
        "all4_original_bad_input_exception_types_and_messages_exact": len(errors) == 8 and [x[1:] for x in errors[:4]] == [x[1:] for x in errors[4:]],
        "all5_real_SQLite_LRU_results_survive_eviction_and_close_exact": lifetime and loaded[0][1] == loaded[1][1],
        "all_real_connections_closed_and_reject_future_SQL_reads": all(row[2]["all_owned_store_closes"] == 1 for row in loaded),
        "all_bindings_restored_after_failure": failed["all_function_bindings_restored"] and byte.byte_sift_matching is original_factory,
        "sources_unchanged": sources == [data_identity(path) for path in paths]}
    result = {"status": "passed" if all(checks.values()) else "failed_verification", "sources": sources, "checks": checks,
        "actual_native_LRU_receipts": [row[1] for row in loaded], "counted": False, "new_object_full_pipeline_pending": True}
    atomic_json(args.output/"RESULT.json", result)
    print({"status": result["status"], "checks": checks}, flush=True)
    if not all(checks.values()):
        raise ValueError("An original immutable feature-read contract failed")


if __name__ == "__main__":
    main()
