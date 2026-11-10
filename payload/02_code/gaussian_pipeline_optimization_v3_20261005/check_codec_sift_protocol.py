"""Exercise complete real raw and encoded requests against the native GPU math."""

import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from codec_sift_matching import CodecObservedWorker, read_server_trace

OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup018/R170/g0"


def vector_identity(value):
    return {"dtype": str(value.dtype), "shape": list(value.shape),
            "bytes": hashlib.sha256(value.tobytes()).hexdigest()}


def ended(worker, failed=False):
    if not failed:
        worker.close()
    else:
        worker.process.stdin.close()
        worker.process.stdout.close()
        code = worker.process.wait(timeout=30)
        worker.log.close()
        if code == 0:
            raise AssertionError("A real server error falsely exited successfully")
    resource = json.loads(worker.resource.read_text(encoding="utf-8"))
    if not resource["closed"] or resource["peak_RSS_bytes"] > resource["maximum_RSS_bytes"]:
        raise AssertionError("The real native worker did not close within its RSS bound")
    return resource


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), *[CODE / name for name in ("codec_sift_matching.py", "sift_payload_input_worker.py",
        "codec_owned_matching_worker.py", "bounded_zstd_payload.py", "integer_sift_worker.py",
        "integer_sift_matching.py", "metric_branch_localization_003.py", "fused_owned_matching_worker.py")],
        ROOT / "05_envs/colmap-4.1.1-cuda/bin/zstd.dll"]
    source_ids = [data_identity(path) for path in sources]
    archive = OUT / "source_snapshot"
    archive.mkdir()
    for index, path in enumerate(sources):
        shutil.copyfile(path, archive / (f"{index:03d}_" + path.name))
    rng = np.random.default_rng(170001)
    fixtures = []
    for number, (nq, counts, kind) in enumerate(((1, [2], "zero"), (31, [17, 33], "repeated"),
            (129, [257, 65], "high"), (65, [33, 51, 17], "random"), (257, [129, 65], "strided"))):
        def values(count):
            data = rng.integers(0, 256, size=(count, 128), dtype=np.uint8)
            if kind == "zero":
                data.fill(0)
            elif kind == "repeated":
                data[:, 0::2] = 0
                data[:, 1::2] %= 8
            elif kind == "high":
                data[:] = np.where(data > 128, 255, 128)
            value = data.astype(np.float32) / 512
            if kind == "strided":
                value = np.concatenate((value, value), axis=1)[:, ::2]
            return value
        fixtures.append({"name": "generic" + str(number), "query": values(nq),
                         "trains": [values(count) for count in counts]})
    database = ROOT / "04_outputs/public_full_e26_scene_v3_20260929T141800Z/attempt02/01_visual_workspace/database.db"
    database_id = data_identity(database)
    with sqlite3.connect("file:" + database.as_posix() + "?mode=ro", uri=True) as connection:
        selected = connection.execute("SELECT image_id,rows,cols,data FROM descriptors WHERE rows >= 2 ORDER BY image_id LIMIT 6").fetchall()
    public = []
    for image_id, rows, columns, data in selected:
        if columns != 128 or not 2 <= rows <= 12000 or len(data) != rows * columns:
            raise ValueError("The actual complete public descriptor contract changed")
        public.append(np.frombuffer(data, np.uint8).reshape(rows, columns).astype(np.float32) / 512)
    if len(public) != 6:
        raise ValueError("All six public descriptor members are required")
    fixtures.extend([{"name": "public0", "query": public[0], "trains": public[1:4]},
                     {"name": "public1", "query": public[3], "trains": [public[4], public[5], public[0]]}])
    sides = {}
    for enabled in (False, True):
        label = "codec" if enabled else "raw"
        folder = OUT / label
        folder.mkdir()
        log = folder / "worker.log"
        worker = CodecObservedWorker(log, enabled)
        rows, errors = [], []
        try:
            for fixture in fixtures:
                before = [vector_identity(value) for value in [fixture["query"], *fixture["trains"]]]
                values = worker.batch(fixture["query"], fixture["trains"])
                rows.append({"name": fixture["name"], "values": [[vector_identity(part) for part in value] for value in values],
                             "inputs_unchanged": before == [vector_identity(value) for value in [fixture["query"], *fixture["trains"]]]})
            value = worker.match(fixtures[1]["query"], fixtures[1]["trains"][0])
            rows.append({"name": "original_nearest_raw_route", "values": [vector_identity(part) for part in value],
                         "inputs_unchanged": True})
            for name, bad in (("float64", fixtures[0]["query"].astype(np.float64)),
                              ("nonfinite", np.full((1, 128), np.nan, np.float32))):
                try:
                    worker.batch(bad, fixtures[0]["trains"])
                except ValueError as error:
                    errors.append({"name": name, "type": type(error).__name__, "message": str(error)})
                else:
                    raise AssertionError("A malformed original host descriptor was accepted")
        finally:
            resource = ended(worker)
        sides[label] = {"rows": rows, "host_errors": errors, "receipt": worker.receipt,
                        "resource": resource, "server": read_server_trace(log) if enabled else None}
    a, b = sides["raw"], sides["codec"]
    body_rows = [row for row in b["receipt"]["wire"] if row["command"] in ("batch", "nearest")]
    checks = {"eight_real_native_GPU_calls_all_complete_vectors_exact": len(a["rows"]) == len(b["rows"]) == 8 and a["rows"] == b["rows"],
              "original_host_errors_exact_and_inputs_preserved": a["host_errors"] == b["host_errors"]
                and len(a["host_errors"]) == 2 and all(row["inputs_unchanged"] for row in a["rows"] + b["rows"]),
              "actual_encoded_and_explicit_raw_paths_reached": any(row["codec"] == "zstd1" for row in body_rows)
                and any(row["codec"] == "raw" for row in body_rows),
              "all_real_decoder_complete_input_hashes_match_client": len(b["server"]["rows"]) == len(body_rows)
                and all(server["raw_sha256"] == client["raw_sha256"] and server["raw_bytes"] == client["raw_payload_bytes"]
                        and server["wire_payload_bytes"] == client["transmitted_payload_bytes"]
                        for server, client in zip(b["server"]["rows"], body_rows)),
              "actual_decoder_stdin_alias_and_native_stop_closed": b["server"]["stdin_alias_restored"]
                and all(side["receipt"]["returncode"] == 0 and side["resource"]["closed"] for side in sides.values()),
              "actual_pipe_write_meter_census": all(sum(row["actual_written_input_bytes"] for row in side["receipt"]["wire"])
                == side["receipt"]["actual_written_input_bytes"] for side in sides.values()),
              "public_original_database_unchanged": database_id == data_identity(database),
              "all_current_sources_and_library_unchanged": source_ids == [data_identity(path) for path in sources]}
    errors = []
    for enabled in (False, True):
        label = "bad_shape_" + str(int(enabled))
        log = OUT / (label + ".log")
        worker = CodecObservedWorker(log, enabled)
        try:
            worker.rpc({"command": "batch", "query_rows": 0, "train_counts": [2]})
        except RuntimeError as error:
            failure = {"type": type(error).__name__, "message": str(error)}
        else:
            raise AssertionError("The real native server shape error was not propagated")
        resource = ended(worker, failed=True)
        errors.append({"enabled": enabled, "error": failure, "wire": worker.receipt["wire"], "resource": resource})
    checks["two_real_original_server_errors_propagated_and_closed"] = (errors[0]["error"] == errors[1]["error"]
        and all(row["resource"]["closed"] for row in errors))
    result = {"status": "native_protocol_components_passed_pending_more_errors_and_full_pairs0" if all(checks.values()) else "failed",
              "checks": checks, "sides": sides, "actual_server_errors": errors,
              "public_database": database_id, "public_descriptor_member_ids": [row[0] for row in selected],
              "sources": source_ids, "complete_component_seconds": time.perf_counter() - started,
              "extra_actual_written_input_bytes": sum(side["receipt"]["actual_written_input_bytes"] for side in sides.values())
                + sum(item["actual_written_input_bytes"] for row in errors for item in row["wire"]),
              "new_optimization_count": 0, "not_full_localization_training_unknown_object_GT_or_speed": True,
              "actual_invalid_codec_and_truncated_wire_full_callbacks_SIGINT_recovery_not_run": True}
    atomic_json(OUT / "RESULT.json", result)
    if not all(checks.values()):
        raise AssertionError("A real complete native protocol component failed")
    print(json.dumps({key: result[key] for key in ("status", "checks", "complete_component_seconds", "extra_actual_written_input_bytes")}), flush=True)


if __name__ == "__main__":
    main()
