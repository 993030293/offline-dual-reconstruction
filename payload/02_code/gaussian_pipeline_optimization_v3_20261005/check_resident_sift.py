"""Verify complete native nearest neighbors, content changes, eviction and faults."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from descriptor_residency_plan import DescriptorResidencyPlan, compact_plan, shape_admitted
from generic_sift_matching import admission
from integer_sift_matching import bytes_exact
from resident_sift_matching import ResidentWorker, resident_sift_transport


def native_equal(query, train, result):
    forward, distance, reverse = result
    bf = cv2.BFMatcher(cv2.NORM_L2)
    pairs, backward = bf.knnMatch(query, train, k=2), bf.match(train, query)
    return all(row[0] == pair[0].trainIdx and row[2] == pair[1].trainIdx
        and float(d[0]) == pair[0].distance and float(d[1]) == pair[1].distance
        for row, d, pair in zip(forward, distance, pairs)) and list(reverse) == [m.trainIdx for m in backward]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), *(Path(__file__).with_name(name) for name in ("descriptor_residency_plan.py",
        "resident_sift_worker.py", "resident_sift_matching.py", "integer_sift_worker.py", "integer_sift_matching.py",
        "generic_sift_worker.py", "generic_sift_matching.py"))]
    identities = [data_identity(path) for path in paths]
    atomic_json(args.output/"PROTOCOL.json", {"sources": identities, "all_original_complete_distances_indices_and_reverse": True,
        "old_resource_budgets_unchanged": True, "content_changes_not_hidden_by_object_identity": True,
        "parameter_tuning_or_object_GT_inputs": False, "actual_three_cost_pairs_pending": True})
    rng, rows = np.random.default_rng(20261005), []
    worker = ResidentWorker(args.output/"worker.log")
    try:
        for nq, nt in ((1, 2), (16, 17), (31, 33), (65, 79), (149, 227), (4, 20003), (12001, 3)):
            q = rng.integers(0, 256, (nq, 128), dtype=np.uint8)
            t = rng.integers(0, 256, (nt, 128), dtype=np.uint8)
            t[1] = t[0]
            q[0] = 0
            if nq > 1:
                q[1] = 255
            q, t = q.astype(np.float32)/512, t.astype(np.float32)/512
            initial = worker.batch(q, [t, t])
            again = worker.batch(np.asfortranarray(q), [np.asfortranarray(t), t.copy()])
            old_wire, hit_wire = worker.receipt["calls"][-2:]
            updated = t.copy()
            updated[0, 0] = ((int(updated[0, 0]*512)+1) % 256)/512
            changed = worker.batch(q, [updated, t])
            rows.append({"query_rows": nq, "train_rows": nt,
                "all_initial_two_complete_native_results": all(native_equal(q, t, result) for result in initial),
                "noncontiguous_and_copy_identity_hits_exact": all(all(np.array_equal(a, b) for a, b in zip(left, right))
                    for left, right in zip(initial, again)),
                "original_duplicate_uses_one_upload_and_full_hit_query_only": old_wire["wire_descriptor_bytes"] == (nq+nt)*128
                    and hit_wire["wire_descriptor_bytes"] == nq*128 and hit_wire["plan"]["missing_count"] == 0,
                "changed_content_native_exact_and_new_identity": native_equal(q, updated, changed[0])
                    and native_equal(q, t, changed[1]) and worker.receipt["calls"][-1]["plan"]["missing_count"] == 1})
    finally:
        worker.close()
    tiny = ResidentWorker(args.output/"tiny_worker.log", cache_bytes=12800)
    q = rng.integers(0, 256, (17, 128), np.uint8).astype(np.float32)/512
    trains = [rng.integers(0, 256, (n, 128), np.uint8).astype(np.float32)/512 for n in (33, 40, 43)]
    evictions = []
    try:
        for slots in ((0, 1), (0, 2), (1, 2), (2, 0), (1, 0)):
            batch = [trains[i] for i in slots]
            result = tiny.batch(q, batch)
            evictions.append({"slots": slots, "all_complete_native_results": all(native_equal(q, t, r) for t, r in zip(batch, result)),
                "evicted_count": tiny.receipt["calls"][-1]["plan"]["evicted_count"],
                "resident_bytes": tiny.receipt["calls"][-1]["plan"]["resident_bytes"]})
    finally:
        tiny.close()
    faults = []
    for name in ("changed_payload", "changed_plan"):
        faulty = ResidentWorker(args.output/(name+"_worker.log"))
        raw = bytes_exact(trains[0]).tobytes()
        key = hashlib.sha256(raw).hexdigest()
        expected = compact_plan(faulty.plan.plan([[key, len(trains[0])]]))
        payload = bytes_exact(q).tobytes()+raw
        if name == "changed_payload":
            payload = payload[:-1]+bytes([payload[-1]^1])
        else:
            expected = dict(expected, resident_bytes=expected["resident_bytes"]+1)
        try:
            faulty.rpc({"command": "resident_batch", "query_rows": len(q), "train_items": [[key, len(trains[0])]],
                "plan": expected}, payload)
        except RuntimeError as error:
            faulty.close()
            faults.append({"name": name, "error": str(error), "returncode": faulty.receipt["returncode"],
                "closed": faulty.stopped and faulty.process.poll() is not None and not faulty.plan.rows,
                "expected_primary_error": "content identity" in str(error) if name == "changed_payload" else "plans differ" in str(error)})
        else:
            faulty.close()
            raise ValueError("An actual corrupt resident request was accepted")
    empty = ResidentWorker(args.output/"empty_worker.log")
    empty.close()
    good = np.zeros((32, 128), np.float32)
    unsupported = {"empty_query": admission(good[:0], [good])[0] == "native",
        "one_train": admission(good, [good[:1]])[0] == "native",
        "non_byte": admission(np.full_like(good, .123456), [good])[0] == "native",
        "matrix_budget": admission(np.zeros((12001, 128), np.float32), [np.zeros((12001, 128), np.float32)])[0] == "native"}
    budgets = [(1, [2]), (12001, [3]), (4, [20003]), (12001, [12001]), (0, [2]),
        (1, [1]), (1, [2]*13), (True, [2]), (1, [True]), (262144, [2]), (32, [262144]),
        (8000, [18000]), (8001, [18000]), (1000, [20000]*12), (1000, [23000]*12)]
    source_directory = Path(__file__).resolve().parent.as_posix()
    if source_directory[1:3] != ":/" or source_directory[0].lower() != "c":
        raise ValueError("The independent original worker oracle must stay in the existing workspace")
    script = ("import json,sys;sys.path.insert(0,sys.argv[1]);from generic_sift_worker import admitted;"
        "cases=json.load(sys.stdin);print(json.dumps([admitted(nq,counts) for nq,counts in cases]))")
    server = subprocess.run(["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
        "/opt/e26_thin_surface_v1/venv/bin/python", "-c", script, "/mnt/c/"+source_directory[3:]],
        input=json.dumps(budgets).encode("ascii"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, check=True)
    original_budget_results = json.loads(server.stdout)
    budget_equal = [shape_admitted(nq, counts) for nq, counts in budgets] == original_budget_results
    import integer_sift_matching as native
    import generic_sift_matching as generic
    original, original_generic = native.DistanceWorker, generic.GenericWorker
    try:
        with resident_sift_transport(True) as restored:
            if native.DistanceWorker is not ResidentWorker or generic.GenericWorker is not ResidentWorker:
                raise ValueError("An owned worker adapter did not bind the exact worker")
            raise RuntimeError("injected_owned_consumer_failure")
    except RuntimeError as error:
        restoration = str(error) == "injected_owned_consumer_failure" and restored["original_worker_factory_restored"]
    check = {"all7_varied_large_small_full_range_tie_oracles": len(rows) == 7 and all(all(v for k, v in row.items() if k not in ("query_rows", "train_rows")) for row in rows),
        "all5_tiny_capacity_complete_native_results_with_real_eviction": len(evictions) == 5
            and all(row["all_complete_native_results"] and row["resident_bytes"] <= 12800 for row in evictions)
            and any(row["evicted_count"] > 0 for row in evictions),
        "both_actual_payload_and_plan_faults_preserve_errors_and_close": len(faults) == 2
            and all(row["closed"] and row["expected_primary_error"] and row["returncode"] != 0 for row in faults),
        "all_original_unsupported_routes_before_execution": all(unsupported.values()),
        "all15_original_shape_and_byte_budget_boundaries_unchanged": budget_equal and len(budgets) == 15,
        "worker_factories_restored_after_consumer_failure": restoration and native.DistanceWorker is original and generic.GenericWorker is original_generic,
        "all_normal_workers_closed_caches_cleared_original_GPU_budget": all(w.receipt["returncode"] == 0
            and w.receipt["shutdown"]["resident_entries_after_clear"] == w.receipt["client_resident_entries_after_close"] == 0
            and w.receipt["shutdown"]["peak_CUDA_reserved_bytes"] <= 4*1024**3 for w in (worker, tiny, empty)),
        "sources_unchanged": identities == [data_identity(path) for path in paths]}
    result = {"status": "passed" if all(check.values()) else "failed_verification", "checks": check, "sources": identities,
        "oracle_rows": rows, "tiny_evictions": evictions, "faults": faults, "unsupported": unsupported,
        "budget_oracle_original_actual_WSL_worker": original_budget_results,
        "workers": [worker.receipt, tiny.receipt, empty.receipt], "counted": False,
        "real_entry_three_cost_pairs_and_non_box_whole_pipeline_pending": True}
    atomic_json(args.output/"RESULT.json", result)
    print({key: result[key] for key in ("status", "checks")}, flush=True)
    if not all(check.values()):
        raise ValueError("An original complete descriptor residency oracle failed")


if __name__ == "__main__":
    main()
