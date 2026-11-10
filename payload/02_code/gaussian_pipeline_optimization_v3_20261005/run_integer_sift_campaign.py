"""Measure three real localization pairs including all owned worker overhead."""

import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import atomic_json, data_identity
from compare_actual_localization import compare
from integer_sift_matching import DistanceWorker, bytes_exact
import cv2
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def lifecycle(output):
    worker = DistanceWorker(output/"lifecycle_worker.log")
    rng = np.random.default_rng(20261005)
    checks = []
    try:
        for nq, nt in ((1, 2), (16, 17), (31, 33), (65, 79), (149, 227)):
            q = rng.integers(0, 256, (nq, 128), dtype=np.uint8)
            t = rng.integers(0, 256, (nt, 128), dtype=np.uint8)
            if nt >= 17:
                t[1] = t[0]
            q, t = q.astype(np.float32)/512, t.astype(np.float32)/512
            forward, distance, reverse = worker.match(q, t)
            matcher = cv2.BFMatcher(cv2.NORM_L2)
            native, native_reverse = matcher.knnMatch(q, t, k=2), matcher.match(t, q)
            checks.append(all(int(row[0]) == pair[0].trainIdx and int(row[2]) == pair[1].trainIdx
                and float(d[0]) == pair[0].distance and float(d[1]) == pair[1].distance
                for row, d, pair in zip(forward, distance, native))
                and list(reverse) == [r.trainIdx for r in native_reverse])
            batched = worker.batch(q, [t, t])[0]
            checks.append(all(np.array_equal(x, y) for x, y in zip((forward, distance, reverse), batched)))
        invalid = np.full((32, 128), .123456, np.float32)
        try:
            bytes_exact(invalid)
        except ValueError:
            rejected = True
        else:
            rejected = False
    finally:
        worker.close()
    result = {"five_padding_tie_and_full_range_native_oracles": checks,
        "non_native_float_component_rejected": rejected,
        "owned_worker_complete_shutdown": worker.process.poll() == 0,
        "worker": worker.receipt}
    atomic_json(output/"LIFECYCLE.json", result)
    if not all(checks) or not rejected or worker.process.poll() != 0:
        raise ValueError("An original exact descriptor distance or worker lifecycle guard failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "reference-cache", "feature-cache", "export", "database", "initial-frames", "reference-run"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(__file__).with_name("integer_sift_matching.py"),
        Path(__file__).with_name("integer_sift_worker.py"), Path(__file__).with_name("localize_integer_sift.py")]
    identity = [data_identity(p) for p in sources]
    (args.output/"sources").mkdir()
    for source in sources:
        shutil.copyfile(source, args.output/"sources"/source.name)
    state = {"status": "running", "sources": identity, "pairs": [], "runs": [], "counted": False,
        "control": "current_Q28_Q47_Q48_Q51_complete_serial_PnP_route", "minimum_complete_process_reduction": .1,
        "all24_and48_exact": True, "original_pose_delta_max": 1e-9,
        "includes_startup_conversion_pipe_and_GPU_work": True}
    atomic_json(args.output/"PROTOCOL.json", state)
    started = time.monotonic()
    try:
        state["lifecycle"] = lifecycle(args.output)
        for trial in range(3):
            pair = args.output/f"trial{trial}"
            pair.mkdir()
            for mode in (["native", "integer"] if trial % 2 == 0 else ["integer", "native"]):
                command = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(pair/(mode+"_process.json")),
                    "--log", str(pair/(mode+".log")), "--maximum-seconds", "900", "--maximum-rss-gib", "8", "--",
                    sys.executable, str(Path(__file__).with_name("localize_integer_sift.py")),
                    "--matching-mode", mode, "--output", str(pair/mode), "--reference-mode", "cache",
                    "--reference-cache-root", str(args.reference_cache), "--feature-mode", "cache",
                    "--feature-cache-root", str(args.feature_cache)]
                for name in ("export", "database", "initial_frames", "reference_run"):
                    command += ["--"+name.replace("_", "-"), str(getattr(args, name))]
                subprocess.run(command, check=True)
                state["runs"].append({"trial": trial, "mode": mode, "process": read(pair/(mode+"_process.json"))})
                atomic_json(args.output/"STATE.json", state)
            entries = {m: pair/m/"distance_entry/reference_entry/query_entry/entry" for m in ("native", "integer")}
            invariants = compare(entries["native"], entries["integer"],
                entries["native"]/"NEIGHBOR_TRACE.json", entries["integer"]/"NEIGHBOR_TRACE.json")
            atomic_json(pair/"INVARIANTS.json", invariants)
            item = {"trial": trial, "all_protected": invariants["all_protected"],
                "whole_seconds": {m: read(pair/(m+"_process.json"))["elapsed_seconds"] for m in entries},
                "match_seconds": {m: read(pair/m/"distance_entry/BYTE_SIFT_TRACE.json")["match_seconds"] for m in entries},
                "worker": read(pair/"integer/INTEGER_SIFT_TRACE.json")["trace"]}
            state["pairs"].append(item)
            atomic_json(args.output/"STATE.json", state)
            print({k: v for k, v in item.items() if k != "worker"}, flush=True)
            if not invariants["all_protected"]:
                raise ValueError("An original real24 geometry or48 matching output changed")
        medians = {kind: {m: statistics.median(r[kind][m] for r in state["pairs"]) for m in ("native", "integer")}
            for kind in ("whole_seconds", "match_seconds")}
        reduction = 1-medians["whole_seconds"]["integer"]/medians["whole_seconds"]["native"]
        checks = {"all_three_full_pairs_exact": len(state["pairs"]) == 3 and all(r["all_protected"] for r in state["pairs"]),
            "whole_process_reduction_at_least10percent": reduction >= .1,
            "all_worker_processes_closed": all(r["worker"]["returncode"] == 0 for r in state["pairs"]),
            "CUDA_within4GiB": all(r["worker"]["shutdown"]["peak_CUDA_reserved_bytes"] <= 4*1024**3 for r in state["pairs"]),
            "sources_unchanged": identity == [data_identity(p) for p in sources]}
        state.update(status="all_actual_stage_and_lifecycle_passed_acceptance_pending" if all(checks.values()) else "failed_verification",
            checks=checks, median_seconds=medians, reduction=reduction, elapsed_seconds=time.monotonic()-started)
        atomic_json(args.output/"RESULT.json", state)
    except BaseException as error:
        state.update(status="failed", error=repr(error), elapsed_seconds=time.monotonic()-started)
        atomic_json(args.output/"STATE.json", state)
        raise
    atomic_json(args.output/"STATE.json", state)


if __name__ == "__main__":
    main()
