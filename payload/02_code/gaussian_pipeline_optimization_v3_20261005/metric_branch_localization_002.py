"""Localize all original development queries in one independently exported R119 map."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

STARTED = time.perf_counter()
from metric_branch_admission import INITIAL, OWNED, ROOT, V1, V1OUT, read
from runtime import atomic_json, data_identity
import integer_sift_matching
import localize_prepared_queries
import psutil

CODE = Path(__file__).parent
RESOURCE_WRAPPER = CODE/"parallel_a_003/owned_matching_worker.py"
REFERENCE = ROOT/"04_outputs/box2_lio_guided_sfm_v1_20261002T014330"
DATABASE = V1OUT/"p23_track_completion_001/database.db"


def linux(path):
    value = Path(path).resolve().as_posix()
    if value[:3].lower() != "c:/":
        raise ValueError("The existing C drive source is required")
    return "/mnt/c/"+value[3:]


class ObservedWorker(integer_sift_matching.DistanceWorker):
    """Keep native RPC ownership and the existing Linux RSS observer."""

    def __init__(self, log):
        log = Path(log)
        self.resource = log.with_name("WSL_RESOURCE.json")
        self.command = ["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
            "/opt/e26_thin_surface_v1/venv/bin/python", "-B", linux(RESOURCE_WRAPPER),
            "--native", linux(CODE/"integer_sift_worker.py"), "--resource", linux(self.resource)]
        self.log = log.open("xb")
        self.stopped = False
        self.owner = threading.get_ident()
        started = time.perf_counter()
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, bufsize=0)
        self.receipt = {"command": self.command, "pid": self.process.pid,
            "creation_time": psutil.Process(self.process.pid).create_time(), "calls": [],
            "resource_receipt": str(self.resource), "counted": False}
        try:
            self.receipt["info"] = self.rpc({"command": "info"})
            self.receipt["info"].pop("payload")
        except BaseException:
            self.process.stdin.close()
            self.process.stdout.close()
            self.process.wait(timeout=30)
            self.log.close()
            raise
        self.receipt["startup_seconds"] = time.perf_counter()-started

    def rpc(self, *args, **kwargs):
        if threading.get_ident() != self.owner:
            raise RuntimeError("Only the original caller owns the matching stdio pipe")
        return super().rpc(*args, **kwargs)

    def close(self):
        super().close()
        if self.receipt["shutdown"]["peak_CUDA_reserved_bytes"] > 4*1024**3:
            raise MemoryError("Original exact matching CUDA4GiB bound exceeded")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.admission.resolve().relative_to(OWNED.resolve())
    args.output.resolve().relative_to(OWNED.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    own = read(args.admission/"RESULT.json")
    if own["status"] != "own_admission_passed_localization_quality_pending" or not all(own["checks"].values()):
        raise ValueError("Own complete admission must pass before localization")
    used, free = map(int, subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.free",
        "--format=csv,noheader,nounits"], text=True).strip().split(","))
    if used > 12*1024 or free < 2*1024 or shutil.disk_usage(OWNED).free < 20*1024**3:
        raise MemoryError("Original device and disk headroom unavailable")
    export = Path(own["export"])
    source_models = sorted(Path(read(export/"TRAINING_ADMISSION.json")["training_opencv_model"]).glob("*.bin"))
    paths = [Path(__file__), RESOURCE_WRAPPER, CODE/"metric_branch_admission.py", *sorted(CODE.glob("localize*.py")),
        CODE/"integer_sift_matching.py", CODE/"integer_sift_worker.py", CODE/"query_preparation_pipeline.py",
        CODE/"native_observation_index.py", CODE/"binary_map_qa.py", CODE/"locked_input_identity.py", V1/"localize_new_map_evaluation.py"]
    inputs = [DATABASE, export/"TRAINING_ADMISSION.json", *source_models, INITIAL/"frame_decisions.jsonl",
        REFERENCE/"00_contract/split_manifest.json"]
    source_ids, input_ids = [data_identity(p) for p in paths], [data_identity(p) for p in inputs]
    atomic_json(args.output/"CONTRACT.json", {"sources": source_ids, "inputs": input_ids, "all24_queries48_requests": True,
        "own_cold_caches": True, "existing_exact_worker_and_geometry": True, "counted": False})
    original, previous, workers = integer_sift_matching.DistanceWorker, sys.argv, []

    def observe(log):
        worker = ObservedWorker(log)
        workers.append(worker)
        return worker

    try:
        integer_sift_matching.DistanceWorker = observe
        sys.argv = [str(Path(localize_prepared_queries.__file__)), "--preparation-mode", "parallel",
            "--output", str(args.output/"e"), "--reference-mode", "cache", "--reference-cache-root", str(args.output/"r"),
            "--feature-mode", "cache", "--feature-cache-root", str(args.output/"f"), "--export", str(export),
            "--database", str(DATABASE), "--initial-frames", str(INITIAL), "--reference-run", str(REFERENCE)]
        localize_prepared_queries.main()
        hits = list((args.output/"entry").rglob("H_LOCALIZATION.json"))
        if len(hits) != 1:
            raise AssertionError("Exactly one own localization receipt is required")
        result = read(hits[0])
        checks = {"original_all24_localized": result["status"] == "all24_localized_and_frozen" and len(result["rows"]) == 24,
            "all48_native_matching_requests": len(workers) == 1 and len(workers[0].receipt["calls"]) == 48,
            "loc_eval_disjoint_all24": all(r["loc_eval_3D_overlap"] == 0 for r in result["rows"]),
            "complete_sources_unchanged": source_ids == [data_identity(p) for p in paths],
            "complete_inputs_unchanged": input_ids == [data_identity(p) for p in inputs],
            "original_worker_stop_wait": len(workers) == 1 and workers[0].stopped and workers[0].receipt["returncode"] == 0,
            "Linux_RSS_observer_closed": len(workers) == 1 and read(workers[0].resource)["closed"]
                and read(workers[0].resource)["peak_RSS_bytes"] <= 32*1024**3}
        atomic_json(args.output/"RESULT.json", {"status": "own_all24_localization_passed_training_quality_pending" if all(checks.values()) else "required_own_localization_guard_failed",
            "checks": checks, "localization": str(hits[0]), "worker": workers[0].receipt if workers else None,
            "complete_seconds": time.perf_counter()-STARTED, "map_pose_comparison_is_not_GT_accuracy": True, "counted": False})
        if not all(checks.values()):
            raise AssertionError("An unchanged complete own localization guard failed")
    finally:
        integer_sift_matching.DistanceWorker = original
        sys.argv = previous
        atomic_json(args.output/"BINDING_RESTORATION.json", {"worker_restored": integer_sift_matching.DistanceWorker is original,
            "argv_restored": sys.argv is previous, "counted": False})


if __name__ == "__main__":
    main()
