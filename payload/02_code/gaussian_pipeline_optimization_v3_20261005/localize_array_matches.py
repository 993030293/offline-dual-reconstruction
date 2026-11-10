"""Run the current complete localization with one private exact array operator."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V1OUT = ROOT/"04_outputs/gaussian_pipeline_optimization_v1_20261005"
CODE = Path(__file__).parent
OWNED = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/am001"
sys.path[:0] = [str(V1)]
from runtime import atomic_json, data_identity
from array_match_geometry import array_match_geometry
import integer_sift_matching
import localize_prepared_queries
from metric_branch_localization_003 import ObservedWorker, RESOURCE_WRAPPER

EXPORT = V1OUT/"new_map_export_003"
DATABASE = V1OUT/"p23_track_completion_001/database.db"
INITIAL = ROOT/"04_outputs/gaussian_pipeline_optimization_v2_20261005/final_raw_cold_001/initial"
REFERENCE = ROOT/"04_outputs/box2_lio_guided_sfm_v1_20261002T014330"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("incumbent", "array"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(OWNED.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    used, free = map(int, subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.free",
        "--format=csv,noheader,nounits"], text=True).strip().split(","))
    if used > 12*1024 or free < 2*1024 or shutil.disk_usage(OWNED).free < 20*1024**3:
        raise MemoryError("The frozen device and disk headroom are unavailable")
    contract_started = time.perf_counter()
    paths = [Path(__file__), CODE/"array_match_geometry.py", CODE/"integer_sift_matching.py",
        CODE/"integer_sift_worker.py", CODE/"localize_integer_sift.py", CODE/"native_observation_index.py", CODE/"binary_map_qa.py",
        CODE/"localize_prepared_queries.py", CODE/"localize_native_observations.py",
        CODE/"localize_with_locked_inputs.py", CODE/"query_preparation_pipeline.py",
        CODE/"locked_input_identity.py", CODE/"metric_branch_localization_003.py", RESOURCE_WRAPPER,
        V1/"localize_new_map_evaluation.py", ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005/observation_index.py",
        ROOT/"02_code/box2_lio_guided_sfm_v1_20261002/feature_localization.py"]
    models = sorted(Path(read(EXPORT/"TRAINING_ADMISSION.json")["training_opencv_model"]).glob("*.bin"))
    inputs = [DATABASE, EXPORT/"TRAINING_ADMISSION.json", *models, INITIAL/"frame_decisions.jsonl",
        REFERENCE/"00_contract/split_manifest.json"]
    sources, identities = [data_identity(p) for p in paths], [data_identity(p) for p in inputs]
    atomic_json(args.output/"CONTRACT.json", {"sources": sources, "inputs": identities, "mode": args.mode,
        "complete_current_R90_R92_R94_R102": True, "cold_private_caches": True, "counted": False})
    geometry_contract_seconds = time.perf_counter()-contract_started
    original, previous, workers = integer_sift_matching.DistanceWorker, sys.argv, []
    def observe(log):
        worker = ObservedWorker(log)
        workers.append(worker)
        return worker
    trace = {"setup_seconds": 0., "shutdown_seconds": 0., "binding_restored": True, "routes": [], "array_neighbors": 0}
    try:
        integer_sift_matching.DistanceWorker = observe
        context = array_match_geometry() if args.mode == "array" else nullcontext(trace)
        with context as trace:
            sys.argv = [str(Path(localize_prepared_queries.__file__)), "--preparation-mode", "parallel",
                "--output", str(args.output/"e"), "--reference-mode", "cache", "--reference-cache-root", str(args.output/"r"),
                "--feature-mode", "cache", "--feature-cache-root", str(args.output/"f"), "--export", str(EXPORT),
                "--database", str(DATABASE), "--initial-frames", str(INITIAL), "--reference-run", str(REFERENCE)]
            localize_prepared_queries.main()
        hits = list((args.output/"e").rglob("H_LOCALIZATION.json"))
        if len(hits) != 1:
            raise AssertionError("Exactly one original full localization output is required")
        result = read(hits[0])
        prep = read(args.output/"e/PREPARATION.json")
        index = prep["complete_original_index_trace"]
        guard_started = time.perf_counter()
        checks = {"all24_native_poses": result["status"] == "all24_localized_and_frozen" and len(result["rows"]) == 24,
            "all48_native_gpu_batches": len(workers) == 1 and len(workers[0].receipt["calls"]) == 48,
            "all48_complete_native_index_calls": index["calls"] == 48 and index["index"]["complete_native_state_unchanged"],
            "query_loc_eval_disjoint": all(r["loc_eval_3D_overlap"] == 0 for r in result["rows"]),
            "full_sources_unchanged": sources == [data_identity(p) for p in paths],
            "full_inputs_unchanged": identities == [data_identity(p) for p in inputs],
            "native_worker_stopped_waited": len(workers) == 1 and workers[0].stopped and workers[0].receipt["returncode"] == 0,
            "native_Linux_RSS_guard_closed": len(workers) == 1 and read(workers[0].resource)["closed"]
                and read(workers[0].resource)["peak_RSS_bytes"] <= 32*1024**3,
            "private_geometry_hook_restored": trace["binding_restored"],
            "actual_array_all48_or_untouched_incumbent": (len(trace["routes"]) == 48
                and all(route == "owned_exact_array" for route in trace["routes"])) if args.mode == "array" else not trace["routes"]}
        final_guard_seconds = time.perf_counter()-guard_started
        atomic_json(args.output/"RESULT.json", {"status": "actual_complete_matches_protection_pending" if all(checks.values()) else "required_guard_failed",
            "mode": args.mode, "checks": checks, "localization": str(hits[0]), "sources": sources, "inputs": identities,
            "trace": trace, "worker": workers[0].receipt if workers else None,
            "complete_CPU_geometry_seconds": index["CPU_geometry_seconds_including_all_snapshot_and_verify"]
                +trace["setup_seconds"]+trace["shutdown_seconds"]+geometry_contract_seconds+final_guard_seconds,
            "geometry_contract_seconds": geometry_contract_seconds, "final_guard_seconds": final_guard_seconds,
            "complete_original_index_trace": index, "complete_seconds": time.perf_counter()-STARTED,
            "map_pose_comparison_is_not_GT_accuracy": True, "counted": False})
        if not all(checks.values()):
            raise AssertionError("An original complete array localization guard failed")
    finally:
        integer_sift_matching.DistanceWorker = original
        sys.argv = previous
        atomic_json(args.output/"BINDING_RESTORATION.json", {"worker_restored": integer_sift_matching.DistanceWorker is original,
            "argv_restored": sys.argv is previous, "counted": False})


if __name__ == "__main__":
    main()
