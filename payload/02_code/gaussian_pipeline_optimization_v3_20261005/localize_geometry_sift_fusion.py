"""Keep prepared geometry common and add the unchanged bounded CPU SIFT broker."""

import argparse
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
OWNED = ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/fusion_geometry_sift_001"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity
from array_match_geometry import array_match_geometry
from native_sift_process import native_sift_process
from fusion_owned_json_write import owned_atomic_json, COMPOSED_SOURCE
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
    parser.add_argument("--mode", choices=("incumbent", "process"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(OWNED.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    used, free = map(int, subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.free",
        "--format=csv,noheader,nounits"], text=True).strip().split(","))
    if used > 12*1024 or free < 2*1024 or shutil.disk_usage(OWNED).free < 20*1024**3:
        raise MemoryError("The original device and disk headroom are unavailable")
    paths = [Path(__file__), CODE/"geometry_sift_fusion_node.py", CODE/"array_match_geometry.py",
        CODE/"native_sift_process.py", CODE/"owned_json_write.py", CODE/"fusion_owned_json_write.py", CODE/"integer_sift_matching.py",
        CODE/"integer_sift_worker.py", CODE/"localize_integer_sift.py", CODE/"native_observation_index.py",
        CODE/"binary_map_qa.py", CODE/"localize_prepared_queries.py", CODE/"localize_native_observations.py",
        CODE/"localize_with_locked_inputs.py", CODE/"query_preparation_pipeline.py",
        CODE/"locked_input_identity.py", CODE/"metric_branch_localization_003.py", RESOURCE_WRAPPER,
        CODE/"sift_process_node_003.py", CODE/"run_bounded_journal_task.py",
        V1/"localize_new_map_evaluation.py", ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005/observation_index.py",
        ROOT/"02_code/box2_lio_guided_sfm_v1_20261002/feature_localization.py"]
    split = read(REFERENCE/"00_contract/split_manifest.json")
    raw = {r["image_name"]: r for r in map(json.loads, (INITIAL/"frame_decisions.jsonl").read_text(encoding="utf-8-sig").splitlines())}
    queries = [REFERENCE/"00_contract/extra_images"/r["image_name"] for r in split["extra_frames"]
        if r["role"] == "H" and raw[r["image_name"]]["coarse_pose_valid"]]
    if len(queries) != 24:
        raise ValueError("The original complete ordered query denominator changed")
    models = sorted(Path(read(EXPORT/"TRAINING_ADMISSION.json")["training_opencv_model"]).glob("*.bin"))
    inputs = [DATABASE, EXPORT/"TRAINING_ADMISSION.json", *models, INITIAL/"frame_decisions.jsonl",
        REFERENCE/"00_contract/split_manifest.json", *queries]
    sources, identities = [data_identity(p) for p in paths], [data_identity(p) for p in inputs]
    for ordinal, path in enumerate(paths):
        shutil.copyfile(path, args.output/f"SOURCE_{ordinal:02d}_{path.name}")
    (args.output/"COMPOSED_OWNED_WRITE_SOURCE.py").write_text(COMPOSED_SOURCE,encoding="utf-8")
    atomic_json(args.output/"CONTRACT.json", {"sources": sources, "inputs": identities, "mode": args.mode,
        "prepared_geometry_common": True, "cold_private_caches": True, "new_optimization_count": 0})
    original, argv, workers = integer_sift_matching.DistanceWorker, sys.argv, []

    def observe(log):
        worker = ObservedWorker(log)
        workers.append(worker)
        return worker

    try:
        integer_sift_matching.DistanceWorker = observe
        with owned_atomic_json(args.output) as write_trace, array_match_geometry() as geometry_trace, native_sift_process(
                queries, enabled=args.mode == "process") as sift_trace:
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
        checks = {
            "all24_original_poses": result["status"] == "all24_localized_and_frozen" and len(result["rows"]) == 24,
            "all48_original_GPU_batches": len(workers) == 1 and len(workers[0].receipt["calls"]) == 48,
            "all48_original_index_calls_and_complete_map": index["calls"] == 48 and index["index"]["complete_native_state_unchanged"],
            "all48_prepared_array_geometry_common": len(geometry_trace["routes"]) == 48
                and all(route == "owned_exact_array" for route in geometry_trace["routes"]),
            "original_loc_eval_partition": all(r["loc_eval_3D_overlap"] == 0 for r in result["rows"]),
            "all_sources_unchanged": sources == [data_identity(p) for p in paths],
            "all_inputs_unchanged": identities == [data_identity(p) for p in inputs],
            "original_worker_stopped_and_waited": len(workers) == 1 and workers[0].stopped and workers[0].receipt["returncode"] == 0,
            "original_Linux_RSS_guard_closed": len(workers) == 1 and read(workers[0].resource)["closed"]
                and read(workers[0].resource)["peak_RSS_bytes"] <= 32*1024**3,
            "all_three_private_hooks_restored": geometry_trace["binding_restored"] and sift_trace["binding_restored"]
                and write_trace["binding_restored"] and write_trace["closed"],
            "all24_process_results_closed_or_original_serial": (
                sift_trace["broker"]["delivered"] == sift_trace["broker"]["submitted"] == 24
                and sift_trace["broker"]["closed"] and sift_trace["broker"]["peak_pending"] == 2
                and len(sift_trace["broker"]["saved_workers"]) == 2)
                if args.mode == "process" else sift_trace["broker"] is None}
        atomic_json(args.output/"RESULT.json", {"status": "full_original_guards_passed" if all(checks.values()) else "failed_verification",
            "mode": args.mode, "checks": checks, "localization": str(hits[0]), "sources": sources, "inputs": identities,
            "trace": sift_trace, "array_geometry_trace": geometry_trace, "write_trace": write_trace,
            "complete_original_index_trace": index, "worker": workers[0].receipt if workers else None,
            "complete_child_seconds": time.perf_counter()-STARTED, "new_optimization_count": 0,
            "whole_parent_window_includes_node_startup_exit_and_closure": True})
        if not all(checks.values()):
            raise AssertionError("A complete fusion localization guard failed")
    finally:
        integer_sift_matching.DistanceWorker = original
        sys.argv = argv
        atomic_json(args.output/"BINDING_RESTORATION.json", {"worker_restored": integer_sift_matching.DistanceWorker is original,
            "argv_restored": sys.argv is argv, "new_optimization_count": 0})


if __name__ == "__main__":
    main()
