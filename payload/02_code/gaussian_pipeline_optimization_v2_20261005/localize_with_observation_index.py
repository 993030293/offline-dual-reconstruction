"""Use the actual24-view localization entry with frozen geometry indexing."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
sys.path.insert(0, str(ROOT/"02_code/box2_lio_guided_sfm_v1_20261002"))
from runtime import atomic_json, data_identity
import feature_localization
import localize_new_map_evaluation
import sift_feature_store
from frozen_neighbors import FrozenCameraNeighbors
from observation_index import FrozenObservationIndex, indexed_match


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["baseline", "index"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    for name in ("export", "database", "initial-frames", "reference-run"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    original_neighbors = sift_feature_store.metadata_neighbors
    original_match = feature_localization.FeatureBank.match
    original_argv = sys.argv
    source_paths = [Path(__file__), Path(__file__).with_name("observation_index.py"),
                    Path(__file__).with_name("frozen_neighbors.py"), Path(feature_localization.__file__),
                    Path(localize_new_map_evaluation.__file__), Path(sift_feature_store.__file__)]
    sources = [data_identity(p) for p in source_paths]
    atomic_json(args.output/"ENTRY_CONTRACT.json", {"mode": args.mode, "sources": sources,
        "inputs": {n: str(getattr(args, n)) for n in ("export", "database", "initial_frames", "reference_run")},
        "maximum_seconds": 900, "maximum_RSS_GiB": 8, "all24_denominator": True,
        "common_baseline": "Q28_snapshot_and_P27_lazy_features_with_single_thread_PnP",
        "geometry_memory_budget_bytes": 128*1024**2, "cold_build_included_in_match_time": True,
        "mutable_map_updates": False, "counted": False})
    neighbor_table = geometry_table = None
    bound_model = None
    calls = []

    def neighbors(store, model, rough, stamp):
        nonlocal neighbor_table, bound_model
        if neighbor_table is None:
            neighbor_table, bound_model = FrozenCameraNeighbors(store, model), model
        elif model is not bound_model:
            raise ValueError("The privately owned localization reconstruction changed")
        return neighbor_table(rough, stamp)

    def match(bank, model, keypoints, descriptors, neighbors, geometric=True):
        nonlocal geometry_table
        started = time.perf_counter()
        if args.mode == "index":
            if geometry_table is None:
                geometry_table = FrozenObservationIndex(model)
            result = indexed_match(bank, model, keypoints, descriptors, neighbors, geometry_table, geometric)
        else:
            result = original_match(bank, model, keypoints, descriptors, neighbors, geometric)
        seconds = time.perf_counter()-started
        encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        calls.append({"call": len(calls), "seconds": seconds, "geometric": geometric,
                      "neighbors": neighbors, "correspondences": len(result[0]),
                      "result_sha256": hashlib.sha256(encoded).hexdigest()})
        return result

    try:
        sift_feature_store.metadata_neighbors = neighbors
        feature_localization.FeatureBank.match = match
        command = [str(Path(localize_new_map_evaluation.__file__))]
        for name in ("export", "database", "initial_frames", "reference_run"):
            command += ["--"+name.replace("_", "-"), str(getattr(args, name))]
        command += ["--output", str(args.output/"localization"), "--feature-storage", "lazy", "--pnp-threads", "1"]
        sys.argv = command
        localize_new_map_evaluation.main()
        verify_started = time.perf_counter()
        if geometry_table is not None:
            geometry_table.verify_unchanged()
        verify_seconds = time.perf_counter()-verify_started
        if sources != [data_identity(p) for p in source_paths]:
            raise ValueError("Source identity changed during localization")
        atomic_json(args.output/"MATCH_TRACE.json", {"status": "actual24_entry_completed", "mode": args.mode,
            "calls": calls, "match_seconds_including_cold_build": sum(c["seconds"] for c in calls),
            "verify_seconds": verify_seconds, "geometry": geometry_table.receipt() if geometry_table else None,
            "frozen_map_verified": True, "source_unchanged": True})
    finally:
        sift_feature_store.metadata_neighbors = original_neighbors
        feature_localization.FeatureBank.match = original_match
        sys.argv = original_argv


if __name__ == "__main__":
    main()
