"""Run the frozen localization entry with an isolated camera-neighbor adapter."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from frozen_neighbors import FrozenCameraNeighbors

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
sys.path.insert(0, str(ROOT/"02_code/box2_lio_guided_sfm_v1_20261002"))
from runtime import atomic_json, data_identity
import feature_localization
import localize_new_map_evaluation
import sift_feature_store


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["baseline", "snapshot"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    for name in ("export", "database", "initial-frames", "reference-run"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    original_neighbors = sift_feature_store.metadata_neighbors
    original_match = feature_localization.FeatureBank.match
    original_argv = sys.argv
    source_paths = [Path(__file__), Path(__file__).with_name("frozen_neighbors.py"),
        Path(localize_new_map_evaluation.__file__), Path(sift_feature_store.__file__),
        Path(feature_localization.__file__)]
    sources = [data_identity(path) for path in source_paths]
    atomic_json(args.output/"ENTRY_CONTRACT.json", {"mode": args.mode, "sources": sources,
        "input_version": "new_map_admission_v1", "feature_storage": "lazy", "PnP_threads": 1,
        "maximum_seconds": 900, "maximum_RSS_bytes": 32*1024**3,
        "neighbor_only_change": True, "snapshot_scope": "one fixed model during this invocation"})
    table = None
    bound_model = None
    calls, matches = [], []
    def select(store, model, rough, stamp):
        nonlocal table, bound_model
        started = time.perf_counter()
        build_seconds = 0.
        if args.mode == "snapshot":
            if table is None:
                table = FrozenCameraNeighbors(store, model)
                bound_model = model
                build_seconds = time.perf_counter()-started
            elif model is not bound_model:
                raise ValueError("A frozen camera snapshot cannot be reused for another map")
            values = table(rough, stamp)
        else:
            values = original_neighbors(store, model, rough, stamp)
        calls.append({"timestamp": stamp, "neighbors": values, "seconds": time.perf_counter()-started,
                      "build_seconds": build_seconds})
        return values
    def match(bank, *values, **kwargs):
        result = original_match(bank, *values, **kwargs)
        encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        matches.append({"call": len(matches), "result_sha256": hashlib.sha256(encoded).hexdigest(),
                        "correspondences": len(result[0]), "edges": len(result[1]), "pairs": len(result[2])})
        return result
    try:
        sift_feature_store.metadata_neighbors = select
        feature_localization.FeatureBank.match = match
        command = [str(Path(localize_new_map_evaluation.__file__))]
        for name in ("export", "database", "initial_frames", "reference_run"):
            command += ["--"+name.replace("_", "-"), str(getattr(args, name))]
        command += ["--output", str(args.output/"localization"), "--feature-storage", "lazy", "--pnp-threads", "1"]
        sys.argv = command
        localize_new_map_evaluation.main()
        if sources != [data_identity(path) for path in source_paths]:
            raise ValueError("Localization source changed during execution")
        atomic_json(args.output/"NEIGHBOR_TRACE.json", {"status": "real_entry_completed", "mode": args.mode,
            "calls": calls, "match_calls": matches, "snapshot": None if table is None else table.receipt(),
            "source_unchanged": True})
    finally:
        sift_feature_store.metadata_neighbors = original_neighbors
        feature_localization.FeatureBank.match = original_match
        sys.argv = original_argv


if __name__ == "__main__":
    main()
