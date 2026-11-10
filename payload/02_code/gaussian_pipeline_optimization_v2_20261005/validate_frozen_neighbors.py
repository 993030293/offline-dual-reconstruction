"""Compare fixed-map neighbor selection over the actual raw query pool."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np
import pycolmap

from frozen_neighbors import FrozenCameraNeighbors

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity
from sift_feature_store import SiftFeatureStore, metadata_neighbors


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    contract = read(args.baseline/"LOCALIZATION_CONTRACT.json")
    localization = read(args.baseline/"H_LOCALIZATION.json")
    sources = contract["source_identity"]
    database = Path(next(s["path"] for s in sources if s["path"].endswith("database.db")))
    raw_path = Path(next(s["path"] for s in sources if s["path"].endswith("frame_decisions.jsonl")))
    raw = [json.loads(line) for line in raw_path.read_text(encoding="utf-8-sig").splitlines()]
    raw_by_name = {r["image_name"]: r for r in raw}
    model = pycolmap.Reconstruction(localization["map_source"])
    names = {im.name for im in model.images.values() if im.has_pose}
    checks = {"new1016_map": len(names) == 1016,
              "original24_queries": len(localization["rows"]) == 24,
              "raw7338_rows": len(raw) == 7338}
    atomic_json(args.output/"PROTOCOL.json", {"maximum_seconds": 600,
        "maximum_RSS_bytes": 8*1024**3, "minimum_stage_speed_gain": .1,
        "paired_trials": 3, "candidate_snapshot_build_included": True,
        "baseline_source": data_identity(V1/"sift_feature_store.py"),
        "candidate_source": data_identity(Path(__file__).with_name("frozen_neighbors.py")),
        "inputs": sources, "raw_query_policy": "all coarse_pose_valid rows; no image quality filtering"})
    with_store = SiftFeatureStore(database, names, "lazy", 32)
    try:
        queries = [(raw_by_name[r["image_name"]], r["camera_header_ns"])
                   for r in localization["rows"]]
        trials = []
        for trial in range(3):
            row = {}
            orders = ("baseline", "candidate") if trial % 2 == 0 else ("candidate", "baseline")
            returned = {}
            for mode in orders:
                start = time.perf_counter()
                if mode == "baseline":
                    values = [metadata_neighbors(with_store, model, rough, stamp) for rough, stamp in queries]
                else:
                    table = FrozenCameraNeighbors(with_store, model)
                    values = [table(rough, stamp) for rough, stamp in queries]
                row[mode+"_seconds"] = time.perf_counter()-start
                returned[mode] = values
            row["neighbors_exact"] = returned["baseline"] == returned["candidate"]
            trials.append(row)
        table = FrozenCameraNeighbors(with_store, model)
        mismatches, tested = [], 0
        for source in raw:
            if not source["coarse_pose_valid"]:
                continue
            expected = metadata_neighbors(with_store, model, source, source["camera_header_ns"])
            actual = table(source, source["camera_header_ns"])
            if expected != actual:
                mismatches.append({"image_name": source["image_name"], "baseline": expected, "candidate": actual})
            tested += 1
            if tested % 250 == 0:
                atomic_json(args.output/"STATE.json", {"status": "checking_all_raw_queries", "tested": tested,
                    "mismatches": len(mismatches), "elapsed_seconds": time.perf_counter()-started})
                print(json.dumps({"tested": tested, "mismatches": len(mismatches)}), flush=True)
            if time.perf_counter()-started > 590:
                raise RuntimeError("Registered neighbor diagnostic time budget exceeded")
        # Identical camera centers and timestamps exercise stable ordering explicitly.
        rotation = SimpleNamespace(matrix=lambda: np.eye(3))
        pose = SimpleNamespace(rotation=rotation)
        image = SimpleNamespace(projection_center=lambda: np.zeros(3), cam_from_world=lambda: pose)
        tie_model = SimpleNamespace(images={i: image for i in reversed(range(12))})
        class TieStore:
            names = {i: "frame_0_100.png" for i in range(12)}
            def __contains__(self, key):
                return key in self.names
        query = {"C_W_m": [0., 0., 0.], "R_WC": np.eye(3).tolist()}
        ties = FrozenCameraNeighbors(TieStore(), tie_model)
        checks["stable_exact_ties"] = ties(query, 100) == metadata_neighbors(TieStore(), tie_model, query, 100) == list(range(8))
        checks["tie_fallback_exercised"] = ties.near_tie_fallbacks == 1
        rejection_checks = {}
        for name, rough, stamp in [
            ("nan_center", {**query, "C_W_m": [float("nan"), 0, 0]}, 100),
            ("bad_rotation_shape", {**query, "R_WC": [1, 2, 3]}, 100),
            ("fractional_time", query, 100.5), ("negative_time", query, -1),
            ("overflow_time", query, 2**63), ("boolean_time", query, True)]:
            try:
                table(rough, stamp)
            except ValueError:
                rejection_checks[name] = True
            else:
                rejection_checks[name] = False
        baseline_seconds = statistics.median(t["baseline_seconds"] for t in trials)
        candidate_seconds = statistics.median(t["candidate_seconds"] for t in trials)
        gain = 1-candidate_seconds/baseline_seconds
        checks.update(all_three_pairs_exact=all(t["neighbors_exact"] for t in trials),
            all_valid_raw_queries_exact=tested == sum(bool(r["coarse_pose_valid"]) for r in raw) and not mismatches,
            stage_gain_at_least10percent=gain >= .1,
            all_invalid_inputs_rejected=all(rejection_checks.values()),
            no_descriptor_payload_loaded=with_store.receipt()["misses"] == 0)
        result = {"status": "neighbor_stage_passed_real_entry_pending" if all(checks.values()) else "failed",
            "checks": checks, "trials": trials, "median_baseline_seconds": baseline_seconds,
            "median_candidate_seconds_including_build": candidate_seconds, "stage_time_reduction": gain,
            "raw_valid_queries_tested": tested, "mismatches": mismatches, "rejection_checks": rejection_checks,
            "snapshot": table.receipt(), "elapsed_seconds": time.perf_counter()-started, "counted": False}
        atomic_json(args.output/"RESULT.json", result)
        atomic_json(args.output/"STATE.json", {"status": result["status"], "counted": False})
        print(json.dumps({k: result[k] for k in ("status", "stage_time_reduction", "raw_valid_queries_tested", "checks")}))
        if not all(checks.values()):
            raise ValueError("A frozen neighbor protection failed")
    finally:
        with_store.close()


if __name__ == "__main__":
    main()
