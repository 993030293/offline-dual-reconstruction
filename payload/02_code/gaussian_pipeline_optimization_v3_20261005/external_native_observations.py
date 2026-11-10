"""Compare unchanged external integer matching with and without native geometry indexing."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import cv2
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2), str(ROOT/"02_code/box2_lio_guided_sfm_v1_20261002")]
from runtime import atomic_json, data_identity
from feature_localization import FeatureBank
from sift_feature_store import SiftFeatureStore
from byte_sift_matching import byte_sift_matching, TransientFloatView
from generic_sift_matching import generic_sift_matching
from native_observation_index import native_observation_index


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "selection-protocol", "oracle", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    prior = read(args.selection_protocol)
    selection = prior["selection"]
    oracle = read(args.oracle/"RESULT.json")
    if len(selection) != 24 or not prior["selection_uses_only_uniform_name_order"] or not all(oracle["checks"].values()):
        raise ValueError("The existing independent-of-geometry selection or native index oracle is missing")
    paths = [Path(__file__), Path(__file__).with_name("native_observation_index.py"), Path(__file__).with_name("binary_map_qa.py"),
        Path(__file__).with_name("generic_sift_matching.py"), Path(__file__).with_name("generic_sift_worker.py"),
        V2/"observation_index.py", V2/"byte_sift_matching.py", V1/"sift_feature_store.py",
        ROOT/"02_code/box2_lio_guided_sfm_v1_20261002/feature_localization.py", args.workspace/"database.db",
        *sorted((args.workspace/"sparse/0").glob("*.bin"))]
    identity = [data_identity(p) for p in paths]
    atomic_json(args.output/"PROTOCOL.json", {"sources_inputs": identity, "selection": selection,
        "both_sides_same_generic_exact_R90_and_byte_SIFT": True, "all_original_features_all24_48": True,
        "all_input_binding_preparation_snapshot_and_worker_cost_included": True, "external_non_box_component_only": True,
        "Oxford_R90_development_status_preserved": True, "no_GT_ROI_PnP_or_new_geometry_quality_claim": True,
        "three_complete_pairs": True, "whole_speed_reported_not_assumed_universal": True, "new_optimization_count": 0})
    pairs = []
    cv2.setNumThreads(8)
    try:
        for trial in range(3):
            runs = {}
            for mode in (("baseline", "index") if trial % 2 == 0 else ("index", "baseline")):
                target = args.output/f"trial{trial}"/mode
                target.mkdir(parents=True, exist_ok=False)
                start = time.perf_counter()
                verified = [data_identity(p) for p in paths]
                model = pycolmap.Reconstruction(args.workspace/"sparse/0")
                outputs = []
                with native_observation_index(target, mode == "index") as geometry:
                    with generic_sift_matching(target/"worker.log") as route:
                        with byte_sift_matching() as byte_trace:
                            store = SiftFeatureStore(args.workspace/"database.db", {r.name for r in model.images.values()}, "lazy", 32)
                            bank = FeatureBank.__new__(FeatureBank)
                            bank.original = store
                            view = TransientFloatView(store, byte_trace)
                            try:
                                for row in selection:
                                    _, keypoints, descriptors = view[row["image_id"]]
                                    for parity in (0, 1):
                                        result = bank.match(model, keypoints[parity::2], descriptors[parity::2], row["neighbors"])
                                        outputs.append({"image_id": row["image_id"], "parity": parity, "output": result})
                                storage = store.receipt()
                            finally:
                                store.close()
                elapsed = time.perf_counter()-start
                runs[mode] = {"outputs": outputs, "whole_seconds": elapsed, "geometry": geometry, "route": route,
                    "inputs_exact": verified == identity, "storage": storage}
                atomic_json(target/"MATCHES.json", outputs)
                atomic_json(target/"TRACE.json", {k: v for k, v in runs[mode].items() if k != "outputs"})
                atomic_json(args.output/"STATE.json", {"status": "running", "trial": trial, "completed_mode": mode, "pairs": pairs, "counted": False})
            checks = {"all48_complete_native_geometry_outputs_exact": len(runs["baseline"]["outputs"]) == 48
                    and runs["baseline"]["outputs"] == runs["index"]["outputs"],
                "all_inputs_unchanged": all(r["inputs_exact"] for r in runs.values()),
                "complete_native_snapshot_verified": runs["index"]["geometry"]["index"]["complete_native_state_unchanged"],
                "all48_original_feature_shapes_admitted": all(len(r["route"]["routes"]) == 48
                    and all(v["route"] == "integer" and v["all_features_preserved"] for v in r["route"]["routes"]) for r in runs.values()),
                "both_owned_GPU_workers_closed_with_original_budget": all(r["route"]["worker"]["returncode"] == 0
                    and r["route"]["worker"]["shutdown"]["peak_CUDA_reserved_bytes"] <= 4*1024**3 for r in runs.values())}
            pairs.append({"trial": trial, "checks": checks,
                "whole_seconds": {m: r["whole_seconds"] for m, r in runs.items()},
                "complete_CPU_geometry_seconds": {m: r["geometry"]["CPU_geometry_seconds_including_all_snapshot_and_verify"] for m, r in runs.items()}})
            print(pairs[-1], flush=True)
            if not all(checks.values()):
                raise ValueError("An actual external native-index identity or output guard failed")
        medians = {stage: {m: statistics.median(p[stage][m] for p in pairs) for m in ("baseline", "index")}
            for stage in ("whole_seconds", "complete_CPU_geometry_seconds")}
        checks = {"all_three_real_non_box_component_pairs_guarded": len(pairs) == 3 and all(all(p["checks"].values()) for p in pairs),
            "sources_and_inputs_unchanged": identity == [data_identity(p) for p in paths]}
        result = {"status": "external_native_index_component_passed" if all(checks.values()) else "failed_verification", "checks": checks,
            "pairs": pairs, "median_seconds": medians, "reductions": {s: 1-v["index"]/v["baseline"] for s, v in medians.items()},
            "sources_inputs": identity, "new_optimization_count": 0, "whole_pipeline_generalization_or_accuracy_claim": False}
        atomic_json(args.output/"RESULT.json", result)
        print({k: result[k] for k in ("status", "checks", "median_seconds", "reductions")}, flush=True)
    except BaseException as error:
        atomic_json(args.output/"FAILURE.json", {"error": repr(error), "pairs": pairs, "new_optimization_count": 0})
        raise


if __name__ == "__main__":
    main()
