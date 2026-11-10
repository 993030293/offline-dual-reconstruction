"""Compare every selected external native match including input and worker costs."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import cv2
import numpy as np
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


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("workspace", "oracle", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    oracle = read(args.oracle/"RESULT.json")
    if not all(oracle["checks"].values()):
        raise ValueError("The frozen generic shape and route oracle is missing")
    paths = [Path(__file__), Path(__file__).with_name("generic_sift_worker.py"),
        Path(__file__).with_name("generic_sift_matching.py"), Path(__file__).with_name("integer_sift_worker.py"),
        Path(__file__).with_name("integer_sift_matching.py"), V2/"byte_sift_matching.py", V1/"sift_feature_store.py",
        ROOT/"02_code/box2_lio_guided_sfm_v1_20261002/feature_localization.py", args.workspace/"database.db",
        *sorted((args.workspace/"sparse/0").glob("*.bin"))]
    identity = [data_identity(p) for p in paths]
    if oracle["sources"] != [data_identity(Path(r["path"])) for r in oracle["sources"]]:
        raise ValueError("An external parameter or matcher source changed after generic oracle freeze")
    model = pycolmap.Reconstruction(args.workspace/"sparse/0")
    available = sorted(model.images, key=lambda i: model.images[i].name)
    query_positions = np.linspace(0, len(available)-1, 24).round().astype(int).tolist()
    selection = [{"image_id": available[index], "image_name": model.images[available[index]].name,
        "neighbors": [available[j] for j in sorted(range(len(available)), key=lambda j: (abs(j-index), j)) if j != index][:4]}
        for index in query_positions]
    pixels = []
    for row in selection:
        path = args.workspace/"images"/row["image_name"]
        decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("A preselected real external image could not decode")
        pixels.append({"file": data_identity(path), "decoded_shape": list(decoded.shape)})
    protocol = {"sources_inputs": identity, "external": "Oxford_Observatory_actual_native_SIFT_database_and_native_model",
        "selection": selection, "selection_uses_only_uniform_name_order": True, "query_split": "existing_all_native_features_even_and_odd",
        "real_images": pixels, "no_features_truncated": True, "GT_or_ROI_or_box_geometry_input": False,
        "all24_queries_all48_geometry_calls": True, "three_whole_component_pairs": True,
        "minimum_whole_component_reduction": .1, "input_hash_model_load_query_prepare_worker_and_store_cost_included": True,
        "PnP_or_external_quality_claim": False, "new_optimization_count": 0,
        "frozen_oracle": str(args.oracle), "existing_refs_are_already_reconstructed_not_new_localization": True}
    atomic_json(args.output/"PROTOCOL.json", protocol)
    del model
    pairs, runs = [], []
    cv2.setNumThreads(8)
    for trial in range(3):
        actual = {}
        for mode in (["native", "integer"] if trial % 2 == 0 else ["integer", "native"]):
            target = args.output/f"trial{trial}"/mode
            target.mkdir(parents=True, exist_ok=False)
            start = time.perf_counter()
            verified = [data_identity(p) for p in paths]
            model = pycolmap.Reconstruction(args.workspace/"sparse/0")
            results = []
            with generic_sift_matching(target/"worker.log", mode == "integer") as route:
                with byte_sift_matching() as byte_trace:
                    store = SiftFeatureStore(args.workspace/"database.db", {r.name for r in model.images.values()}, "lazy", 32)
                    bank = FeatureBank.__new__(FeatureBank)
                    bank.original = store
                    query_view = TransientFloatView(store, byte_trace)
                    try:
                        for row in selection:
                            _, keypoints, descriptors = query_view[row["image_id"]]
                            for parity in (0, 1):
                                result = bank.match(model, keypoints[parity::2], descriptors[parity::2], row["neighbors"])
                                results.append({"image_id": row["image_id"], "parity": parity, "output": result})
                        storage = store.receipt()
                    finally:
                        store.close()
            del model
            seconds = time.perf_counter()-start
            actual[mode] = {"results": results, "seconds": seconds, "route": route, "storage": storage,
                "inputs_exact": verified == identity}
            atomic_json(target/"MATCHES.json", results)
            atomic_json(target/"TRACE.json", {k: v for k, v in actual[mode].items() if k != "results"})
            runs.append({"trial": trial, "mode": mode, "seconds": seconds})
            atomic_json(args.output/"STATE.json", {"status": "running", "runs": runs, "new_optimization_count": 0})
        rows = actual["integer"]["route"]["routes"]
        worker = actual["integer"]["route"].get("worker", {})
        checks = {"all48_full_geometric_matches_exact": len(actual["native"]["results"]) == 48
            and actual["native"]["results"] == actual["integer"]["results"],
            "all_sources_inputs_exact": all(r["inputs_exact"] for r in actual.values()),
            "all48_admitted_without_feature_truncation": len(rows) == 48 and all(r["route"] == "integer" and r["all_features_preserved"] for r in rows),
            "owned_worker_closed_and_original_CUDA_budget": worker.get("returncode") == 0
                and worker["shutdown"]["peak_CUDA_reserved_bytes"] <= 4*1024**3}
        pairs.append({"trial": trial, "checks": checks, "whole_component_seconds": {m: v["seconds"] for m, v in actual.items()},
            "maximum_real_train_rows": max(n for r in rows for n in r["train_rows"]),
            "peak_CUDA_reserved_bytes": worker.get("shutdown", {}).get("peak_CUDA_reserved_bytes")})
        print(pairs[-1], flush=True)
        if not all(checks.values()):
            raise ValueError("An external actual SIFT component identity, geometry or lifecycle guard failed")
    medians = {m: statistics.median(p["whole_component_seconds"][m] for p in pairs) for m in ("native", "integer")}
    reduction = 1-medians["integer"]/medians["native"]
    checks = {"all_three_complete_external_pairs_guarded": all(all(p["checks"].values()) for p in pairs),
        "external_whole_component_reduction_at_least10percent": reduction >= .1,
        "sources_inputs_unchanged": identity == [data_identity(p) for p in paths]}
    result = {"status": "external_exact_matcher_component_passed" if all(checks.values()) else "failed_verification",
        "checks": checks, "pairs": pairs, "median_seconds": medians, "reduction": reduction,
        "scope": "real_non_box_outdoor_native_SIFT_matching_component_only", "new_optimization_count": 0,
        "whole_pipeline_generalization_or_accuracy_claim": False}
    atomic_json(args.output/"RESULT.json", result)
    print({k: result[k] for k in ("status", "checks", "median_seconds", "reduction")}, flush=True)
    if not all(checks.values()):
        raise ValueError("An external matcher cost or final identity guard failed")


if __name__ == "__main__":
    main()
