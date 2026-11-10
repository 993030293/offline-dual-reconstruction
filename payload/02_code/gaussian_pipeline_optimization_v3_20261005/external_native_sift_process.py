"""Check uniformly selected public images without pose or object priors."""

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
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from native_sift_process import NativeSiftProcess, options


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005/sp001")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    directory = ROOT/"04_outputs/public_full_e26_scene_v3_20260929T141800Z/attempt02/01_visual_workspace/images"
    inventory = sorted(directory.glob("*.png"))
    if len(inventory) != 360:
        raise ValueError("The existing public image inventory changed")
    paths = [inventory[i] for i in np.linspace(0, len(inventory)-1, 6).round().astype(int)]
    sources = [Path(__file__), Path(__file__).with_name("native_sift_process.py")]
    identities, inputs = [data_identity(p) for p in sources], [data_identity(p) for p in paths]
    protocol = {"scope": "six_uniform_name_order_images_existing_Oxford_development_component",
        "images": inputs, "source_image_inventory_count": len(inventory), "GT_pose_ROI_box_prior_used": False,
        "full360_or_full_pipeline_claim": False, "blind_test": False,
        "orders": [["incumbent", "process"], ["process", "incumbent"], ["incumbent", "process"]],
        "each_complete_batch_contains_constructor_input_decode_source_verify_transport_and_close": True}
    atomic_json(args.output/"PROTOCOL.json", protocol)
    cv2.setNumThreads(8)
    pairs, worker_traces = [], []
    for ordinal, order in enumerate(protocol["orders"]):
        actual, costs = {}, {}
        for mode in order:
            before = time.perf_counter()
            if mode == "incumbent":
                extractor = pycolmap.Sift(options(), pycolmap.Device.cpu)
                values = []
                for path in paths:
                    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                    if gray is None:
                        raise ValueError("A declared public image failed to decode")
                    values.append(extractor.extract(gray))
            else:
                broker = NativeSiftProcess(paths)
                try:
                    extractor = broker.bind(pycolmap.Sift(options(), pycolmap.Device.cpu))
                    values = [extractor.extract(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)) for path in paths]
                finally:
                    broker.close()
                worker_traces.append(broker.trace)
            actual[mode] = values
            costs[mode] = time.perf_counter()-before
        compare_started = time.perf_counter()
        checks = {"all6_complete_native_keypoints_descriptors_dtype_shape_and_bytes_exact": all(
            len(a) == len(b) and all(x.shape == y.shape and x.dtype == y.dtype and x.tobytes() == y.tobytes()
                for x, y in zip(a, b)) for a, b in zip(actual["incumbent"], actual["process"])),
            "all_inputs_and_sources_unchanged": inputs == [data_identity(p) for p in paths] and identities == [data_identity(p) for p in sources],
            "two_processes_closed_and_all6_delivered": broker.trace["closed"] and broker.trace["submitted"] == broker.trace["delivered"] == 6
                and len(broker.trace["saved_workers"]) == 2}
        pair = {"trial": ordinal, "order": order, "checks": checks, "complete_batch_seconds": costs,
            "complete_compare_seconds": time.perf_counter()-compare_started,
            "feature_counts": [len(v[0]) for v in actual["incumbent"]]}
        pairs.append(pair)
        if not all(checks.values()):
            raise ValueError("The public native CPU feature component changed")
    medians = {mode: statistics.median(p["complete_batch_seconds"][mode] for p in pairs) for mode in ("incumbent", "process")}
    checks = {"all_three_six_image_component_pairs_exact_and_closed": len(pairs) == 3 and all(all(p["checks"].values()) for p in pairs),
        "final_sources_and_inputs_unchanged": identities == [data_identity(p) for p in sources] and inputs == [data_identity(p) for p in paths]}
    atomic_json(args.output/"RESULT.json", {"status": "external_six_image_component_passed" if all(checks.values()) else "failed_verification",
        "checks": checks, "protocol": protocol, "pairs": pairs, "worker_traces": worker_traces, "sources": identities,
        "medians": medians, "diagnostic_batch_reduction": 1-medians["process"]/medians["incumbent"],
        "whole_seconds": time.perf_counter()-started, "new_optimization_count": 0,
        "claim_boundary": "public native feature bytes only; not independent pose GT, full dataset speed, training quality or unseen object generalization"})
    print(json.dumps({"status": "external_six_image_component_passed", "medians": medians,
        "diagnostic_reduction": 1-medians["process"]/medians["incumbent"], "counted": False}))


if __name__ == "__main__":
    main()
