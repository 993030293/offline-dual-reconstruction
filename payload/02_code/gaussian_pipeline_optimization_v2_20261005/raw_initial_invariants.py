"""Compare complete raw-selection decisions and output bytes without using timing as proof."""

import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def compare_initial_outputs(baseline, candidate):
    baseline, candidate = Path(baseline), Path(candidate)
    manifests = [read(root/"INITIAL_FRAME_MANIFEST.json") for root in (baseline, candidate)]
    receipts = [read(root/"SELECTION_RECEIPT.json") for root in (baseline, candidate)]
    checks = {name+"_exact": manifests[0][name] == manifests[1][name]
        for name in ("selected_frames", "camera_intrinsics", "policy")}
    checks["all_original_input_denominators"] = all((r["raw_frame_count"], r["selected_count"], r["matching_candidate_pair_count"],
        r["quality_and_coarse_pose_eligible_count"], r["excluded_overlap"]) == (7338, 1002, 11583, 6797, 0) for r in receipts)
    for name in ("frame_decisions.jsonl", "initial_matching_pairs.txt", "EXCLUDED_EVALUATION_IDS.json", "RAW_SENSOR_CONTRACT.json"):
        checks[name+"_bytes_exact"] = (baseline/name).read_bytes() == (candidate/name).read_bytes()
    names = sorted(p.name for p in (baseline/"images").iterdir() if p.is_file())
    checks["all1002_image_names_exact"] = (len(names) == 1002 and names == sorted(r["image_name"] for r in manifests[0]["selected_frames"]) and
        names == sorted(p.name for p in (candidate/"images").iterdir() if p.is_file()))
    checks["all1002_actual_JPEG_bytes_exact"] = all((baseline/"images"/name).read_bytes() == (candidate/"images"/name).read_bytes() for name in names)
    with np.load(baseline/"raw_body_odometry.npz", allow_pickle=False) as left, np.load(candidate/"raw_body_odometry.npz", allow_pickle=False) as right:
        checks["all_recorded_raw_odometry_arrays_exact"] = left.files == right.files and all(np.array_equal(left[name], right[name]) for name in left.files)
    return {"all_protected": all(checks.values()), "checks": checks,
        "raw_frames": 7338, "selected": 1002, "candidate_pairs": 11583,
        "pose_producer_or_absolute_accuracy_claim": False}
