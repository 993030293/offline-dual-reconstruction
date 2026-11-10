"""Compare all actual query pixels, features, matches, supports and poses."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def compare(left, right, left_trace, right_trace):
    runs = [json.loads((p/"localization/H_LOCALIZATION.json").read_text(encoding="utf-8")) for p in (left, right)]
    traces = [json.loads(p.read_text(encoding="utf-8")) for p in (left_trace, right_trace)]
    a, b = runs
    checks = {"all24_localized": all(r["status"] == "all24_localized_and_frozen" and len(r["rows"]) == 24 for r in runs),
              "same_ordered_names": [r["image_name"] for r in a["rows"]] == [r["image_name"] for r in b["rows"]],
              "same_map_and_full_K": a["map_source"] == b["map_source"] and a["PINHOLE_camera"] == b["PINHOLE_camera"]}
    trace_keys = "match_calls" if "match_calls" in traces[0] else "calls"
    hashes = [[r["result_sha256"] for r in t[trace_keys]] for t in traces]
    checks["all48_actual_match_outputs_exact"] = len(hashes[0]) == len(hashes[1]) == 48 and hashes[0] == hashes[1]
    records = []
    for x, y in zip(a["rows"], b["rows"]):
        name = x["image_name"]
        with np.load(left/"localization/feature_cache"/(Path(name).stem+".npz")) as p, np.load(right/"localization/feature_cache"/(Path(name).stem+".npz")) as q:
            feature_equal = set(p.files) == set(q.files) and all(np.array_equal(p[k], q[k]) for k in p.files)
        delta = max(float(np.max(np.abs(np.asarray(x[k])-np.asarray(y[k])))) for k in ("R_CW", "t_CW"))
        excluded = ("image_reference", "R_CW", "t_CW", "camera_center_m", "loc_inlier_reprojection_px", "eval_reprojection_px_all_unused_correspondences")
        discrete_equal = {k: v for k, v in x.items() if k not in excluded} == {k: v for k, v in y.items() if k not in excluded}
        centers = []
        for r in (x, y):
            translation = np.asarray(r["t_CW"])
            expected = -np.asarray(r["R_CW"]).T@translation
            centers.append(bool(np.max(np.abs(expected-r["camera_center_m"])) <= 64*np.finfo(np.float64).eps*(1+np.linalg.norm(translation))))
        stats_equal = all(x[k]["count"] == y[k]["count"] and np.isfinite(list(x[k].values())).all()
                          and np.isfinite(list(y[k].values())).all()
                          for k in ("loc_inlier_reprojection_px", "eval_reprojection_px_all_unused_correspondences"))
        pixel_equal = data_identity(Path(x["image_reference"]))["sha256"] == data_identity(Path(y["image_reference"]))["sha256"]
        records.append({"image_name": name, "features_exact": bool(feature_equal), "reference_pixels_and_bytes_exact": pixel_equal,
                        "pose_max_delta": delta, "pose_within1e_9": delta <= 1e-9,
                        "discrete_and_matching_results_exact": discrete_equal, "center_oracle": all(centers),
                        "diagnostic_stat_counts_and_finiteness": bool(stats_equal)})
    checks["all24_invariants"] = len(records) == 24 and all(all(v for v in r.values() if isinstance(v, bool)) for r in records)
    protocols = [json.loads((p/"localization/LOCALIZATION_CONTRACT.json").read_text()) for p in (left, right)]
    checks["same_frozen_inputs"] = protocols[0]["source_identity"] == protocols[1]["source_identity"]
    checks["same_serial_PnP_and_lazy_features"] = all(p["PnP_threads"] == 1 and p["feature_storage"] == "lazy" for p in protocols)
    return {"checks": checks, "records": records, "all_protected": all(checks.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "baseline-trace", "candidate-trace", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = compare(args.baseline, args.candidate, args.baseline_trace, args.candidate_trace)
    result.update(status="actual_localization_invariants_passed" if result["all_protected"] else "failed", counted=False)
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "checks": result["checks"]}))
    if not result["all_protected"]:
        raise ValueError("A complete actual localization invariant failed")


if __name__ == "__main__":
    main()
