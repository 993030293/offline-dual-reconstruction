"""Verify actual localization invariants before accepting neighbor acceleration."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

V1 = Path(__file__).resolve().parents[2]/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("diagnostic", "baseline", "candidate", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    diagnostic = read(args.diagnostic/"RESULT.json")
    runs = {name: read(path/"localization/H_LOCALIZATION.json") for name, path in
            (("baseline", args.baseline), ("candidate", args.candidate))}
    traces = {name: read(path/"NEIGHBOR_TRACE.json") for name, path in
              (("baseline", args.baseline), ("candidate", args.candidate))}
    entries = [read(path/"ENTRY_CONTRACT.json") for path in (args.baseline, args.candidate)]
    protocols = [read(path/"localization/LOCALIZATION_CONTRACT.json") for path in (args.baseline, args.candidate)]
    left, right = runs["baseline"], runs["candidate"]
    checks = {
        "diagnostic_all_guards_passed": diagnostic["status"] == "neighbor_stage_passed_real_entry_pending" and all(diagnostic["checks"].values()),
        "candidate_source_matches_diagnostic": data_identity(Path(__file__).with_name("frozen_neighbors.py")) == read(args.diagnostic/"PROTOCOL.json")["candidate_source"],
        "explicit_baseline_and_snapshot": [e["mode"] for e in entries] == ["baseline", "snapshot"],
        "same_adapter_and_reference_sources": entries[0]["sources"] == entries[1]["sources"],
        "source_unchanged": all(t["source_unchanged"] for t in traces.values()),
        "same_frozen_input_identities": protocols[0]["source_identity"] == protocols[1]["source_identity"],
        "same_single_thread_PnP": all(p["PnP_threads"] == 1 for p in protocols),
        "same_lazy_feature_storage": all(p["feature_storage"] == "lazy" for p in protocols),
        "all24_localized": all(r["status"] == "all24_localized_and_frozen" and len(r["rows"]) == 24 for r in runs.values()),
        "same_ordered_names": [r["image_name"] for r in left["rows"]] == [r["image_name"] for r in right["rows"]],
        "same_intrinsics_and_map": left["PINHOLE_camera"] == right["PINHOLE_camera"] and left["map_source"] == right["map_source"],
        "all48_real_match_outputs_exact": traces["baseline"]["match_calls"] == traces["candidate"]["match_calls"] and len(traces["baseline"]["match_calls"]) == 48,
        "all24_actual_neighbors_exact": [{k: c[k] for k in ("timestamp", "neighbors")} for c in traces["baseline"]["calls"]]
            == [{k: c[k] for k in ("timestamp", "neighbors")} for c in traces["candidate"]["calls"]] and len(traces["candidate"]["calls"]) == 24,
        "all1016_snapshot_cameras": traces["candidate"]["snapshot"]["camera_count"] == 1016,
    }
    records = []
    for a, b in zip(left["rows"], right["rows"]):
        name = a["image_name"]
        item = {"image_name": name}
        feature_name = Path(name).stem+".npz"
        with np.load(args.baseline/"localization/feature_cache"/feature_name) as x, np.load(args.candidate/"localization/feature_cache"/feature_name) as y:
            item["features_exact"] = set(x.files) == set(y.files) and all(np.array_equal(x[k], y[k]) for k in x.files)
        item["reference_bytes_exact"] = data_identity(Path(a["image_reference"]))["sha256"] == data_identity(Path(b["image_reference"]))["sha256"]
        item["pose_max_delta"] = max(float(np.max(np.abs(np.asarray(a[k])-np.asarray(b[k])))) for k in ("R_CW", "t_CW"))
        item["pose_within1e_9"] = item["pose_max_delta"] <= 1e-9
        pose_derived = ("camera_center_m", "loc_inlier_reprojection_px",
                        "eval_reprojection_px_all_unused_correspondences")
        excluded = ("image_reference", "R_CW", "t_CW", *pose_derived)
        item["remaining_discrete_and_matching_results_exact"] = {k: v for k, v in a.items() if k not in excluded} == {k: v for k, v in b.items() if k not in excluded}
        center_checks = []
        for record in (a, b):
            translation = np.asarray(record["t_CW"])
            expected = -np.asarray(record["R_CW"]).T@translation
            roundoff = 64*np.finfo(np.float64).eps*(1+np.linalg.norm(translation))
            center_checks.append(bool(np.max(np.abs(expected-record["camera_center_m"])) <= roundoff))
        item["derived_centers_satisfy_same_pose_oracle"] = all(center_checks)
        item["derived_stat_counts_exact_and_values_finite"] = all(
            a[k]["count"] == b[k]["count"] and np.isfinite(list(a[k].values())).all()
            and np.isfinite(list(b[k].values())).all() for k in pose_derived[1:])
        item["derived_stats_max_delta_diagnostic_only"] = max(
            abs(a[k][q]-b[k][q]) for k in pose_derived[1:] for q in a[k] if q != "count")
        item["passed"] = all(v for k, v in item.items() if isinstance(v, bool))
        records.append(item)
    checks["all24_pixels_features_pose_geometry_invariants"] = len(records) == 24 and all(r["passed"] for r in records)
    acceptance = {"status": "accepted_stage_pending_round2_combination" if all(checks.values()) else "failed",
        "mechanism": "Q28_FROZEN_CAMERA_NEIGHBORS", "checks": checks, "records": records,
        "stage_time_reduction": diagnostic["stage_time_reduction"],
        "median_baseline_seconds": diagnostic["median_baseline_seconds"],
        "median_candidate_seconds_including_build": diagnostic["median_candidate_seconds_including_build"],
        "real_entry_neighbor_seconds": {name: sum(c["seconds"] for c in t["calls"]) for name, t in traces.items()},
        "real_entry_elapsed_seconds": {name: r["elapsed_seconds"] for name, r in runs.items()},
        "stage_accepted": all(checks.values()), "round2_combination_accepted": False, "counted_final": False,
        "new_pose_accuracy_claim": False, "total_SfM_speed_claim": False,
        "assessor_correction": "The original1e-9 pose guard remains. Pose-derived floating diagnostics are not a second bitwise pose requirement; exact matches, geometric edges, support counts and input pixels remain required. Original failed acceptance is preserved.",
        "evidence": [data_identity(args.diagnostic/"RESULT.json"), data_identity(args.baseline/"NEIGHBOR_TRACE.json"),
                     data_identity(args.candidate/"NEIGHBOR_TRACE.json")], "source": data_identity(Path(__file__))}
    atomic_json(args.output/"ACCEPTANCE.json", acceptance)
    print(json.dumps({k: acceptance[k] for k in ("status", "stage_time_reduction", "checks")}))
    if not all(checks.values()):
        raise ValueError("Neighbor acceleration failed an original real-entry guard")


if __name__ == "__main__":
    main()
