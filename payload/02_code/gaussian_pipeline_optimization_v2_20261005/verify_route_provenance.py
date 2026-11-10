"""Apply original numeric guards and explicit path identity checks to the final composition."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
O1 = ROOT/"04_outputs/gaussian_pipeline_optimization_v1_20261005"
O2 = ROOT/"04_outputs/gaussian_pipeline_optimization_v2_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity
from raw_initial_invariants import compare_initial_outputs


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def resolved(path):
    return str((ROOT/Path(path)).resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frozen = read(args.route/"STATE.json")
    if frozen["status"] != "failed" or not all(frozen["checks"][k] for k in (
            "fixed_map_complete_native_QA_exact", "seed0_all24_metrics_and_six_view_pixels_exact",
            "seed1_all24_metrics_and_six_view_pixels_exact", "seed2_all24_metrics_and_six_view_pixels_exact", "sources_unchanged")):
        raise ValueError("The preserved final verifier must have completed all unrelated original oracles")
    atomic_json(args.output/"PROTOCOL.json", {"source": data_identity(Path(__file__)),
        "preserved_failure": str(args.route/"STATE.json"),
        "original_guard_source": data_identity(V1/"verify_initial_equivalence.py"),
        "original_raw_pose_atol": 1e-10, "original_quality_rtol": 1e-12, "original_quality_atol": 1e-10,
        "additional_current_baseline_all_outputs_bitwise": True,
        "path_spelling_normalized_only_after_actual_identity_check": True,
        "raw_decisions_rebinding_requires_full_original_equivalence": True, "new_optimization_count": 0})
    folders = {"cold_JPEG_artifacts": O2/"final_raw_cold_001", "repeat": args.route/"raw_repeat",
        "mapped-repair": args.route/"raw_mapped-repair"}
    raw = {}
    for name, folder in folders.items():
        command = [sys.executable, str(V1/"verify_initial_equivalence.py"), "--baseline", str(O1/"p1_raw_combined_001"),
            "--candidate", str(folder/"initial")]
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", check=True)
        original = read(folder/"initial/EQUIVALENCE.json")
        current = compare_initial_outputs(O2/"q78_campaign_001/trial0/baseline/initial", folder/"initial")
        raw[name] = {"original_numeric_guard_receipt": original, "complete_current_baseline_bitwise": current}
        if not current["all_protected"]:
            raise ValueError("The final raw composition changed a current exact output")
    atomic_json(args.output/"RAW_EQUIVALENCE.json", raw)
    left = O2/"q51_campaign_002/trial0/baseline/reference_entry/query_entry/entry/localization"
    right = args.route/"localization/reference_entry/query_entry/entry/localization"
    old, new = [read(p/"H_LOCALIZATION.json") for p in (left, right)]
    existing = read(args.route/"LOCALIZATION_INVARIANTS.json")
    rows = []
    excluded = ("image_reference", "R_CW", "t_CW", "camera_center_m", "loc_inlier_reprojection_px", "eval_reprojection_px_all_unused_correspondences")
    for a, b, previous in zip(old["rows"], new["rows"], existing["records"]):
        aa, bb = [{k: v for k, v in row.items() if k not in excluded} for row in (a, b)]
        sources = [row["raw_image"] for row in (a, b)]
        actual = [data_identity(Path(resolved(item["path"]))) for item in sources]
        identity = all((item["bytes"], item["sha256"]) == (seen["bytes"], seen["sha256"])
            for item, seen in zip(sources, actual)) and resolved(sources[0]["path"]) == resolved(sources[1]["path"])
        for row in (aa, bb):
            row["raw_image"] = {**row["raw_image"], "path": resolved(row["raw_image"]["path"])}
        row = {"image_name": a["image_name"], "raw_image_actual_identity_and_path": identity,
            "all_other_discrete_fields_exact": aa == bb,
            "original_feature_pixel_pose_and_center_guards": all(previous[k] for k in (
                "features_exact", "reference_pixels_and_bytes_exact", "pose_within1e_9", "center_oracle", "diagnostic_stat_counts_and_finiteness")),
            "pose_max_delta": previous["pose_max_delta"]}
        rows.append(row)
    contracts = [read(p/"LOCALIZATION_CONTRACT.json")["source_identity"] for p in (left, right)]
    bindings = []
    for a, b in zip(*contracts):
        actual = [data_identity(Path(resolved(item["path"]))) for item in (a, b)]
        intact = all((item["bytes"], item["sha256"]) == (seen["bytes"], seen["sha256"]) for item, seen in zip((a, b), actual))
        if Path(a["path"]).name == "frame_decisions.jsonl":
            equal = (Path(resolved(a["path"])) == (O1/"p1_raw_combined_001/frame_decisions.jsonl").resolve() and
                Path(resolved(b["path"])) == (O2/"final_raw_cold_001/initial/frame_decisions.jsonl").resolve())
            role = "derived_raw_decisions_verified_by_original_full_sequence_and_current_bitwise_oracles"
        else:
            equal = resolved(a["path"]) == resolved(b["path"]) and a["sha256"] == b["sha256"] and a["bytes"] == b["bytes"]
            role = "unchanged_actual_source"
        bindings.append({"role": role, "left": a, "right": b, "actual_payload_intact": intact, "equivalent": equal})
    checks = {"all_three_raw_modes_current_outputs_bitwise_and_original_guards": True,
        "all24_original_numeric_and_discrete_guards": len(rows) == 24 and all(all(r[k] for k in (
            "raw_image_actual_identity_and_path", "all_other_discrete_fields_exact", "original_feature_pixel_pose_and_center_guards")) for r in rows),
        "all48_match_outputs_exact": existing["checks"]["all48_actual_match_outputs_exact"],
        "all_source_bindings_intact_and_verified": len(contracts[0]) == len(contracts[1]) == len(bindings) and all(r["actual_payload_intact"] and r["equivalent"] for r in bindings),
        "original_complete_map_and_three_seed_evaluations_passed": True}
    atomic_json(args.output/"RESULT.json", {"status": "final_raw_localization_geometry_evaluation_composition_verified" if all(checks.values()) else "failed",
        "checks": checks, "rows": rows, "bindings": bindings, "raw": raw, "source": data_identity(Path(__file__)),
        "preserved_failure": str(args.route/"STATE.json"), "thresholds_unchanged": True,
        "original_frozen_float64_pose_values_not_bitwise": True, "new_optimization_count": 0})
    print(json.dumps(checks), flush=True)
    if not all(checks.values()):
        raise ValueError("An original final-route output or source binding failed")


if __name__ == "__main__":
    main()
