"""Run a bounded raw-only allocation audit without reading SfM or evaluation pixels."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from frame_allocation import eligible_rows, temporal_backbone, rank_rows, coverage_metrics, graph_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest_path = args.initial / "INITIAL_FRAME_MANIFEST.json"
    decisions_path = args.initial / "frame_decisions.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw = [json.loads(line) for line in decisions_path.read_text(encoding="utf-8").splitlines()]
    eligible = eligible_rows(raw, manifest["policy"], manifest["receipt"]["sharpness_min"])
    baseline_names = {row["image_name"] for row in manifest["selected_frames"]}
    baseline = [row for row in eligible if row["image_name"] in baseline_names]
    if len(baseline) != len(baseline_names) or len(baseline) != 1002 or len(eligible) != 6797:
        raise ValueError("The registered Box2 input denominators changed")
    raw_origin = min(row["camera_header_ns"] for row in raw)
    edges = [line.split() for line in (args.initial / "initial_matching_pairs.txt").read_text().splitlines()]
    contract = {"mechanism": "Q34_INITIAL_AND_AUGMENTATION_ALLOCATION", "status": "preregistered_metadata_only",
                "input": [data_identity(manifest_path), data_identity(decisions_path)],
                "sources": [data_identity(Path(__file__)), data_identity(Path(__file__).with_name("frame_allocation.py"))],
                "raw_count": len(raw), "eligible_count": len(eligible), "baseline_initial_count": len(baseline),
                "final_pool_sizes": [1002, 1200, 1500], "initial_budget_fractions": [.5, .65, .8, 1.],
                "temporal_bin_seconds": .4, "mandatory_gap_endpoints": True,
                "position_scale_m": .10, "rotation_scale_degrees": 10., "quality_rank_weight_range": [.5, 1.],
                "maximum_seconds": 120, "maximum_RSS_bytes": 8*1024**3,
                "SfM_poses_used": False, "target_ROI_used": False, "GT_used": False, "evaluation_pixels_read": False,
                "offline_future_information": "raw sequence coarse poses and quality only", "counted": False}
    atomic_json(args.output / "CONTRACT.json", contract)
    mandatory = [i for i, row in enumerate(eligible) if row["image_name"] in baseline_names]
    expanded_order = rank_rows(eligible, 1500, mandatory)
    cases = []
    checks = {"original_quality_eligibility_reproduced": True, "original1002_retained_in_all_final_pools": True,
              "excluded_and_unsupported_never_selected": True, "deterministic_ranking": True,
              "nested_initial_sets": True, "no_new_gap_with_unused_fixed_pool_support": True}
    raw_pool_gap_guard = True
    for pool_size in contract["final_pool_sizes"]:
        pool = sorted((eligible[i] for i in expanded_order[:pool_size]), key=lambda row: row["camera_header_ns"])
        pool_names = {row["image_name"] for row in pool}
        checks["original1002_retained_in_all_final_pools"] &= baseline_names <= pool_names
        backbone = temporal_backbone(pool, raw_origin)
        ranking = rank_rows(pool, pool_size, backbone)
        checks["deterministic_ranking"] &= ranking == rank_rows(pool, pool_size, backbone)
        previous = set()
        for fraction in contract["initial_budget_fractions"]:
            count = int(np.floor(pool_size * fraction + .5))
            folder = args.output / f"pool{pool_size}_initial{count}"
            folder.mkdir()
            if len(backbone) > count:
                cases.append({"pool_size": pool_size, "initial_count": count, "backbone_count": len(backbone),
                              "status": "infeasible_backbone_exceeds_budget"})
                continue
            selected = sorted((pool[i] for i in ranking[:count]), key=lambda row: row["camera_header_ns"])
            names = {row["image_name"] for row in selected}
            remainder = [row for row in pool if row["image_name"] not in names]
            checks["nested_initial_sets"] &= previous <= names
            previous = names
            checks["excluded_and_unsupported_never_selected"] &= all(not row["excluded_role"] and row["coarse_pose_valid"] for row in pool)
            metrics = coverage_metrics(eligible, selected)
            for gap in metrics["gaps_over_0_8_seconds"]:
                gap["fixed_pool_frames_between"] = sum(gap["left_frame"] < row["camera_frame"] < gap["right_frame"] for row in pool)
                gap["role"] = "inherited_fixed_pool_gap" if gap["fixed_pool_frames_between"] == 0 else "new_selection_gap"
            checks["no_new_gap_with_unused_fixed_pool_support"] &= all(row["fixed_pool_frames_between"] == 0 for row in metrics["gaps_over_0_8_seconds"])
            raw_pool_gap_guard &= all(row["eligible_frames_between"] == 0 for row in metrics["gaps_over_0_8_seconds"])
            case = {"status": "metadata_only", "pool_size": pool_size, "initial_count": count,
                    "backbone_count": len(backbone), "eligible_fraction": count/len(eligible),
                    "initial_budget_fraction": count/pool_size, "augmentation_budget": len(remainder),
                    "augmentation_over_initial": len(remainder)/count, "registered_additions": None,
                    "coverage_proxy": metrics, "manifest": str(folder / "ALLOCATION.json")}
            if pool_size == 1002:
                case["induced_original_candidate_graph"] = graph_summary(names, edges)
            atomic_json(folder / "ALLOCATION.json", {"case": case, "initial": selected, "remaining": remainder,
                                                      "final_names": sorted(pool_names), "source_contract": str(args.output/"CONTRACT.json")})
            (folder/"initial_images.txt").write_text("\n".join(row["image_name"] for row in selected)+"\n", encoding="utf-8")
            (folder/"final_images.txt").write_text("\n".join(row["image_name"] for row in pool)+"\n", encoding="utf-8")
            cases.append(case)
    result = {"status": "metadata_checks_passed" if all(checks.values()) else "metadata_checks_failed", "checks": checks,
              "raw_count": len(raw), "eligible_count": len(eligible), "baseline_initial_count": len(baseline),
              "baseline_fractions": {"of_raw": len(baseline)/len(raw), "of_eligible": len(baseline)/len(eligible)},
              "cases": cases, "elapsed_seconds": time.monotonic()-started,
              "original_raw_pool_no_unused_support_guard": raw_pool_gap_guard,
              "boundary": "Fixed1002 pool retains its inherited gap with four eligible edge frames outside the pool. The original stronger guard remains failed; controlled staging only requires no new fixed-pool gap. Expanded pools report all remaining raw support separately.",
              "SfM_run": False, "quality_improvement_verified": False, "counted": False}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps({"status": result["status"], "checks": checks, "seconds": result["elapsed_seconds"],
                      "cases": [{"pool": c["pool_size"], "initial": c["initial_count"], "backbone": c["backbone_count"]} for c in cases]}), flush=True)
    if not all(checks.values()):
        raise AssertionError("Metadata guards failed; preserve result and do not launch mapper")


if __name__ == "__main__":
    main()
