"""Consolidate actual accepted mechanisms and related combination oracles without new gains."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
O2 = ROOT/"04_outputs/gaussian_pipeline_optimization_v2_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity
from raw_initial_invariants import compare_initial_outputs
from compare_actual_localization import compare


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    target = O2/"FINAL_ROUTE_ACCEPTANCE.json"
    if target.exists():
        raise FileExistsError(target)
    registry = {"Q28": "q28_acceptance_002", "Q30": "q30_campaign_002", "Q31": "q31_acceptance_001",
        "Q33": "q33_campaign_001", "Q47": "q47_acceptance_001", "Q48": "q48_acceptance_001",
        "Q49": "q49_native_observations_001", "Q50": "q50_acceptance_001", "Q51": "q51_acceptance_001",
        "Q53": "q53_acceptance_001", "Q55": "q55_resource_repair_acceptance_001",
        "Q58": "camera_execution_acceptance_002", "Q59": "camera_execution_acceptance_002", "Q60": "camera_execution_acceptance_002",
        "Q61": "q61_acceptance_002", "Q69": "viewer_execution_acceptance_001", "Q71": "viewer_execution_acceptance_001",
        "Q76": "q76_acceptance_001", "Q77": "viewer_execution_acceptance_001", "Q78": "q78_acceptance_001"}
    items = []
    for mechanism, folder in registry.items():
        receipt = O2/folder/"ACCEPTANCE.json"
        value = read(receipt)
        if not value.get("checks") or not all(value["checks"].values()):
            raise ValueError("An original mechanism acceptance failed: "+mechanism)
        items.append({"mechanism": mechanism, "acceptance": str(receipt), "original_status": value["status"],
            "all_original_checks_passed": True})
    provenance = read(O2/"final_route_provenance_001/RESULT.json")
    browser = read(O2/"final_viewer_browser_003/BROWSER_VALIDATION.json")
    visual = read(O2/"final_viewer_browser_003/VISUAL_REVIEW.json")
    for source in browser["bound_sources"]:
        if data_identity(Path(source["path"])) != source:
            raise ValueError("An actual final browser source changed")
    raw = compare_initial_outputs(O2/"final_raw_cold_001/initial", O2/"final_entry_initial_001/initial")
    qa = read(O2/"final_entry_qa_001/QA.json")
    left = O2/"final_route_checks_001/localization/reference_entry/query_entry/entry"
    right = O2/"final_entry_localization_001/reference_entry/query_entry/entry"
    localization = compare(left, right, left/"NEIGHBOR_TRACE.json", right/"NEIGHBOR_TRACE.json")
    records = []
    excluded = ("image_reference", "R_CW", "t_CW", "camera_center_m", "loc_inlier_reprojection_px", "eval_reprojection_px_all_unused_correspondences")
    for a, b, old in zip(read(left/"localization/H_LOCALIZATION.json")["rows"], read(right/"localization/H_LOCALIZATION.json")["rows"], localization["records"]):
        aa, bb = [{k: v for k, v in row.items() if k not in excluded} for row in (a, b)]
        for row in (aa, bb):
            source = row["raw_image"]
            path = (ROOT/source["path"]).resolve()
            actual = data_identity(path)
            if any(source[key] != actual[key] for key in ("bytes", "sha256")):
                raise ValueError("A final entry raw image identity changed")
            row["raw_image"] = {**source, "path": str(path)}
        records.append({"image_name": a["image_name"], "all_discrete_fields_and_actual_path_exact": aa == bb,
            "all_original_feature_pixel_pose_center_guards": all(old[k] for k in ("features_exact", "reference_pixels_and_bytes_exact", "pose_within1e_9", "center_oracle", "diagnostic_stat_counts_and_finiteness"))})
    contracts = [read(p/"localization/LOCALIZATION_CONTRACT.json")["source_identity"] for p in (left, right)]
    bindings = []
    for a, b in zip(*contracts):
        actual = [data_identity((ROOT/item["path"]).resolve()) for item in (a, b)]
        intact = all((item["bytes"], item["sha256"]) == (seen["bytes"], seen["sha256"]) for item, seen in zip((a, b), actual))
        same = a["sha256"] == b["sha256"] and a["bytes"] == b["bytes"]
        if Path(a["path"]).name != "frame_decisions.jsonl":
            same = same and (ROOT/a["path"]).resolve() == (ROOT/b["path"]).resolve()
        bindings.append({"left": a, "right": b, "actual_payload_intact": intact, "payloads_exact": same})
    checks = {"twenty_unique_new_mechanisms": len(registry) == len(set(registry)) == 20,
        "raw_localization_map_and_all_three_seed_evaluation_combinations": all(provenance["checks"].values()),
        "fresh_model_viewer_actual104_pixels_projection_and_controls": browser["status"] == "current_models_actual_combination_passed_visual_pending" and
            len(browser["pixels"]) == 104 and all(row["equal"] for row in browser["pixels"]) and all(row["passed"] for row in browser["checks"]) and len(browser["controls"]) == 2,
        "four_actual_current_drone_views_self_reviewed": all(visual["checks"].values()) and len(visual["captures"]) == 4,
        "actual_complete_common_prefix_and_archived_native_recovery": all(read(O2/"camera_shared_prefix_recovery_001/RESULT.json")["checks"].values()) and
            all(read(O2/"q76_acceptance_001/ACCEPTANCE.json")["checks"].values()),
        "unified_entry_actual_raw_output_exact": raw["all_protected"],
        "unified_entry_actual_complete_QA": qa["status"] == "full_native_projection_QA_passed" and qa["identity_count"] == 959120,
        "unified_entry_actual_all24_48_localization_protected": len(records) == 24 and all(r["all_discrete_fields_and_actual_path_exact"] and r["all_original_feature_pixel_pose_center_guards"] for r in records) and localization["checks"]["all48_actual_match_outputs_exact"],
        "unified_entry_source_bindings_actual_and_exact": len(contracts[0]) == len(contracts[1]) == len(bindings) and all(r["actual_payload_intact"] and r["payloads_exact"] for r in bindings)}
    for label in ("baseline", "combined"):
        native = read(O2/f"camera_quality_training_001/seed0/evaluation/EVALUATION.json")["results"][label]["model"]
        displayed = read(O2/f"viewer_execution_001/release/{label}/ASSET_MANIFEST.json")["scene"]
        checks[label+"_current_evaluated_model_displayed"] = native["sha256"] == displayed["source_ply_sha256"]
    if not all(checks.values()):
        atomic_json(O2/"FINAL_ROUTE_FAILED.json", {"checks": checks})
        raise ValueError("The complete actual route acceptance has an unresolved guard")
    atomic_json(target, {"status": "twenty_effective_mechanisms_and_related_combinations_verified", "additional_accepted_count": 20,
        "prior_accepted_count": 15, "full_route_combination_accepted": True, "mechanisms": items, "checks": checks,
        "entry": str(Path(__file__).with_name("effective_route.py")), "entry_invariants": {"raw": raw, "localization": records, "source_bindings": bindings},
        "sources": [data_identity(Path(__file__)), data_identity(Path(__file__).with_name("effective_route.py"))],
        "preview_url": "http://127.0.0.1:8210/release/combined/cases/packed/index.html?mode=box",
        "scope": "fixed_actual_map_reused_with_related_raw_localization_QA_training_evaluation_recovery_and_viewer_combinations",
        "full_new_SfM_or_LIO_replay_executed": False, "raw_to_new_map_end_to_end_speedup_claim": False,
        "independent_geometry_or_quality_improvement_claim": False, "independent_review": False,
        "repeated_cache_first_costs_and_memory_tradeoffs_preserved": True, "all_original_failed_candidates_retained": True,
        "document_text_review_separate": True})
    print(json.dumps({"status": "twenty_effective_mechanisms_and_related_combinations_verified", "count": 20, "checks": checks}), flush=True)


if __name__ == "__main__":
    main()
