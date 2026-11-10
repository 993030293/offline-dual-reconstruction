"""Accept one scoped vector-statistics mechanism after actual reviewer exit."""

from pathlib import Path
import time

import psutil

import profile_quality_corner_cost as reference
from vector_gray_stats_evidence import cost_snapshot

ROOT = reference.ROOT
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"


def main():
    started = time.perf_counter()
    base = OUT / "vector_gray_stats_001"
    if (base / "ACCEPTANCE.json").exists():
        raise ValueError("The immutable vector acceptance receipt already exists")
    review = reference.load(base / "r1/RESULT.json")
    if review["status"] != "separate_root_self_review_passed_post_exit_final_cost_and_text_pending" or not all(review["checks"].values()):
        raise ValueError("The separate exactness and cost self-review failed")
    if reference.data_identity(Path(review["source"]["path"])) != review["source"]:
        raise ValueError("The actual frozen reviewer source changed")
    for source in review["sources_verified"] + review["related_ids"]:
        if reference.data_identity(Path(source["path"])) != source:
            raise ValueError("An actual frozen scientific or audit source changed")
    for file in ("LEASE_R1.json", "GUARD_R1.json"):
        row = reference.load(base / file)
        if row["returncode"] != 0:
            raise ValueError("The actual reviewer did not exit successfully")
        identities = list(row.get("owned_process_identities", []))
        identities.extend({"pid": row[prefix + "_pid"], "creation_time": row[prefix + "_creation_time"]}
                          for prefix in ("guard", "child") if prefix + "_creation_time" in row)
        for item in identities:
            try:
                if psutil.Process(item["pid"]).create_time() == item["creation_time"]:
                    raise ValueError("The exact successful reviewer instance is still alive")
            except psutil.NoSuchProcess:
                pass
    parent = reference.load(ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json")
    if parent["additional_accepted_count"] != 26 or any(name.startswith("R144") for name in parent["accepted_mechanisms"]):
        raise ValueError("The original parent count changed or this family was already counted")
    complete = reference.load(base / "c0/RESULT.json")
    cost = cost_snapshot(base, complete, time.perf_counter() - started)
    if cost["net_reductions"]["phase_seconds"] < .05 or cost["whole_median_ratio"] > 1.01 or any(
            pair["whole_seconds"]["vector"] > 1.1 * pair["whole_seconds"]["current"] for pair in complete["pairs"]):
        raise ValueError("The original after-known-cost phase or whole gate failed")
    receipt = {"status": "accepted_scoped_exact_SIMD_gray_statistics_under_existing_V2_V4",
        "mechanism": "R144_V2_EXACT_SIMD_GRAY_STATISTICS", "new_optimization_count": 1,
        "parent_accepted_count_before": 26, "parent_goal_stays_active": True,
        "local_effect_confirmed": True, "generalization_pending": True,
        "control": "accepted_R147_R149_R141_R104_Q78_R99_Q50_Q53_P18",
        "raw_frames": 7338, "eligible_frames": 6797, "selected_frames": 1002, "candidate_pairs": 11583,
        "fresh_complete_pairs": 3, "all_original_six_fields_decisions_cache_K_coarse_poses_JPEG_and_pairs_exact": True,
        "generic_stat_cases": 21, "generic_complete_quality_cases": 48, "actual_RAW_cases": 32, "public_RGB_derived_cases": 32,
        "original_R144_scalar_negative_preserved": str(OUT / "gm001/p0/RESULT.json"),
        "generic": str(base / "g0/RESULT.json"), "complete128": str(base / "s0/RESULT.json"),
        "complete7338": str(base / "c0/RESULT.json"), "actual_recovery_external": str(base / "v0/RESULT.json"),
        "review": str(base / "r1/RESULT.json"), "review_type": "separate_root_self_review_not_independent_agent",
        "failed_r0_preserved": str(base / "r0/FAILURE.json"), "owned_build_telemetry_cleanup": str(base / "tc0/RESULT.json"),
        "cost_before_publisher_exit": cost, "production_default_changed": False,
        "no_box_geometry_or_semantic_tuning": True, "not_sensor_RAW_GT_or_strict_blind_generalization": True,
        "source": reference.data_identity(Path(__file__)), "publication_seconds_before_write": time.perf_counter() - started}
    reference.atomic_json(base / "ACCEPTANCE.json", receipt)
    print({"status": receipt["status"], "new_optimization_count": 1, "phase_net": cost["net_reductions"]["phase_seconds"],
        "whole_net": cost["net_reductions"]["whole_seconds"], "RSS_medians_bytes": review["peak_tree_RSS_medians_bytes"],
        "source_count": review["current_source_identity_count"], "snapshot_count": review["executed_source_snapshot_count"]}, flush=True)


if __name__ == "__main__":
    main()
