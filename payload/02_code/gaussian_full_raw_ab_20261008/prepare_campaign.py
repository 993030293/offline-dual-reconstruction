"""Freeze raw cases, comparison semantics and the complete prior module registry."""

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def main():
    output = ROOT / "04_outputs/gaussian_full_raw_ab_20261008"
    output.mkdir(parents=True, exist_ok=True)
    records = {
        "c0_july": "e25_dynamic_box_development_20260727/recon_collection_20260724_211610_855569_0.mcap",
        "c1": "e26_orbit_capture_20260818_v1/raw_mcap/recon_collection_20260818_202249_217576/recon_collection_20260818_202249_217576_0.mcap",
        "box1": "e26_orbit_capture_20260820_box_pair_v1/raw_mcap/20260820A_Box1/recon_collection_20260820_170154_044592_0.mcap",
        "box2": "e26_orbit_capture_20260820_box_pair_v1/raw_mcap/20260820B_Box2/recon_collection_20260820_170743_689578_0.mcap",
    }
    cases = []
    for name, suffix in records.items():
        raw = Path("E:/lidar_reconstruction_video_data/raw_recordings") / suffix
        if not raw.is_file():
            raise FileNotFoundError(raw)
        stat = raw.stat()
        cases.append({"case": name, "raw": str(raw), "bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns, "fresh_full_hash_verified": False,
            "coarse_route": "fresh_raw_IMU_deskew_local_map_ICP" if name == "c0_july" else "fresh_raw_embedded_odometry",
            "output": str(output / name), "baseline": "pending", "optimized": "pending"})
    receipt_paths = [ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/FINAL_ROUTE_ACCEPTANCE.json",
                     ROOT / "04_outputs/gaussian_pipeline_optimization_v2_20261005/FINAL_ROUTE_ACCEPTANCE.json",
                     ROOT / "research/2026-10-05-gaussian-optimization-round3-50/CURRENT_STATE.json"]
    previous = [json.loads(path.read_text(encoding="utf-8-sig")) for path in receipt_paths]
    modules = []
    for row in previous[0]["mechanisms"]:
        modules.append({"id": row["id"], "round": 1, "prior_evidence": str(ROOT / row["evidence"])})
    for row in previous[1]["mechanisms"]:
        modules.append({"id": row["mechanism"], "round": 2, "prior_evidence": row["acceptance"]})
    for name, evidence in zip(previous[2]["accepted_mechanisms"], previous[2]["acceptance_receipts"], strict=True):
        modules.append({"id": name, "round": 3, "prior_evidence": str(ROOT / evidence)})
    if len(modules) != 75 or len({row["id"] for row in modules}) != 75:
        raise ValueError("The full prior accepted mechanism registry is not exactly 75 unique items")
    for row in modules:
        row.update(current_campaign_status="not_yet_exercised", cases={},
                   prior_acceptance_is_not_new_sequence_acceptance=True)
    atomic_json(output / "MODULE_REGISTRY.json", {"modules": modules, "accepted_before_campaign": 75,
        "fresh_module_effectiveness_verified": 0, "no_inferred_pass_from_aggregate_speed": True})
    atomic_json(output / "CASES.json", {"cases": cases, "created_unix": time.time(),
        "raw_recordings_are_read_only": True, "all_derived_outputs_on_C": True,
        "original_four_scene_Tesla_not_selected": True})
    atomic_json(output / "METRIC_CONTRACT.json", {"schema_version": 1,
        "complete_wall_clock": "all fresh stage child durations and necessary transitions; exclude operator idle; report controller elapsed separately",
        "stage_timing_repeat_count": 1, "statistically_stable_speed_claim_allowed": False,
        "resource_scope": "owned process-tree peak RSS and sampled GPU usage; sampling cadence recorded per stage",
        "quality": {"evaluation_count_per_case": 24, "strata": "raw-index midpoints", "excluded_next_frame": True,
                    "strict_blind_test": False, "PSNR_delta_min_db": -.1, "SSIM_delta_min": -.002,
                    "localization_failures_kept_in_denominator": True,
                    "camera_source": "new map PnP per side, never historical SfM",
                    "target_region": "common raw sensor pixels, frozen before rendering comparison"},
        "scale": {"world_unit": "meters", "independent_reference_available": False,
                  "allowed": ["new_Sim3_scale", "raw_coarse_position_alignment_residual_m", "LiDAR_consistency_diagnostic_m"],
                  "not_available": ["absolute_size_accuracy", "ground_truth_ATE", "ground_truth_RPE"]},
        "training": {"reference_iterations": 10000, "gsplat_iterations": 12000, "seed": 0,
                     "resolution_divisor": 2, "different_iteration_counts_are_policy_comparison": True,
                     "backend_same_iteration_speed_claim_allowed": False},
        "initial_frames": "unchanged original quality/motion/coverage gates; cap 1200; no required reduction",
        "registration": "all original initial images remain denominator; failed images and weak support retained separately",
        "runtime_sources": [data_identity(Path(__file__)), *[data_identity(path) for path in receipt_paths[:2]]],
        "historical_results_used_as_new_runtime_inputs": False})
    print(json.dumps({"status": "campaign_cases_and_registry_frozen", "cases": list(records), "modules": len(modules)}))


if __name__ == "__main__":
    main()
