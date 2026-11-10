"""Promote actual verified cache, motion and shared-build gains independently."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "visual-review", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    result, visual = read(args.campaign/"BROWSER_RESULT.json"), read(args.visual_review)
    expected = {"Q69", "Q71", "Q77"}
    checks = {"three_actual_native_browser_mechanisms_passed": set(result["acceptable_mechanisms"]) == expected and not result["failures"],
        "all208_original_static_pixels_and_projection_matrices_exact": len(result["pixels"]) == 208 and all(row["equal"] for row in result["pixels"]),
        "all_lifecycle_integrity_and_navigation_guards_passed": all(row["passed"] for row in result["checks"]),
        "actual_free_flight_orbit_drone_path_and_both_ranges": {row["label"] for row in result["controls"]} == {"motion", "shared", "shared-reference"},
        "actual_RTX5060Ti_hardware_renderer": all("RTX 5060 Ti" in row["actual_GL_renderer"] for row in result["cacheWarm"]),
        "real_six_drone_views_visual_self_review": visual["method"] == "actual_browser_render_image_self_review"
            and visual["all_six_actual_drone_renders_reviewed"] and not visual["material_candidate_target_regression_observed"],
        "bound_browser_sources_unchanged": all(data_identity(Path(row["path"]))["sha256"] == row["sha256"] for row in result["bound_sources"])}
    args.output.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "stage_accepted_combination_pending" if all(checks.values()) else "failed_verification", "checks": checks,
        "stage_accepted_count": 3 if all(checks.values()) else 0, "full_route_combination_pending": True,
        "mechanisms": ["Q69_OPTIONAL_MOTION_RESOLUTION_WITH_EXACT_STATIC_RESTORATION", "Q71_VERIFIED_PERSISTENT_IMMUTABLE_ASSET_CACHE", "Q77_SHARED_IDENTICAL_NATIVE_RENDERER_BUILD"],
        "Q69": result["Q69"], "Q69_actual_frame_pacing": result["Q69_pacing"], "Q71": result["Q71"], "Q71_cold": result["cacheCold"],
        "Q77_component_storage": result["sources"]["Q77_shared_build"], "visual_review": visual,
        "rejected_Q68_not_counted": True, "entire_GPU_work_power_or_bundle_reductions_not_claimed": True,
        "all24_preview_capture_positions_are_distinct_from_all24_training_quality_development_frames": True,
        "receipts": [str(args.campaign/"BROWSER_RESULT.json"), str(args.visual_review)], "assessor_source": data_identity(Path(__file__))}
    atomic_json(args.output/"ACCEPTANCE.json", receipt)
    if not all(checks.values()):
        raise AssertionError("An actual browser execution acceptance protection failed")
    print({"status": receipt["status"], "stage_accepted_count": receipt["stage_accepted_count"]}, flush=True)


if __name__ == "__main__":
    main()
