"""Record map, prior and training-support roles without deleting a camera."""

import argparse
import importlib
import json
from pathlib import Path
import sys
import time

import pycolmap
import psutil

from run_metric_refinement import bind_supported_priors, projection_validity, verified_pose_gap_names
from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "initial-frames", "pose-gap-decisions", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--phase", choices=["before_BA", "after_BA"], required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "02_code/box2_lio_guided_sfm_v1_20261002"))
    common = importlib.import_module("common")
    sys.path.insert(0, str(common.PACKAGE / "code/validated_fresh_runtime"))
    from metric_admission import camera_geometry_metrics

    paths = [args.model / name for name in ("cameras.bin", "images.bin", "points3D.bin")]
    paths.extend(args.model / name for name in ("frames.bin", "rigs.bin") if (args.model / name).exists())
    source = [data_identity(path) for path in paths]
    manifest_path = args.initial_frames / "INITIAL_FRAME_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    decision_path = args.initial_frames / "frame_decisions.jsonl"
    raw = [json.loads(line) for line in decision_path.read_text(encoding="utf-8-sig").splitlines()]
    lookup = {row["image_name"]: row for row in raw if row["decision"] in
              ("eligible_not_selected", "selected_initial_sfm_pending")
              and row["coarse_pose_valid"] and not row["excluded_role"]}
    allowed_gaps = verified_pose_gap_names(args.initial_frames, args.pose_gap_decisions)
    model = pycolmap.Reconstruction(args.model)
    priors, gaps = bind_supported_priors(model, lookup, allowed_gaps)
    common.validate_prior_rows(model, priors)
    prior_names = {row["image_name"] for row in priors}
    initial = {row["image_name"] for row in manifest["selected_frames"]}
    registered = {image.name for image in model.images.values() if image.has_pose}
    excluded = {row["image_name"] for row in raw if row["excluded_role"]}
    if registered & excluded:
        raise ValueError("A protected development frame entered the visual training map")
    policy_path = common.PACKAGE / "config/METRIC_ALIGNMENT_POLICY.json"
    policy = json.loads(policy_path.read_text())["camera_geometry_admission"]
    rows = []
    for image in sorted(model.images.values(), key=lambda value: value.image_id):
        if not image.has_pose:
            continue
        support = camera_geometry_metrics(model, image, policy)
        rows.append({"image_id": image.image_id, "frame_id": image.frame_id,
            "image_name": image.name, "in_original_initial_set": image.name in initial,
            "visual_BA_role": "variable_pose_with_fixed_intrinsics",
            "position_prior_role": "raw_supported" if image.name in prior_names else "verified_pose_gap_no_prior",
            "position_prior_sigma_m": .15 if image.name in prior_names else None,
            "support": support, "training_support_candidate": support["passed"],
            "training_export_authorized_by_this_receipt": False})
    validity = projection_validity(model)
    memory = psutil.Process().memory_info()
    peak_rss = max(memory.rss, getattr(memory, "peak_wset", memory.rss))
    if peak_rss > 32 * 1024**3 or time.monotonic() - started > 120:
        raise MemoryError("The read-only role audit exceeded its registered resource budget")
    if source != [data_identity(path) for path in paths]:
        raise RuntimeError("Source map changed during read-only role audit")
    if not validity["valid"]:
        atomic_json(args.output / "PROJECTION_FAILURE.json", validity)
        raise ValueError("The role audit found invalid observed projections")
    result = {"status": "roles_and_projection_verified_export_pending", "phase": args.phase,
        "source_model": str(args.model), "source_identity": source,
        "raw_manifest": data_identity(manifest_path), "raw_decisions": data_identity(decision_path),
        "pose_gap_decisions": data_identity(args.pose_gap_decisions),
        "policy_source": data_identity(policy_path), "camera_support_policy": policy,
        "raw_candidate_count": len(raw), "initial_count": len(initial), "registered_count": len(registered),
        "registered_initial_count": len(initial & registered), "initial_not_registered": sorted(initial - registered),
        "added_registered_names": sorted(registered - initial),
        "position_prior_count": len(priors), "pose_gap_names": gaps,
        "protected_development_exclusion_count": len(excluded),
        "support_pass_count": sum(row["training_support_candidate"] for row in rows),
        "support_fail_names": [row["image_name"] for row in rows if not row["training_support_candidate"]],
        "all_map_cameras_retained": True, "all_observations_retained": True,
        "projection": validity, "rows": rows, "counted": False,
        "source": data_identity(Path(__file__)), "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": peak_rss,
        "old_full_frame_failures_relabelled": False, "independent_accuracy_claim": False}
    atomic_json(args.output / "TRAINING_ROLES.json", result)
    print(json.dumps({key: result[key] for key in ("status", "phase", "initial_count", "registered_count",
        "position_prior_count", "support_pass_count", "support_fail_names")}))


if __name__ == "__main__":
    main()
