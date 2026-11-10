"""Add quality-supported temporal views around actual remaining SfM failures."""

import argparse
import importlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

import cv2
import pycolmap

from runtime import atomic_json, data_identity
from run_pair_recovery import observations, summarize_models
from run_sfm_preflight import database_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--purpose", choices=["registration", "pose_gap", "geometry_support"], default="registration")
    parser.add_argument("--support-audit", type=Path)
    parser.add_argument("--threads", type=int, default=8, choices=[2, 8])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source_receipt = args.recovery/"RESULT_REPARSED.json"
    if not source_receipt.exists():
        source_receipt = args.recovery/"RESULT.json"
    source_result = json.loads(source_receipt.read_text())
    source_model = Path(source_result["models"][0]["path"])
    previous = pycolmap.Reconstruction(source_model)
    old_observations = observations(previous)
    registered_before = sorted(image.name for image in previous.images.values() if image.has_pose)
    failures = set(source_result["models"][0]["remaining_failures"])
    trigger_names = set(failures)
    support_policy = None
    support_audit_identity = None
    if args.purpose == "geometry_support":
        if args.support_audit is None:
            raise ValueError("Geometry support recovery requires the real source-map support audit")
        audit = json.loads((args.support_audit / "RESULT.json").read_text())
        actual_ids = [data_identity(source_model / name)["sha256"] for name in ("cameras.bin", "images.bin", "points3D.bin")]
        if actual_ids != [row["sha256"] for row in audit["source_identity"]]:
            raise ValueError("Support audit belongs to a different reconstruction")
        policy_path = Path(audit["policy_source"]["path"])
        if data_identity(policy_path)["sha256"] != audit["policy_source"]["sha256"]:
            raise ValueError("Frozen geometry support policy changed")
        support_policy = json.loads(policy_path.read_text())["camera_geometry_admission"]
        sys.path.insert(0, str(policy_path.parents[1] / "code/validated_fresh_runtime"))
        from metric_admission import camera_geometry_metrics
        measured = [camera_geometry_metrics(previous, image, support_policy)
            for image in previous.images.values() if image.has_pose]
        recorded = json.loads((args.support_audit / "CAMERA_SUPPORT.json").read_text())["before"]
        if {row["image_name"]: row for row in measured} != {row["image_name"]: row for row in recorded}:
            raise ValueError("The registered-camera support audit does not reproduce")
        weak_names = {row["image_name"] for row in measured if not row["passed"]}
        if not weak_names:
            raise ValueError("No registered camera has a measured geometry deficit")
        trigger_names |= weak_names
        support_audit_identity = data_identity(args.support_audit / "RESULT.json")
    elif args.support_audit is not None:
        raise ValueError("A support audit is only meaningful for geometry_support recovery")
    rows = [json.loads(line) for line in (args.initial/"frame_decisions.jsonl").read_text().splitlines()]
    by_name = {row["image_name"]: row for row in rows}
    with sqlite3.connect(f"file:{(args.recovery/'database.db').as_posix()}?mode=ro", uri=True) as connection:
        old_images = dict(connection.execute("SELECT image_id,name FROM images"))
        camera_ids = [row[0] for row in connection.execute("SELECT camera_id FROM cameras")]
    if len(camera_ids) != 1:
        raise ValueError("Augmentation expects the verified single fixed OPENCV camera")
    old_names = set(old_images.values())
    bins = [(-.8, -.4), (-.4, -.15), (-.15, 0), (0, .15), (.15, .4), (.4, .8)]
    additions = {}
    decisions = []
    if args.purpose == "pose_gap":
        from pose_gap_candidates import select_pose_gap_candidates
        manifest = json.loads((args.initial / "INITIAL_FRAME_MANIFEST.json").read_text())
        candidates, decisions = select_pose_gap_candidates(rows, manifest["policy"], manifest["receipt"]["sharpness_min"])
        additions = {row["image_name"]: row for row in candidates if row["image_name"] not in old_names}
        if len(additions) > 64:
            raise ValueError("Pose-gap diagnostics exceed their registered64-image budget; do not silently truncate")
        atomic_json(args.output / "POSE_GAP_DECISIONS.json", {"rows": decisions,
            "unchanged_quality_policy": manifest["policy"], "sharpness_min": manifest["receipt"]["sharpness_min"],
            "position_priors_added": 0, "extrapolation_used": False})
        shutil.copyfile(Path(__file__).with_name("pose_gap_candidates.py"), args.output / "pose_gap_candidates.py")
    else:
        candidates = [row for row in rows if row["decision"] in ("eligible_not_selected", "selected_initial_sfm_pending")
                      and row["image_name"] not in old_names]
        for name in sorted(trigger_names):
            timestamp = by_name[name]["camera_header_ns"]
            for low, high in bins:
                supported = [row for row in candidates if low*1e9 <= row["camera_header_ns"]-timestamp < high*1e9]
                best = min(supported, key=lambda row: (-row["quality"]["quality_score"], row["camera_frame"])) if supported else None
                decisions.append({"failed_image": name if name in failures else None,
                                  "trigger_image": name, "trigger_kind": "unregistered" if name in failures else "registered_geometry_deficit",
                                  "relative_time_bin_seconds": [low, high],
                                  "eligible_candidates": len(supported), "chosen": best["image_name"] if best else None})
                if best:
                    additions[best["image_name"]] = best
    if not additions:
        raise ValueError("No additional image passes the unchanged quality and temporal support gates")
    if len(additions) > 64:
        raise ValueError("The bounded augmentation exceeds64new images; refuse silent truncation")
    if any(row["excluded_role"] or (args.purpose != "pose_gap" and not row["coarse_pose_valid"]) for row in additions.values()):
        raise AssertionError("Excluded or unsupported image entered augmentation")
    contract = {"source_recovery_receipt": data_identity(source_receipt), "original_images": sorted(old_names), "original_count": len(old_names),
        "registered_before": registered_before, "remaining_failures_before": sorted(failures),
        "selection_rule": "all nonexcluded raw-quality-passing frames without coarse time support, no extrapolated priors" if args.purpose == "pose_gap" else "best existing raw-quality score per six fixed relative-time bins around each actual unregistered image",
        "purpose": args.purpose,
        "selection_trigger_images": sorted(trigger_names), "support_audit": support_audit_identity,
        "registered_camera_support_policy": support_policy,
        "CPU_threads": args.threads,
        "time_bins_seconds": bins, "decisions": decisions, "additions": list(additions.values()),
        "unchanged_initial_quality_gates": True, "old_frames_removed": [], "GT_used": False,
        "old_SfM_used": False, "source_model_role": "fresh raw-only reconstruction from this execution",
        "original_registration_guard": f"all original {len(old_names)} images remain required; additions must actually register to count as support",
        "feature_policy": "extract only new names in a copy of the verified feature/match database",
        "pair_policy": "new image with every old/new image; unchanged pairs remain cached",
        "maximum_matching_or_mapping_seconds": 600, "source": data_identity(Path(__file__))}
    atomic_json(args.output/"CONTRACT.json", contract)
    shutil.copyfile(Path(__file__), args.output/"source_snapshot.py")
    image_root = args.output/"images"
    image_root.mkdir()
    source_images = Path(source_result.get("image_root", str(args.recovery / "images")))
    if not source_images.is_dir():
        source_images = args.initial / "images"
    for name in sorted(old_names):
        shutil.copyfile(source_images/name, image_root/name)
    project = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project/"02_code/box2_initial_pose_quality_v2_20261002"))
    baseline = importlib.import_module("initial_selection")
    sys.path.insert(0, str(baseline.TOOLS))
    from mcap_audit import McapIndex
    from camera_motion import decode_bayer
    from color_contract import demosaic_rggb8_to_bgr
    cv2.setNumThreads(1)
    index = McapIndex(baseline.RAW)
    by_frame = {row["camera_frame"]: row for row in additions.values()}
    for frame, message in index.iter_selected_messages(baseline.CAMERA_TOPIC, by_frame):
        stamp, _, raw = decode_bayer(message["payload"])
        row = by_frame[frame]
        if int(stamp) != row["camera_header_ns"]:
            raise ValueError("Raw augmentation image timestamp mismatch")
        ok, encoded = cv2.imencode(".jpg", demosaic_rggb8_to_bgr(raw), [cv2.IMWRITE_JPEG_QUALITY, baseline.POLICY["jpeg_quality"]])
        if not ok:
            raise RuntimeError("Augmentation JPEG encoding failed")
        (image_root/row["image_name"]).write_bytes(encoded.tobytes())
    all_names = sorted(old_names | set(additions))
    if sorted(path.name for path in image_root.iterdir()) != all_names:
        raise AssertionError("Augmentation export is incomplete")
    atomic_json(args.output/"IMAGE_IDENTITIES.json", [data_identity(image_root/name) for name in all_names])
    new_list = args.output/"new_images.txt"
    new_list.write_text("\n".join(sorted(additions))+"\n", encoding="utf-8")
    pairs = [(a, b) for i, a in enumerate(all_names) for b in all_names[i+1:] if a in additions or b in additions]
    pair_list = args.output/"new_pairs.txt"
    pair_list.write_text("\n".join(a+" "+b for a, b in pairs)+"\n", encoding="utf-8")
    database = args.output/"database.db"
    shutil.copyfile(args.recovery/"database.db", database)
    binary = project/"05_envs/colmap-4.1.1-cuda/bin/colmap.exe"
    timings = {}

    def run(label, command):
        begin = time.monotonic()
        with (args.output/(label+".log")).open("w") as log:
            result = subprocess.run(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT, timeout=600)
        timings[label] = time.monotonic()-begin
        if result.returncode:
            atomic_json(args.output/"FAILURE.json", {"stage": label, "returncode": result.returncode, "timings": timings})
            raise RuntimeError("Augmentation stage failed")

    run("features", [binary, "feature_extractor", "--database_path", database, "--image_path", image_root,
        "--image_list_path", new_list, "--ImageReader.existing_camera_id", camera_ids[0],
        "--FeatureExtraction.use_gpu", "1", "--FeatureExtraction.gpu_index", "0", "--FeatureExtraction.num_threads", args.threads,
        "--FeatureExtraction.max_image_size", "1600", "--SiftExtraction.max_num_features", "8192", "--default_random_seed", "0"])
    run("matching", [binary, "matches_importer", "--database_path", database, "--match_list_path", pair_list,
        "--match_type", "pairs", "--FeatureMatching.use_gpu", "1", "--FeatureMatching.gpu_index", "0",
        "--FeatureMatching.num_threads", args.threads, "--TwoViewGeometry.random_seed", "0", "--default_random_seed", "0"])
    geometry = database_summary(database, all_names)
    atomic_json(args.output/"MATCHING.json", geometry)
    root = args.output/"models"
    root.mkdir()
    run("mapping", [binary, "mapper", "--database_path", database, "--image_path", image_root,
        "--input_path", source_model, "--output_path", root, "--Mapper.fix_existing_frames", "1",
        "--Mapper.num_threads", args.threads, "--Mapper.random_seed", "0", "--default_random_seed", "0",
        "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0", "--Mapper.ba_refine_sensor_from_rig", "0", "--Mapper.max_runtime_seconds", "590"])
    models = summarize_models(root, dict(enumerate(all_names)), failures, {"registered": registered_before}, old_observations)
    for model in models:
        current = pycolmap.Reconstruction(model["path"])
        names = {image.name for image in current.images.values() if image.has_pose}
        model.update(original_registered_count=len(names & old_names), additions_registered=sorted(names & set(additions)),
                     original_failures_remaining=sorted(old_names-names))
        if args.purpose == "geometry_support":
            support = [camera_geometry_metrics(current, image, support_policy)
                for image in current.images.values() if image.has_pose]
            model["registered_camera_support_passed"] = all(row["passed"] for row in support)
            model["weak_registered_images_remaining"] = [row for row in support if not row["passed"]]
            model["support_trigger_images"] = sorted(trigger_names)
    result = {"status": "augmentation_evaluated", "models": models, "added_count": len(additions), "original_count": len(old_names),
        "purpose": args.purpose, "image_root": str(image_root),
        "original_registration_guard_passed": any(model["original_registered_count"] == len(old_names) for model in models),
        "timings": timings, "elapsed_seconds": time.monotonic()-started, "counted": False,
        "metric_accuracy_claim": False, "old_input_denominator_unchanged": True}
    if args.purpose == "pose_gap":
        result["pose_gap_registration_guard_passed"] = any(
            len(model["additions_registered"]) == len(additions) and not model["original_registered_frames_lost"]
            for model in models)
        result["original_failed_route_relabelled_as_passed"] = False
        result["boundary"] = "The pose-gap guard only measures recovery of previously excluded no-odom images; original route registration failures remain separately reported"
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
