"""Retry real failed-camera pairs with geometric-guided matching in a copied DB."""

import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time

import pycolmap

from runtime import atomic_json, data_identity
from run_pair_recovery import observations, summarize_models
from run_sfm_preflight import database_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategy", choices=["guided", "feature_escalation"], default="guided")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source = json.loads((args.source_run / "RESULT.json").read_text())
    best = max(source["models"], key=lambda row: row["registered_count"])
    model_path = Path(best["path"])
    model = pycolmap.Reconstruction(model_path)
    registered = sorted(image.name for image in model.images.values() if image.has_pose)
    failures = set(best["remaining_failures"])
    image_root = Path(source.get("image_root", str(args.source_run / "images")))
    enrich = args.strategy == "feature_escalation"
    if not failures:
        raise ValueError("No actual failed image triggers this diagnostic")
    raw = {row["image_name"]: row for line in (args.initial / "frame_decisions.jsonl").read_text().splitlines()
           for row in [json.loads(line)]}
    source_db = args.source_run / "database.db"
    with sqlite3.connect(f"file:{source_db.as_posix()}?mode=ro", uri=True) as db:
        images = dict(db.execute("SELECT image_id,name FROM images"))
        camera_ids = [row[0] for row in db.execute("SELECT camera_id FROM cameras")]
        nonempty = {pid for pid, in db.execute("SELECT pair_id FROM two_view_geometries WHERE rows>0")}
    if len(camera_ids) != 1:
        raise ValueError("Expected the frozen single OPENCV camera")
    failed_ids = {iid for iid, name in images.items() if name in failures}
    if any(iid in model.images and model.images[iid].has_pose for iid in failed_ids):
        raise AssertionError("Never replace features belonging to a registered camera")
    ids = sorted(images)
    pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i+1:]
             if (images[a] in failures or images[b] in failures) and
             (enrich or 2147483647*a+b in nonempty or abs(raw[images[a]]["camera_header_ns"]-raw[images[b]]["camera_header_ns"]) <= 2e9)]
    contract = {"source_database": data_identity(source_db), "source_receipt": data_identity(args.source_run / "RESULT.json"),
        "source": data_identity(Path(__file__)), "input_count": len(images), "failed_before": sorted(failures),
        "candidate": args.strategy,
        "pair_policy": "all pairs involving replaced features" if enrich else "existing nonempty verified pair or absolute timestamp difference <=2 seconds; at least one unregistered endpoint",
        "requested_pairs": len(pairs), "features_unchanged": not enrich,
        "registered_image_features_unchanged": True,
        "failed_feature_policy": {"max_image_size": 1600, "max_num_features": 8192, "peak_threshold": .003} if enrich else None,
        "ratio_distance_RANSAC_thresholds_unchanged": True,
        "original_mapper_inlier_and_triangulation_thresholds_unchanged": True,
        "existing_cameras_fixed": True, "calibration_fixed": True, "all_input_frames_required": True,
        "no_GT_or_legacy_SfM": True, "matching_and_mapping_timeout_each_seconds": 600,
        "acceptance": f"all original{len(images)} cameras in one map; no old cameras lost; common observation P95 increase<=0.05px; existing poses fixed"}
    atomic_json(args.output / "CONTRACT.json", contract)
    shutil.copyfile(Path(__file__), args.output / "source_snapshot.py")
    pair_file = args.output / "retry_pairs.txt"
    pair_file.write_text("\n".join(images[a]+" "+images[b] for a, b in pairs)+"\n", encoding="utf-8")
    database = args.output / "database.db"
    shutil.copyfile(source_db, database)
    pair_ids = [(2147483647*a+b,) for a, b in pairs]
    with sqlite3.connect(database) as db:
        db.executemany("DELETE FROM matches WHERE pair_id=?", pair_ids)
        db.executemany("DELETE FROM two_view_geometries WHERE pair_id=?", pair_ids)
        if enrich:
            db.executemany("DELETE FROM keypoints WHERE image_id=?", [(iid,) for iid in failed_ids])
            db.executemany("DELETE FROM descriptors WHERE image_id=?", [(iid,) for iid in failed_ids])
    binary = Path(__file__).resolve().parents[2] / "05_envs/colmap-4.1.1-cuda/bin/colmap.exe"
    timings = {}

    def run(label, command):
        begin = time.monotonic()
        with (args.output / (label+".log")).open("w") as log:
            result = subprocess.run(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT, timeout=600)
        timings[label] = time.monotonic()-begin
        if result.returncode:
            atomic_json(args.output / "FAILURE.json", {"stage": label, "returncode": result.returncode, "timings": timings})
            raise RuntimeError("Guided recovery failed; source DB and model are unchanged")

    if enrich:
        names_path = args.output / "reextract_images.txt"
        names_path.write_text("\n".join(sorted(failures))+"\n", encoding="utf-8")
        run("features", [binary, "feature_extractor", "--database_path", database, "--image_path", image_root,
            "--image_list_path", names_path, "--ImageReader.existing_camera_id", camera_ids[0],
            "--FeatureExtraction.use_gpu", "1", "--FeatureExtraction.gpu_index", "0", "--FeatureExtraction.num_threads", "8",
            "--FeatureExtraction.max_image_size", "1600", "--SiftExtraction.max_num_features", "8192",
            "--SiftExtraction.peak_threshold", ".003", "--default_random_seed", "0"])
    run("matching", [binary, "matches_importer", "--database_path", database, "--match_list_path", pair_file,
        "--match_type", "pairs", "--FeatureMatching.guided_matching", "0" if enrich else "1", "--FeatureMatching.use_gpu", "1",
        "--FeatureMatching.gpu_index", "0", "--FeatureMatching.num_threads", "8",
        "--TwoViewGeometry.random_seed", "0", "--default_random_seed", "0"])
    with sqlite3.connect(database) as db:
        db.execute("ATTACH DATABASE ? AS previous", (str(source_db),))
        for table in ("images", "cameras", "keypoints", "descriptors"):
            predicate = ""
            if enrich and table in ("keypoints", "descriptors"):
                predicate = " WHERE image_id NOT IN ("+",".join(str(iid) for iid in sorted(failed_ids))+")"
            if db.execute(f"SELECT COUNT(*) FROM (SELECT * FROM {table}{predicate} EXCEPT SELECT * FROM previous.{table}{predicate})").fetchone()[0]:
                raise AssertionError("Non-pair content unexpectedly changed")
            if db.execute(f"SELECT COUNT(*) FROM (SELECT * FROM previous.{table}{predicate} EXCEPT SELECT * FROM {table}{predicate})").fetchone()[0]:
                raise AssertionError("Original non-pair content lost")
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise AssertionError("Copied match database integrity failure")
    atomic_json(args.output / "MATCHING.json", database_summary(database, list(images.values())))
    root = args.output / "models"
    root.mkdir()
    run("mapping", [binary, "mapper", "--database_path", database, "--image_path", image_root,
        "--input_path", model_path, "--output_path", root, "--Mapper.fix_existing_frames", "1",
        "--Mapper.num_threads", "8", "--Mapper.random_seed", "0", "--default_random_seed", "0",
        "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0", "--Mapper.ba_refine_sensor_from_rig", "0", "--Mapper.max_runtime_seconds", "590"])
    models = summarize_models(root, images, failures, {"registered": registered}, observations(model))
    passed = any(row["registered_count"] == len(images) and not row["original_registered_frames_lost"] for row in models)
    receipt = {"status": "guided_recovery_evaluated", "models": models, "timings": timings,
        "original_registration_guard_passed": passed, "all_input_denominator": len(images),
        "image_root": str(image_root), "strategy": args.strategy, "elapsed_seconds": time.monotonic()-started,
        "counted": False, "metric_accuracy_claim": False, "source_unchanged": data_identity(source_db) == contract["source_database"]}
    atomic_json(args.output / "RESULT.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
