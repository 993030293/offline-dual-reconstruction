"""Test a global match fallback only for failed images, retaining every denominator."""

import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time

import pycolmap

from runtime import atomic_json, data_identity
from run_sfm_preflight import database_summary


def observations(model):
    return {(int(image.image_id), index, int(point.point3D_id))
            for image in model.images.values() for index, point in enumerate(image.points2D) if point.has_point3D()}


def summarize_models(root, images, failed, best, old_observations):
    models = []
    candidates = [root] + [path for path in sorted(root.iterdir()) if path.is_dir()]
    for path in candidates:
        if not (path/"images.bin").exists():
            continue
        current = pycolmap.Reconstruction(path)
        registered = {image.name for image in current.images.values() if image.has_pose}
        new_observations = observations(current)
        models.append({"path": str(path), "registered_count": len(registered),
            "recovered_original_failures": sorted(registered & failed),
            "remaining_failures": sorted(set(images.values())-registered),
            "original_registered_frames_lost": sorted(set(best["registered"])-registered),
            "old_observations": len(old_observations), "retained_old_observations": len(old_observations & new_observations),
            "new_total_observations": len(new_observations), "points": current.num_points3D(),
            "mean_reprojection_error_px": current.compute_mean_reprojection_error()})
    if not models:
        raise RuntimeError("Mapper returned without a readable model; inspect native output layout and log")
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--input-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--analyze-existing", action="store_true")
    args = parser.parse_args()
    if not args.analyze_existing:
        args.output.mkdir(parents=True, exist_ok=False)
    prior = json.loads((args.source_run/"guided/RESULT.json").read_text())
    best = max(prior["models"], key=lambda row: row["registered_count"])
    source_model = Path(best["path"])
    model = pycolmap.Reconstruction(source_model)
    old_observations = observations(model)
    failed = set(best["failed_images"])
    source_db = args.source_run/"guided/database.db"
    with sqlite3.connect(f"file:{source_db.as_posix()}?mode=ro", uri=True) as connection:
        images = dict(connection.execute("SELECT image_id,name FROM images"))
        attempted = {row[0] for row in connection.execute("SELECT pair_id FROM matches")}
    if args.analyze_existing:
        result = json.loads((args.output/"RESULT.json").read_text())
        result["models"] = summarize_models(args.output/"models", images, failed, best, old_observations)
        result["original_registration_guard_passed"] = max(row["registered_count"] for row in result["models"]) == len(images)
        result["parser_correction"] = {"cause": "warm mapper writes the resumed model directly at output_path",
            "original_incorrect_receipt_preserved": "RESULT.json", "native_matching_or_mapping_repeated": False,
            "source": data_identity(Path(__file__))}
        target = args.output/"RESULT_REPARSED.json"
        if target.exists():
            raise FileExistsError(target)
        atomic_json(target, result)
        print(json.dumps(result), flush=True)
        return
    ids = sorted(images)
    new_pairs = [(a, b) for i, a in enumerate(ids) for b in ids[i+1:]
                 if (images[a] in failed or images[b] in failed) and 2147483647*a+b not in attempted]
    pair_file = args.output/"failure_pairs.txt"
    pair_file.write_text("\n".join(images[a]+" "+images[b] for a, b in new_pairs)+"\n", encoding="utf-8")
    project = Path(__file__).resolve().parents[2]
    binary = project/"05_envs/colmap-4.1.1-cuda/bin/colmap.exe"
    atomic_json(args.output/"CONTRACT.json", {"source_run": str(args.source_run), "original_count": len(images),
        "registered_before": best["registered_count"], "failures_before": sorted(failed),
        "selection_information": "actual guided SfM registration failures only; exhaustive result not used to choose pairs",
        "fallback": "all previously unattempted pairs involving a failed image", "new_requested_pairs": len(new_pairs),
        "all_original_frames_retained": True, "original_all_frame_registration_guard_unchanged": True,
        "matching_timeout_seconds": 600, "mapping_timeout_seconds": 600,
        "fixed_existing_camera_poses": True, "intrinsics_refinement": False,
        "partial_recovery_may_be_recorded_but_does_not_pass_original_guard": True,
        "source_database": data_identity(source_db), "source": data_identity(Path(__file__))})
    shutil.copyfile(Path(__file__), args.output/"source_snapshot.py")
    database = args.output/"database.db"
    shutil.copyfile(source_db, database)
    timings = {}

    def run(label, command):
        before = time.monotonic()
        with (args.output/(label+".log")).open("w") as log:
            result = subprocess.run(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT, timeout=600)
        timings[label] = time.monotonic()-before
        if result.returncode:
            atomic_json(args.output/"FAILURE.json", {"stage": label, "returncode": result.returncode, "timings": timings})
            raise RuntimeError("Recovery stage failed; frozen source remains intact")

    run("matching", [binary, "matches_importer", "--database_path", database, "--match_list_path", pair_file,
        "--match_type", "pairs", "--FeatureMatching.use_gpu", "1", "--FeatureMatching.gpu_index", "0",
        "--FeatureMatching.num_threads", "8", "--TwoViewGeometry.random_seed", "0", "--default_random_seed", "0"])
    geometry = database_summary(database, list(images.values()))
    atomic_json(args.output/"MATCHING.json", geometry)
    root = args.output/"models"
    root.mkdir()
    run("mapping", [binary, "mapper", "--database_path", database, "--image_path", args.input_images,
        "--input_path", source_model, "--output_path", root, "--Mapper.fix_existing_frames", "1",
        "--Mapper.num_threads", "8", "--Mapper.random_seed", "0", "--default_random_seed", "0",
        "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0", "--Mapper.ba_refine_sensor_from_rig", "0",
        "--Mapper.max_runtime_seconds", "590"])
    models = summarize_models(root, images, failed, best, old_observations)
    maximum = max([row["registered_count"] for row in models], default=0)
    receipt = {"status": "recovery_evaluated", "models": models, "timings": timings,
        "new_pairs_requested": len(new_pairs), "total_attempted_pairs": geometry["attempted_matching_pairs"],
        "original_registration_guard_passed": maximum == len(images),
        "metric_accuracy_claim": False, "counted": False,
        "boundary": "partial recovery never relabels the failed all-frame preflight as passed"}
    atomic_json(args.output/"RESULT.json", receipt)
    print(json.dumps({"status": receipt["status"], "registered": maximum,
                      "original_guard_passed": receipt["original_registration_guard_passed"]}), flush=True)


if __name__ == "__main__":
    main()
