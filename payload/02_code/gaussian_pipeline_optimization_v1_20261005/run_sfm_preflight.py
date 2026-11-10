"""Run isolated, shared-feature exhaustive and coarse-guided fresh SfM controls."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time
import traceback

import cv2
import numpy as np
import psutil
import pycolmap

from runtime import atomic_json, data_identity


def database_summary(path, names):
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as connection:
        images = dict(connection.execute("SELECT image_id,name FROM images").fetchall())
        keys = connection.execute("SELECT image_id,rows FROM keypoints").fetchall()
        matches = connection.execute("SELECT pair_id,rows FROM matches").fetchall()
        geometries = connection.execute("SELECT pair_id,rows,config FROM two_view_geometries").fetchall()
    if sorted(images.values()) != sorted(names) or len(keys) != len(names):
        raise ValueError("Feature database does not cover exactly the registered input images")
    adjacency = {image_id: set() for image_id in images}
    verified = []
    for pair_id, count, config in geometries:
        if count < 15:
            continue
        a, b = divmod(pair_id, 2147483647)
        adjacency[a].add(b)
        adjacency[b].add(a)
        verified.append({"image_a": images[a], "image_b": images[b], "inliers": count, "configuration": config})
    remaining = set(images)
    components = []
    while remaining:
        reached = {min(remaining)}
        queue = list(reached)
        for item in queue:
            for neighbor in adjacency[item]-reached:
                reached.add(neighbor)
                queue.append(neighbor)
        remaining -= reached
        components.append(sorted(images[item] for item in reached))
    return {"image_count": len(images), "keypoint_counts": {images[i]: count for i, count in keys},
            "attempted_matching_pairs": len(matches), "stored_two_view_rows": len(geometries),
            "geometric_pairs_at_least_15_inliers": len(verified), "verified_pairs": verified,
            "geometric_configuration_counts": dict(Counter(row["configuration"] for row in verified)),
            "component_sizes": sorted(map(len, components), reverse=True), "components": components}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=64, choices=[64, 256, 1002])
    parser.add_argument("--preflight-source", type=Path)
    parser.add_argument("--augmentation-source", type=Path)
    parser.add_argument("--mapper-seconds", type=int, default=600, choices=[600, 1200])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    prerequisite = None
    augmentation_prerequisite = None
    if args.count == 256:
        if args.preflight_source is None:
            raise ValueError("A passed 64-frame interface receipt is required before expansion")
        prerequisite = json.loads((args.preflight_source/"RESULT.json").read_text(encoding="utf-8"))
        if not prerequisite["registration_guard_passed"] or prerequisite["best_registered_counts"] != {"exhaustive": 64, "guided": 64}:
            raise ValueError("The 64-frame registration guard did not pass")
    if args.count == 1002:
        if args.augmentation_source is None:
            raise ValueError("Full-route expansion requires the actual 256-frame recovery and observation audit")
        augmentation_prerequisite = json.loads((args.augmentation_source/"RESULT.json").read_text())
        audit = json.loads((args.augmentation_source/"observation_audit/RESULT.json").read_text())
        if not augmentation_prerequisite["original_registration_guard_passed"] or not audit["guard_passed"]:
            raise ValueError("Adaptive recovery has not passed registration and observation protection")
    project = Path(__file__).resolve().parents[2]
    binary = project/"05_envs/colmap-4.1.1-cuda/bin/colmap.exe"
    source = args.input/"INITIAL_FRAME_MANIFEST.json"
    manifest = json.loads(source.read_text(encoding="utf-8"))
    rows = manifest["selected_frames"]
    selected = None
    if args.count == 64:
        for begin in range(len(rows)-args.count+1):
            candidate = rows[begin:begin+args.count]
            if max(np.diff([row["camera_header_ns"] for row in candidate])) <= 800_000_000:
                selected = candidate
                break
        selection_rule = "earliest 64 consecutive initial frames with all adjacent header gaps at most 0.8 seconds"
    elif args.count == 256:
        selected = [rows[i] for i in np.linspace(0, len(rows)-1, args.count, dtype=int)]
        selection_rule = "uniform ordered indices across the complete 1002-frame initial set, including temporal gaps and loop revisits"
    else:
        if len(rows) != args.count:
            raise ValueError("Full-route image denominator differs from the frozen 1002-frame initial set")
        selected = rows
        selection_rule = "all 1002 raw-only selected images, no reduction or old SfM initialization"
    if selected is None:
        raise ValueError("No 64-frame continuous window satisfies the registered temporal support rule")
    names = [row["image_name"] for row in selected]
    name_set = set(names)
    guided_pairs = []
    for line in (args.input/"initial_matching_pairs.txt").read_text().splitlines():
        a, b = line.split()
        if a in name_set and b in name_set:
            guided_pairs.append((a, b))
    pairing_rule = "induced subgraph of frozen initial temporal/position candidates"
    if args.count == 256:
        edges = set()
        positions = np.asarray([row["C_W_m"] for row in selected])
        for i in range(len(selected)):
            for j in range(i+1, min(len(selected), i+9)):
                edges.add((i, j))
            for j in np.argsort(np.linalg.norm(positions-positions[i], axis=1))[:17]:
                if abs(int(j)-i) > 8:
                    edges.add(tuple(sorted((i, int(j)))))
        guided_pairs = [(names[i], names[j]) for i, j in sorted(edges)]
        pairing_rule = "same 8 temporal / 16 coarse spatial neighbor rule regenerated on the 256-frame diagnostic subset"
    camera_info = manifest["camera_intrinsics"]
    matrix = np.asarray(camera_info["camera_matrix"]).reshape(3, 3)
    distortion = np.asarray(camera_info["distortion"])
    if camera_info["distortion_model"] != "plumb_bob" or len(distortion) != 5 or distortion[4] != 0:
        raise ValueError("This preflight supports the existing k3=0 OpenCV calibration only")
    params = [matrix[0, 0], matrix[1, 1], matrix[0, 2]+.5, matrix[1, 2]+.5, *distortion[:4]]
    camera = pycolmap.Camera(model="OPENCV", width=camera_info["width"], height=camera_info["height"], params=params)
    points = np.array([[x, y, z] for x in [-.8, 0, .8] for y in [-.6, 0, .6] for z in [2., 3.]])
    pixel, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), matrix, distortion)
    camera_error = float(np.linalg.norm(camera.img_from_cam(points)-(pixel.reshape(-1, 2)+.5), axis=1).max())
    if camera_error > 1e-6:
        raise AssertionError("OpenCV integer-center to COLMAP corner-coordinate conversion failed")
    (args.output/"image_list.txt").write_text("\n".join(names)+"\n", encoding="utf-8")
    pair_file = args.output/"guided_pairs.txt"
    pair_file.write_text("\n".join(a+" "+b for a, b in guided_pairs)+"\n", encoding="utf-8")
    contract = {"status": "registered_before_feature_extraction", "input": data_identity(source),
        "selection": selection_rule, "guided_pair_rule": pairing_rule,
        "prerequisite_64_run": str(args.preflight_source) if prerequisite else None,
        "prerequisite_adaptive_256_run": str(args.augmentation_source) if augmentation_prerequisite else None,
        "input_images": [data_identity(args.input/"images"/name) for name in names],
        "frames": [row["camera_frame"] for row in selected],
        "camera": {"model": "OPENCV", "width": camera_info["width"], "height": camera_info["height"], "params": params},
        "pixel_convention": "raw OpenCV integer-center to COLMAP corner-center by adding 0.5 to principal point",
        "camera_projection_oracle_error_px": camera_error, "intrinsics_refinement": False,
        "shared_features": {"SIFT_max_features": 8192, "maximum_image_size": 1600, "GPU": 0, "threads": 8},
        "exhaustive_requested_pairs": args.count*(args.count-1)//2 if args.count != 1002 else None,
        "exhaustive_run": args.count != 1002, "guided_requested_pairs": len(guided_pairs),
        "binary": data_identity(binary), "pycolmap": pycolmap.__version__, "pycolmap_has_cuda": pycolmap.has_cuda,
        "feature_and_match_execution": "existing CUDA CLI; CPU-only pycolmap reads results",
        "feature_seconds_limit": 300, "matching_seconds_limit_per_case": 600, "mapper_seconds_limit_per_case": args.mapper_seconds,
        "RAM_bytes_limit_per_process": 32*1024**3, "DB_writer_policy": "sequential isolated copies of closed shared feature database",
        "registration_guard": f"guided registers all {args.count}; exhaustive comparison required only for the 64/256 diagnostics",
        "speed_acceptance": "this single trial is interface/cost preflight; three trials and larger subset required",
        "poses_initialized_from_old_SfM": False, "metric_alignment_performed": False, "old_Sim3_used": False,
        "GT_used": False, "strict_blind_test": False, "source": data_identity(Path(__file__))}
    atomic_json(args.output/"CONTRACT.json", contract)
    shutil.copyfile(Path(__file__), args.output/"run_sfm_preflight_snapshot.py")
    started = time.monotonic()
    state = {"status": "running", "controller_pid": os.getpid(), "stages": []}
    atomic_json(args.output/"STATE.json", state)

    def run_stage(name, command, timeout):
        stage = {"name": name, "command": list(map(str, command)), "status": "running", "timeout_seconds": timeout}
        state["stages"].append(stage)
        stage_started = time.monotonic()
        log_path = args.output/(name+".log")
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT)
            stage["pid"] = process.pid
            stage["log"] = str(log_path)
            last_report = 0.
            while process.poll() is None:
                elapsed = time.monotonic()-stage_started
                rss = psutil.Process(process.pid).memory_info().rss
                stage["peak_rss_bytes"] = max(stage.get("peak_rss_bytes", 0), rss)
                if elapsed > timeout or rss > 32*1024**3:
                    process.kill()
                    process.wait()
                    stage.update(status="budget_exceeded", elapsed_seconds=elapsed)
                    atomic_json(args.output/"STATE.json", state)
                    raise RuntimeError(f"Stage {name} exceeded its registered budget")
                if elapsed-last_report >= 5:
                    stage["elapsed_seconds"] = elapsed
                    atomic_json(args.output/"STATE.json", state)
                    last_report = elapsed
                time.sleep(.5)
        stage.update(status="completed" if process.returncode == 0 else "failed", returncode=process.returncode,
                     elapsed_seconds=time.monotonic()-stage_started)
        atomic_json(args.output/"STATE.json", state)
        print(json.dumps({"stage": name, "status": stage["status"], "seconds": stage["elapsed_seconds"]}), flush=True)
        if process.returncode != 0:
            raise RuntimeError(f"Stage {name} failed; inspect its preserved log")
        return stage

    try:
        features = args.output/"features.db"
        run_stage("features", [binary, "feature_extractor", "--database_path", features,
            "--image_path", args.input/"images", "--image_list_path", args.output/"image_list.txt",
            "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "OPENCV",
            "--ImageReader.camera_params", ",".join(format(value, ".17g") for value in params),
            "--FeatureExtraction.use_gpu", "1", "--FeatureExtraction.gpu_index", "0",
            "--FeatureExtraction.num_threads", "8", "--FeatureExtraction.max_image_size", "1600",
            "--SiftExtraction.max_num_features", "8192", "--default_random_seed", "0"], 300)
        with sqlite3.connect(features) as database:
            if database.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Shared feature database integrity failed")
            database.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        atomic_json(args.output/"FEATURES.json", database_summary(features, names))
        results = {}
        for label in (["guided"] if args.count == 1002 else ["exhaustive", "guided"]):
            folder = args.output/label
            folder.mkdir()
            database = folder/"database.db"
            shutil.copyfile(features, database)
            command = [binary, "exhaustive_matcher" if label == "exhaustive" else "matches_importer",
                "--database_path", database, "--FeatureMatching.use_gpu", "1", "--FeatureMatching.gpu_index", "0",
                "--FeatureMatching.num_threads", "8", "--TwoViewGeometry.random_seed", "0", "--default_random_seed", "0"]
            if label == "guided":
                command += ["--match_list_path", pair_file, "--match_type", "pairs"]
            matching = run_stage(label+"_matching", command, 600)
            geometry = database_summary(database, names)
            atomic_json(folder/"MATCHING.json", geometry)
            model_root = folder/"models"
            model_root.mkdir()
            mapping = run_stage(label+"_mapping", [binary, "mapper", "--database_path", database,
                "--image_path", args.input/"images", "--output_path", model_root,
                "--Mapper.num_threads", "8", "--Mapper.random_seed", "0", "--default_random_seed", "0",
                "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_principal_point", "0",
                "--Mapper.ba_refine_extra_params", "0", "--Mapper.ba_refine_sensor_from_rig", "0",
                "--Mapper.max_runtime_seconds", str(args.mapper_seconds-10)], args.mapper_seconds)
            models = []
            for path in sorted(model_root.iterdir()):
                if not path.is_dir() or not (path/"images.bin").exists():
                    continue
                model = pycolmap.Reconstruction(path)
                registered = sorted(image.name for image in model.images.values() if image.has_pose)
                models.append({"path": str(path), "registered_count": len(registered), "registered": registered,
                    "failed_images": sorted(name_set-set(registered)), "points": model.num_points3D(),
                    "observations": model.compute_num_observations(),
                    "mean_reprojection_error_px": model.compute_mean_reprojection_error(),
                    "coordinate_scale": "visual_arbitrary_scale_metric_prior_not_yet_applied"})
            results[label] = {"matching_seconds": matching["elapsed_seconds"], "mapping_seconds": mapping["elapsed_seconds"],
                              "database": str(database), "matching": geometry, "models": models}
            atomic_json(folder/"RESULT.json", results[label])
        counts = {label: max([model["registered_count"] for model in value["models"]], default=0)
                  for label, value in results.items()}
        result = {"status": "preflight_completed", "results": results, "best_registered_counts": counts,
            "registration_guard_passed": counts["guided"] == args.count and counts["guided"] >= counts.get("exhaustive", 0),
            "full_route": args.count == 1002, "exhaustive_comparison_available": "exhaustive" in counts,
            "metric_precision_verified": False, "speedup_accepted": False, "counted": False,
            "elapsed_seconds": time.monotonic()-started}
        atomic_json(args.output/"RESULT.json", result)
        state.update(status="preflight_completed", best_registered_counts=counts, elapsed_seconds=result["elapsed_seconds"])
        atomic_json(args.output/"STATE.json", state)
        print(json.dumps({"status": result["status"], "registered": counts, "registration_guard": result["registration_guard_passed"]}), flush=True)
    except BaseException as error:
        state.update(status="failed", error=repr(error), traceback=traceback.format_exc(), elapsed_seconds=time.monotonic()-started)
        atomic_json(args.output/"STATE.json", state)
        raise


if __name__ == "__main__":
    main()
