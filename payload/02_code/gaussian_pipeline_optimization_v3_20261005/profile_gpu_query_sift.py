"""Screen complete frozen queries through CPU workers or the existing GPU CLI."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import cv2
import numpy as np
import psutil
import pycolmap

from gpu_query_node_config import ROOT, CODE, OUT, V1, atomic_json, data_identity, read
from native_sift_process import extract_worker, initialize_worker, options
from query_feature_cache import option_identity

REFERENCE = ROOT / "04_outputs/box2_lio_guided_sfm_v1_20261002T014330"
INITIAL = ROOT / "04_outputs/gaussian_pipeline_optimization_v2_20261005/final_raw_cold_001/initial"
EXPORT = ROOT / "04_outputs/gaussian_pipeline_optimization_v1_20261005/new_map_export_003"
COLMAP = ROOT / "05_envs/colmap-4.1.1-cuda/bin/colmap.exe"


def queries():
    split = read(REFERENCE / "00_contract/split_manifest.json")
    raw = {r["image_name"]: r for r in map(json.loads,
        (INITIAL / "frame_decisions.jsonl").read_text(encoding="utf-8-sig").splitlines())}
    paths = [REFERENCE / "00_contract/extra_images" / r["image_name"]
        for r in split["extra_frames"] if r["role"] == "H" and raw[r["image_name"]]["coarse_pose_valid"]]
    if len(paths) != 24 or len({p.stem for p in paths}) != 24:
        raise ValueError("The complete frozen query order and names must be preserved")
    return paths


def alive(identity):
    try:
        return psutil.Process(identity["pid"]).create_time() == identity["creation_time"]
    except psutil.NoSuchProcess:
        return False


def validate(keypoints, descriptors):
    if (keypoints.ndim != 2 or keypoints.shape[1] not in (4, 6) or len(keypoints) < 30
            or descriptors.shape != (len(keypoints), 128) or keypoints.dtype != np.float32
            or descriptors.dtype != np.float32 or not np.isfinite(keypoints).all()
            or not np.isfinite(descriptors).all() or (descriptors < 0).any()
            or (descriptors > 255 / 512).any()
            or not np.array_equal(descriptors * 512, np.rint(descriptors * 512))):
        raise ValueError("A complete native query feature format is unsupported")


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cpu_process", "gpu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(OUT.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    result = {"status": "starting", "mode": args.mode, "counted": False,
        "scope": "all24_feature_screen_not_complete_localization_or_quality", "rows": [], "owners": []}
    try:
        paths = queries()
        source_paths = [Path(__file__), CODE / "gpu_query_node.py", CODE / "gpu_query_node_config.py",
            CODE / "native_sift_process.py", CODE / "run_bounded_journal_task.py", COLMAP,
            ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005/query_feature_cache.py",
            ROOT / "02_code/box2_lio_guided_sfm_v1_20261002/feature_localization.py"]
        model_path = Path(read(EXPORT / "TRAINING_ADMISSION.json")["training_opencv_model"])
        input_paths = [*paths, REFERENCE / "00_contract/split_manifest.json",
            INITIAL / "frame_decisions.jsonl", EXPORT / "TRAINING_ADMISSION.json", *sorted(model_path.glob("*.bin"))]
        phase_started = time.perf_counter()
        sources, inputs = [data_identity(p) for p in source_paths], [data_identity(p) for p in input_paths]
        selected = options()
        result.update(sources=sources, inputs=inputs, original_CPU_options=option_identity(selected.todict()),
            original_query_paths=list(map(str, paths)))
        arrays, owners, traces = {}, [], []
        result["owners"] = owners
        if args.mode == "cpu_process":
            with ProcessPoolExecutor(max_workers=2, initializer=initialize_worker,
                    mp_context=multiprocessing.get_context("spawn")) as pool:
                for path, value in zip(paths, pool.map(extract_worker, map(str, paths), chunksize=1)):
                    arrays[path.name] = (value.pop("keypoints"), value.pop("descriptors"))
                    traces.append({"path": str(path), **value})
                    if value["worker"] not in owners:
                        owners.append(value["worker"])
            result["CPU_workers_actually_gone"] = all(not alive(owner) for owner in owners)
            result["CPU_worker_trace"] = traces
        else:
            model = pycolmap.Reconstruction(model_path)
            if len(model.cameras) != 1:
                raise ValueError("The frozen diagnostic requires the original single camera")
            camera = next(iter(model.cameras.values()))
            gray_root = args.output / "gray_images"
            gray_root.mkdir()
            gray_rows = []
            for path in paths:
                gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if gray is None or gray.shape != (camera.height, camera.width):
                    raise ValueError("Original full camera and grayscale dimensions differ")
                target = gray_root / (path.stem + ".png")
                if not cv2.imwrite(str(target), gray):
                    raise IOError("An owned lossless gray image could not be written")
                back = cv2.imread(str(target), cv2.IMREAD_GRAYSCALE)
                if not np.array_equal(gray, back):
                    raise AssertionError("The complete original grayscale pixels changed")
                gray_rows.append({"original": path.name, "gpu_name": target.name, "shape": list(gray.shape),
                    "gray_sha256": hashlib.sha256(gray.tobytes()).hexdigest()})
            database = args.output / "gpu_features.db"
            command = [str(COLMAP), "feature_extractor", "--database_path", str(database),
                "--image_path", str(gray_root), "--ImageReader.single_camera", "1",
                "--ImageReader.camera_model", camera.model_name,
                "--ImageReader.camera_params", ",".join(format(float(v), ".17g") for v in camera.params),
                "--descriptor_normalization", "l1_root", "--FeatureExtraction.type", "SIFT",
                "--FeatureExtraction.use_gpu", "1", "--FeatureExtraction.gpu_index", "0",
                "--FeatureExtraction.max_image_size", "-1", "--FeatureExtraction.num_threads", "8",
                "--default_random_seed", "0"]
            for name in ("max_num_features", "first_octave", "num_octaves", "octave_resolution", "peak_threshold",
                    "edge_threshold", "estimate_affine_shape", "max_num_orientations", "upright", "domain_size_pooling",
                    "dsp_min_scale", "dsp_max_scale", "dsp_num_scales"):
                value = getattr(selected.sift, name)
                command.extend(["--SiftExtraction." + name, str(int(value)) if isinstance(value, bool) else str(value)])
            result.update(command=command, gray_rows=gray_rows)
            with (args.output / "colmap_extract.log").open("xb") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                owner = {"pid": child.pid, "creation_time": psutil.Process(child.pid).create_time()}
                owners.append(owner)
                try:
                    returncode = child.wait(timeout=300)
                finally:
                    if child.poll() is None:
                        child.terminate()
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait(timeout=10)
            result.update(CLI_returncode=returncode, CLI_actually_gone=not alive(owner))
            body = (args.output / "colmap_extract.log").read_text(encoding="utf-8", errors="replace")
            result["actual_GPU_extractor_log"] = "Creating SIFT GPU feature extractor" in body
            if returncode or not result["actual_GPU_extractor_log"] or "Creating SIFT CPU feature extractor" in body:
                raise RuntimeError("The existing CLI did not complete the requested actual GPU extractor")
            with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
                rows = connection.execute("SELECT images.name,keypoints.rows,keypoints.cols,keypoints.data,"
                    "descriptors.rows,descriptors.cols,descriptors.data FROM images JOIN keypoints USING(image_id) "
                    "JOIN descriptors USING(image_id)").fetchall()
                cameras = connection.execute("SELECT model,width,height,params FROM cameras").fetchall()
            names = {row["gpu_name"]: row["original"] for row in gray_rows}
            if len(rows) != 24 or set(names) != {r[0] for r in rows} or len(cameras) != 1:
                raise AssertionError("Every original query and exactly one camera must be read back")
            cm, width, height, params = cameras[0]
            if (cm != camera.model.value or width != camera.width or height != camera.height
                    or not np.array_equal(np.frombuffer(params, np.float64), camera.params)):
                raise AssertionError("The original camera metadata was not preserved in the private database")
            for name, kr, kc, kb, dr, dc, db in rows:
                if kr != dr or dc != 128:
                    raise ValueError("A complete original descriptor row is missing")
                arrays[names[name]] = (np.frombuffer(kb, np.float32).reshape(kr, kc).copy(),
                    np.frombuffer(db, np.uint8).reshape(dr, dc).astype(np.float32) / 512.)
        result["owners"] = owners
        features = args.output / "features"
        features.mkdir()
        for path in paths:
            keypoints, descriptors = arrays[path.name]
            validate(keypoints, descriptors)
            target = features / (path.stem + ".npz")
            np.savez_compressed(target, keypoints=keypoints, descriptors=descriptors)
            with np.load(target, allow_pickle=False) as stored:
                if not np.array_equal(keypoints, stored["keypoints"]) or not np.array_equal(descriptors, stored["descriptors"]):
                    raise AssertionError("A complete native feature cache round trip changed")
            result["rows"].append({"name": path.name, "keypoints": len(keypoints),
                "keypoint_columns": keypoints.shape[1], "keypoint_dtype": str(keypoints.dtype),
                "descriptor_dtype": str(descriptors.dtype), "feature_artifact": str(target),
                "all_format_and_cache_bits_valid": True})
        checks = {"all24_queries_full_format_and_roundtrip": len(result["rows"]) == 24,
            "all_sources_unchanged": sources == [data_identity(p) for p in source_paths],
            "all_inputs_unchanged": inputs == [data_identity(p) for p in input_paths],
            "all_actual_worker_owners_gone": bool(owners) and all(not alive(owner) for owner in owners),
            "requested_GPU_or_original_CPU_route": result.get("actual_GPU_extractor_log", False)
                if args.mode == "gpu" else result.get("CPU_workers_actually_gone", False)}
        result.update(checks=checks, complete_phase_seconds=time.perf_counter() - phase_started,
            whole_seconds=time.perf_counter() - started,
            status="screen_passed_complete_localization_quality_pending" if all(checks.values()) else "required_screen_guard_failed")
        if not all(checks.values()):
            raise AssertionError("A frozen all-query screen guard failed")
        print(json.dumps({"mode": args.mode, "status": result["status"], "complete_phase_seconds": result["complete_phase_seconds"]}), flush=True)
    except BaseException as error:
        result.update(status="failed", error=repr(error), whole_seconds=time.perf_counter() - started)
        raise
    finally:
        atomic_json(args.output / "RESULT.json", result)


if __name__ == "__main__":
    main()
