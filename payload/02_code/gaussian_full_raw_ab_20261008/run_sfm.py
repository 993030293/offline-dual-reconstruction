"""Build one fresh sparse map using native serial or isolated sharded matching."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback

import numpy as np
import psutil
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"),
               str(ROOT / "02_code/gaussian_pipeline_optimization_v3_20261005")]
from runtime import data_identity
from atomic_json_windows import atomic_json
from run_sfm_preflight import database_summary
from native_sfm_match_shards import partition_pairs, merge_databases
from gpu_lease import gpu_lease
from process_ownership import capture_owned_tree


def process_resources(records, tracked):
    """Treat an observed command exit as a resource-sampling race, not a job failure."""
    rss = 0
    for process in records:
        if process.poll() is not None:
            continue
        try:
            parent = psutil.Process(process.pid)
            born = parent.create_time()
            if process.pid in tracked and tracked[process.pid] != born:
                raise RuntimeError('A native command process identity changed during monitoring')
            tracked[process.pid] = born
            owned, rejected = capture_owned_tree(process.pid, born)
            if rejected:
                previous = {(row['pid'], row['creation_time']): row for row in
                    getattr(process, 'ancestry_ownership_rejections', [])}
                previous.update({(row['pid'], row['creation_time']): row for row in rejected})
                process.ancestry_ownership_rejections = list(previous.values())
        except psutil.NoSuchProcess:
            if process.poll() is None:
                raise
            continue
        for item in owned:
            try:
                born = item.create_time()
                tracked[item.pid] = born
                rss += item.memory_info().rss
            except psutil.NoSuchProcess:
                # A descendant may exit between enumeration and resource sampling.
                continue
    return rss


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    manifest = json.loads((args.initial / "INITIAL_FRAME_MANIFEST.json").read_text())
    names = [row["image_name"] for row in manifest["selected_frames"]]
    source_images = args.initial / "images"
    binary = ROOT / "05_envs/colmap-4.1.1-cuda/bin/colmap.exe"
    info = manifest["camera_intrinsics"]
    k = np.asarray(info["camera_matrix"]).reshape(3, 3)
    distortion = info["distortion"]
    if info["distortion_model"] != "plumb_bob" or len(distortion) != 5 or distortion[4] != 0:
        raise ValueError("The original OpenCV calibration contract is unsupported")
    params = [k[0, 0], k[1, 1], k[0, 2] + .5, k[1, 2] + .5, *distortion[:4]]
    (args.output / "image_list.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    pairs = (args.initial / "initial_matching_pairs.txt").read_text().splitlines()
    pair_file = args.output / "guided_pairs.txt"
    pair_file.write_text("\n".join(pairs) + "\n", encoding="utf-8")
    state = {"status": "starting", "controller_pid": os.getpid(), "stages": [],
             "initial_count": len(names), "initial_pairs": len(pairs), "mode": args.mode}
    atomic_json(args.output / "CONTRACT.json", {"initial": data_identity(args.initial / "INITIAL_FRAME_MANIFEST.json"),
        "matching_pairs": data_identity(pair_file), "camera_params": list(map(float, params)),
        "mode": args.mode, "source": data_identity(Path(__file__)), "binary": data_identity(binary),
        "old_SfM_used": False, "fresh_database": True, "feature_max_size": 1600, "feature_max_number": 8192,
        "intrinsics_fixed": True, "mapper_limit_seconds": args.mapper_seconds,
        "GPU_training_concurrent": False, "CPU_mapping_may_overlap_other_side_training": True,
        "shards_use_independent_SQLite_databases": True,
        "baseline_uses_same_candidate_graph_as_optimized": True})

    def native_jobs(label, commands, maximum, lease=None):
        folder = args.output / label
        folder.mkdir()
        records, streams, tracked = [], [], {}
        began = time.perf_counter()
        stage = {"name": label, "status": "running", "commands": [list(map(str, command)) for command in commands]}
        state["stages"].append(stage)
        try:
            for number, command in enumerate(commands):
                stream = (folder / (str(number) + ".log")).open("x", encoding="utf-8")
                streams.append(stream)
                process = subprocess.Popen(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                records.append(process)
                if lease is not None:
                    lease['command_drain_verified'] = False
                tracked[process.pid] = psutil.Process(process.pid).create_time()
            stage["owned_births"] = [{"pid": pid, "birth": birth} for pid, birth in tracked.items()]
            while any(process.poll() is None for process in records):
                elapsed = time.perf_counter() - began
                rss = process_resources(records, tracked)
                stage['ancestry_ownership_rejections'] = [row for process in records
                    for row in getattr(process, 'ancestry_ownership_rejections', [])]
                stage.update(elapsed_seconds=elapsed, peak_tree_RSS_bytes=max(stage.get("peak_tree_RSS_bytes", 0), rss))
                state["status"] = label
                atomic_json(args.output / "STATE.json", state)
                if elapsed > maximum or rss > 32 * 1024**3:
                    raise RuntimeError("The registered stage time or memory budget was exceeded")
                failed = [process.returncode for process in records if process.poll() not in (None, 0)]
                if failed:
                    raise ChildProcessError("A native stage failed: " + repr(failed))
                time.sleep(.5)
            if any(process.returncode != 0 for process in records):
                raise ChildProcessError("A native stage exited unsuccessfully")
            stage.update(status="completed", returncodes=[process.returncode for process in records],
                         elapsed_seconds=time.perf_counter() - began)
        except BaseException:
            stage.update(status="failed", elapsed_seconds=time.perf_counter() - began,
                         traceback=traceback.format_exc())
            state.update(status='failed_native_stage', failed_stage=label)
            raise
        finally:
            for pid, birth in reversed(list(tracked.items())):
                try:
                    item = psutil.Process(pid)
                    if item.create_time() == birth and item.is_running():
                        item.kill()
                        item.wait(timeout=10)
                except psutil.NoSuchProcess:
                    pass
            for process in records:
                process.wait(timeout=10)
            for stream in streams:
                stream.close()
            stage['owned_births'] = [{'pid': pid, 'birth': birth} for pid, birth in tracked.items()]
            stage['returncodes'] = [process.returncode for process in records]
            live = []
            for pid, birth in tracked.items():
                try:
                    item = psutil.Process(pid)
                    if item.create_time() == birth and item.is_running():
                        live.append({'pid': pid, 'birth': birth})
                except psutil.NoSuchProcess:
                    pass
            stage['command_drain_verified'] = not live
            stage['still_live_owned_births'] = live
            if live:
                stage.update(status='failed', failure_reason='owned_process_tree_not_drained')
                state.update(status='failed_native_stage', failed_stage=label)
            if lease is not None:
                lease['command_drain_verified'] = stage['command_drain_verified']
                lease['owned_process_identities'] = [
                    {'pid': pid, 'creation_time': birth} for pid, birth in tracked.items()]
            atomic_json(args.output / "STATE.json", state)
            if live:
                raise RuntimeError('A finite native stage did not drain its owned process tree')
        print(json.dumps(stage), flush=True)
        return stage

    def jobs(label, commands, maximum):
        if label == "guided_mapping":
            return native_jobs(label, commands, maximum)
        with gpu_lease(args.output / (label + "_GPU_LEASE.json"), label) as lease:
            result = native_jobs(label, commands, maximum, lease)
        result["GPU_queue_wait_seconds"] = lease["wait_seconds"]
        atomic_json(args.output / "STATE.json", state)
        return result

    features = args.output / "features.db"
    jobs("features", [[binary, "feature_extractor", "--database_path", features,
        "--image_path", source_images, "--image_list_path", args.output / "image_list.txt",
        "--ImageReader.single_camera", "1", "--ImageReader.camera_model", "OPENCV",
        "--ImageReader.camera_params", ",".join(format(float(value), ".17g") for value in params),
        "--FeatureExtraction.use_gpu", "1", "--FeatureExtraction.gpu_index", "0",
        "--FeatureExtraction.num_threads", "8", "--FeatureExtraction.max_image_size", "1600",
        "--SiftExtraction.max_num_features", "8192", "--default_random_seed", "0"]], 900)
    with sqlite3.connect(features) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("The fresh feature database failed integrity validation")
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    atomic_json(args.output / "FEATURES.json", database_summary(features, names))
    guided = args.output / "guided"
    guided.mkdir()
    database = guided / "database.db"
    shutil.copyfile(features, database)
    commands = []
    partition = None
    matching_began = time.perf_counter()
    if args.mode == "optimized":
        subsets, partition = partition_pairs(database, pairs)
        paths = []
        for number, subset in enumerate(subsets):
            path = database if number == 0 else guided / "shard1.db"
            if number:
                shutil.copyfile(features, path)
            subset_file = guided / ("pairs" + str(number) + ".txt")
            subset_file.write_text("\n".join(subset) + "\n", encoding="utf-8")
            paths.append((path, subset_file))
    else:
        paths = [(database, pair_file)]
    for path, match_list in paths:
        commands.append([binary, "matches_importer", "--database_path", path,
            "--match_list_path", match_list, "--match_type", "pairs", "--FeatureMatching.use_gpu", "1",
            "--FeatureMatching.gpu_index", "0", "--FeatureMatching.num_threads", "8",
            "--TwoViewGeometry.random_seed", "0", "--default_random_seed", "0"])
    jobs("guided_matching", commands, 1800)
    merge = merge_databases(database, paths[1][0]) if len(paths) > 1 else None
    matching_seconds = time.perf_counter() - matching_began
    geometry = database_summary(database, names)
    atomic_json(guided / "MATCHING.json", geometry)
    models_root = guided / "models"
    models_root.mkdir()
    mapping = jobs("guided_mapping", [[binary, "mapper", "--database_path", database,
        "--image_path", source_images, "--output_path", models_root,
        "--Mapper.num_threads", "8", "--Mapper.random_seed", "0", "--default_random_seed", "0",
        "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0", "--Mapper.ba_refine_sensor_from_rig", "0",
        "--Mapper.max_runtime_seconds", str(args.mapper_seconds - 10)]], args.mapper_seconds)
    models = []
    for path in sorted(models_root.iterdir()):
        if not path.is_dir() or not (path / "images.bin").exists():
            continue
        model = pycolmap.Reconstruction(path)
        registered = sorted(image.name for image in model.images.values() if image.has_pose)
        models.append({"path": str(path), "registered_count": len(registered), "registered": registered,
            "failed_images": sorted(set(names) - set(registered)), "remaining_failures": sorted(set(names) - set(registered)),
            "points": model.num_points3D(), "observations": model.compute_num_observations(),
            "mean_reprojection_error_px": model.compute_mean_reprojection_error()})
    if not models:
        raise RuntimeError("No readable fresh model was produced")
    models.sort(key=lambda row: (-row["registered_count"], row["path"]))
    result = {"status": "fresh_SfM_completed_registration_guard_pending", "models": models,
        "matching_seconds": matching_seconds, "mapping_seconds": mapping["elapsed_seconds"],
        "database": str(database), "image_root": str(source_images), "initial_count": len(names),
        "best_registered_count": models[0]["registered_count"], "registration_fraction": models[0]["registered_count"] / len(names),
        "shard_partition": partition, "shard_merge": merge, "elapsed_seconds": time.perf_counter() - started,
        "all_initial_registered": models[0]["registered_count"] == len(names), "GT_used": False,
        "old_SfM_reused": False, "scale": "visual_arbitrary_scale"}
    atomic_json(guided / "RESULT.json", result)
    atomic_json(args.output / "RESULT.json", result)
    state.update(status=result["status"], best_registered_count=result["best_registered_count"], elapsed_seconds=result["elapsed_seconds"])
    atomic_json(args.output / "STATE.json", state)
    print(json.dumps({key: result[key] for key in ("status", "best_registered_count", "initial_count", "elapsed_seconds")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("baseline", "optimized"), required=True)
    parser.add_argument("--mapper-seconds", type=int, default=3600)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException:
        if args.output.exists():
            atomic_json(args.output / "FAILURE.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
