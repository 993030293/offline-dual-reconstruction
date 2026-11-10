"""Recover failed image components with bounded coarse-pose pair expansion."""

import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity
from run_pair_recovery import observations, summarize_models
from run_sfm_preflight import database_summary

PAIR_BASE = 2147483647


def pairs_for_failures(images, failed, attempted, raw, temporal, spatial):
    """Use coarse poses to order candidates while leaving a recorded full fallback."""
    ids = np.asarray(sorted(images), dtype=np.int64)
    rows = [raw[images[int(i)]] for i in ids]
    if any(not r["coarse_pose_valid"] or r["excluded_role"] for r in rows):
        raise ValueError("Pair retrieval requires supported training-frame coarse poses")
    stamps = np.asarray([r["camera_header_ns"] for r in rows], dtype=np.int64)
    centers = np.asarray([r["C_W_m"] for r in rows], dtype=np.float64)
    forwards = np.asarray([np.asarray(r["R_WC"])[:, 2] for r in rows], dtype=np.float64)
    if not np.isfinite(centers).all() or not np.isfinite(forwards).all():
        raise ValueError("Coarse pair metadata is nonfinite")
    selected = set()
    for slot, image_id in enumerate(ids):
        if images[int(image_id)] not in failed:
            continue
        time_order = np.argsort(np.abs(stamps-stamps[slot]), kind="stable")
        scores = np.linalg.norm(centers-centers[slot], axis=1)+.5*(1-np.clip(forwards@forwards[slot], -1, 1))
        pose_order = np.argsort(scores, kind="stable")
        others = set(time_order[:temporal+1]) | set(pose_order[:spatial+1])
        for other in others:
            a, b = sorted((int(image_id), int(ids[other])))
            if a != b and PAIR_BASE*a+b not in attempted:
                selected.add((a, b))
    return sorted(selected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "input-images", "initial-frames", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--mode", choices=["baseline", "progressive"], required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    prior = json.loads((args.source_run/"guided/RESULT.json").read_text())
    best = max(prior["models"], key=lambda r: r["registered_count"])
    source_model = Path(best["path"])
    model = pycolmap.Reconstruction(source_model)
    original_observations = observations(model)
    original_poses = {im.name: im.cam_from_world().matrix().copy() for im in model.images.values() if im.has_pose}
    source_db = args.source_run/"guided/database.db"
    with sqlite3.connect(source_db.resolve().as_uri()+"?mode=ro", uri=True) as connection:
        images = dict(connection.execute("SELECT image_id,name FROM images"))
        attempted = {p for p, in connection.execute("SELECT pair_id FROM matches")}
    raw_path = args.initial_frames/"frame_decisions.jsonl"
    raw = {r["image_name"]: r for r in map(json.loads, raw_path.read_text(encoding="utf-8-sig").splitlines())}
    failed = set(best["failed_images"])
    sources = [data_identity(p) for p in [source_db, raw_path, Path(__file__), V1/"run_pair_recovery.py"]]
    contract = {"mode": args.mode, "sources": sources, "original_denominator": len(images),
        "initial_registered": model.num_reg_images(), "initial_failures": sorted(failed),
        "pair_schedule": [[128, 96], [256, 192]], "full_fallback_recorded_not_executed_automatically": True,
        "selection_uses": "current registration failures, raw coarse pose and timestamps only",
        "GT_or_exhaustive_pairs_used_to_rank": False, "minimum_registered_for_diagnostic_comparison": 1000,
        "original_all1002_registration_guard_unchanged": True, "maximum_matching_seconds_per_batch": 600,
        "maximum_mapping_seconds_per_batch": 600, "maximum_total_seconds": 1200,
        "maximum_output_bytes": 30*1024**3, "minimum_free_bytes": 20*1024**3,
        "fixed_existing_frames": True, "fixed_intrinsics": True,
        "counted": False, "absolute_pose_accuracy_claim": False}
    atomic_json(args.output/"CONTRACT.json", contract)
    shutil.copyfile(Path(__file__), args.output/"source_snapshot.py")
    if shutil.disk_usage(args.output).free < contract["minimum_free_bytes"]:
        raise OSError("Insufficient recovery disk headroom")
    database = args.output/"database.db"
    with sqlite3.connect(source_db.resolve().as_uri()+"?mode=ro", uri=True) as reader:
        with sqlite3.connect(database) as writer:
            reader.backup(writer)
    binary = ROOT/"05_envs/colmap-4.1.1-cuda/bin/colmap.exe"
    timings, batches = {}, []

    def run(label, command):
        if time.monotonic()-started >= contract["maximum_total_seconds"]:
            raise TimeoutError("Recovery total wall-clock budget exhausted")
        before = time.monotonic()
        with (args.output/(label+".log")).open("x", encoding="utf-8") as log:
            process = subprocess.run(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT, timeout=600)
        timings[label] = time.monotonic()-before
        if process.returncode:
            raise RuntimeError("Native recovery stage failed: "+label)

    source = source_model
    active = set(failed)
    schedules = [(len(images), len(images))] if args.mode == "baseline" else [(128, 96), (256, 192)]
    try:
        for batch_index, (temporal, spatial) in enumerate(schedules):
            if args.mode == "baseline":
                ids = sorted(images)
                pairs = [(a, b) for slot, a in enumerate(ids) for b in ids[slot+1:]
                    if (images[a] in active or images[b] in active) and PAIR_BASE*a+b not in attempted]
            else:
                pairs = pairs_for_failures(images, active, attempted, raw, temporal, spatial)
            if not pairs:
                raise ValueError("A failed recovery component has no new permitted candidate pairs")
            batch = args.output/f"batch{batch_index}"
            batch.mkdir()
            pair_file = batch/"pairs.txt"
            pair_file.write_text("\n".join(images[a]+" "+images[b] for a, b in pairs)+"\n", encoding="utf-8")
            row = {"batch": batch_index, "status": "matching", "temporal": temporal, "spatial": spatial,
                "failed_names_before": sorted(active), "requested_pairs": len(pairs)}
            batches.append(row)
            atomic_json(args.output/"STATE.json", {"status": "running", "batches": batches, "timings": timings})
            run(f"matching{batch_index}", [binary, "matches_importer", "--database_path", database,
                "--match_list_path", pair_file, "--match_type", "pairs", "--FeatureMatching.use_gpu", "1",
                "--FeatureMatching.gpu_index", "0", "--FeatureMatching.num_threads", "8",
                "--TwoViewGeometry.random_seed", "0", "--default_random_seed", "0"])
            with sqlite3.connect(database.resolve().as_uri()+"?mode=ro", uri=True) as connection:
                attempted = {p for p, in connection.execute("SELECT pair_id FROM matches")}
            row["status"] = "mapping"
            atomic_json(args.output/"STATE.json", {"status": "running", "batches": batches, "timings": timings})
            destination = batch/"model"
            destination.mkdir()
            run(f"mapping{batch_index}", [binary, "mapper", "--database_path", database,
                "--image_path", args.input_images, "--input_path", source, "--output_path", destination,
                "--Mapper.fix_existing_frames", "1", "--Mapper.num_threads", "8",
                "--Mapper.random_seed", "0", "--default_random_seed", "0",
                "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_principal_point", "0",
                "--Mapper.ba_refine_extra_params", "0", "--Mapper.ba_refine_sensor_from_rig", "0",
                "--Mapper.max_runtime_seconds", "590"])
            summaries = summarize_models(destination, images, failed, best, original_observations)
            selected = max(summaries, key=lambda r: r["registered_count"])
            recovered = pycolmap.Reconstruction(Path(selected["path"]))
            registered = {im.name for im in recovered.images.values() if im.has_pose}
            active = set(images.values())-registered
            pose_delta = max(float(np.max(np.abs(im.cam_from_world().matrix()-original_poses[im.name])))
                for im in recovered.images.values() if im.has_pose and im.name in original_poses)
            same_intrinsics = all(np.array_equal(c.params, model.cameras[i].params) for i, c in recovered.cameras.items())
            row.update(status="completed", summary=selected, original_pose_max_delta=pose_delta,
                original_intrinsics_exact=same_intrinsics)
            atomic_json(args.output/"STATE.json", {"status": "running", "batches": batches, "timings": timings})
            if selected["original_registered_frames_lost"] or pose_delta > 1e-9 or not same_intrinsics:
                raise ValueError("Progressive recovery changed a frozen existing camera or calibration")
            source = Path(selected["path"])
            if selected["registered_count"] >= contract["minimum_registered_for_diagnostic_comparison"]:
                break
        result = {"status": "actual_recovery_completed_diagnostic_guards_pending", "mode": args.mode,
            "batches": batches, "timings": timings, "requested_pairs": sum(b["requested_pairs"] for b in batches),
            "original_denominator": len(images), "final": batches[-1]["summary"],
            "original_all_frame_registration_guard": not active, "remaining_failures": sorted(active),
            "elapsed_seconds_including_preparation": time.monotonic()-started,
            "source_inputs_unchanged": sources == [data_identity(p) for p in [source_db, raw_path, Path(__file__), V1/"run_pair_recovery.py"]],
            "three_pairs_and_final_quality_pending": True, "counted": False}
        atomic_json(args.output/"RESULT.json", result)
        atomic_json(args.output/"STATE.json", {"status": result["status"], "timings": timings, "batches": batches})
        print(json.dumps({"registered": result["final"]["registered_count"], "pairs": result["requested_pairs"],
            "timings": timings, "counted": False}), flush=True)
    except BaseException as error:
        atomic_json(args.output/"STATE.json", {"status": "failed", "error": repr(error), "timings": timings,
            "batches": batches, "elapsed_seconds": time.monotonic()-started, "counted": False})
        raise


if __name__ == "__main__":
    main()
