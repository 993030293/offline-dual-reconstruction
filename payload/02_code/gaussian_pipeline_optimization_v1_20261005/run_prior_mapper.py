"""Run the installed prior-aware mapper with bound and read-back metric positions."""

import argparse
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time

import numpy as np
import pycolmap

from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2, choices=[2, 8])
    parser.add_argument("--maximum-seconds", type=int, default=600, choices=[600, 1200])
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--include-raw-candidates", action="store_true")
    parser.add_argument("--preflight-receipt", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[2]
    binary = project/"05_envs/colmap-4.1.1-cuda/bin/colmap.exe"
    rows = json.loads((args.initial/"INITIAL_FRAME_MANIFEST.json").read_text())["selected_frames"]
    if args.include_raw_candidates:
        rows = [json.loads(line) for line in (args.initial / "frame_decisions.jsonl").read_text().splitlines()]
        rows = [row for row in rows if row["coarse_pose_valid"] and not row["excluded_role"]
                and row["decision"] in ("eligible_not_selected", "selected_initial_sfm_pending")]
    lookup = {row["image_name"]: row for row in rows}
    database = args.output/"database.db"
    shutil.copyfile(args.database, database)
    bindings = []
    with pycolmap.Database.open(database) as connection:
        if connection.read_all_pose_priors():
            raise ValueError("Source database already contains priors; refuse ambiguous provenance")
        images = connection.read_all_images()
        if len(images) > 256:
            if args.preflight_receipt is None:
                raise ValueError("A full run requires the prior-factor activation preflight")
            preflight = json.loads(args.preflight_receipt.read_text())
            if not preflight.get("all_expected_position_factors_activated") or preflight["registered_images"] < 64:
                raise ValueError("The small-scale native position factor preflight did not pass")
        for image in sorted(images, key=lambda image: image.image_id):
            row = lookup[image.name]
            if not row["coarse_pose_valid"] or row["excluded_role"]:
                raise ValueError("Unsupported or evaluation image cannot receive a training prior")
            prior = pycolmap.PosePrior(corr_data_id=image.data_id, position=np.asarray(row["C_W_m"]),
                position_covariance=np.eye(3)*.15**2, coordinate_system=pycolmap.PosePriorCoordinateSystem.CARTESIAN)
            written_id = connection.write_pose_prior(prior)
            bindings.append({"image_id": image.image_id, "name": image.name, "prior_id": written_id,
                             "position_m": row["C_W_m"]})
        loaded = connection.read_all_pose_priors()
        by_id = {image.image_id: image for image in images}
        for prior in loaded:
            image = by_id[prior.corr_data_id.id]
            if prior.corr_data_id != image.data_id or not np.array_equal(prior.position, lookup[image.name]["C_W_m"]):
                raise AssertionError("Stored prior binding or position differs from the actual image identity")
            if not np.array_equal(prior.position_covariance, np.eye(3)*.15**2):
                raise AssertionError("Stored covariance changed")
        if len(loaded) != len(images):
            raise AssertionError("Prior read-back count mismatch")
    with sqlite3.connect(database) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Prior database integrity check failed")
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    root = args.output/"models"
    root.mkdir()
    image_root = args.image_root or args.initial / "images"
    if any(not (image_root / image.name).is_file() for image in images):
        raise ValueError("Input database image is missing from the declared image root")
    command = [binary, "pose_prior_mapper", "--database_path", database, "--image_path", image_root,
        "--log_level", "2", "--log_target", "stderr",
        "--output_path", root, "--Mapper.num_threads", str(args.threads), "--Mapper.random_seed", "0",
        "--default_random_seed", "0", "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_principal_point", "0",
        "--Mapper.ba_refine_extra_params", "0", "--Mapper.ba_refine_sensor_from_rig", "0",
        "--use_robust_loss_on_prior_position", "1", "--prior_position_loss_scale", "7.82",
        "--overwrite_priors_covariance", "0", "--Mapper.max_runtime_seconds", str(args.maximum_seconds-10)]
    contract = {"source_database": data_identity(args.database), "prior_bindings": bindings, "readback_verified": True,
        "sigma_m": .15, "covariance_is_assumed_not_sensor_estimate": True, "coordinate_system": "CARTESIAN",
        "position_frame": "recorded odom camera optical center in metres", "recorded_odometry_is_not_GT": True,
        "source": data_identity(Path(__file__)), "command": list(map(str, command)), "old_SfM_model_used": False,
        "native_log_verbosity": 2, "CLI_prior_loss_scale": 7.82,
        "include_raw_candidates": args.include_raw_candidates,
        "preflight_receipt": data_identity(args.preflight_receipt) if args.preflight_receipt else None,
        "CLI_scale_is_kept_distinct_from_python_BA_scale": True,
        "maximum_seconds": args.maximum_seconds, "counted": False,
        "acceptance": "real all-frame registration and finite map; inspect native prior BA reports and alignment fallbacks; explicit final-factor audit required before claiming complete activation"}
    atomic_json(args.output/"CONTRACT.json", contract)
    shutil.copyfile(Path(__file__), args.output/"source_snapshot.py")
    environment = dict(os.environ, GLOG_v="2")
    started = time.monotonic()
    with (args.output/"mapper.log").open("w") as stream:
        process = subprocess.Popen(list(map(str, command)), stdout=stream, stderr=subprocess.STDOUT, env=environment)
        atomic_json(args.output/"STATE.json", {"status": "running", "controller_pid": os.getpid(), "child_pid": process.pid})
        try:
            code = process.wait(timeout=args.maximum_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            atomic_json(args.output/"STATE.json", {"status": "watchdog_timeout", "elapsed_seconds": time.monotonic()-started})
            raise
    log = (args.output/"mapper.log").read_text(errors="replace")
    models = []
    for path in [root]+[path for path in root.iterdir() if path.is_dir()]:
        if not (path/"images.bin").is_file():
            continue
        model = pycolmap.Reconstruction(path)
        names = {image.name for image in model.images.values() if image.has_pose}
        errors = [float(np.linalg.norm(image.projection_center()-lookup[image.name]["C_W_m"])) for image in model.images.values() if image.has_pose]
        models.append({"path": str(path), "registered": len(names), "failed_images": sorted(set(image.name for image in images)-names),
            "points": model.num_points3D(), "mean_reprojection_error_px": model.compute_mean_reprojection_error(),
            "raw_prior_consistency_m_not_GT": {"p50": float(np.quantile(errors, .5)), "p95": float(np.quantile(errors, .95)), "max": max(errors)},
            "poses_finite": all(np.isfinite(image.cam_from_world().matrix()).all() for image in model.images.values() if image.has_pose)})
    result = {"status": "prior_mapper_completed_activation_audit_pending" if code == 0 else "prior_mapper_failed",
        "returncode": code, "elapsed_seconds": time.monotonic()-started, "models": models,
        "native_prior_BA_report_count": log.count("Pose Prior Bundle adjustment report"),
        "alignment_fallback_warning_count": log.count("Alignment w.r.t. prior positions failed"),
        "linear_solver_warning_count": log.count("Linear solver failure"),
        "all_images_registered": any(model["registered"] == len(images) for model in models),
        "final_factor_audit_completed": False, "counted": False}
    atomic_json(args.output/"RESULT.json", result)
    atomic_json(args.output/"STATE.json", result)
    print(json.dumps(result), flush=True)
    if code:
        raise RuntimeError("Native prior mapper failed")


if __name__ == "__main__":
    main()
