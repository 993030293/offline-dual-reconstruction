"""Run a fresh recording through the original or accepted raw initial stages."""

import argparse
from collections import Counter
from contextlib import ExitStack
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

import cv2
import numpy as np
import psutil

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"
V3 = ROOT / "02_code/gaussian_pipeline_optimization_v3_20261005"
sys.path[:0] = [str(V1), str(V2), str(V3), str(V3 / "parallel_a"),
               str(ROOT / "02_code/box2_initial_pose_quality_v2_20261002")]

from runtime import atomic_json, data_identity
import initial_selection as baseline
import quality_engine
import selection_engine
sys.path.insert(0, str(baseline.TOOLS))
from mcap_audit import McapIndex, Cdr, decode_camera_info, decode_tf_static
from camera_motion import decode_bayer
from color_contract import demosaic_rggb8_to_bgr, COLOR_CONTRACT_VERSION
from indexed_candidate_pairs import candidate_pairs
from parallel_raw_export import prepared_exports
from single_copy_raw_reader import iter_single_copy_messages
from pose_interpolation import coarse_camera_at_many


def run(args):
    if args.output.exists():
        raise FileExistsError("Fresh raw stages require a new output directory")
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    process = psutil.Process()
    state = {"status": "initializing", "pid": os.getpid(), "birth": process.create_time(),
             "raw": str(args.raw), "mode": args.mode, "case": args.case}

    def update(status, **values):
        state.update(status=status, elapsed_seconds=time.perf_counter() - started, **values)
        state["rss_bytes"] = process.memory_info().rss
        atomic_json(args.output / "STATE.json", state)
        print(json.dumps(state), flush=True)

    cv2.setNumThreads(1)
    index_started = time.perf_counter()
    index = McapIndex(args.raw)
    total = index.message_count(baseline.CAMERA_TOPIC)
    timings = {"raw_index_seconds": time.perf_counter() - index_started}
    if total < 100:
        raise ValueError("This comparison requires an adequately sized raw capture")
    evaluation = sorted({int((number + .5) * total / 24) for number in range(24)})
    excluded = {frame: "development_evaluation" for frame in evaluation}
    for frame in evaluation:
        if frame + 1 < total:
            excluded[frame + 1] = "development_evaluation_adjacent"
    policy = dict(baseline.POLICY)
    contract = {"schema_version": 1, "case": args.case, "mode": args.mode,
        "raw_file": str(args.raw.resolve()), "raw_bytes": args.raw.stat().st_size,
        "raw_mtime_ns": args.raw.stat().st_mtime_ns, "raw_frame_count": total,
        "policy": policy, "evaluation_rule": "24 raw-index stratum midpoints and next-frame exclusion",
        "evaluation_frames": evaluation, "excluded_frames": sorted(excluded),
        "strict_blind_test": False, "old_runtime_inputs_used": False, "quality_cache_enabled": False,
        "JPEG_cache_enabled": False, "SfM_or_GT_for_initial_selection": False,
        "coarse_pose_source": "fresh_raw_LiDAR_IMU_motion" if args.coarse_trajectory else "raw_recorded_odometry_decoded_fresh_producer_unconfirmed",
        "new_LIO_solver_executed": False, "color_contract": COLOR_CONTRACT_VERSION,
        "explicit_external_body_camera_calibration": data_identity(args.body_camera_calibration) if args.body_camera_calibration else None,
        "opencv": cv2.__version__, "numpy": np.__version__,
        "sources": [data_identity(Path(__file__)), data_identity(Path(baseline.__file__)),
                    data_identity(Path(quality_engine.__file__)), data_identity(Path(selection_engine.__file__))]}
    atomic_json(args.output / "CONTRACT.json", contract)
    snapshot = args.output / "source_snapshot"
    snapshot.mkdir()
    for source in contract["sources"]:
        shutil.copyfile(source["path"], snapshot / Path(source["path"]).name)
    atomic_json(args.output / "EXCLUDED_EVALUATION_IDS.json", {"frames": excluded,
        "evaluation_frames": evaluation, "frozen_before_scoring": True})
    update("reading_raw_sensor_contract", raw_frame_count=total)
    module_trace = {}
    before = time.perf_counter()
    stamps, poses, covariance, frames = [], [], [], set()
    if args.coarse_trajectory:
        receipt = json.loads((args.coarse_trajectory.parent / "COARSE_RECEIPT.json").read_text())
        if (Path(receipt["raw_file"]).resolve() != args.raw.resolve()
                or receipt["SfM_used"] or receipt["GT_used"]):
            raise ValueError("The coarse motion is not fresh raw-only motion for this recording")
        with np.load(args.coarse_trajectory) as motion:
            stamps = motion["timestamp_ns"].tolist()
            poses = motion["pose_xyz_xyzw"].tolist()
        frames.add(("odom", "base_link"))
        module_trace["raw_LiDAR_IMU_motion"] = data_identity(args.coarse_trajectory)
    else:
        for message in index.iter_messages("/localization/pose/odom"):
            cdr = Cdr(message["payload"])
            sec, ns, parent = cdr.header()
            child = cdr.string()
            stamps.append(sec * 1000000000 + ns)
            poses.append([cdr.f64() for _ in range(7)])
            covariance.append([cdr.f64() for _ in range(36)])
            frames.add((parent, child))
    if frames != {("odom", "base_link")}:
        raise ValueError("The original body/world frame contract differs")
    transforms = []
    for message in index.iter_messages("/tf_static"):
        transforms.extend(decode_tf_static(message["payload"]))
    if args.body_camera_calibration:
        from scipy.spatial.transform import Rotation
        declared = json.loads(args.body_camera_calibration.read_text())
        entry = declared["T_base_link_camera_optical"]
        if entry["source_frame"] != "camera_optical" or entry["target_frame"] != "base_link":
            raise ValueError("The explicit external calibration has incompatible frame directions")
        values = np.asarray(entry["pose_xyz_xyzw"], dtype=np.float64)
        if values.shape != (7,) or not np.isfinite(values).all() or abs(np.linalg.norm(values[3:])-1.) > 1e-9:
            raise ValueError("The explicit external rigid calibration is malformed")
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = Rotation.from_quat(values[3:]).as_matrix()
        extrinsic[:3, 3] = values[:3]
        camera_edges = [row for row in transforms if "camera_optical" in (row["parent"], row["child"])]
        if camera_edges:
            recorded, _ = baseline.static_transform(transforms)
            if not np.allclose(recorded, extrinsic, rtol=0, atol=1e-10):
                raise ValueError("The explicit calibration contradicts recorded camera extrinsics")
        chain = ["camera_optical", "explicit_external_rig_calibration", "base_link"]
        module_trace["external_body_camera_calibration"] = {
            "identity": data_identity(args.body_camera_calibration), "consumed_field": "T_base_link_camera_optical",
            "recorded_camera_TF_present": bool(camera_edges), "independent_validation": "unconfirmed",
            "historical_trajectory_or_SfM_inputs_read": False}
    else:
        extrinsic, chain = baseline.static_transform(transforms)
    decoder = decode_camera_info
    if args.mode == "optimized":
        from calibration_body_cache import CalibrationBodyCache
        decoder = CalibrationBodyCache(decode_camera_info, Cdr, True)
        module_trace["R99"] = decoder.trace
    camera_info, camera_info_count = None, 0
    for message in index.iter_messages("/sensing/camera/camera_info"):
        info = decoder(message["payload"])
        if camera_info is None:
            camera_info = info
        for key in ("frame_id", "height", "width", "distortion_model", "distortion", "camera_matrix"):
            if camera_info[key] != info[key]:
                raise ValueError("Raw calibration changes within this recording")
        camera_info_count += 1
    curve = baseline.CoarseCameraCurve(stamps, poses, extrinsic)
    timings["raw_pose_and_calibration_seconds"] = time.perf_counter() - before
    atomic_json(args.output / "RAW_SENSOR_CONTRACT.json", {"raw_file": str(args.raw),
        "camera_info": camera_info, "camera_info_count": camera_info_count,
        "T_body_camera": extrinsic.tolist(), "extrinsic_chain": chain,
        "odometry_count": len(stamps), "odometry_frames": sorted(frames),
        "odometry_covariance_all_zero": bool(np.all(np.asarray(covariance) == 0)) if covariance else None,
        "odometry_producer": "raw_IMU_deskew_and_local_map_translation_ICP" if args.coarse_trajectory else "unconfirmed",
        "odometry_role": "estimated_motion_not_GT"})
    np.savez(args.output / "raw_body_odometry.npz", timestamp_ns=np.asarray(stamps, dtype=np.int64),
             pose_xyz_xyzw=np.asarray(poses))

    def messages(numbers):
        if args.mode == "optimized":
            return iter_single_copy_messages(index, baseline.CAMERA_TOPIC, numbers)
        return index.iter_selected_messages(baseline.CAMERA_TOPIC, numbers)

    def progress(count):
        if count % 250 == 0 or count == total:
            update("scoring", completed=count, total=total)

    before = time.perf_counter()
    with ExitStack() as stack:
        if args.mode == "optimized":
            from exact_bayer_gray import raw_gray_quality_execution
            from planar_corner import planar_corner_execution
            from vector_gray_stats import vector_gray_stats_execution
            from quality_histogram import histogram_quality_execution
            libs = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005"
            module_trace["R149"] = stack.enter_context(planar_corner_execution(libs / "qc001/b0/native_planar_corner.dll"))
            module_trace["R144"] = stack.enter_context(vector_gray_stats_execution(libs / "vector_gray_stats_001/b0/native_vector_gray_stats.dll"))
            module_trace["R147_R141"] = stack.enter_context(raw_gray_quality_execution(
                libs / "bg001/b0/native_bayer_gray.dll", demosaic_rggb8_to_bgr))
            module_trace["R104"] = stack.enter_context(histogram_quality_execution(True))
            scorer, converter, workers = quality_engine.compact_quality, lambda raw: raw, 4
        else:
            scorer, converter, workers = baseline.quality, demosaic_rggb8_to_bgr, 1
        scores, detail = quality_engine.score_messages(messages(range(total)), decode_bayer,
            converter, scorer, workers=workers, capacity=16, cache=None,
            maximum_seconds=3600, progress=progress)
    timings["raw_image_scoring_wall_seconds"] = time.perf_counter() - before
    timings["raw_image_scoring_detail"] = detail
    before = time.perf_counter()
    rows = []
    batch = None
    if args.mode == "optimized":
        batch = coarse_camera_at_many(curve, [score["camera_header_ns"] for score in scores],
                                     policy["maximum_interpolation_span_ns"])
    for number, score in enumerate(scores):
        frame, stamp = score["camera_frame"], score["camera_header_ns"]
        row = {key: score[key] for key in ("camera_frame", "camera_header_ns", "optical_frame", "width", "height", "quality")}
        row.update(image_name=f"frame_{frame:05d}_{stamp}.jpg", excluded_role=excluded.get(frame), coarse_pose_valid=False)
        if row["optical_frame"] != camera_info["frame_id"] or (row["width"], row["height"]) != (camera_info["width"], camera_info["height"]):
            raise ValueError("Raw image and raw calibration identities differ")
        if batch is not None:
            row.update(batch[number])
        else:
            try:
                row.update(curve.at(stamp, policy["maximum_interpolation_span_ns"]), coarse_pose_valid=True)
            except ValueError as error:
                if str(error) not in ("outside_coarse_time_support", "coarse_interpolation_gap_too_large"):
                    raise
                row["pose_rejection"] = str(error)
        rows.append(row)
    timings["coarse_pose_interpolation_seconds"] = time.perf_counter() - before
    reference = [row["quality"]["sharpness"] for row in rows if row["coarse_pose_valid"] and not row["excluded_role"]]
    threshold = float(np.quantile(reference, policy["sharpness_quantile"]))
    before = time.perf_counter()
    if args.mode == "optimized":
        from spatial_novelty import spatial_novelty_execution
        with spatial_novelty_execution(True) as trace:
            selected, eligible = selection_engine.choose_batched(rows, policy, threshold, baseline)
        module_trace["R107"] = trace
    else:
        selected, eligible = baseline.choose(rows, policy, threshold)
    timings["selection_seconds"] = time.perf_counter() - before
    update("exporting", selected=len(selected), eligible=len(eligible))
    images = args.output / "images"
    images.mkdir()
    chosen = {row["camera_frame"]: row for row in selected}
    evaluation_rows = {row["camera_frame"]: row for row in rows if row["camera_frame"] in evaluation}
    evaluation_dir = args.output / "evaluation_images"
    evaluation_dir.mkdir()
    export_ids = sorted(set(chosen) | set(evaluation))
    before = time.perf_counter()
    export_trace = {}
    if args.mode == "optimized":
        exports = prepared_exports(messages(export_ids), decode_bayer, demosaic_rggb8_to_bgr,
            policy["jpeg_quality"], workers=4, capacity=8, encode_in_workers=True, trace=export_trace)
    else:
        def serial_exports():
            for frame, message in messages(export_ids):
                stamp, _, raw = decode_bayer(message["payload"])
                ok, encoded = cv2.imencode(".jpg", demosaic_rggb8_to_bgr(raw),
                    [cv2.IMWRITE_JPEG_QUALITY, policy["jpeg_quality"]])
                yield frame, stamp, ok, encoded
        exports = serial_exports()
    try:
        for frame, stamp, ok, encoded in exports:
            row = chosen.get(frame, evaluation_rows.get(frame))
            if not ok or int(stamp) != row["camera_header_ns"]:
                raise ValueError("Fresh JPEG export changed its original timestamp or failed")
            folder = images if frame in chosen else evaluation_dir
            path = folder / row["image_name"]
            with path.open("xb") as stream:
                stream.write(encoded.tobytes())
            row["image_sha256"] = data_identity(path)["sha256"]
    finally:
        exports.close()
    module_trace["Q53"] = export_trace
    timings["decode_encode_write_seconds"] = time.perf_counter() - before
    before = time.perf_counter()
    pairs, pair_trace = candidate_pairs(selected, args.mode == "optimized")
    module_trace["Q50"] = pair_trace
    with (args.output / "initial_matching_pairs.txt").open("x", encoding="utf-8") as stream:
        for i, j in sorted(pairs):
            stream.write(selected[i]["image_name"] + " " + selected[j]["image_name"] + "\n")
    timings["candidate_pair_generation_seconds"] = time.perf_counter() - before
    with (args.output / "frame_decisions.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    atomic_json(args.output / "EVALUATION_FRAMES.json", {"rows": list(evaluation_rows.values()),
        "role": "development_evaluation_excluded_before_quality_selection"})
    receipt = {"status": "fresh_initial_completed_sfm_not_run", "raw_frame_count": total,
        "quality_and_coarse_pose_eligible_count": len(eligible), "selected_count": len(selected),
        "sharpness_min": threshold, "decision_counts": dict(Counter(row["decision"] for row in rows)),
        "matching_candidate_pair_count": len(pairs), "excluded_overlap": len(set(chosen) & set(excluded)),
        "evaluation_export_count": len(evaluation_rows), "quality_cache_hits": detail["cache_hits"],
        "timings": timings, "elapsed_seconds": time.perf_counter() - started,
        "module_trace": module_trace, "SfM_run": False, "Gaussian_retrained": False,
        "GT_used": False, "old_sfm_inputs_used": False, "full_LIO_solver_run": False,
        "raw_LiDAR_IMU_coarse_motion_solver_run": bool(args.coarse_trajectory)}
    atomic_json(args.output / "INITIAL_FRAME_MANIFEST.json", {"receipt": receipt, "selected_frames": selected,
        "coarse_pose_convention": "T_world_camera; metric odom world; optical R_WC columns",
        "camera_intrinsics": camera_info, "policy": policy})
    atomic_json(args.output / "SELECTION_RECEIPT.json", receipt)
    update("completed_initial_selection", selected_count=len(selected), timings=timings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--mode", choices=("baseline", "optimized"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coarse-trajectory", type=Path)
    parser.add_argument("--body-camera-calibration", type=Path)
    args = parser.parse_args()
    try:
        run(args)
    except BaseException:
        if args.output.exists():
            atomic_json(args.output / "FAILURE.json", {"status": "failed", "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
