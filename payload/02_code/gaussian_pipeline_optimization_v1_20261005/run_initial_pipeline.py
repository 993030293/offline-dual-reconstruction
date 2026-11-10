"""Rebuild initial frames from raw measurements with isolated, bounded optimizations."""

import argparse
from collections import Counter
import importlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import cv2
import numpy as np
import psutil

from quality_engine import compact_quality, score_messages, ScoreCache
from runtime import atomic_json, data_identity
from selection_engine import choose_batched


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--workers", type=int, default=4, choices=[1, 4, 8])
    parser.add_argument("--scorer", choices=["baseline", "compact"], default="compact")
    parser.add_argument("--selector", choices=["baseline", "batched"], default="batched")
    parser.add_argument("--reader", choices=["indexed", "mapped"], default="indexed")
    parser.add_argument("--pose-mode", choices=["scalar", "batched"], default="scalar")
    parser.add_argument("--reuse-export", type=Path)
    args = parser.parse_args()
    if args.reuse_export is not None:
        raise ValueError("Export caching requires a separate verified implementation")
    args.output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project/"02_code/box2_initial_pose_quality_v2_20261002"))
    baseline = importlib.import_module("initial_selection")
    sys.path.insert(0, str(baseline.TOOLS))
    from mcap_audit import McapIndex, Cdr, decode_camera_info, decode_tf_static
    from camera_motion import decode_bayer
    from color_contract import demosaic_rggb8_to_bgr, COLOR_CONTRACT_VERSION
    started = time.perf_counter()
    cv2.setNumThreads(1)
    state = {"status": "starting", "pid": os.getpid(), "input_role": "raw_only_no_SfM_or_GT"}
    atomic_json(args.output/"STATE.json", state)
    exclusion_source = project/"04_outputs/box2_initial_pose_quality_v2_20261002T193102/01_initial_frames/EXCLUDED_EVALUATION_IDS.json"
    excluded = json.loads(exclusion_source.read_text(encoding="utf-8"))
    exclusions = {int(key): value for key, value in excluded["frames"].items()}
    atomic_json(args.output/"EXCLUDED_EVALUATION_IDS.json", excluded)
    sources = [Path(__file__), Path(__file__).with_name("quality_engine.py"),
               Path(__file__).with_name("selection_engine.py"), Path(baseline.__file__),
               baseline.TOOLS/"camera_motion.py", baseline.TOOLS/"color_contract.py", baseline.TOOLS/"mcap_audit.py"]
    if args.reader == "mapped":
        sources.append(Path(__file__).with_name("mapped_raw_reader.py"))
    if args.pose_mode == "batched":
        sources.append(Path(__file__).with_name("pose_interpolation.py"))
    identities = [data_identity(path) for path in sources]
    atomic_json(args.output/"CONTRACT.json", {"policy": baseline.POLICY, "sources": identities,
        "intervention": {"workers": args.workers, "scorer": args.scorer, "selector": args.selector,
                         "cache": str(args.cache) if args.cache else None, "reader": args.reader,
                         "pose_mode": args.pose_mode},
        "maximum_wall_seconds": 1800, "rss_budget_bytes": 32*1024**3,
        "raw_only_selection": True, "old_decisions_used_only_after_run_for_equivalence": True,
        "strict_blind_test": False, "production_inputs_modified": False})
    snapshot = args.output/"source_snapshot"
    snapshot.mkdir()
    for source in sources:
        shutil.copyfile(source, snapshot/source.name)

    def update(status, **values):
        state.update(status=status, elapsed_seconds=time.perf_counter()-started, **values)
        memory = psutil.Process().memory_info()
        state["peak_rss_bytes"] = getattr(memory, "peak_wset", memory.rss)
        atomic_json(args.output/"STATE.json", state)
        if state["elapsed_seconds"] > 1800 or state["peak_rss_bytes"] > 32*1024**3:
            raise RuntimeError("Initial pipeline exceeded its frozen resource budget")

    before = time.perf_counter()
    index = McapIndex(baseline.RAW)
    timings = {"raw_index_seconds": time.perf_counter()-before}

    def image_messages(numbers):
        if args.reader == "mapped":
            from mapped_raw_reader import iter_mapped_messages
            return iter_mapped_messages(index, baseline.CAMERA_TOPIC, numbers)
        return index.iter_selected_messages(baseline.CAMERA_TOPIC, numbers)
    before = time.perf_counter()
    stamps, poses, covariance, frames = [], [], [], set()
    for message in index.iter_messages("/localization/pose/odom"):
        cdr = Cdr(message["payload"])
        sec, ns, parent = cdr.header()
        child = cdr.string()
        stamps.append(sec*1_000_000_000+ns)
        poses.append([cdr.f64() for _ in range(7)])
        covariance.append([cdr.f64() for _ in range(36)])
        frames.add((parent, child))
    if frames != {("odom", "base_link")}:
        raise ValueError("Raw world/body frame differs from the supported contract")
    transforms = []
    for message in index.iter_messages("/tf_static"):
        transforms.extend(decode_tf_static(message["payload"]))
    extrinsic, chain = baseline.static_transform(transforms)
    camera_info = None
    camera_info_count = 0
    for message in index.iter_messages("/sensing/camera/camera_info"):
        info = decode_camera_info(message["payload"])
        if camera_info is None:
            camera_info = info
        for key in ("frame_id", "height", "width", "distortion_model", "distortion", "camera_matrix"):
            if camera_info[key] != info[key]:
                raise ValueError("Raw camera calibration changes during this recording")
        camera_info_count += 1
    curve = baseline.CoarseCameraCurve(stamps, poses, extrinsic)
    timings["raw_pose_and_calibration_seconds"] = time.perf_counter()-before
    atomic_json(args.output/"RAW_SENSOR_CONTRACT.json", {"raw_file": str(baseline.RAW),
        "raw_bytes": baseline.RAW.stat().st_size, "camera_info_source": "raw_mcap_camera_info",
        "camera_info": camera_info, "camera_info_count": camera_info_count,
        "T_body_camera": extrinsic.tolist(), "extrinsic_chain": chain,
        "odometry_count": len(stamps), "odometry_frames": sorted(frames),
        "odometry_covariance_all_zero": bool(np.all(np.asarray(covariance) == 0)),
        "odometry_producer": "unconfirmed", "odometry_role": "recorded_estimate_not_GT",
        "color_contract": COLOR_CONTRACT_VERSION, "sfm_or_gt_used_for_selection": False})
    np.savez(args.output/"raw_body_odometry.npz", timestamp_ns=np.asarray(stamps, dtype=np.int64),
             pose_xyz_xyzw=np.asarray(poses))
    cache_contract = {"version": 1, "color": COLOR_CONTRACT_VERSION, "opencv": cv2.__version__,
        "numpy": np.__version__, "scorer": args.scorer,
        "sources": [item["sha256"] for item in identities if Path(item["path"]).name in
                    ("quality_engine.py", "initial_selection.py", "camera_motion.py", "color_contract.py")]}
    cache = ScoreCache(args.cache, cache_contract) if args.cache else None
    total = index.message_count(baseline.CAMERA_TOPIC)
    before = time.perf_counter()

    def progress(count):
        if count % 250 == 0 or count == total:
            update("scoring", completed=count, total=total)
            print(json.dumps({"status": "scoring", "frames": count, "total": total}), flush=True)

    scores, score_timings = score_messages(image_messages(range(total)),
        decode_bayer, demosaic_rggb8_to_bgr, compact_quality if args.scorer == "compact" else baseline.quality,
        workers=args.workers, capacity=16, cache=cache, maximum_seconds=1500, progress=progress)
    timings["raw_image_scoring_wall_seconds"] = time.perf_counter()-before
    timings["raw_image_scoring_detail"] = score_timings
    before = time.perf_counter()
    rows = []
    batch_poses = None
    if args.pose_mode == "batched":
        from pose_interpolation import coarse_camera_at_many
        batch_poses = coarse_camera_at_many(curve, [score["camera_header_ns"] for score in scores],
                                            baseline.POLICY["maximum_interpolation_span_ns"])
    for score_index, score in enumerate(scores):
        frame, stamp = score["camera_frame"], score["camera_header_ns"]
        row = {key: score[key] for key in ("camera_frame", "camera_header_ns", "optical_frame", "width", "height", "quality")}
        row.update(image_name=f"frame_{frame:05d}_{stamp}.jpg", excluded_role=exclusions.get(frame), coarse_pose_valid=False)
        if row["optical_frame"] != "camera_optical" or [row["width"], row["height"]] != [camera_info["width"], camera_info["height"]]:
            raise ValueError("Raw image differs from calibration identity/dimensions")
        if batch_poses is not None:
            row.update(batch_poses[score_index])
        else:
            try:
                row.update(curve.at(stamp, baseline.POLICY["maximum_interpolation_span_ns"]), coarse_pose_valid=True)
            except ValueError as error:
                if str(error) not in ("outside_coarse_time_support", "coarse_interpolation_gap_too_large"):
                    raise
                row["pose_rejection"] = str(error)
        rows.append(row)
    timings["coarse_pose_interpolation_seconds"] = time.perf_counter()-before
    reference = [row["quality"]["sharpness"] for row in rows if row["coarse_pose_valid"] and not row["excluded_role"]]
    threshold = float(np.quantile(reference, baseline.POLICY["sharpness_quantile"]))
    before = time.perf_counter()
    if args.selector == "batched":
        selected, eligible = choose_batched(rows, baseline.POLICY, threshold, baseline)
    else:
        selected, eligible = baseline.choose(rows, baseline.POLICY, threshold)
    timings["selection_seconds"] = time.perf_counter()-before
    update("exporting", selected=len(selected))
    images = args.output/"images"
    images.mkdir()
    selected_by_id = {row["camera_frame"]: row for row in selected}
    before = time.perf_counter()
    for exported, (frame, message) in enumerate(image_messages(selected_by_id), 1):
        stamp, _, raw = decode_bayer(message["payload"])
        row = selected_by_id[frame]
        if int(stamp) != row["camera_header_ns"]:
            raise ValueError("Raw image identity changed before export")
        ok, encoded = cv2.imencode(".jpg", demosaic_rggb8_to_bgr(raw),
                                 [cv2.IMWRITE_JPEG_QUALITY, baseline.POLICY["jpeg_quality"]])
        if not ok:
            raise RuntimeError("Raw image JPEG export failed")
        target = images/row["image_name"]
        with target.open("xb") as stream:
            stream.write(encoded.tobytes())
        row["image_sha256"] = data_identity(target)["sha256"]
        if exported % 100 == 0:
            update("exporting", exported=exported)
    timings["decode_encode_write_seconds"] = time.perf_counter()-before
    before = time.perf_counter()
    positions = np.asarray([row["C_W_m"] for row in selected])
    pairs = set()
    for i, row in enumerate(selected):
        for j in range(i+1, min(len(selected), i+9)):
            pairs.add((i, j))
        for j in np.argsort(np.linalg.norm(positions-positions[i], axis=1))[:17]:
            if abs(int(j)-i) > 8:
                pairs.add(tuple(sorted((i, int(j)))))
    with (args.output/"initial_matching_pairs.txt").open("x", encoding="utf-8") as stream:
        for i, j in sorted(pairs):
            stream.write(selected[i]["image_name"]+" "+selected[j]["image_name"]+"\n")
    timings["candidate_pair_generation_seconds"] = time.perf_counter()-before
    with (args.output/"frame_decisions.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
    times = np.asarray([row["camera_header_ns"] for row in selected], dtype=np.int64)
    gaps = np.diff(times)/1e9
    unresolved = [{"left": selected[i]["image_name"], "right": selected[i+1]["image_name"], "gap_seconds": float(gap)}
                  for i, gap in enumerate(gaps) if gap > baseline.POLICY["bridge_gap_seconds"]]
    receipt = {"status": "initial_frames_exported_sfm_not_run", "raw_frame_count": total,
        "quality_and_coarse_pose_eligible_count": len(eligible), "selected_count": len(selected),
        "sharpness_min": threshold, "decision_counts": dict(Counter(row["decision"] for row in rows)),
        "selection_reason_counts": dict(Counter(row["selection_reason"] for row in selected)),
        "matching_candidate_pair_count": len(pairs), "matching_pairs_verified_by_features": False,
        "selected_max_gap_seconds": float(gaps.max()), "unresolved_quality_supported_time_gaps": unresolved,
        "excluded_overlap": len(set(selected_by_id) & set(exclusions)), "timings": timings,
        "elapsed_seconds": time.perf_counter()-started, "SfM_run": False, "Gaussian_retrained": False,
        "GT_used": False, "old_sfm_inputs_used": False, "counted_as_usable_optimizations": []}
    atomic_json(args.output/"INITIAL_FRAME_MANIFEST.json", {"receipt": receipt, "selected_frames": selected,
        "coarse_pose_convention": "T_world_camera; R_WC optical axes; C_W in odom metric coordinates",
        "camera_intrinsics": camera_info, "policy": baseline.POLICY})
    atomic_json(args.output/"SELECTION_RECEIPT.json", receipt)
    update("completed_initial_selection", timings=timings)
    print(json.dumps({"status": state["status"], "elapsed_seconds": state["elapsed_seconds"],
                      "selected_count": len(selected), "timings": timings}), flush=True)


if __name__ == "__main__":
    main()
