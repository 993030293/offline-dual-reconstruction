"""Probe or recover fixed problematic cameras with the native learned matcher."""

import argparse
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

import numpy as np
import psutil
import pycolmap

from prepare_native_lightglue import MODEL_SHA
from runtime import atomic_json, data_identity
from run_pair_recovery import observations, summarize_models


def gpu_usage():
    text = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.free", "--format=csv,noheader,nounits"], text=True, timeout=10)
    rows = text.strip().splitlines()
    if len(rows) != 1:
        raise ValueError("This diagnostic assumes exactly one verified GPU")
    return tuple(int(x.strip()) for x in rows[0].split(","))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-run", "initial", "weights", "feature-probe", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--recover", action="store_true")
    parser.add_argument("--smoke", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    started = time.monotonic()
    source = json.loads((args.source_run/"RESULT.json").read_text())
    probe = json.loads((args.feature_probe/"RESULT.json").read_text())
    if any(row.get("descriptor_type_verified") != "SIFT" for row in probe["rows"]):
        raise ValueError("An explicit native SIFT type audit is required")
    weights = args.weights/"sift-lightglue.onnx"
    if data_identity(weights)["sha256"] != MODEL_SHA:
        raise ValueError("Only the official checksum-bound native model is allowed")
    if args.recover:
        if args.smoke is None:
            raise ValueError("Recovery requires the real native matcher smoke")
        smoke = json.loads((args.smoke/"RESULT.json").read_text())
        if not smoke["native_match_smoke_passed"]:
            raise ValueError("Native learned matching has not produced verified geometry")
        smoke_contract = json.loads((args.smoke/"CONTRACT.json").read_text())
        if smoke_contract.get("device") != args.device:
            raise ValueError("Recovery requires a smoke using the same explicit execution device")
    best = max(source["models"], key=lambda row: row["registered_count"])
    problem = sorted(set(best["remaining_failures"]+[r["image_name"] for r in best["weak_registered_images_remaining"]]))
    if len(problem) != 5 or set(problem) != {r["name"] for r in probe["rows"]}:
        raise ValueError("This branch is bounded to the five frozen problem cameras")
    model_path = Path(best["path"]).resolve()
    previous = pycolmap.Reconstruction(model_path)
    registered = {im.name for im in previous.images.values() if im.has_pose}
    raw = {r["image_name"]: r for line in (args.initial/"frame_decisions.jsonl").read_text().splitlines()
           for r in [json.loads(line)]}
    source_db = (args.source_run/"database.db").resolve()
    with sqlite3.connect(source_db.as_uri()+"?mode=ro&immutable=1", uri=True) as db:
        images = dict(db.execute("SELECT image_id,name FROM images"))
    if len(images) != 1021:
        raise ValueError("The frozen all-frame denominator changed")
    ids = {name: iid for iid, name in images.items()}
    pairs = set()
    for name in problem:
        stamp = raw[name]["camera_header_ns"]
        candidates = [other for other in registered if other != name]
        if args.recover:
            before = sorted((n for n in candidates if raw[n]["camera_header_ns"] < stamp), key=lambda n: -raw[n]["camera_header_ns"])[:16]
            after = sorted((n for n in candidates if raw[n]["camera_header_ns"] >= stamp), key=lambda n: raw[n]["camera_header_ns"])[:16]
            neighbours = before+after
        else:
            neighbours = [min(candidates, key=lambda n: (abs(raw[n]["camera_header_ns"]-stamp), n))]
        pairs.update(tuple(sorted((ids[name], ids[n]))) for n in neighbours)
    pairs = sorted(pairs)
    if not 1 <= len(pairs) <= (160 if args.recover else 5):
        raise ValueError("Native matcher pair budget exceeded")
    gpu_start, gpu_free = gpu_usage()
    gpu_budget = min(12288, gpu_free-2048)
    if gpu_budget < 2048:
        raise RuntimeError("Insufficient isolated GPU headroom")
    args.output.mkdir(parents=True, exist_ok=False)
    binary = Path(__file__).resolve().parents[2]/"05_envs/colmap-4.1.1-cuda/bin/colmap.exe"
    contract = {"candidate": "P20_NATIVE_SIFT_LIGHTGLUE", "stage": "recovery" if args.recover else "smoke",
        "source_database": data_identity(source_db), "source_receipt": data_identity(args.source_run/"RESULT.json"),
        "binary": data_identity(binary), "weights": data_identity(weights), "source": data_identity(Path(__file__)),
        "all_input_denominator": 1021, "problem_cameras": problem, "pairs": [[images[a], images[b]] for a,b in pairs],
        "maximum_seconds_each": 600, "maximum_rss_bytes": 32*1024**3,
        "initial_gpu_used_mib": gpu_start, "additional_gpu_budget_mib": gpu_budget,
        "feature_indices_and_xy_unchanged": True, "confidence": .1, "native_geometric_verification": True,
        "single_database_writer": True, "existing_camera_poses_fixed": True, "calibration_fixed": True,
        "acceptance": "1021 in one map; no old camera loss; original camera support policy; common observation P95 increase <=0.05px",
        "environment_installation": False, "counted": False}
    contract["device"] = args.device
    atomic_json(args.output/"CONTRACT.json", contract)
    database = args.output/"database.db"
    shutil.copyfile(source_db, database)
    pair_ids = [(2147483647*a+b,) for a,b in pairs]
    with sqlite3.connect(database) as db:
        before = {pid: db.execute("SELECT rows FROM two_view_geometries WHERE pair_id=?", (pid,)).fetchone() for pid, in pair_ids}
        for table in ("matches", "two_view_geometries"):
            db.executemany("DELETE FROM "+table+" WHERE pair_id=?", pair_ids)
    pair_file = args.output/"pairs.txt"
    pair_file.write_text("\n".join(images[a]+" "+images[b] for a,b in pairs)+"\n")
    timings = {}
    def run(label, command):
        begin = time.monotonic()
        atomic_json(args.output/(label+"_COMMAND.json"), {"command": list(map(str, command))})
        with (args.output/(label+".log")).open("w") as log:
            process = subprocess.Popen(list(map(str, command)), stdout=log, stderr=subprocess.STDOUT)
            peak = 0
            peak_gpu = gpu_start
            last_gpu_check = 0
            try:
                while process.poll() is None:
                    try:
                        parent = psutil.Process(process.pid)
                        peak = max(peak, sum(p.memory_info().rss for p in [parent]+parent.children(recursive=True)))
                    except psutil.NoSuchProcess:
                        pass
                    if time.monotonic()-begin > 600 or peak > 32*1024**3:
                        raise RuntimeError("Native diagnostic exceeded its fixed time or RAM budget")
                    if time.monotonic()-last_gpu_check >= 2:
                        peak_gpu = max(peak_gpu, gpu_usage()[0])
                        last_gpu_check = time.monotonic()
                        if peak_gpu-gpu_start > gpu_budget:
                            raise RuntimeError("Native diagnostic exceeded its sampled GPU budget")
                    time.sleep(.5)
            except Exception as error:
                process.kill()
                process.wait()
                atomic_json(args.output/"FAILURE.json", {"stage": label, "error": repr(error),
                    "peak_rss_bytes": peak, "peak_sampled_total_gpu_mib": peak_gpu})
                raise
        timings[label] = {"seconds": time.monotonic()-begin, "peak_rss_bytes": peak,
            "peak_sampled_total_gpu_mib": peak_gpu, "GPU_sampling_interval_seconds": 2, "returncode": process.returncode}
        if process.returncode:
            atomic_json(args.output/"FAILURE.json", {"stage": label, "timings": timings, "source_unchanged": data_identity(source_db)==contract["source_database"]})
            raise RuntimeError("Native matcher diagnostic failed; inspect the preserved log")
    run("matching", [binary, "matches_importer", "--database_path", database, "--match_list_path", pair_file,
        "--match_type", "pairs", "--FeatureMatching.type", "SIFT_LIGHTGLUE", "--FeatureMatching.use_gpu", "1" if args.device == "cuda" else "0",
        "--FeatureMatching.gpu_index", "0", "--FeatureMatching.num_threads", "1", "--FeatureMatching.guided_matching", "0",
        "--FeatureMatching.skip_geometric_verification", "0", "--SiftMatching.lightglue_model_path", weights,
        "--SiftMatching.lightglue_min_score", ".1", "--TwoViewGeometry.random_seed", "0", "--default_random_seed", "0"])
    rows = []
    with sqlite3.connect(database) as db:
        db.execute("ATTACH DATABASE ? AS previous", (str(source_db),))
        for table in ("images", "cameras", "keypoints", "descriptors", "frames", "rigs"):
            for left, right in ((table, "previous."+table), ("previous."+table, table)):
                if db.execute(f"SELECT COUNT(*) FROM (SELECT * FROM {left} EXCEPT SELECT * FROM {right})").fetchone()[0]:
                    raise ValueError("Native matching changed non-pair input records")
        for (a,b), (pid,) in zip(pairs, pair_ids):
            match = db.execute("SELECT rows,cols,data FROM matches WHERE pair_id=?", (pid,)).fetchone()
            verified = db.execute("SELECT rows,cols,data FROM two_view_geometries WHERE pair_id=?", (pid,)).fetchone()
            counts = [db.execute("SELECT rows FROM keypoints WHERE image_id=?", (iid,)).fetchone()[0] for iid in (a,b)]
            for record in (match, verified):
                if record is not None and record[0]:
                    n, cols, blob = record
                    if cols != 2 or len(blob) != n*8:
                        raise ValueError("Invalid native point-index layout")
                    indices = np.frombuffer(blob, dtype=np.uint32).reshape(n,2)
                    if not (indices < np.array(counts)).all():
                        raise ValueError("Native matches contain out-of-range point indices")
            rows.append({"images": [images[a], images[b]], "previous_verified": before[pid][0] if before[pid] else 0,
                "predicted": match[0] if match else 0, "verified": verified[0] if verified else 0})
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("The copied match database failed integrity validation")
    result = {"status": "native_matching_verified", "native_match_smoke_passed": any(r["verified"] >= 15 for r in rows),
        "rows": rows, "timings": timings, "non_pair_records_unchanged": True, "all_input_denominator": 1021,
        "source_unchanged": data_identity(source_db)==contract["source_database"], "counted": False}
    atomic_json(args.output/"RESULT.json", result)
    if args.recover and result["native_match_smoke_passed"]:
        root = args.output/"models"
        root.mkdir()
        run("mapping", [binary, "mapper", "--database_path", database, "--image_path", source["image_root"],
            "--input_path", model_path, "--output_path", root, "--Mapper.fix_existing_frames", "1",
            "--Mapper.num_threads", "8", "--Mapper.random_seed", "0", "--default_random_seed", "0",
            "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_principal_point", "0",
            "--Mapper.ba_refine_extra_params", "0", "--Mapper.ba_refine_sensor_from_rig", "0", "--Mapper.max_runtime_seconds", "590"])
        result["models"] = summarize_models(root, images, set(best["remaining_failures"]), {"registered": sorted(registered)}, observations(previous))
        policy_path = Path(__file__).resolve().parents[2]/"05_packages/e26_video_to_target_m3_nollm_v1r23_20260927_installed_candidate/config/METRIC_ALIGNMENT_POLICY.json"
        sys.path.insert(0, str(policy_path.parents[1]/"code/validated_fresh_runtime"))
        from metric_admission import camera_geometry_metrics
        policy = json.loads(policy_path.read_text())["camera_geometry_admission"]
        for row in result["models"]:
            model = pycolmap.Reconstruction(row["path"])
            support = [camera_geometry_metrics(model, im, policy) for im in model.images.values() if im.has_pose]
            row["weak_cameras"] = [r for r in support if not r["passed"]]
            row["registration_and_support_passed"] = row["registered_count"]==1021 and not row["original_registered_frames_lost"] and not row["weak_cameras"]
        result.update(status="native_recovery_evaluated_common_observation_audit_pending", timings=timings,
            geometry_policy=data_identity(policy_path), elapsed_seconds=time.monotonic()-started)
        atomic_json(args.output/"RESULT.json", result)
    print(json.dumps({"status": result["status"], "native_match_smoke_passed": result["native_match_smoke_passed"],
        "pairs": len(rows), "timings": timings, "models": result.get("models", [])}))


if __name__ == "__main__":
    main()
