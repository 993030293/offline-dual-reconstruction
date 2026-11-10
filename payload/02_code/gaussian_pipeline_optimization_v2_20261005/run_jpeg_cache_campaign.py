"""Compare three complete raw exports with exact encoded-artifact reuse."""

import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

V1 = Path(__file__).resolve().parents[2]/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "quality-cache", "validation"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    validation = read(args.validation)
    if validation["status"] != "numeric_lifecycle_passed" or not all(validation["checks"].values()):
        raise ValueError("The encoder lifecycle prerequisite did not pass")
    cache_source = Path(__file__).with_name("encoded_jpeg_cache.py")
    if validation["sources"][1] != data_identity(cache_source):
        raise ValueError("The validated encoder adapter changed")
    if shutil.disk_usage(args.output.parent).free < 20*1024**3:
        raise MemoryError("Insufficient disk headroom")
    args.output.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__), cache_source, Path(__file__).with_name("run_initial_with_jpeg_cache.py")]
    source_paths += [V1/name for name in ("run_initial_pipeline.py", "runtime.py", "quality_engine.py",
        "selection_engine.py", "pose_interpolation.py")]
    root = V1.parents[1]
    source_paths.append(root/"02_code/box2_initial_pose_quality_v2_20261002/initial_selection.py")
    native_tools = root/"05_packages/e26_video_to_target_m3_nollm_v1r23_20260927_installed_candidate/code/reference_sources/e26_c2_support/frontend/tools"
    source_paths += [native_tools/name for name in ("camera_motion.py", "color_contract.py", "mcap_audit.py")]
    sources = [data_identity(path) for path in source_paths]
    snapshots = args.output/"source_snapshots"
    snapshots.mkdir()
    for path in source_paths:
        shutil.copyfile(path, snapshots/path.name)
    state = {"status": "running", "mechanism": "Q33_EXACT_JPEG_ARTIFACT_REUSE", "sources": sources,
        "maximum_seconds_per_trial": 180, "maximum_RSS_bytes": 8*1024**3,
        "maximum_output_bytes": 8*1024**3, "required_export_reduction": .1,
        "runs": [], "checks": {}, "counted": False}
    atomic_json(args.output/"CONTRACT.json", state)
    started = time.monotonic()
    def run(name, cached):
        if sources != [data_identity(path) for path in source_paths]:
            raise ValueError("The registered cache source changed")
        command = [sys.executable, str(Path(__file__).with_name("run_initial_with_jpeg_cache.py")),
            "--quality-cache", str(args.quality_cache), "--output", str(args.output/name)]
        if cached:
            command += ["--jpeg-cache", str(args.output/"jpeg_cache")]
        bounded = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(args.output/(name+"_process.json")),
            "--log", str(args.output/(name+".log")), "--maximum-seconds", "180", "--maximum-rss-gib", "8", "--", *command]
        row = {"name": name, "cached": cached, "command": bounded, "status": "running"}
        state["runs"].append(row)
        atomic_json(args.output/"STATE.json", state)
        with (args.output/(name+"_controller.log")).open("x") as log:
            process = subprocess.Popen(bounded, stdout=log, stderr=subprocess.STDOUT)
            row["pid"] = process.pid
            atomic_json(args.output/"STATE.json", state)
            code = process.wait()
        row.update(returncode=code, status="completed" if code == 0 else "failed")
        if code:
            raise RuntimeError("A bounded raw export failed")
        initial = args.output/name/"initial"
        receipt = read(initial/"SELECTION_RECEIPT.json")
        artifact = read(args.output/name/"JPEG_CACHE_RESULT.json")
        row.update(export_seconds=receipt["timings"]["decode_encode_write_seconds"],
            wall_seconds=receipt["elapsed_seconds"], cache=artifact["cache"],
            source_unchanged=artifact["sources_unchanged"])
        if (receipt["raw_frame_count"], receipt["selected_count"], receipt["matching_candidate_pair_count"], receipt["excluded_overlap"]) != (7338, 1002, 11583, 0):
            raise ValueError("The original input denominator or exclusions changed")
        if sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file()) > 8*1024**3:
            raise MemoryError("The registered output budget was exceeded")
        atomic_json(args.output/"STATE.json", state)
        print(json.dumps({key: row[key] for key in ("name", "export_seconds", "wall_seconds")}), flush=True)
    try:
        run("cold_fill", True)
        if state["runs"][0]["cache"]["misses"] != 1002 or state["runs"][0]["cache"]["hits"]:
            raise ValueError("The cold fill unexpectedly reused artifacts")
        for seed in range(3):
            for name in (["cache", "reference"] if seed == 1 else ["reference", "cache"]):
                run(f"pair{seed}_{name}", name == "cache")
        oracle_dir = args.output/"pair0_reference/initial"
        oracle = read(oracle_dir/"INITIAL_FRAME_MANIFEST.json")
        canonical_files = ("frame_decisions.jsonl", "initial_matching_pairs.txt", "EXCLUDED_EVALUATION_IDS.json",
            "RAW_SENSOR_CONTRACT.json")
        exact = True
        for row in state["runs"]:
            initial = args.output/row["name"]/"initial"
            manifest = read(initial/"INITIAL_FRAME_MANIFEST.json")
            exact &= manifest["selected_frames"] == oracle["selected_frames"]
            exact &= manifest["camera_intrinsics"] == oracle["camera_intrinsics"] and manifest["policy"] == oracle["policy"]
            exact &= all((initial/name).read_bytes() == (oracle_dir/name).read_bytes() for name in canonical_files)
            actual_names = sorted(path.name for path in (initial/"images").iterdir())
            exact &= actual_names == sorted(frame["image_name"] for frame in oracle["selected_frames"])
            for frame in manifest["selected_frames"]:
                exact &= data_identity(initial/"images"/frame["image_name"])["sha256"] == frame["image_sha256"]
        pairs = state["runs"][1:]
        reference = [r["export_seconds"] for r in pairs if not r["cached"]]
        candidate = [r["export_seconds"] for r in pairs if r["cached"]]
        reduction = 1-statistics.median(candidate)/statistics.median(reference)
        state["checks"] = {"all7338_decisions_1002_image_bytes_and_11583_pairs_exact": bool(exact),
            "all_six_actual_entries_completed": len(pairs) == 6 and all(r["status"] == "completed" for r in pairs),
            "all_warm_entries_hit1002": all(r["cache"]["hits"] == 1002 and r["cache"]["misses"] == 0
                and r["cache"]["closed"] for r in pairs if r["cached"]),
            "median_whole_export_improves10percent": reduction >= .1,
            "sources_unchanged": all(r["source_unchanged"] for r in state["runs"])}
        state.update(status="stage_guards_passed_combination_pending" if all(state["checks"].values()) else "rejected",
            median_reference_export_seconds=statistics.median(reference), median_cache_export_seconds=statistics.median(candidate),
            median_export_reduction=reduction, elapsed_seconds=time.monotonic()-started,
            cold_fill_seconds=state["runs"][0]["export_seconds"], total_speed_claim=False)
        atomic_json(args.output/"ACCEPTANCE.json", state)
    except BaseException as error:
        state.update(status="failed", error=repr(error), elapsed_seconds=time.monotonic()-started)
        raise
    finally:
        atomic_json(args.output/"STATE.json", state)


if __name__ == "__main__":
    main()
