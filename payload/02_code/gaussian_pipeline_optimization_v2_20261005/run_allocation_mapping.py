"""Compare fresh one-stage and staged SfM on exactly the same final evidence."""

import argparse
import importlib.util
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
import pycolmap

V1 = Path(__file__).resolve().parents[1] / "gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity
from sqlite_evidence import compare_database_content


def model_summary(folder, required, support_function, policy):
    paths = [folder] if (folder/"images.bin").is_file() else sorted(p for p in folder.iterdir() if p.is_dir() and (p/"images.bin").is_file())
    summaries = []
    for path in paths:
        model = pycolmap.Reconstruction(path)
        registered = {image.name for image in model.images.values() if image.has_pose}
        if not registered <= required:
            raise ValueError("Mapper registered a frame outside the current-stage input list")
        rows = [support_function(model, image, policy) for image in model.images.values() if image.has_pose]
        tracks = [point.track.length() for point in model.points3D.values()]
        summaries.append({"path": str(path), "registered_count": len(registered), "registered_names": sorted(registered),
                          "remaining_failures": sorted(required-registered), "points": model.num_points3D(),
                          "observations": model.compute_num_observations(),
                          "mean_reprojection_error_px": float(model.compute_mean_reprojection_error()),
                          "track_length_p10_median_p90": np.quantile(tracks, [.1, .5, .9]).tolist() if tracks else [],
                          "geometry_support_pass_names": sorted(row["image_name"] for row in rows if row["passed"]),
                          "geometry_support_fail_rows": [row for row in rows if not row["passed"]],
                          "coordinate_scale": "fresh_visual_arbitrary_scale_before_metric_BA"})
    return sorted(summaries, key=lambda item: (-item["registered_count"], item["path"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--source-database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-completed-control", type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[2]
    binary = project/"05_envs/colmap-4.1.1-cuda/bin/colmap.exe"
    if shutil.disk_usage(args.output).free < 20*1024**3:
        raise RuntimeError("The experiment must leave20GiB free")
    allocation = json.loads(args.allocation.read_text(encoding="utf-8"))
    metadata = json.loads((args.allocation.parent.parent/"RESULT.json").read_text(encoding="utf-8"))
    if not all(metadata["checks"].values()) or allocation["case"]["pool_size"] != 1002 or allocation["case"]["initial_count"] != 651:
        raise ValueError("The registered651/1002 coarse-only allocation did not pass its scoped guards")
    final_names = set(allocation["final_names"])
    initial_names = {row["image_name"] for row in allocation["initial"]}
    if final_names != initial_names | {row["image_name"] for row in allocation["remaining"]}:
        raise ValueError("Allocation does not exactly partition the frozen final pool")
    manifest_path = args.initial/"INITIAL_FRAME_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if final_names != {row["image_name"] for row in manifest["selected_frames"]}:
        raise ValueError("The final frame denominator changed")
    image_root = (args.initial/"images").resolve()
    images = []
    for row in manifest["selected_frames"]:
        identity = data_identity(image_root/row["image_name"])
        if identity["sha256"] != row["image_sha256"]:
            raise ValueError("Original raw image artifact changed")
        images.append(identity)
    with sqlite3.connect(f"file:{args.source_database.resolve().as_posix()}?mode=ro", uri=True) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("Source feature database integrity failed")
        if final_names != {row[0] for row in connection.execute("SELECT name FROM images")}:
            raise ValueError("Feature database input denominator differs")
        database_counts = {table: connection.execute("SELECT count(*) FROM "+table).fetchone()[0]
                           for table in ("images", "keypoints", "descriptors", "matches", "two_view_geometries")}
    package = project/"05_packages/e26_video_to_target_m3_nollm_v1r23_20260927_installed_candidate"
    policy_path = package/"config/METRIC_ALIGNMENT_POLICY.json"
    support_source = package/"code/validated_fresh_runtime/metric_admission.py"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))["camera_geometry_admission"]
    spec = importlib.util.spec_from_file_location("frozen_camera_support", support_source)
    support_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(support_module)
    sources = [data_identity(path) for path in (Path(__file__), V1/"runtime.py", V1/"run_bounded_task.py", binary, support_source,
                                               Path(__file__).with_name("sqlite_evidence.py"))]
    source_database_identity = data_identity(args.source_database)
    contract = {"mechanism": "Q34_INITIAL_AND_AUGMENTATION_ALLOCATION", "status": "registered_before_mapping",
                "allocation": data_identity(args.allocation), "manifest": data_identity(manifest_path),
                "source_database": source_database_identity, "database_counts": database_counts,
                "input_images": images, "sources": sources, "support_policy": policy, "policy_source": data_identity(policy_path),
                "all_final_frames": sorted(final_names), "staged_initial_frames": sorted(initial_names),
                "cases": ["one_shot1002", "staged651_then1002"], "independent_processes": True,
                "shared_feature_and_matching_cost": "excluded equally; reused fixed evidence isolates mapping schedule",
                "source_models_or_SfM_poses_used": False, "old_metric_alignment_used": False, "GT_used": False,
                "maximum_seconds_per_mapper": 1200, "maximum_group_seconds": 3600, "maximum_RSS_bytes": 32*1024**3,
                "maximum_output_bytes": 10*1024**3, "CPU_threads": 8,
                "initialization": "fresh visual mapper separately for each case",
                "continuation": "mapper with triangulation and existing frames variable; fixed intrinsics",
                "seed": 0, "diagnostic_not_three_pair_acceptance": True, "counted": False,
                "reused_completed_control": str(args.reuse_completed_control) if args.reuse_completed_control else None,
                "database_guard": "read transactions include committed WAL; all schemas, rows and blobs exactly equal; physical transaction counters recorded separately"}
    atomic_json(args.output/"CONTRACT.json", contract)
    started = time.monotonic()
    state = {"status": "running", "controller_pid": os.getpid(), "stages": [], "counted": False}
    atomic_json(args.output/"STATE.json", state)
    results = {}

    def run_stage(case_folder, label, names, database, input_model=None):
        if time.monotonic()-started > 3600:
            raise RuntimeError("Whole group budget exceeded")
        image_list = case_folder/(label+"_images.txt")
        image_list.write_text("\n".join(sorted(names))+"\n", encoding="utf-8")
        model_root = case_folder/label
        model_root.mkdir()
        command = [str(binary), "mapper", "--database_path", str(database), "--image_path", str(image_root),
                   "--output_path", str(model_root), "--Mapper.image_list_path", str(image_list),
                   "--Mapper.num_threads", "8", "--Mapper.random_seed", "0", "--default_random_seed", "0",
                   "--Mapper.ba_refine_focal_length", "0", "--Mapper.ba_refine_principal_point", "0",
                   "--Mapper.ba_refine_extra_params", "0", "--Mapper.ba_refine_sensor_from_rig", "0",
                   "--Mapper.fix_existing_frames", "0", "--Mapper.max_runtime_seconds", "1190"]
        if input_model:
            command.extend(["--input_path", str(input_model)])
        receipt, log = case_folder/(label+"_process.json"), case_folder/(label+".log")
        wrapper = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(receipt), "--log", str(log),
                   "--maximum-seconds", "1200", "--maximum-rss-gib", "32", "--", *command]
        stage = {"case": case_folder.name, "label": label, "input_count": len(names), "status": "running", "receipt": str(receipt)}
        state["stages"].append(stage)
        atomic_json(args.output/"STATE.json", state)
        completed = subprocess.run(wrapper)
        measured = json.loads(receipt.read_text(encoding="utf-8"))
        stage.update(status=measured["status"], seconds=measured["elapsed_seconds"], peak_tree_RSS_bytes=measured["peak_tree_RSS_bytes"])
        atomic_json(args.output/"STATE.json", state)
        if completed.returncode:
            raise RuntimeError("A bounded mapping child failed; keep its failure and outputs")
        summaries = model_summary(model_root, names, support_module.camera_geometry_metrics, policy)
        atomic_json(case_folder/(label+"_MODELS.json"), {"models": summaries, "required_count": len(names)})
        if not summaries:
            raise ValueError("No registered model was produced")
        stage.update(best_registered=summaries[0]["registered_count"], failed_count=len(summaries[0]["remaining_failures"]))
        atomic_json(args.output/"STATE.json", state)
        print(json.dumps(stage), flush=True)
        return summaries[0], measured["elapsed_seconds"]

    try:
        for name in contract["cases"]:
            folder = args.output/name
            folder.mkdir()
            if name == "one_shot1002" and args.reuse_completed_control:
                previous = args.reuse_completed_control.resolve()
                old_contract = json.loads((previous/"CONTRACT.json").read_text(encoding="utf-8"))
                old_state = json.loads((previous/"STATE.json").read_text(encoding="utf-8"))
                old_process = json.loads((previous/name/"initial_process.json").read_text(encoding="utf-8"))
                if (old_state["error"] != "ValueError('Read-only mapping unexpectedly changed its database evidence')"
                    or old_process["status"] != "completed" or old_process["returncode"] != 0):
                    raise ValueError("Reuse is limited to the completed control stopped by the known counter verifier")
                for field in ("allocation", "manifest", "source_database", "input_images", "all_final_frames", "staged_initial_frames", "support_policy", "policy_source"):
                    if old_contract[field] != contract[field]:
                        raise ValueError("Completed control input or policy identity changed: "+field)
                for source in old_contract["sources"]:
                    original_path = Path(source["path"])
                    verified_path = previous/"run_allocation_mapping_snapshot.py" if original_path.name == Path(__file__).name else original_path
                    if data_identity(verified_path)["sha256"] != source["sha256"]:
                        raise ValueError("Completed control source identity changed")
                old_images = (previous/name/"initial_images.txt").read_text(encoding="utf-8").splitlines()
                if set(old_images) != final_names or len(old_images) != len(final_names):
                    raise ValueError("Completed control did not receive all fixed input names")
                database_check = compare_database_content(args.source_database, previous/name/"database.db")
                summaries = model_summary(previous/name/"initial", final_names, support_module.camera_geometry_metrics, policy)
                old_summaries = json.loads((previous/name/"initial_MODELS.json").read_text(encoding="utf-8"))["models"]
                if summaries != old_summaries:
                    raise ValueError("Completed control model/support readback changed")
                results[name] = {"final": summaries[0], "mapping_seconds": old_process["elapsed_seconds"],
                                 "reused_after_verified_counter_only_failure": True, "reused_from": str(previous),
                                 "database_evidence": database_check, "database_copy_seconds": None}
                atomic_json(folder/"RESULT.json", results[name])
                state["stages"].append({"case": name, "status": "reused_completed_control_verified", "seconds": old_process["elapsed_seconds"],
                                        "best_registered": summaries[0]["registered_count"], "reused_from": str(previous)})
                atomic_json(args.output/"STATE.json", state)
                continue
            database = folder/"database.db"
            copy_started = time.monotonic()
            shutil.copyfile(args.source_database, database)
            copy_seconds = time.monotonic()-copy_started
            if data_identity(database)["sha256"] != source_database_identity["sha256"]:
                raise ValueError("Isolated database copy differs")
            compare_database_content(args.source_database, database)
            if name == "one_shot1002":
                final, seconds = run_stage(folder, "initial", final_names, database)
                results[name] = {"final": final, "mapping_seconds": seconds, "database_copy_seconds": copy_seconds}
            else:
                initial, initial_seconds = run_stage(folder, "initial", initial_names, database)
                final, continuation_seconds = run_stage(folder, "continuation", final_names, database, initial["path"])
                results[name] = {"initial": initial, "final": final, "mapping_seconds": initial_seconds+continuation_seconds,
                                 "initial_seconds": initial_seconds, "continuation_seconds": continuation_seconds,
                                 "initial_registered_lost": sorted(set(initial["registered_names"])-set(final["registered_names"])),
                                 "database_copy_seconds": copy_seconds}
            results[name]["database_evidence"] = compare_database_content(args.source_database, database)
            atomic_json(folder/"RESULT.json", results[name])
            if sum(path.stat().st_size for path in args.output.rglob("*") if path.is_file()) > 10*1024**3:
                raise RuntimeError("Experiment output exceeded its disk budget")
        control, staged = results["one_shot1002"], results["staged651_then1002"]
        missing = sorted(set(control["final"]["registered_names"])-set(staged["final"]["registered_names"]))
        lost_support = sorted(set(control["final"]["geometry_support_pass_names"])-set(staged["final"]["geometry_support_pass_names"]))
        unchanged = all(data_identity(Path(row["path"])) == row for row in sources)
        checks = {"all1002_final_input_frames_retained": True, "no_control_registered_camera_lost": not missing,
                  "no_control_supported_camera_lost": not lost_support,
                  "no_staged_initial_camera_lost": not staged["initial_registered_lost"],
                  "mapping_time_improves10percent": staged["mapping_seconds"] <= .9*control["mapping_seconds"],
                  "source_database_unchanged": data_identity(args.source_database) == source_database_identity,
                  "sources_unchanged": unchanged}
        result = {"status": "diagnostic_complete", "checks": checks, "results": results,
                  "missing_control_registered_names": missing, "lost_control_supported_names": lost_support,
                  "mapping_time_reduction": 1-staged["mapping_seconds"]/control["mapping_seconds"],
                  "elapsed_seconds": time.monotonic()-started, "counted": False,
                  "boundary": "One fixed-evidence diagnostic pair; no new-frame benefit, metric precision, or3DGS quality claim"}
        atomic_json(args.output/"RESULT.json", result)
        state.update(status="diagnostic_complete", checks=checks, elapsed_seconds=result["elapsed_seconds"])
        atomic_json(args.output/"STATE.json", state)
        print(json.dumps({"status": result["status"], "checks": checks, "seconds": result["elapsed_seconds"]}), flush=True)
    except BaseException as error:
        state.update(status="failed", error=repr(error), traceback=traceback.format_exc(), elapsed_seconds=time.monotonic()-started)
        atomic_json(args.output/"STATE.json", state)
        raise


if __name__ == "__main__":
    main()
