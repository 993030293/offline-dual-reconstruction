"""Verify serial accepted-route compositions against complete frozen output oracles."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import pycolmap

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = Path(__file__).parent
O1 = ROOT/"04_outputs/gaussian_pipeline_optimization_v1_20261005"
O2 = ROOT/"04_outputs/gaussian_pipeline_optimization_v2_20261005"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity
from raw_initial_invariants import compare_initial_outputs
from compare_actual_localization import compare
from native_observation_checks import projection_validity_native, observation_identity_native
from run_metric_refinement import projection_validity, observation_identity


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def mounted(path):
    return "/mnt/c/"+str(Path(path).resolve()).replace("\\", "/")[3:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-cold", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), V2/"run_effective_initial.py", V2/"evaluate_effective_combination.py",
        V2/"localize_with_byte_sift.py", V2/"native_observation_checks.py"]
    identities = [data_identity(p) for p in sources]
    state = {"status": "running", "sources": identities, "runs": [], "checks": {},
        "one_serial_composition_pass_not_new_benchmark": True, "new_optimization_count": 0,
        "existing_exact_map_reused_no_new_SfM_or_LIO": True,
        "common_raw_quality_cache_already_warm": True}
    atomic_json(args.output/"PROTOCOL.json", state)

    def run(name, command, linux=False):
        prefix = ["wsl", "-d", "E26-C2-LIV-20260907-134802", "--"] if linux else []
        python = "/opt/e26_thin_surface_v1/venv/bin/python" if linux else sys.executable
        cast = mounted if linux else str
        monitored = prefix+[python, cast(V1/"run_bounded_task.py"), "--receipt", cast(args.output/(name+"_process.json")),
            "--log", cast(args.output/(name+".log")), "--maximum-seconds", "600", "--maximum-rss-gib", "16", "--"]+command
        row = {"name": name, "status": "running", "command": monitored}
        state["runs"].append(row)
        atomic_json(args.output/"STATE.json", state)
        with (args.output/(name+"_controller.log")).open("x", encoding="utf-8") as log:
            process = subprocess.Popen(monitored, stdout=log, stderr=subprocess.STDOUT)
            row["pid"] = process.pid
            atomic_json(args.output/"STATE.json", state)
            code = process.wait()
        row.update(status="completed" if code == 0 else "failed", returncode=code)
        atomic_json(args.output/"STATE.json", state)
        print(json.dumps({"name": name, "status": row["status"]}), flush=True)
        if code:
            raise RuntimeError("An actual final composition failed: "+name)

    try:
        raw = {"cold_JPEG_artifacts": args.raw_cold}
        cache = args.output/"JPEG_cache"
        before = time.perf_counter()
        shutil.copytree(O2/"q33_campaign_001/jpeg_cache", cache)
        state["JPEG_cache_copy"] = {"seconds": time.perf_counter()-before,
            "bytes": sum(p.stat().st_size for p in cache.rglob("*") if p.is_file()), "preexisting_cache_not_modified": True}
        for mode in ("repeat", "mapped-repair"):
            target = args.output/("raw_"+mode)
            command = [sys.executable, str(V2/"run_effective_initial.py"), "--mode", mode,
                "--quality-cache", str(O1/"p1_shared_quality_cache"), "--output", str(target)]
            if mode == "repeat":
                command += ["--jpeg-cache", str(cache)]
            run("raw_"+mode, command)
            raw[mode] = target
        raw_results = {name: compare_initial_outputs(O1/"p1_raw_combined_001", folder/"initial") for name, folder in raw.items()}
        atomic_json(args.output/"RAW_INVARIANTS.json", raw_results)
        state["checks"]["all_three_raw_modes_all7338_1002_11583_exact"] = all(r["all_protected"] for r in raw_results.values())
        reference_cache, feature_cache = args.output/"reference_cache", args.output/"feature_cache"
        shutil.copytree(O2/"q47_reference_cache_001", reference_cache)
        shutil.copytree(O2/"q48_query_cache_002", feature_cache)
        loc = args.output/"localization"
        run("localization", [sys.executable, str(V2/"localize_with_byte_sift.py"), "--descriptor-mode", "bytes",
            "--output", str(loc), "--reference-mode", "cache", "--reference-cache-root", str(reference_cache),
            "--feature-mode", "cache", "--feature-cache-root", str(feature_cache), "--export", str(O1/"new_map_export_003"),
            "--database", str(O1/"p23_track_completion_001/database.db"), "--initial-frames", str(args.raw_cold/"initial"),
            "--reference-run", str(ROOT/"04_outputs/box2_lio_guided_sfm_v1_20261002T014330")])
        relative = Path("reference_entry/query_entry/entry")
        left = O2/"q51_campaign_002/trial0/baseline"/relative
        right = loc/relative
        comparison = compare(left, right, left/"NEIGHBOR_TRACE.json", right/"NEIGHBOR_TRACE.json")
        atomic_json(args.output/"LOCALIZATION_INVARIANTS.json", comparison)
        state["checks"]["all24_poses_pixels_features_48_matches_exact"] = comparison["all_protected"]
        model = pycolmap.Reconstruction(O1/"new_map_export_003/training_opencv_model")
        native = projection_validity_native(model)
        qa = {"projection_exact": native == projection_validity(model),
            "all959120_observations_valid": native["valid"] and native["observation_count"] == 959120,
            "identity_order_exact": observation_identity_native(model) == observation_identity(model)}
        atomic_json(args.output/"MAP_QA.json", {"checks": qa, "native": native, "fresh_coordinate_table": True})
        state["checks"]["fixed_map_complete_native_QA_exact"] = all(qa.values())
        for seed in range(3):
            pair = O2/f"camera_quality_training_001/seed{seed}"
            run(f"evaluation_seed{seed}", ["/opt/e26_thin_surface_v1/venv/bin/python", mounted(V2/"evaluate_effective_combination.py"),
                "--runtime", mounted(ROOT/"04_outputs/box2_3dgs_display_v1_20260930"), "--pair", mounted(pair),
                "--localization", mounted(O1/"p27_newmap_localization_lazy_serial_001/H_LOCALIZATION.json"),
                "--reference-run", mounted(ROOT/"04_outputs/box2_lio_guided_sfm_v1_20261002T014330"),
                "--mode", "optimized", "--folder", "evaluation_effective_combination_001"], linux=True)
            old, new = read(pair/"evaluation/EVALUATION.json"), read(pair/"evaluation_effective_combination_001/EVALUATION.json")
            exact = all(old[key] == new[key] for key in ("results", "candidate_minus_control", "training_iterations_by_label",
                "evaluation_protocol", "all_frames_included_in_denominator", "development_metric_guard_passed"))
            image_exact = (pair/"evaluation/SIX_VIEW_COMPARISON.jpg").read_bytes() == (pair/"evaluation_effective_combination_001/SIX_VIEW_COMPARISON.jpg").read_bytes()
            state["checks"][f"seed{seed}_all24_metrics_and_six_view_pixels_exact"] = exact and image_exact
            atomic_json(args.output/"STATE.json", state)
            if not exact or not image_exact:
                raise ValueError("The composed Gaussian reader and SSIM evaluator changed an original result")
        state["checks"]["sources_unchanged"] = identities == [data_identity(p) for p in sources]
        if not all(state["checks"].values()):
            raise ValueError("A final complete output oracle failed")
        state["status"] = "raw_localization_map_QA_and_three_seed_evaluation_combinations_passed"
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        atomic_json(args.output/"STATE.json", state)
        raise
    atomic_json(args.output/"STATE.json", state)
    atomic_json(args.output/"RESULT.json", state)
    print(json.dumps({"status": state["status"], "checks": state["checks"]}), flush=True)


if __name__ == "__main__":
    main()
