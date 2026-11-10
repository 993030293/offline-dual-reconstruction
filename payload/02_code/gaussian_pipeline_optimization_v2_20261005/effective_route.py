"""Provide explicit stage commands for the verified fixed-input engineering route."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = Path(__file__).parent
sys.path.insert(0, str(V1))
from runtime import atomic_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def mounted(path):
    value = str(Path(path).resolve()).replace("\\", "/")
    if value[1:3] != ":/" or value[0].lower() != "c":
        raise ValueError("The verified local route requires its existing C drive workspace")
    return "/mnt/c/"+value[3:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"research/2026-10-05-gaussian-optimization-round2/configs/EFFECTIVE_ROUTE.json")
    parser.add_argument("--stage", choices=["preflight", "initial", "localization", "qa", "train", "evaluate", "archive", "preview"], required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--initial-mode", choices=["cold", "repeat", "mapped-repair"], default="cold")
    parser.add_argument("--jpeg-cache", type=Path)
    parser.add_argument("--initial-frames", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", choices=[0, 1, 2], type=int, default=0)
    args = parser.parse_args()
    config = read(args.config)
    if (config["schema_version"], config["raw_count"], config["initial_count"], config["training_count"],
            config["development_evaluation_count"], config["iterations"]) != (1, 7338, 1002, 1016, 24, 12000) or len(set(config["mechanisms"])) != 20:
        raise ValueError("This entry only covers its verified20 fixed-input contract")
    paths = {name: ROOT/config[name] for name in ("camera_config", "export", "dataset", "runtime", "reference_run", "database",
        "initial_frames", "quality_cache", "reference_cache", "feature_cache", "gsplat_build", "localization", "pair", "preview_start")}
    if not all(path.exists() for path in paths.values()):
        raise FileNotFoundError("A verified stage input or installed runtime path is missing")
    if args.stage == "preflight":
        initial, localization = read(paths["initial_frames"]/"INITIAL_FRAME_MANIFEST.json"), read(paths["localization"])
        if len(initial["selected_frames"]) != 1002 or len(localization["rows"]) != 24 or not all(row["status"] == "localized" for row in localization["rows"]):
            raise ValueError("The explicit initial or development denominator changed")
        print(json.dumps({"status": "verified_entry_inputs_available", "mechanisms": config["mechanisms"],
            "initial": 1002, "training": 1016, "development": 24, "full_new_SfM_not_run": True, "preview_url": config["preview_url"]}))
        return
    if args.stage == "preview":
        subprocess.run(["powershell", "-NoProfile", "-File", str(paths["preview_start"])], check=True)
        return
    if args.output is None or args.output.exists() or not args.output.resolve().is_relative_to(ROOT/"04_outputs"):
        raise ValueError("Every mutable stage requires a new owned output under04_outputs")
    if args.stage == "archive":
        from checkpoint_archive import encode_checkpoint
        if args.checkpoint is None or read(args.checkpoint.with_suffix(".json"))["status"] != "complete":
            raise ValueError("Archival requires a real complete native checkpoint and its receipt")
        args.output.mkdir(parents=True, exist_ok=False)
        receipt = encode_checkpoint(args.checkpoint, args.output/"checkpoint.ptgz")
        atomic_json(args.output/"ARCHIVE.json", receipt)
        print(json.dumps({"status": "lossless_transport_archive_created_native_restore_still_required"}))
        return
    if args.stage == "qa":
        import pycolmap
        from native_observation_checks import projection_validity_native, observation_identity_native
        model = pycolmap.Reconstruction(paths["export"]/"training_opencv_model")
        result = projection_validity_native(model)
        if not result["valid"] or model.num_reg_images() != 1016 or result["observation_count"] != 959120:
            raise ValueError("The complete fixed-map geometry guard failed")
        identity = observation_identity_native(model)
        args.output.mkdir(parents=True, exist_ok=False)
        atomic_json(args.output/"QA.json", {"status": "full_native_projection_QA_passed", "result": result,
            "identity_count": len(identity), "fresh_coordinate_table": True, "independent_accuracy_claim": False})
        print(json.dumps({"status": "full_native_projection_QA_passed", "observations": len(identity)}))
        return
    command = [sys.executable]
    if args.stage == "initial":
        command += [str(V2/"run_effective_initial.py"), "--mode", args.initial_mode, "--quality-cache", str(paths["quality_cache"]), "--output", str(args.output)]
        if args.jpeg_cache:
            command += ["--jpeg-cache", str(args.jpeg_cache)]
    elif args.stage == "localization":
        command += [str(V2/"localize_with_byte_sift.py"), "--descriptor-mode", "bytes", "--output", str(args.output),
            "--reference-mode", "cache", "--reference-cache-root", str(paths["reference_cache"]), "--feature-mode", "cache",
            "--feature-cache-root", str(paths["feature_cache"])]
        for name in ("export", "database", "reference_run"):
            command += ["--"+name.replace("_", "-"), str(paths[name])]
        command += ["--initial-frames", str(args.initial_frames or paths["initial_frames"])]
    else:
        prefix = ["wsl", "-d", config["WSL_distribution"], "--"]
        command = [config["WSL_python"]]
        if args.stage == "train":
            command += [mounted(V2/"train_camera_execution.py"), "--camera-config", mounted(paths["camera_config"]),
                "--runtime", mounted(paths["runtime"]), "--dataset", mounted(paths["dataset"]), "--output", mounted(args.output),
                "--projection", "full", "--seed", str(args.seed), "--iterations", "12000", "--maximum-seconds", "1800",
                "--input-version", config["input_version"], "--expected-image-count", "1016", "--gsplat-build", mounted(paths["gsplat_build"]),
                "--vectorized-ply", "--verify-native-ply"]
        else:
            if args.output.parent.resolve() != paths["pair"].resolve() or not args.output.name.startswith("evaluation"):
                raise ValueError("The exact evaluator writes a unique evaluation child of its fixed pair")
            command += [mounted(V2/"evaluate_effective_combination.py"), "--runtime", mounted(paths["runtime"]),
                "--pair", mounted(paths["pair"]), "--localization", mounted(paths["localization"]), "--reference-run", mounted(paths["reference_run"]),
                "--mode", "optimized", "--folder", args.output.name]
        monitored = [config["WSL_python"], mounted(V1/"run_bounded_task.py"), "--receipt", mounted(args.output.with_name(args.output.name+"_process.json")),
            "--log", mounted(args.output.with_name(args.output.name+".log")), "--maximum-seconds", "1900", "--maximum-rss-gib", "32", "--"]+command
        subprocess.run(prefix+monitored, check=True)
        return
    monitored = [sys.executable, str(V1/"run_bounded_task.py"), "--receipt", str(args.output.with_name(args.output.name+"_process.json")),
        "--log", str(args.output.with_name(args.output.name+".log")), "--maximum-seconds", "900", "--maximum-rss-gib", "8", "--"]+command
    subprocess.run(monitored, check=True)


if __name__ == "__main__":
    main()
