"""Expose only locally verified round-three stages with explicit immutable inputs."""

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["qa", "archive", "localization"], required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--model", type=Path, default=ROOT/"04_outputs/gaussian_pipeline_optimization_v1_20261005/new_map_export_003/training_opencv_model")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to(ROOT/"04_outputs/gaussian_pipeline_optimization_v3_20261005"):
        raise ValueError("The round-three entry requires a new owned versioned output")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.stage == "localization":
        import json
        import subprocess
        config = json.loads((ROOT/"research/2026-10-05-gaussian-optimization-round2/configs/EFFECTIVE_ROUTE.json").read_text())
        command = [sys.executable, str(Path(__file__).with_name("localize_integer_sift.py")),
            "--matching-mode", "integer", "--output", str(args.output.resolve()/"entry"), "--reference-mode", "cache",
            "--reference-cache-root", str(ROOT/config["reference_cache"]), "--feature-mode", "cache",
            "--feature-cache-root", str(ROOT/config["feature_cache"])]
        for key in ("export", "database", "initial_frames", "reference_run"):
            command += ["--"+key.replace("_", "-"), config[key]]
        atomic_json(args.output/"CONTRACT.json", {"source": data_identity(Path(__file__)), "command": command,
            "mechanism": "R90_BATCHED_EXACT_INTEGER_SIFT_DISTANCE", "counted": False})
        subprocess.run(command, check=True, cwd=ROOT)
        return
    if args.stage == "archive":
        import json
        from zstd_checkpoint import encode_checkpoint
        if args.checkpoint is None or json.loads(args.checkpoint.with_suffix(".json").read_text())["status"] != "complete":
            raise ValueError("Archival requires a complete native checkpoint and its original receipt")
        receipt = encode_checkpoint(args.checkpoint, args.output/"checkpoint.archive")
        atomic_json(args.output/"ARCHIVE.json", receipt)
        print({"status": "complete_native_zstd_transport_created", "actual_native_restore_required": True})
        return
    from binary_map_qa import projection_validity_binary
    inputs = sorted(args.model.glob("*.bin"))
    if not all((args.model/name).is_file() for name in ("cameras.bin", "images.bin", "points3D.bin")):
        raise ValueError("The complete native immutable model inputs are missing")
    paths = [Path(__file__), Path(__file__).with_name("binary_map_qa.py")]
    sources, identity = [data_identity(path) for path in paths], [data_identity(path) for path in inputs]
    atomic_json(args.output/"CONTRACT.json", {"sources": sources, "input": identity,
        "mechanism": "R85_NATIVE_BINARY_OBSERVATION_ARRAY_QA", "complete_native_files": True,
        "in_memory_BA_QA_supported": False, "independent_GT_accuracy_claim": False, "counted": False})
    before = time.perf_counter()
    result = projection_validity_binary(args.model)
    elapsed = time.perf_counter()-before
    if not result["valid"] or result["observation_count"] != 959120:
        atomic_json(args.output/"QA.json", {"status": "failed_original_complete_projection_guard", "result": result})
        raise ValueError("The actual fixed-map QA failed its original denominator or geometry guard")
    unchanged = identity == [data_identity(path) for path in inputs] and sources == [data_identity(path) for path in paths]
    if not unchanged:
        raise ValueError("A read-only native QA input or implementation changed during the entry")
    atomic_json(args.output/"QA.json", {"status": "full_native_binary_QA_passed", "result": result,
        "algorithm_seconds": elapsed, "fresh_disk_reads_and_coordinates": True, "sources_inputs_intact": unchanged,
        "mechanism": "R85_NATIVE_BINARY_OBSERVATION_ARRAY_QA", "new_map_or_quality_gain_claim": False})
    print({"status": "full_native_binary_QA_passed", "observations": result["observation_count"], "seconds": elapsed})


if __name__ == "__main__":
    main()
