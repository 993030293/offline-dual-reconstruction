"""Execute the existing complete evaluator with only final artifact conversion adapted."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import atomic_json, data_identity
import evaluate_dag_models
from device_artifact_picture import device_artifact_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("native", "device"), required=True)
    for name in ("runtime", "pair", "localization", "reference-run", "input-version"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    args = parser.parse_args()
    paths = [Path(__file__), Path(__file__).with_name("device_artifact_picture.py"),
        Path(evaluate_dag_models.__file__), V1/"evaluate_pair.py", V2/"reference_ssim.py", V2/"vectorized_gaussian_read.py"]
    sources = [data_identity(path) for path in paths]
    old = sys.argv
    try:
        with device_artifact_evaluation(args.mode == "device") as trace:
            sys.argv = [str(Path(evaluate_dag_models.__file__)), "--runtime", args.runtime, "--pair", args.pair,
                "--localization", args.localization, "--reference-run", args.reference_run,
                "--input-version", args.input_version, "--iterations", str(args.iterations)]
            evaluate_dag_models.main()
        if sources != [data_identity(path) for path in paths]:
            raise ValueError("The final display artifact source changed during the full evaluator")
        atomic_json(Path(args.pair)/"evaluation/ARTIFACTS.json", {"status": "complete_evaluation_artifact_protection_pending",
            "mode": args.mode, "sources": sources, "trace": trace,
            "complete_artifact_conversion_seconds_including_source_setup": trace["composition_seconds"]+sum(c["seconds"] for c in trace["calls"]),
            "actual_device_to_host_tensor_bytes": sum(c["device_to_host_tensor_bytes"] for c in trace["calls"]),
            "original_float_tensor_bytes": sum(c["original_float_tensor_bytes"] for c in trace["calls"]),
            "training_or_rendering_math_changed": False, "counted": False})
    finally:
        sys.argv = old


if __name__ == "__main__":
    main()
