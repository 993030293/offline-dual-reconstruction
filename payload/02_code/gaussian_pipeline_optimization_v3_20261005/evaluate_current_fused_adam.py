"""Record the owned Linux identity while calling the unchanged all24 evaluator."""

import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/fused_adam_current_001"
sys.path.insert(0, str(V1))
from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-owned-entry", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    expected = V1 / "evaluate_pair.py"
    if args.original_owned_entry.resolve() != expected.resolve() or remaining.count("--pair") != 1:
        raise ValueError("Only the original registered evaluator and unique owned pair are allowed")
    pair = Path(remaining[remaining.index("--pair") + 1])
    pair.resolve().relative_to(OUT.resolve())
    output = pair / "evaluation"
    if output.exists():
        raise FileExistsError("Never overwrite an actual current Adam evaluation")
    sources = [data_identity(Path(__file__)), data_identity(expected)]
    specification = importlib.util.spec_from_file_location("owned_original_fused_evaluator", expected)
    original = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(original)
    original_json, original_argv = original.atomic_json, sys.argv
    import psutil
    process = psutil.Process()

    def publish(path, value):
        if Path(path) == output / "EVALUATION_CONTRACT.json":
            atomic_json(output / "PROCESS_IDENTITY.json", {
                "scope": "WSL", "pid": process.pid, "creation_time": process.create_time()})
        original_json(path, value)

    try:
        original.atomic_json = publish
        sys.argv = [str(expected), *remaining]
        original.main()
        if sources != [data_identity(Path(row["path"])) for row in sources]:
            raise ValueError("An original evaluation source changed")
        atomic_json(output / "REGISTERED_EVALUATOR.json", {"status": "original_all24_evaluator_completed",
            "sources": sources, "original_evaluation_function_changed": False, "new_optimization_count": 0})
    finally:
        original.atomic_json, sys.argv = original_json, original_argv


if __name__ == "__main__":
    main()
