"""Compare original and vectorized readback of one model through every actual view."""

import argparse
from contextlib import ExitStack
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
import evaluate_pair
from vectorized_gaussian_read import vectorized_gaussian_read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    candidate = (args.pair/"vectorized/model/point_cloud/iteration_12000/point_cloud.ply").resolve()
    original_activate, original_argv = evaluate_pair.activate_reference, sys.argv
    stack = ExitStack()
    traces = []

    def activate(*values, **kwargs):
        metadata = original_activate(*values, **kwargs)
        from scene.gaussian_model import GaussianModel
        original = GaussianModel.load_ply
        trace = stack.enter_context(vectorized_gaussian_read())
        fast = GaussianModel.load_ply

        def load(model, path, *arguments, **options):
            method = fast if Path(path).resolve() == candidate else original
            return method(model, path, *arguments, **options)

        GaussianModel.load_ply = load
        traces.append(trace)
        return metadata

    try:
        evaluate_pair.activate_reference = activate
        sys.argv = [str(Path(evaluate_pair.__file__)), "--pair", str(args.pair), "--control-label", "reference",
            "--candidate-label", "vectorized", "--iterations", "12000", *remaining]
        evaluate_pair.main()
        atomic_json(args.pair/"evaluation/GAUSSIAN_READ_TRACE.json", {"actual_read_trace": traces,
            "sources": [data_identity(Path(__file__)), data_identity(Path(__file__).with_name("vectorized_gaussian_read.py"))], "counted": False})
    finally:
        stack.close()
        evaluate_pair.activate_reference, sys.argv = original_activate, original_argv


if __name__ == "__main__":
    main()
