"""Preserve the existing evaluator while validating the declared dataset version."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import activate_reference, atomic_json, data_identity
import evaluate_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "pair", "localization", "reference-run", "input-version"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    args = parser.parse_args()
    pair = Path(args.pair)
    for label in ("baseline", "combined"):
        manifest = json.loads((pair/label/"INPUT_MANIFEST.json").read_text())
        state = json.loads((pair/label/"STATE.json").read_text())
        if manifest["input_version"] != args.input_version or state["status"] != "completed" or state["iteration"] != args.iterations:
            raise ValueError("An actual model version or completed iteration budget differs from this evaluator")
    localization = Path(args.localization)
    if args.input_version == "C212_frozen":
        if localization.resolve() != (Path(args.reference_run)/"05_evaluation/C_augmented_212/H_LOCALIZATION.json").resolve():
            raise ValueError("The preserved C212 evaluator only supports its own exact frozen localization")
        extra = []
    else:
        if json.loads(localization.read_text()).get("input_version") != args.input_version:
            raise ValueError("The explicitly supplied localization belongs to a different map version")
        extra = ["--localization", str(localization)]
    metadata = activate_reference(Path(args.runtime))
    import torch
    from utils import loss_utils
    from reference_ssim import ReferenceSSIM
    from vectorized_gaussian_read import vectorized_gaussian_read
    cache = ReferenceSSIM(loss_utils.create_window)
    original_ssim, original_activate, original_argv = loss_utils.ssim, evaluate_pair.activate_reference, sys.argv
    free, _ = torch.cuda.mem_get_info()
    budget = min(12*1024**3, free-2*1024**3)
    if budget <= 0:
        raise MemoryError("The complete evaluator requires its preserved CUDA headroom")
    torch.cuda.reset_peak_memory_stats()
    try:
        evaluate_pair.activate_reference = lambda runtime: metadata
        loss_utils.ssim = cache
        sys.argv = [str(Path(evaluate_pair.__file__)), "--runtime", args.runtime, "--pair", args.pair,
            "--reference-run", args.reference_run, "--control-label", "baseline", "--candidate-label", "combined",
            "--iterations", str(args.iterations), "--evaluation-folder", "evaluation", "--input-version", args.input_version, *extra]
        before = time.perf_counter()
        with vectorized_gaussian_read(True) as reads:
            evaluate_pair.main()
        if torch.cuda.max_memory_reserved() > budget:
            raise MemoryError("The complete evaluator exceeded its preserved GPU resource budget")
        atomic_json(pair/"evaluation/COMBINATION.json", {"status": "actual_preserved_Q30_Q61_evaluation_completed",
            "native_Gaussian_reads": reads, "reference_cache": cache.receipt(), "elapsed_seconds": time.perf_counter()-before,
            "peak_CUDA_reserved_bytes": torch.cuda.max_memory_reserved(), "CUDA_budget_bytes": budget,
            "source": data_identity(Path(__file__)), "input_version": args.input_version,
            "original_calibrations_localization_and_metrics_unchanged": True, "new_optimization_count": 0})
    finally:
        loss_utils.ssim, evaluate_pair.activate_reference, sys.argv = original_ssim, original_activate, original_argv
        cache.clear()


if __name__ == "__main__":
    main()
