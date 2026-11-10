"""Exercise exact Gaussian readback and optional reference statistics in one native evaluator."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import activate_reference, atomic_json, data_identity
import evaluate_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "pair", "localization", "reference-run"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--mode", choices=["baseline", "optimized"], required=True)
    parser.add_argument("--folder", required=True)
    args = parser.parse_args()
    metadata = activate_reference(args.runtime)
    import torch
    from utils import loss_utils
    from reference_ssim import ReferenceSSIM
    from vectorized_gaussian_read import vectorized_gaussian_read
    cache = ReferenceSSIM(loss_utils.create_window)
    original_ssim, original_activate, original_argv = loss_utils.ssim, evaluate_pair.activate_reference, sys.argv
    paths = [Path(__file__), Path(__file__).with_name("reference_ssim.py"),
             Path(__file__).with_name("vectorized_gaussian_read.py"), Path(evaluate_pair.__file__)]
    identities = [data_identity(path) for path in paths]
    free, _ = torch.cuda.mem_get_info()
    budget = min(12*1024**3, free-2*1024**3)
    if budget <= 0:
        raise MemoryError("The actual combined evaluator requires its original CUDA headroom")
    torch.cuda.reset_peak_memory_stats()

    def already_active(runtime):
        if Path(runtime).resolve() != args.runtime.resolve():
            raise ValueError("The combined evaluator runtime changed")
        return metadata

    try:
        evaluate_pair.activate_reference = already_active
        if args.mode == "optimized":
            loss_utils.ssim = cache
        sys.argv = [str(Path(evaluate_pair.__file__)), "--runtime", str(args.runtime), "--pair", str(args.pair),
            "--reference-run", str(args.reference_run), "--localization", str(args.localization),
            "--control-label", "baseline", "--candidate-label", "combined", "--iterations", "12000",
            "--evaluation-folder", args.folder, "--input-version", "new_map_admission_v1"]
        before = time.perf_counter()
        with vectorized_gaussian_read(args.mode == "optimized") as reads:
            evaluate_pair.main()
        if identities != [data_identity(path) for path in paths] or torch.cuda.max_memory_reserved() > budget:
            raise ValueError("A combined readback identity or memory protection failed")
        atomic_json(args.pair/args.folder/"COMBINATION.json", {"status": "actual_combined_evaluation_completed_metric_oracle_pending",
            "mode": args.mode, "sources": identities, "native_Gaussian_reads": reads, "reference_cache": cache.receipt(),
            "elapsed_seconds": time.perf_counter()-before, "peak_CUDA_reserved_bytes": torch.cuda.max_memory_reserved(),
            "CUDA_budget_bytes": budget, "training_loss_and_render_formula_unchanged": True, "new_optimization_count": 0})
    finally:
        loss_utils.ssim, evaluate_pair.activate_reference, sys.argv = original_ssim, original_activate, original_argv
        cache.clear()


if __name__ == "__main__":
    main()
