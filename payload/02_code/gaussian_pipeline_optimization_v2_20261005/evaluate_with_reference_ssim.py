"""Profile evaluation-only reference reuse inside the actual frozen evaluator."""

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import activate_reference, atomic_json, data_identity
import evaluate_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["baseline", "cached"], required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--pair", type=Path, required=True)
    parser.add_argument("--localization", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--folder", required=True)
    args = parser.parse_args()
    metadata = activate_reference(args.runtime)
    import torch
    import psutil
    from utils import loss_utils
    from reference_ssim import ReferenceSSIM
    native = loss_utils.ssim
    cache = ReferenceSSIM(loss_utils.create_window)
    torch.cuda.reset_peak_memory_stats()
    free, _ = torch.cuda.mem_get_info()
    budget = min(12*1024**3, free-2*1024**3)
    if budget <= 0:
        raise MemoryError("Insufficient CUDA headroom for the registered evaluation")
    calls = []
    def measured(a, b, *values, **kwargs):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = (native if args.mode == "baseline" else cache)(a, b, *values, **kwargs)
        torch.cuda.synchronize()
        calls.append({"call": len(calls), "shape": list(a.shape), "seconds": time.perf_counter()-started,
                      "value": float(result)})
        if torch.cuda.max_memory_reserved() > budget:
            raise MemoryError("Evaluation exceeded the frozen CUDA budget")
        return result
    original_activate = evaluate_pair.activate_reference
    original_argv = sys.argv
    def already_active(runtime):
        if Path(runtime).resolve() != args.runtime.resolve():
            raise ValueError("The evaluator attempted to change an already fixed runtime")
        return metadata
    started = time.perf_counter()
    sources = [Path(__file__), Path(__file__).with_name("reference_ssim.py"),
               Path(evaluate_pair.__file__), Path(loss_utils.__file__)]
    identities = [data_identity(path) for path in sources]
    try:
        evaluate_pair.activate_reference = already_active
        loss_utils.ssim = measured
        sys.argv = [str(Path(evaluate_pair.__file__)), "--runtime", str(args.runtime),
            "--reference-run", str(args.reference_run), "--pair", str(args.pair),
            "--control-label", "reference", "--candidate-label", "combined",
            "--control-iterations", "10000", "--candidate-iterations", "12000",
            "--evaluation-folder", args.folder, "--localization", str(args.localization),
            "--input-version", "new_map_admission_v1"]
        evaluate_pair.main()
        if identities != [data_identity(path) for path in sources]:
            raise ValueError("A frozen evaluator source changed during the run")
        receipt = {"status": "actual_evaluation_completed", "mode": args.mode, "sources": identities,
            "ssim_calls": calls, "ssim_seconds_including_cold_statistics": sum(c["seconds"] for c in calls),
            "elapsed_seconds": time.perf_counter()-started, "cache": cache.receipt(),
            "peak_CUDA_reserved_bytes": torch.cuda.max_memory_reserved(), "CUDA_budget_bytes": budget,
            "RSS_after_evaluation_bytes": psutil.Process().memory_info().rss,
            "training_loss_changed": False, "rendering_changed": False, "counted": False}
        atomic_json(args.pair/args.folder/"SSIM_PROFILE.json", receipt)
    finally:
        loss_utils.ssim = native
        evaluate_pair.activate_reference = original_activate
        sys.argv = original_argv
        cache.clear()


if __name__ == "__main__":
    main()
