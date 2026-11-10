"""Check reference-statistic reuse against the installed official SSIM."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import activate_reference, atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = activate_reference(args.runtime)
    import torch
    from utils.loss_utils import ssim, create_window
    from reference_ssim import ReferenceSSIM
    torch.manual_seed(31)
    cache = ReferenceSSIM(create_window, maximum_bytes=4*1024**2)
    records = []
    for device in ("cpu", "cuda"):
        for shape, size, sliced, average in [((1, 3, 31, 47), 11, False, True),
                ((2, 1, 32, 40), 7, False, False), ((1, 3, 33, 45), 11, True, True),
                ((1, 3, 11, 11), 11, False, True)]:
            a, b = [torch.rand(shape, device=device) for _ in range(2)]
            if sliced:
                a, b = a[..., ::2, 1::2], b[..., ::2, 1::2]
            original = ssim(a, b, size, average)
            first, second = cache(a, b, size, average), cache(a, b, size, average)
            error = max(float((value-original).abs().max()) for value in (first, second))
            records.append({"device": device, "shape": shape, "window": size, "sliced": sliced,
                "size_average": average, "maximum_error": error, "passed": error <= 1e-7})
    a, b = [torch.rand((1, 3, 23, 27), device="cuda") for _ in range(2)]
    before = cache(a, b)
    b.zero_()
    after = cache(a, b)
    mutation_error = float((after-ssim(a, b)).abs())
    tiny = ReferenceSSIM(create_window, maximum_bytes=4*a.numel()*a.element_size())
    tensors = [torch.rand_like(a) for _ in range(4)]
    eviction_errors = [float((tiny(a, ref)-ssim(a, ref)).abs()) for ref in tensors+[tensors[0]]]
    tiny_budget = ReferenceSSIM(create_window, maximum_bytes=1)
    oversize_error = float((tiny_budget(a, b)-ssim(a, b)).abs())
    try:
        cache(a.requires_grad_(), b)
    except ValueError:
        gradient_rejected = True
    else:
        gradient_rejected = False
    checks = {"all_reference_formula_cases": all(r["passed"] for r in records),
        "real_inplace_mutation_invalidated": mutation_error <= 1e-7 and not torch.equal(before, after),
        "real_eviction_preserves_results": max(eviction_errors) <= 1e-7 and tiny.evictions >= 4,
        "memory_bound_observed": tiny.peak_bytes <= tiny.maximum_bytes,
        "oversize_explicit_uncached_path": oversize_error <= 1e-7 and tiny_budget.oversize == 1 and not tiny_budget.entries,
        "training_gradient_use_rejected": gradient_rejected}
    atomic_json(args.output/"RESULT.json", {"status": "numeric_passed_real_evaluation_pending" if all(checks.values()) else "failed",
        "checks": checks, "cases": records, "mutation_error": mutation_error,
        "eviction_max_error": max(eviction_errors), "cache": cache.receipt(), "runtime": metadata,
        "adapter_source": data_identity(Path(__file__).with_name("reference_ssim.py")), "counted": False})
    print(checks)
    if not all(checks.values()):
        raise ValueError("Reference SSIM reuse failed its fixed numerical contract")


if __name__ == "__main__":
    main()
