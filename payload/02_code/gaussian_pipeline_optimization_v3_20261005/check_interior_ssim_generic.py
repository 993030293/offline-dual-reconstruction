"""Check the fixed interior-reuse scope against the unchanged Q30 evaluator."""

import time

STARTED = time.perf_counter()

import argparse
import json
from pathlib import Path
import shutil
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import activate_reference, atomic_json, data_identity
from mapped_image_linux_guard import LinuxRSSGuard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(args.output).free < 20 * 1024**3:
        raise MemoryError("The original free-disk reserve is unavailable")
    owner = psutil.Process()
    atomic_json(args.output / "PROCESS_IDENTITY.json", {"scope": "WSL", "pid": owner.pid,
                "creation_time": owner.create_time()})
    with LinuxRSSGuard(args.output, atomic_json, maximum_seconds=120, maximum_bytes=8 * 1024**3) as guard:
        runtime = activate_reference(ROOT / "04_outputs/box2_3dgs_display_v1_20260930")
        import torch
        from utils import loss_utils
        from reference_ssim import ReferenceSSIM
        from interior_ssim_reuse import InteriorSSIM
        sources = [Path(__file__), CODE / "interior_ssim_reuse.py", V2 / "reference_ssim.py",
                   Path(loss_utils.__file__), V1 / "runtime.py", CODE / "mapped_image_linux_guard.py"]
        identities = [data_identity(path) for path in sources]
        copies = args.output / "source_copies"
        copies.mkdir()
        for index, path in enumerate(sources):
            shutil.copyfile(path, copies / (f"{index:02d}_" + path.name))
        free, _ = torch.cuda.mem_get_info()
        budget = min(12 * 1024**3, free - 2 * 1024**3)
        if budget <= 0:
            raise MemoryError("The original CUDA headroom is unavailable")
        torch.cuda.reset_peak_memory_stats()
        dimensions = [(34, 53), (45, 71), (39, 61), (80, 112), (103, 89), (620, 812)]
        patterns = ["random", "near_constant", "same", "extremes", "zeros", "ones"]
        atomic_json(args.output / "CONTRACT.json", {"runtime": runtime, "sources": identities,
            "dimensions_before_results": dimensions, "patterns_before_results": patterns,
            "absolute_scalar_tolerance": 1e-7, "current_Q30_reference_baseline": True,
            "full_and_target_window_and_original_mean_unchanged": True,
            "no_full_model_or_complete_three_pair_claim": True, "CUDA_budget_bytes": budget,
            "CPU_float64_other_window_and_small_crop_use_original_Q30": True})
        original_cpu_rng = torch.random.get_rng_state().clone()
        original_cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
        generator = torch.Generator(device="cuda").manual_seed(134001)
        rows = []
        baseline = ReferenceSSIM(loss_utils.create_window)
        candidate = InteriorSSIM(loss_utils.create_window)

        def compare(name, left, right, window_size=11, size_average=True, required_route=None):
            input_bits = (left.clone(), right.clone(), left._version, right._version, left.stride(), right.stride())
            before_reused = candidate.reused
            expected = baseline(left, right, window_size, size_average)
            actual = candidate(left, right, window_size, size_average)
            finite = torch.isfinite(expected)
            delta = float((expected[finite] - actual[finite]).abs().max().item()) if finite.any() else 0.0
            checks = {"original_scalar_absolute_tolerance": delta <= 1e-7,
                      "original_return_shape_dtype_device": actual.shape == expected.shape and actual.dtype == expected.dtype
                          and actual.device == expected.device,
                      "original_nonfinite_propagation": torch.equal(torch.isnan(actual), torch.isnan(expected))
                          and torch.equal(torch.isposinf(actual), torch.isposinf(expected))
                          and torch.equal(torch.isneginf(actual), torch.isneginf(expected)),
                      "original_all_input_bits_layout_and_versions": torch.equal(left.contiguous().view(torch.uint8), input_bits[0].contiguous().view(torch.uint8))
                          and torch.equal(right.contiguous().view(torch.uint8), input_bits[1].contiguous().view(torch.uint8))
                          and (left._version, right._version, left.stride(), right.stride()) == input_bits[2:],
                      "original_cache_total_budget": candidate.receipt()["total_accounted_bytes"] <= 1024**3}
            if required_route is not None:
                checks["preregistered_route"] = (candidate.reused > before_reused) == required_route
            row = {"name": name, "shape": list(left.shape), "maximum_absolute_scalar_error": delta,
                   "checks": checks, "interior_reused": candidate.reused > before_reused}
            rows.append(row)
            atomic_json(args.output / "STATE.json", {"status": "running", "rows": rows, "counted": False})
            if not all(checks.values()):
                raise AssertionError("An original SSIM scalar, input, route or resource protection failed")

        try:
            with torch.no_grad():
                for index, ((h, w), pattern) in enumerate(zip(dimensions, patterns)):
                    right = torch.rand((1, h, w, 3), generator=generator, device="cuda").permute(0, 3, 1, 2)
                    left = torch.rand((1, 3, h, w), generator=generator, device="cuda")
                    if pattern == "near_constant":
                        right.fill_(.5)
                        left.mul_(1e-5).add_(.5)
                    elif pattern == "same":
                        left = right.contiguous()
                    elif pattern == "extremes":
                        left = (left > .5).float()
                        right = (right > .5).float().permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)
                    elif pattern == "zeros":
                        left.zero_()
                        right.zero_()
                    elif pattern == "ones":
                        left.fill_(1)
                        right.fill_(1)
                    compare(f"{index}_{pattern}_full", left, right, required_route=False)
                    for label, region in (("interior", (3, h-4, 7, w-6)), ("top_left", (0, h-7, 0, w-8)),
                                          ("bottom_right", (5, h, 4, w)), ("11pixel", (2, 13, 3, 14)),
                                          ("small", (2, 12, 3, 14))):
                        candidate.clear()
                        compare(f"{index}_{pattern}_{label}_reset_full", left, right, required_route=False)
                        y1, y2, x1, x2 = region
                        compare(f"{index}_{pattern}_{label}", left[:, :, y1:y2, x1:x2], right[:, :, y1:y2, x1:x2],
                                required_route=label != "small")
                    candidate.clear()
                    compare(f"{index}_before_mutation", left, right, required_route=False)
                    left.add_(.001)
                    compare(f"{index}_prediction_mutation", left[:, :, 2:-2, 3:-3], right[:, :, 2:-2, 3:-3], required_route=False)
                    candidate.clear()
                    compare(f"{index}_before_reference_mutation", left, right, required_route=False)
                    right.add_(.002)
                    compare(f"{index}_reference_mutation", left[:, :, 2:-2, 3:-3], right[:, :, 2:-2, 3:-3], required_route=False)
                    candidate.clear()
                    compare(f"{index}_size_average_false_full", left, right, size_average=False, required_route=False)
                    compare(f"{index}_size_average_false_crop", left[:, :, 2:-2, 3:-3], right[:, :, 2:-2, 3:-3],
                            size_average=False, required_route=True)
                for dtype, device, window in ((torch.float32, "cpu", 11), (torch.float64, "cpu", 11),
                                             (torch.float64, "cuda", 11), (torch.float32, "cuda", 3)):
                    left = torch.arange(3*23*37, device=device, dtype=dtype).reshape(1, 3, 23, 37) / (3*23*37)
                    right = left.flip(-1)
                    candidate.clear()
                    compare(f"original_route_{dtype}_{device}_{window}", left, right, window_size=window, required_route=False)
                    compare(f"original_route_crop_{dtype}_{device}_{window}", left[:, :, 1:-1, 2:-2], right[:, :, 1:-1, 2:-2],
                            window_size=window, required_route=False)
            for kind in ("requires_grad", "shape", "window"):
                left = torch.zeros((1, 3, 13, 17), device="cuda")
                right = left.clone()
                if kind == "requires_grad":
                    left.requires_grad_(True)
                elif kind == "shape":
                    right = right[:, :, :-1]
                errors = []
                for algorithm in (baseline, candidate):
                    try:
                        algorithm(left, right, 2 if kind == "window" else 11)
                    except BaseException as error:
                        errors.append((type(error).__name__, str(error)))
                    else:
                        errors.append(None)
                checks = {"original_exception_type_message": errors[0] is not None and errors[0] == errors[1]}
                rows.append({"name": "error_" + kind, "checks": checks, "errors": errors})
                if not all(checks.values()):
                    raise AssertionError("The original SSIM rejection changed")
            torch.cuda.synchronize()
            checks = {"all_preregistered_generic_rows": all(all(row["checks"].values()) for row in rows),
                      "all_sources_unchanged": identities == [data_identity(path) for path in sources],
                      "global_Torch_CPU_and_all_CUDA_RNG_unchanged": torch.equal(original_cpu_rng, torch.random.get_rng_state())
                          and all(torch.equal(a, b) for a, b in zip(original_cuda_rng, torch.cuda.get_rng_state_all())),
                      "actual_CUDA_budget": torch.cuda.max_memory_reserved() <= budget}
            result = {"status": "generic_passed" if all(checks.values()) else "failed_generic", "checks": checks,
                      "rows": rows, "cache": candidate.receipt(), "peak_CUDA_reserved_bytes": torch.cuda.max_memory_reserved(),
                      "complete_child_seconds": time.perf_counter() - STARTED, "new_optimization_count": 0}
            atomic_json(args.output / "RESULT.json", result)
            print(json.dumps({"status": result["status"], "checks": checks, "rows": len(rows),
                              "maximum_error": max(row.get("maximum_absolute_scalar_error", 0) for row in rows)}), flush=True)
            if not all(checks.values()):
                raise AssertionError("A generic source, RNG or resource protection failed")
        except BaseException as error:
            result = {"status": "negative_generic_first_required_failure", "error": repr(error), "rows": rows,
                      "cache": candidate.receipt(), "complete_child_seconds": time.perf_counter() - STARTED,
                      "new_optimization_count": 0, "formal_pairs_external_recovery_training": "not_run"}
            atomic_json(args.output / "RESULT.json", result)
            print(json.dumps({"status": result["status"], "error": result["error"], "last_row": rows[-1] if rows else None}), flush=True)
            raise
        finally:
            candidate.clear()
            baseline.clear()


if __name__ == "__main__":
    main()
