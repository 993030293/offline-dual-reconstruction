"""Check optional SSIM forward/backward values and measure isolated CUDA kernels."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

from runtime import activate_reference, atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reference_identity = activate_reference(args.runtime)
    sys.path[:0] = [str(args.source), str(args.library)]
    import torch
    from utils.loss_utils import ssim
    from fused_ssim import fused_ssim
    torch.manual_seed(0)
    contract = {"reference": reference_identity, "extension": [data_identity(p) for p in args.library.glob("*.so")],
        "cases": ["independent_random", "nearly_constant", "near_identical", "zero", "one"],
        "sizes": [[64, 80], [620, 812]], "padding": "same", "dtype": "float32",
        "forward_absolute_tolerance": 1e-4, "gradient_max_absolute_tolerance": 3e-6,
        "gradient_relative_L2_tolerance": .005, "relative_gradient_denominator_floor": 1e-7,
        "relative_gradient_gate_cases": ["independent_random", "nearly_constant", "near_identical"],
        "constant_identity_gradient_gate": "absolute only; relative error against zero is undefined",
        "timing_repetitions": 3, "timed_forward_backward_steps": 30, "warmup_steps": 5,
        "kernel_gain_is_not_training_gain": True, "maximum_total_seconds": 120,
        "implementation": data_identity(Path(__file__))}
    atomic_json(args.output/"CONTRACT.json", contract)
    started = time.monotonic()
    rows = []
    for height, width in contract["sizes"]:
        shape = (1, 3, height, width)
        for case in contract["cases"]:
            a, b = torch.rand(shape, device="cuda"), torch.rand(shape, device="cuda")
            if case == "nearly_constant":
                a, b = .5+.001*(a-.5), .5+.001*(b-.5)
            elif case == "near_identical":
                b = (a+.005*(b-.5)).clamp(0, 1)
            elif case in ("zero", "one"):
                a = torch.zeros(shape, device="cuda") if case == "zero" else torch.ones(shape, device="cuda")
                b = a.clone()
            a.requires_grad_(True)
            reference = ssim(a, b)
            old_gradient = torch.autograd.grad(reference, a)[0]
            candidate = fused_ssim(a, b, padding="same")
            new_gradient = torch.autograd.grad(candidate, a)[0]
            forward = float((candidate-reference).abs())
            gradient = float((new_gradient-old_gradient).abs().max())
            relative = float(torch.linalg.vector_norm(new_gradient-old_gradient)/torch.linalg.vector_norm(old_gradient).clamp_min(1e-7))
            passed = forward <= 1e-4 and gradient <= 3e-6 and (case in ("zero", "one") or relative <= .005)
            rows.append({"size": [height, width], "case": case, "forward_absolute_error": forward,
                         "gradient_max_absolute_error": gradient, "gradient_relative_L2_error": relative, "passed": passed})
            if time.monotonic()-started > 120:
                raise TimeoutError("SSIM numeric validation exceeded its budget")
    if not all(row["passed"] for row in rows):
        atomic_json(args.output/"RESULT.json", {"status": "numeric_gate_failed", "rows": rows, "counted": False})
        raise AssertionError("Optional SSIM is not accepted by the registered numeric gate")
    a = torch.rand((1, 3, 620, 812), device="cuda", requires_grad=True)
    b = torch.rand_like(a)
    records = []
    functions = {"reference": ssim, "fused": fused_ssim}
    for repetition in range(3):
        for name in (["reference", "fused"] if repetition != 1 else ["fused", "reference"]):
            function = functions[name]
            for _ in range(5):
                torch.autograd.grad(function(a, b), a)
            torch.cuda.synchronize()
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            before = time.perf_counter()
            start_event.record()
            for _ in range(30):
                torch.autograd.grad(function(a, b), a)
            end_event.record()
            torch.cuda.synchronize()
            records.append({"variant": name, "repetition": repetition,
                "wall_ms_per_step": (time.perf_counter()-before)/30*1000,
                "GPU_ms_per_step": start_event.elapsed_time(end_event)/30})
    medians = {name: statistics.median(row["wall_ms_per_step"] for row in records if row["variant"] == name) for name in functions}
    receipt = {"status": "numeric_and_kernel_timing_completed_training_pending", "rows": rows, "timing": records,
        "median_wall_ms": medians, "kernel_wall_reduction_fraction": 1-medians["fused"]/medians["reference"],
        "elapsed_seconds": time.monotonic()-started, "training_gain_verified": False, "counted": False}
    atomic_json(args.output/"RESULT.json", receipt)
    print(json.dumps({"status": receipt["status"], "median_wall_ms": medians}), flush=True)


if __name__ == "__main__":
    main()
