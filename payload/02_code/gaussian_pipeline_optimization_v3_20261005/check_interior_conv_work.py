"""Check the native census and actual view-finisher on multiple generic images."""

import argparse
from pathlib import Path
import shutil
import sys
import time

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"),
               str(ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005")]
from runtime import activate_reference, atomic_json, data_identity
from mapped_image_linux_guard import LinuxRSSGuard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/iscw001")
    args.output.mkdir(parents=True, exist_ok=False)
    with LinuxRSSGuard(args.output, atomic_json, maximum_seconds=120, maximum_bytes=8 * 1024**3):
        activate_reference(ROOT / "04_outputs/box2_3dgs_display_v1_20260930")
        import torch
        import torch.nn.functional as functional
        from utils import loss_utils
        from reference_ssim import ReferenceSSIM
        from interior_ssim_reuse import InteriorSSIM
        from evaluate_interior_conv_work import finish_coupled_view, composed_main
        from ssim_conv_work import native_conv_work

        paths = [Path(__file__), CODE / "evaluate_interior_conv_work.py", CODE / "ssim_conv_work.py",
                 CODE / "interior_ssim_reuse.py", CODE / "view_major_evaluation.py", Path(loss_utils.__file__),
                 ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005/reference_ssim.py"]
        sources = [data_identity(p) for p in paths]
        for index, path in enumerate(paths):
            shutil.copyfile(path, args.output / (f"SOURCE_{index:02d}_" + path.name))
        (args.output / "COMPOSED_SOURCE.py").write_text(composed_main(), encoding="utf-8")
        original = functional.conv2d
        before_rng = (torch.random.get_rng_state().tolist(), [r.tolist() for r in torch.cuda.get_rng_state_all()])
        rows = []
        with torch.no_grad(), native_conv_work() as trace:
            for ordinal, (height, width) in enumerate(((23, 37), (61, 73), (94, 127))):
                reference = torch.arange(height * width * 3, device="cuda", dtype=torch.float32).reshape(
                    1, height, width, 3).permute(0, 3, 1, 2) / (height * width * 3)
                native, candidate = ReferenceSSIM(loss_utils.create_window), InteriorSSIM(loss_utils.create_window)
                finished = {"native": [], "interior": []}
                for model in range(2):
                    prediction = reference.contiguous().flip(-1) * (.8 + .01 * model)
                    for name, left, right in (("full", prediction, reference),
                        ("target", prediction[:, :, 2:-2, 3:-3], reference[:, :, 2:-2, 3:-3])):
                        expected, actual = native(left, right), candidate(left, right)
                        rows.append({"view": ordinal, "model": model, "region": name,
                                     "absolute_error": float((expected - actual).abs().item())})
                finish_coupled_view(native, finished["native"])
                finish_coupled_view(candidate, finished["interior"])
                if native.entries or candidate.cache.entries or candidate.last is not None:
                    raise AssertionError("The actual view-finisher retains released tensors")
                if len(finished["native"]) != 1 or len(finished["interior"]) != 1:
                    raise AssertionError("A completed view has an invalid finisher denominator")
                native.clear()
                candidate.clear()
            left, weight = torch.ones((1, 1, 7, 9), device="cuda"), torch.ones((1, 1, 3, 3), device="cuda")
            value = functional.conv2d(left, weight)
            if trace["rows"][-1]["output_elements"] != 35 or tuple(value.shape) != (1, 1, 5, 7):
                raise AssertionError("The real native convolution output census is incorrect")
            try:
                functional.conv2d(left, torch.ones((2, 3, 3, 3), device="cuda"))
            except RuntimeError:
                pass
            else:
                raise AssertionError("The original invalid convolution did not fail")
        torch.cuda.synchronize()
        recomputed = sum(row.get("output_elements", 0) for row in trace["rows"] if row.get("completed")
                         and row.get("device", "").startswith("cuda"))
        checks = {"all12_original_generic_scalars": len(rows) == 12 and all(r["absolute_error"] <= 1e-7 for r in rows),
                  "native_successful_CUDA_census_recomputes": recomputed == trace["successful_CUDA_output_elements"],
                  "original_invalid_native_call_propagated": trace["failed_calls"] == 1,
                  "actual_native_binding_restored": functional.conv2d is original and trace["original_binding_restored"],
                  "global_Torch_RNG_unchanged": before_rng == (torch.random.get_rng_state().tolist(),
                      [r.tolist() for r in torch.cuda.get_rng_state_all()]),
                  "all_sources_unchanged": sources == [data_identity(p) for p in paths],
                  "actual_CUDA_headroom": torch.cuda.max_memory_reserved() <= 12 * 1024**3
                      and torch.cuda.mem_get_info()[0] >= 2 * 1024**3}
        atomic_json(args.output / "RESULT.json", {"status": "generic_bridge_and_census_passed" if all(checks.values()) else "failed",
            "checks": checks, "rows": rows, "trace": trace, "sources": sources,
            "complete_child_seconds": time.perf_counter() - STARTED, "new_optimization_count": 0})
        if not all(checks.values()):
            raise AssertionError("A real census or generic consumer protection failed")
        print("generic_bridge_and_census_passed", flush=True)


if __name__ == "__main__":
    main()

