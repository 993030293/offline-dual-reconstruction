"""Check nonfinite propagation, noncontiguous inputs and stale-pair rejection."""

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
    owner = psutil.Process()
    atomic_json(args.output / "PROCESS_IDENTITY.json", {"scope": "WSL", "pid": owner.pid, "creation_time": owner.create_time()})
    with LinuxRSSGuard(args.output, atomic_json, maximum_seconds=120, maximum_bytes=8 * 1024**3):
        activate_reference(ROOT / "04_outputs/box2_3dgs_display_v1_20260930")
        import torch
        from utils import loss_utils
        from reference_ssim import ReferenceSSIM
        from interior_ssim_reuse import InteriorSSIM
        paths = [Path(__file__), CODE / "interior_ssim_reuse.py", V2 / "reference_ssim.py", Path(loss_utils.__file__)]
        identities = [data_identity(path) for path in paths]
        copies = args.output / "source_copies"
        copies.mkdir()
        for index, path in enumerate(paths):
            shutil.copyfile(path, copies / (f"{index:02d}_" + path.name))
        atomic_json(args.output / "CONTRACT.json", {"sources": identities,
            "cases_before_results": ["noncontiguous_prediction", "nan", "positive_inf", "negative_inf",
                "interleaved_different_pair", "mismatched_alias_coordinates", "different_shape_batch_channels", "oversize_uncached"],
            "absolute_scalar_tolerance": 1e-7, "no_protection_or_scope_changes_after_first_generic": True})
        original_cpu_rng = torch.random.get_rng_state().clone()
        original_cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
        rows = []

        def check(name, baseline, candidate, left, right, reuse):
            before = candidate.reused
            expected, actual = baseline(left, right), candidate(left, right)
            finite = torch.isfinite(expected)
            delta = float((expected[finite] - actual[finite]).abs().max().item()) if finite.any() else 0.0
            checks = {"original_absolute_scalar": delta <= 1e-7,
                      "original_nonfinite": torch.equal(torch.isnan(expected), torch.isnan(actual))
                          and torch.equal(torch.isposinf(expected), torch.isposinf(actual))
                          and torch.equal(torch.isneginf(expected), torch.isneginf(actual)),
                      "declared_reuse_route": (candidate.reused > before) == reuse,
                      "bounded_cache": candidate.receipt()["total_accounted_bytes"] <= candidate.maximum_bytes}
            rows.append({"name": name, "checks": checks, "maximum_absolute_scalar_error": delta})
            if not all(checks.values()):
                raise AssertionError("An extended SSIM propagation or stale-pair protection failed")

        try:
            with torch.no_grad():
                for name, value in (("noncontiguous_prediction", None), ("nan", float("nan")),
                                    ("positive_inf", float("inf")), ("negative_inf", -float("inf"))):
                    left = torch.linspace(0, 1, 37*53*3, device="cuda").reshape(1, 37, 53, 3).permute(0, 3, 1, 2)
                    right = left.flip(-2).clone()
                    if value is not None:
                        left[:, :, 13, 19] = value
                    baseline, candidate = ReferenceSSIM(loss_utils.create_window), InteriorSSIM(loss_utils.create_window)
                    check(name + "_full", baseline, candidate, left, right, False)
                    check(name + "_crop", baseline, candidate, left[:, :, 3:-4, 5:-6], right[:, :, 3:-4, 5:-6], True)
                    baseline.clear()
                    candidate.clear()
                left = torch.linspace(0, 1, 37*53*3, device="cuda").reshape(1, 3, 37, 53)
                right = left.flip(-1)
                baseline, candidate = ReferenceSSIM(loss_utils.create_window), InteriorSSIM(loss_utils.create_window)
                check("interleaved_first", baseline, candidate, left, right, False)
                other_left, other_right = left.clone(), right.clone()
                check("interleaved_second", baseline, candidate, other_left, other_right, False)
                check("interleaved_old_crop", baseline, candidate, left[:, :, 3:-4, 5:-6], right[:, :, 3:-4, 5:-6], False)
                candidate.clear()
                check("alias_reset_full", baseline, candidate, left, right, False)
                check("mismatched_alias_coordinates", baseline, candidate, left[:, :, 3:-4, 5:-6], right[:, :, 4:-3, 6:-5], False)
                baseline.clear()
                candidate.clear()
                for batch, channels in ((2, 3), (1, 2)):
                    left = torch.ones((batch, channels, 19, 27), device="cuda")
                    right = left * .8
                    baseline, candidate = ReferenceSSIM(loss_utils.create_window), InteriorSSIM(loss_utils.create_window)
                    check(f"original_batch{batch}_channels{channels}", baseline, candidate, left, right, False)
                    baseline.clear()
                    candidate.clear()
                left = torch.ones((1, 3, 37, 53), device="cuda")
                right = left * .9
                baseline = ReferenceSSIM(loss_utils.create_window, maximum_bytes=1024)
                candidate = InteriorSSIM(loss_utils.create_window, maximum_bytes=1024)
                check("oversize_uncached_full", baseline, candidate, left, right, False)
                check("oversize_uncached_crop", baseline, candidate, left[:, :, 3:-4, 5:-6], right[:, :, 3:-4, 5:-6], False)
                candidate.clear()
                baseline.clear()
                checks = {"all17_extra_cases": len(rows) == 17 and all(all(row["checks"].values()) for row in rows),
                          "all_sources_unchanged": identities == [data_identity(path) for path in paths],
                          "global_CPU_and_all_CUDA_RNG_unchanged": torch.equal(original_cpu_rng, torch.random.get_rng_state())
                              and all(torch.equal(a, b) for a, b in zip(original_cuda_rng, torch.cuda.get_rng_state_all()))}
                result = {"status": "generic_extra_passed" if all(checks.values()) else "failed_extra",
                          "checks": checks, "rows": rows, "new_optimization_count": 0,
                          "complete_child_seconds": time.perf_counter() - STARTED}
                atomic_json(args.output / "RESULT.json", result)
                print(json.dumps({"status": result["status"], "checks": checks, "rows": len(rows)}), flush=True)
                if not all(checks.values()):
                    raise AssertionError("The extra SSIM source or RNG verification failed")
        except BaseException as error:
            atomic_json(args.output / "FAILURE.json", {"status": "negative_extra", "error": repr(error), "rows": rows,
                        "new_optimization_count": 0, "complete_child_seconds": time.perf_counter() - STARTED})
            raise


if __name__ == "__main__":
    main()
