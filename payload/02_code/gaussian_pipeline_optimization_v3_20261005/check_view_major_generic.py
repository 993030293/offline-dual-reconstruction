"""Check source composition and exact reference-statistic last-use lifetimes."""

import time

STARTED = time.perf_counter()

import ast
import json
from pathlib import Path
import shutil
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/vm001/g0"
sys.path[:0] = [str(V1), str(V2)]
from runtime import activate_reference, atomic_json, data_identity
from mapped_image_linux_guard import LinuxRSSGuard


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    owner = psutil.Process()
    atomic_json(OUT / "PROCESS_IDENTITY.json", {"scope": "WSL", "pid": owner.pid, "creation_time": owner.create_time()})
    with LinuxRSSGuard(OUT, atomic_json, maximum_seconds=120, maximum_bytes=8 * 1024**3):
        activate_reference(ROOT / "04_outputs/box2_3dgs_display_v1_20260930")
        import torch
        from utils import loss_utils
        from reference_ssim import ReferenceSSIM
        from view_major_evaluation import compose_view_major, finish_reference_view
        import evaluate_pair
        composed = compose_view_major(evaluate_pair.main)
        ast.parse(composed)
        (OUT / "COMPOSED_VIEW_MAJOR.py").write_text(composed, encoding="utf-8")
        paths = [Path(__file__), CODE / "view_major_evaluation.py", CODE / "evaluate_view_major.py",
                 CODE / "evaluate_interior_ssim.py", V1 / "evaluate_pair.py", V2 / "reference_ssim.py", Path(loss_utils.__file__)]
        sources = [data_identity(path) for path in paths]
        copies = OUT / "source_copies"
        copies.mkdir()
        for index, path in enumerate(paths):
            shutil.copyfile(path, copies / (f"{index:02d}_" + path.name))
        atomic_json(OUT / "CONTRACT.json", {"sources": sources, "minimum_actual_CUDA_reserved_peak_reduction": .2,
            "absolute_SSIM_tolerance": 1e-7, "whole_median_cannot_regress": True, "whole_each_maximum_ratio": 1.1,
            "patterns_before_results": ["random", "constant", "noncontiguous", "mutation", "nan"],
            "dimensions_before_results": [[19, 27], [31, 43], [620, 812]],
            "component_only_no_actual_model_or_GT_claim": True})
        before_cpu = torch.random.get_rng_state().clone()
        before_cuda = [state.clone() for state in torch.cuda.get_rng_state_all()]
        generator = torch.Generator(device="cuda").manual_seed(135001)
        rows = []
        for pattern in ("random", "constant", "noncontiguous", "mutation", "nan"):
            references, predictions = [], []
            for h, w in ((19, 27), (31, 43), (620, 812)):
                reference = torch.rand((1, 3, h, w), generator=generator, device="cuda")
                if pattern == "constant":
                    reference.fill_(.5)
                if pattern == "noncontiguous":
                    reference = reference.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)
                if pattern == "nan":
                    reference[:, :, 9, 11] = float("nan")
                references.append(reference)
            for label in range(2):
                predictions.append([reference.clone() + (label + 1) * .001 for reference in references])
            caches = {mode: ReferenceSSIM(loss_utils.create_window) for mode in ("native", "view")}
            results = {mode: {} for mode in caches}
            trace = []
            with torch.no_grad():
                for label in range(2):
                    for index, reference in enumerate(references):
                        predicted = predictions[label][index]
                        results["native"][label, index] = [caches["native"](predicted, reference),
                            caches["native"](predicted[:, :, 2:-2, 3:-3], reference[:, :, 2:-2, 3:-3])]
                for index, reference in enumerate(references):
                    for label in range(2):
                        predicted = predictions[label][index]
                        results["view"][label, index] = [caches["view"](predicted, reference),
                            caches["view"](predicted[:, :, 2:-2, 3:-3], reference[:, :, 2:-2, 3:-3])]
                    finish_reference_view(caches["view"], trace)
            errors = []
            nonfinite = []
            for key in results["native"]:
                for expected, actual in zip(results["native"][key], results["view"][key]):
                    errors.append(float((expected - actual).abs().item()) if torch.isfinite(expected) else 0.0)
                    nonfinite.append(bool(torch.equal(torch.isnan(expected), torch.isnan(actual))
                                         and torch.equal(torch.isposinf(expected), torch.isposinf(actual))
                                         and torch.equal(torch.isneginf(expected), torch.isneginf(actual))))
            checks = {"all_original_scalar_values": max(errors) <= 1e-7 and all(nonfinite),
                      "all_three_last_use_reference_lifetimes_released": len(trace) == 3 and all(row["released_reference_entries"] == 2 for row in trace),
                      "original_reference_reuse_hits_and_misses": caches["native"].hits == caches["view"].hits == 6
                          and caches["native"].misses == caches["view"].misses == 6,
                      "window_retained_across_views": len(caches["view"].windows) == 1,
                      "last_use_clears_only_reference_entries": not caches["view"].entries and caches["view"].resident_bytes == 0}
            if pattern == "mutation":
                references[-1].add_(.1)
                predictions[0][-1].add_(.01)
                after = caches["view"](predictions[0][-1], references[-1])
                expected = caches["native"](predictions[0][-1], references[-1])
                checks["changed_reference_prediction_recomputed_exactly"] = bool(torch.equal(after, expected))
                finish_reference_view(caches["view"], trace)
            for cache in caches.values():
                cache.clear()
                cache.clear()
                if cache.entries or cache.windows or cache.resident_bytes:
                    raise AssertionError("The original idempotent clear retained cache resources")
            rows.append({"pattern": pattern, "checks": checks, "maximum_absolute_scalar_error": max(errors)})
            if not all(checks.values()):
                atomic_json(OUT / "FAILURE.json", {"rows": rows, "new_optimization_count": 0})
                raise AssertionError("An original last-use reference statistic contract changed")
        for name in ("requires_grad", "shape", "window"):
            left = torch.zeros((1, 3, 17, 23), device="cuda")
            right = left.clone()
            if name == "requires_grad":
                left.requires_grad_(True)
            if name == "shape":
                right = right[:, :, :-1]
            failures = []
            for mode in ("native", "view"):
                cache = ReferenceSSIM(loss_utils.create_window)
                try:
                    cache(left, right, 2 if name == "window" else 11)
                except BaseException as error:
                    failures.append((type(error).__name__, str(error)))
                else:
                    failures.append(None)
                finally:
                    if mode == "view":
                        finish_reference_view(cache, [])
                    cache.clear()
            checks = {"original_error_and_close": failures[0] is not None and failures[0] == failures[1]}
            rows.append({"error": name, "checks": checks})
            if not all(checks.values()):
                raise AssertionError("The original reference error propagation changed")
        checks = {"all_eight_preregistered_generic_groups": len(rows) == 8 and all(all(row["checks"].values()) for row in rows),
                  "all_original_sources_unchanged": sources == [data_identity(path) for path in paths],
                  "original_CPU_and_all_CUDA_RNG": torch.equal(before_cpu, torch.random.get_rng_state())
                      and all(torch.equal(a, b) for a, b in zip(before_cuda, torch.cuda.get_rng_state_all())),
                  "private_consumer_dependencies_and_original_ops_preserved": composed.count("_finish_view()") == 1
                      and composed.count("predicted = render(") == 1 and composed.count('row["full_image"] = metrics(') == 1
                      and composed.count('row["target"] = metrics(') == 1}
        result = {"status": "generic_passed" if all(checks.values()) else "failed_generic", "checks": checks,
                  "rows": rows, "complete_child_seconds": time.perf_counter() - STARTED, "new_optimization_count": 0}
        atomic_json(OUT / "RESULT.json", result)
        print(json.dumps({"status": result["status"], "checks": checks, "groups": len(rows)}), flush=True)
        if not all(checks.values()):
            raise AssertionError("A private complete consumer, RNG or source protection failed")


if __name__ == "__main__":
    main()
