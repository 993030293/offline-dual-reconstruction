"""Regress the real full-image ROI alias before repeating the full evaluator."""

import time

STARTED = time.perf_counter()

import ast
import argparse
import inspect
import json
from pathlib import Path
import shutil
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/vm001/g1"
sys.path[:0] = [str(V1), str(V2)]
from runtime import activate_reference, atomic_json, data_identity
from mapped_image_linux_guard import LinuxRSSGuard


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=False)
    owner = psutil.Process()
    atomic_json(OUT / "PROCESS_IDENTITY.json", {"scope": "WSL", "pid": owner.pid, "creation_time": owner.create_time()})
    with LinuxRSSGuard(OUT, atomic_json, maximum_seconds=120, maximum_bytes=8 * 1024**3):
        metadata = activate_reference(ROOT / "04_outputs/box2_3dgs_display_v1_20260930")
        import torch
        from utils import loss_utils
        from reference_ssim import ReferenceSSIM
        from view_major_evaluation import finish_reference_view
        import check_view_major_generic
        paths = [Path(__file__), CODE / "view_major_evaluation.py", CODE / "evaluate_view_major.py",
                 CODE / "evaluate_interior_ssim.py", V1 / "evaluate_pair.py", V2 / "reference_ssim.py",
                 CODE / "check_view_major_generic.py", Path(loss_utils.__file__)]
        sources = [data_identity(path) for path in paths]
        original = json.loads((OUT.parent / "g0/CONTRACT.json").read_text(encoding="utf-8"))["sources"][1]
        if (sources[1]["bytes"], sources[1]["sha256"]) != (original["bytes"], original["sha256"]):
            raise ValueError("The view-major algorithm changed while repairing its verifier")
        copies = OUT / "source_copies"
        copies.mkdir()
        for index, path in enumerate(paths):
            shutil.copyfile(path, copies / (f"{index:02d}_" + path.name))
        atomic_json(OUT / "CONTRACT.json", {"sources": sources, "real_case_before_rerun": "target_view_alias_is_full_image",
            "original_source_core_unchanged": True, "minimum_actual_CUDA_reserved_peak_reduction": .2})
        left = torch.ones((1, 3, 37, 53), device="cuda") * .8
        right = torch.ones_like(left) * .9
        cache = ReferenceSSIM(loss_utils.create_window)
        values = [cache(left, right), cache(left[:, :, :, :], right[:, :, :, :]),
                  cache(left * .95, right), cache((left * .95)[:, :, :, :], right[:, :, :, :])]
        keys = list(cache.entries)
        trace = []
        finish_reference_view(cache, trace)
        checks = {"complete_ROI_has_one_original_reference_key": len(keys) == 1,
                  "original_whole_crop_values_exact": torch.equal(values[0], values[1]) and torch.equal(values[2], values[3]),
                  "original_full_alias_hit_miss_counts": cache.hits == 3 and cache.misses == 1,
                  "one_actual_original_entry_released": trace[0]["released_reference_entries"] == 1
                      and not cache.entries and cache.resident_bytes == 0}
        cache.clear()
        if not all(checks.values()):
            raise AssertionError("The real complete-ROI alias regression failed")
        source = inspect.getsource(check_view_major_generic.main)
        namespace = dict(vars(check_view_major_generic))
        rerun = OUT / "full_generic"
        namespace.update(OUT=rerun)
        def already_active(runtime):
            if Path(runtime).resolve() != (ROOT / "04_outputs/box2_3dgs_display_v1_20260930").resolve():
                raise ValueError("The explicitly activated runtime differs from the generic contract")
            return metadata
        namespace["activate_reference"] = already_active
        exec(compile(source, "<private-current-view-major-generic-regression>", "exec"), namespace)
        namespace["main"]()
        full = json.loads((rerun / "RESULT.json").read_text(encoding="utf-8"))
        checks["all_previous_eight_generic_groups_current_verifier"] = full["status"] == "generic_passed" and all(full["checks"].values())
        checks["all_current_sources_unchanged"] = sources == [data_identity(path) for path in paths]
        result = {"status": "generic_passed" if all(checks.values()) else "failed_generic", "checks": checks,
                  "full_generic": str(rerun / "RESULT.json"), "original_algorithm_unchanged": True,
                  "complete_child_seconds": time.perf_counter() - STARTED, "new_optimization_count": 0}
        atomic_json(OUT / "RESULT.json", result)
        print(json.dumps(result), flush=True)
        if not all(checks.values()):
            raise AssertionError("A current exact alias/source/generic regression failed")


if __name__ == "__main__":
    main()
