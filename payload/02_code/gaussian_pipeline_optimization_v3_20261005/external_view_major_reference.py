"""Check original SSIM lifetimes on all existing external photographs."""

import time

STARTED = time.perf_counter()

import json
from pathlib import Path
import shutil
import sys

import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/vm001/x0"
IMAGES = ROOT / "04_outputs/public_full_e26_scene_v3_20260929T141800Z/attempt02/01_visual_workspace/images"
sys.path[:0] = [str(V1), str(V2)]
from runtime import activate_reference, atomic_json, data_identity
from mapped_image_linux_guard import LinuxRSSGuard


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    if shutil.disk_usage(OUT).free < 20 * 1024**3:
        raise MemoryError("The original free-disk reserve is unavailable")
    owner = psutil.Process()
    atomic_json(OUT / "PROCESS_IDENTITY.json", {"scope": "WSL", "pid": owner.pid, "creation_time": owner.create_time()})
    with LinuxRSSGuard(OUT, atomic_json, maximum_seconds=600, maximum_bytes=8 * 1024**3):
        activate_reference(ROOT / "04_outputs/box2_3dgs_display_v1_20260930")
        import numpy as np
        import torch
        from PIL import Image
        from utils import loss_utils
        from utils.general_utils import PILtoTorch
        from reference_ssim import ReferenceSSIM
        from view_major_evaluation import finish_reference_view
        paths = sorted(IMAGES.glob("*.png"))
        if len(paths) != 360:
            raise ValueError("The complete existing Oxford photograph denominator changed")
        sources = [Path(__file__), CODE / "view_major_evaluation.py", V2 / "reference_ssim.py",
                   Path(loss_utils.__file__), Path(sys.modules[PILtoTorch.__module__].__file__)]
        identities = [data_identity(path) for path in paths]
        source_ids = [data_identity(path) for path in sources]
        frozen = json.loads((OUT.parent / "g2/CONTRACT.json").read_text(encoding="utf-8"))["sources"][1]
        if (frozen["bytes"], frozen["sha256"]) != (source_ids[1]["bytes"], source_ids[1]["sha256"]):
            raise ValueError("The external component uses a different frozen lifetime algorithm")
        copies = OUT / "source_copies"
        copies.mkdir()
        for index, path in enumerate(sources):
            shutil.copyfile(path, copies / (f"{index:02d}_" + path.name))
        atomic_json(OUT / "CONTRACT.json", {"sources": source_ids, "inputs": identities, "full360_photographs": True,
            "component_only_two_synthetic_tensor_variants_no_trained_external_model": True,
            "predictor_before_results": ["reference*0.99+0.002", "reference*0.98+0.003"],
            "region_before_results": "whole every6 otherwise y1=H//7,y2=6H//7,x1=W//5,x2=4W//5",
            "original_half_resolution_PILtoTorch_and_Q30": True, "absolute_SSIM_tolerance": 1e-7,
            "Oxford_nonblind_development_not_GT_or_new_object_pipeline": True})
        free, _ = torch.cuda.mem_get_info()
        budget = min(12 * 1024**3, free - 2 * 1024**3)
        if budget <= 0:
            raise MemoryError("The original CUDA headroom is unavailable")
        torch.cuda.reset_peak_memory_stats()
        cpu_rng = torch.random.get_rng_state().clone()
        cuda_rng = [state.clone() for state in torch.cuda.get_rng_state_all()]
        rows = []
        with torch.no_grad():
            for index, (path, expected_id) in enumerate(zip(paths, identities)):
                if data_identity(path) != expected_id:
                    raise ValueError("An external original photograph changed")
                with Image.open(path) as photo:
                    with photo.convert("RGB") as rgb:
                        tensor = PILtoTorch(rgb, (round(rgb.width/2), round(rgb.height/2)))
                reference = tensor.to("cuda")[None]
                h, w = reference.shape[2:]
                region = (0, h, 0, w) if index % 6 == 0 else (h//7, 6*h//7, w//5, 4*w//5)
                y1, y2, x1, x2 = region
                baseline, candidate = ReferenceSSIM(loss_utils.create_window), ReferenceSSIM(loss_utils.create_window)
                values = {mode: [] for mode in ("native", "view")}
                reference_before = reference.clone()
                for mode, cache in (("native", baseline), ("view", candidate)):
                    for scale, offset in ((.99, .002), (.98, .003)):
                        predicted = reference * scale + offset
                        values[mode].append(float(cache(predicted, reference).item()))
                        values[mode].append(float(cache(predicted[:, :, y1:y2, x1:x2], reference[:, :, y1:y2, x1:x2]).item()))
                trace = []
                expected_entries = len(baseline.entries)
                finish_reference_view(candidate, trace)
                maximum = max(abs(a-b) for a, b in zip(values["native"], values["view"]))
                checks = {"all_original_Q30_scalars": maximum <= 1e-7,
                          "original_reference_bits_unchanged": torch.equal(reference.contiguous().view(torch.uint8), reference_before.contiguous().view(torch.uint8)),
                          "original_alias_dependent_entry_count_released": trace[0]["released_reference_entries"] == expected_entries,
                          "all_finished_reference_statistics_released_window_retained": not candidate.entries and candidate.resident_bytes == 0 and len(candidate.windows) == 1,
                          "same_original_hits_and_misses": baseline.hits == candidate.hits and baseline.misses == candidate.misses,
                          "actual_image_source_unchanged": data_identity(path) == expected_id}
                rows.append({"name": path.name, "shape": list(reference.shape), "region": list(region), "checks": checks,
                             "maximum_absolute_SSIM_error": maximum, "released_reference_entries": expected_entries})
                baseline.clear()
                candidate.clear()
                if not all(checks.values()):
                    atomic_json(OUT / "FAILURE.json", {"rows": rows, "new_optimization_count": 0})
                    raise AssertionError("An original external full-image statistic or lifetime changed")
                if index % 60 == 59:
                    atomic_json(OUT / "STATE.json", {"status": "running", "completed360": index+1, "rows": rows})
                    print(json.dumps({"completed360": index+1}), flush=True)
        checks = {"all360_complete_photograph_components": len(rows) == 360 and all(all(row["checks"].values()) for row in rows),
                  "all_original_inputs_and_sources_unchanged": identities == [data_identity(path) for path in paths]
                      and source_ids == [data_identity(path) for path in sources],
                  "original_CPU_and_all_CUDA_RNG": torch.equal(cpu_rng, torch.random.get_rng_state())
                      and all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all())),
                  "actual_original_CUDA_budget": torch.cuda.max_memory_reserved() <= budget}
        result = {"status": "external_component_passed" if all(checks.values()) else "failed_component", "checks": checks,
                  "rows": rows, "component_only_no_external_trained_model_GT_or_speed_claim": True,
                  "maximum_absolute_SSIM_error": max(row["maximum_absolute_SSIM_error"] for row in rows),
                  "complete_child_seconds": time.perf_counter() - STARTED, "peak_CUDA_reserved_bytes": torch.cuda.max_memory_reserved(),
                  "new_optimization_count": 0}
        atomic_json(OUT / "RESULT.json", result)
        print(json.dumps({key: result[key] for key in ("status", "checks", "maximum_absolute_SSIM_error", "complete_child_seconds")}), flush=True)
        if not all(checks.values()):
            raise AssertionError("A complete external source, RNG or resource protection failed")


if __name__ == "__main__":
    main()
