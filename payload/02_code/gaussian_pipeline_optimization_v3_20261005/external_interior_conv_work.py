"""Check the frozen SSIM component on all existing public photographs."""

from pathlib import Path
import shutil
import sys
import time

STARTED = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
sys.path[:0] = [str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"),
               str(ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005")]
from runtime import activate_reference, atomic_json, data_identity
from mapped_image_linux_guard import LinuxRSSGuard


def main():
    output = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/iscw001/x0"
    images = ROOT / "04_outputs/public_full_e26_scene_v3_20260929T141800Z/attempt02/01_visual_workspace/images"
    paths = sorted(images.glob("*.png"))
    if len(paths) != 360:
        raise ValueError("The original full public photograph denominator changed")
    output.mkdir(parents=True, exist_ok=False)
    with LinuxRSSGuard(output, atomic_json, maximum_seconds=300, maximum_bytes=8 * 1024**3):
        activate_reference(ROOT / "04_outputs/box2_3dgs_display_v1_20260930")
        import numpy as np
        from PIL import Image
        import torch
        from utils import loss_utils
        from reference_ssim import ReferenceSSIM
        from interior_ssim_reuse import InteriorSSIM
        from evaluate_interior_conv_work import finish_coupled_view
        from ssim_conv_work import native_conv_work
        sources = [Path(__file__), CODE / "interior_ssim_reuse.py", CODE / "evaluate_interior_conv_work.py",
                   CODE / "ssim_conv_work.py", CODE / "view_major_evaluation.py", Path(loss_utils.__file__)]
        source_ids, inputs = [data_identity(p) for p in sources], [data_identity(p) for p in paths]
        for ordinal, path in enumerate(sources):
            shutil.copyfile(path, output / (f"SOURCE_{ordinal:02d}_" + path.name))
        atomic_json(output / "CONTRACT.json", {"sources": source_ids, "inputs": inputs,
            "full360_public_component_only": True, "dimensions": [67, 91], "crop": [17, 29],
            "synthetic_prediction_formula": ["reference*.99+.002", "reference*.98+.003"],
            "original_scalar_absolute_tolerance": 1e-7, "strict_blind_or_GT_or_external_model": False})
        cpu_rng, cuda_rng = torch.random.get_rng_state().tolist(), [r.tolist() for r in torch.cuda.get_rng_state_all()]
        rows = []
        with torch.no_grad(), native_conv_work() as work:
            for index, (path, identity) in enumerate(zip(paths, inputs)):
                with Image.open(path) as photo:
                    with photo.convert("RGB") as rgb:
                        with rgb.resize((91, 67)) as resized:
                            data = np.array(resized)
                reference = torch.from_numpy(data).permute(2, 0, 1).float().div(255).to("cuda")[None]
                before = reference.clone()
                native, candidate = ReferenceSSIM(loss_utils.create_window), InteriorSSIM(loss_utils.create_window)
                maximum = 0.0
                for scale, offset in ((.99, .002), (.98, .003)):
                    prediction = reference * scale + offset
                    for left, right in ((prediction, reference), (prediction[:, :, 5:22, 7:36], reference[:, :, 5:22, 7:36])):
                        expected, actual = native(left, right), candidate(left, right)
                        maximum = max(maximum, float((expected - actual).abs().item()))
                finish_coupled_view(native, [])
                finish_coupled_view(candidate, [])
                if maximum > 1e-7 or not torch.equal(before, reference) or candidate.last is not None or candidate.cache.entries:
                    raise AssertionError("A public component numerical or reference lifetime protection failed")
                native.clear()
                candidate.clear()
                rows.append({"name": path.name, "maximum_absolute_SSIM_error": maximum})
                if data_identity(path) != identity:
                    raise ValueError("An original public photograph changed during consumption")
        torch.cuda.synchronize()
        checks = {"all360_existing_photographs_included": len(rows) == 360,
                  "all1440_full_and_crop_scalars_original_tolerance": all(r["maximum_absolute_SSIM_error"] <= 1e-7 for r in rows),
                  "all_original_sources_and_inputs_unchanged": source_ids == [data_identity(p) for p in sources]
                      and inputs == [data_identity(p) for p in paths],
                  "original_Torch_RNG_unchanged": cpu_rng == torch.random.get_rng_state().tolist()
                      and cuda_rng == [r.tolist() for r in torch.cuda.get_rng_state_all()],
                  "native_census_restored": work["original_binding_restored"],
                  "actual_GPU_budget": torch.cuda.max_memory_reserved() <= 12 * 1024**3
                      and torch.cuda.mem_get_info()[0] >= 2 * 1024**3}
        atomic_json(output / "RESULT.json", {"status": "all360_component_only_passed" if all(checks.values()) else "failed",
            "checks": checks, "rows": rows, "work": work, "sources": source_ids, "inputs": inputs,
            "complete_child_seconds": time.perf_counter() - STARTED, "new_optimization_count": 0})
        if not all(checks.values()):
            raise AssertionError("An original full public component protection failed")
        print("all360_component_only_passed", flush=True)


if __name__ == "__main__":
    main()

