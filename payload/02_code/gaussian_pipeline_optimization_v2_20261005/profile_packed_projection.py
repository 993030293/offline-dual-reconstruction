"""Check complete render, parameter and densification gradients on actual views."""

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import activate_reference, atomic_json, data_identity
from full_intrinsics import Pinhole, apply_to_camera
from evaluate_pair import mounted_path
from packed_projection import packed_projection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "build", "model", "localization", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    activate_reference(args.runtime)
    from gsplat_adapter import activate_gsplat, render
    build = activate_gsplat(args.build)
    import torch
    from scene.gaussian_model import GaussianModel
    from scene.cameras import Camera
    from types import SimpleNamespace

    free, total = torch.cuda.mem_get_info()
    gpu_budget = min(12*1024**3, free-2*1024**3)
    if gpu_budget <= 0:
        raise MemoryError("Packed profile has insufficient GPU headroom")

    localization = json.loads(args.localization.read_text())
    if len(localization["rows"]) != 24 or not all(r["status"] == "localized" for r in localization["rows"]):
        raise ValueError("Packed projection requires the complete actual24 development cameras")
    sources = [Path(__file__), Path(__file__).with_name("packed_projection.py"), args.model, args.localization]
    identities = [data_identity(p) for p in sources]
    atomic_json(args.output/"PROTOCOL.json", {"sources_and_inputs": identities, "build": build,
        "minimum_memory_reduction": .1, "maximum_stage_time_ratio": 1.1,
        "forward_max_abs": 1e-5, "gradient_atol": 2e-6, "gradient_rtol": 1e-4,
        "radii_and_visibility_exact": True, "three_paired_full24_render_backward_passes": True,
        "packed_densifier_scatter_not_asset_packing": True, "three_seed_training_pending": True,
        "maximum_seconds": 600, "maximum_GPU_bytes": 12*1024**3, "counted": False})
    intr = localization["PINHOLE_camera"]
    original = Pinhole(intr["width"], intr["height"], *intr["params"])
    scaled = original.resized(round(original.width/2), round(original.height/2))
    gaussian = GaussianModel(3)
    gaussian.load_ply(str(args.model))
    gaussian.active_sh_degree = 3
    parameters = [gaussian._xyz, gaussian._scaling, gaussian._rotation, gaussian._opacity,
                  gaussian._features_dc, gaussian._features_rest]
    cameras = []
    for row in localization["rows"]:
        with Image.open(mounted_path(row["image_reference"])) as photo:
            camera = Camera((scaled.width, scaled.height), 0, np.asarray(row["R_CW"]).T,
                np.asarray(row["t_CW"]), 2*math.atan(original.width/(2*original.fx)),
                2*math.atan(original.height/(2*original.fy)), None, photo.convert("RGB"), None,
                row["image_name"], 0, data_device="cpu")
        cameras.append(apply_to_camera(camera, scaled))
    pipeline = SimpleNamespace(antialiasing=False, compute_cov3D_python=False, convert_SHs_python=False)
    background = torch.zeros(3, device="cuda")
    rows, pairs = [], []
    for trial in range(3):
        timings, peaks = {}, {}
        for index, camera in enumerate(cameras):
            values = {}
            for mode in (["dense", "packed"] if trial % 2 == 0 else ["packed", "dense"]):
                for parameter in parameters:
                    parameter.grad = None
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                baseline_bytes = torch.cuda.memory_allocated()
                started = time.perf_counter()
                with packed_projection(mode == "packed"):
                    value = render(camera, gaussian, pipeline, background)
                    value["render"].square().mean().backward()
                    torch.cuda.synchronize()
                timings[mode] = timings.get(mode, 0.)+time.perf_counter()-started
                peaks[mode] = max(peaks.get(mode, 0), torch.cuda.max_memory_allocated()-baseline_bytes)
                if torch.cuda.max_memory_reserved() > gpu_budget:
                    raise MemoryError("Packed profile exceeded its admitted GPU budget")
                values[mode] = {"render": value["render"].detach().cpu(), "radii": value["radii"].detach().cpu(),
                    "proxy_gradient": value["viewspace_points"].grad.detach().cpu(),
                    "gradients": [p.grad.detach().cpu() for p in parameters]}
                del value
            exact = torch.equal(values["dense"]["radii"], values["packed"]["radii"])
            forward_error = float((values["dense"]["render"]-values["packed"]["render"]).abs().max())
            gradients = [torch.allclose(a, b, atol=2e-6, rtol=1e-4) for a, b in zip(
                [values["dense"]["proxy_gradient"], *values["dense"]["gradients"]],
                [values["packed"]["proxy_gradient"], *values["packed"]["gradients"]])]
            record = {"trial": trial, "view": index, "name": camera.image_name, "radii_exact": exact,
                "forward_max_abs": forward_error, "all_seven_gradient_groups_protected": all(gradients)}
            rows.append(record)
            atomic_json(args.output/"STATE.json", {"status": "running", "rows": rows, "pairs": pairs})
            if not exact or forward_error > 1e-5 or not all(gradients):
                atomic_json(args.output/"FAILURE.json", {"status": "failed_numerical_guard", "row": record,
                    "gradient_groups": gradients, "counted": False})
                atomic_json(args.output/"STATE.json", {"status": "failed_numerical_guard", "rows": rows, "pairs": pairs})
                raise AssertionError("Packed actual-view projection violated a registered numerical guard")
        item = {"trial": trial, "full24_forward_backward_seconds": timings, "peak_stage_allocation_bytes": peaks}
        pairs.append(item)
        print(json.dumps(item), flush=True)
    median_time = {m: statistics.median(p["full24_forward_backward_seconds"][m] for p in pairs) for m in ("dense", "packed")}
    median_memory = {m: statistics.median(p["peak_stage_allocation_bytes"][m] for p in pairs) for m in ("dense", "packed")}
    checks = {"all72_actual_view_protections": len(rows) == 72,
        "memory_reduces10percent": median_memory["packed"] <= .9*median_memory["dense"],
        "time_protected": median_time["packed"] <= 1.1*median_time["dense"],
        "sources_and_inputs_unchanged": identities == [data_identity(p) for p in sources]}
    result = {"status": "stage_profile_passed_full_training_pending" if all(checks.values()) else "failed",
        "checks": checks, "median_seconds": median_time, "median_peak_stage_bytes": median_memory,
        "pairs": pairs, "rows": rows, "counted": False, "three_seed_training_pending": True,
        "new_quality_or_accuracy_claim": False}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "pairs")}), flush=True)
    if not all(checks.values()):
        raise AssertionError("Packed stage profile failed its original resource protections")


if __name__ == "__main__":
    main()
