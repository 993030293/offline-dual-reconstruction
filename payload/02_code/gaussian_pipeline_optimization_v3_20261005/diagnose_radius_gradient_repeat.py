"""Compare unmodified gradient repeatability with the failed fused-radius observation."""

import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path[:0] = [str(V1), str(ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005")]
from runtime import activate_reference, atomic_json, data_identity
from full_intrinsics import Pinhole, apply_to_camera
from evaluate_pair import mounted_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "model", "localization", "build", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["TRITON_CACHE_DIR"] = str(args.output/"triton_cache")
    activate_reference(args.runtime)
    import torch
    import gsplat_adapter
    from fused_projection_radius import fused_radius_adapter
    from scene.cameras import Camera
    from scene.gaussian_model import GaussianModel
    from arguments import PipelineParams
    from vectorized_gaussian_read import vectorized_gaussian_read
    from utils.loss_utils import l1_loss, ssim
    gsplat_adapter.activate_gsplat(args.build)
    paths = [Path(__file__), Path(__file__).with_name("fused_projection_radius.py"), args.model, args.localization]
    identities = [data_identity(p) for p in paths]
    atomic_json(args.output/"PROTOCOL.json", {"sources_inputs": identities, "same_previously_failed_camera_index": 21,
        "six_same_parameter_forward_backward_no_optimizer": True, "native_native_control": True,
        "gradient_atol_unchanged": 2e-6, "gradient_rtol_unchanged": 1e-4,
        "diagnostic_not_candidate_acceptance": True, "new_optimization_count": 0})
    local = json.loads(args.localization.read_text())
    row, intr = local["rows"][21], local["PINHOLE_camera"]
    native = Pinhole(intr["width"], intr["height"], *intr["params"])
    scaled = native.resized(round(native.width/2), round(native.height/2))
    with Image.open(mounted_path(row["image_reference"])) as photo:
        camera = Camera((scaled.width, scaled.height), 0, np.asarray(row["R_CW"]).T, np.asarray(row["t_CW"]),
            2*math.atan(native.width/(2*native.fx)), 2*math.atan(native.height/(2*native.fy)), None,
            photo.convert("RGB"), None, row["image_name"], 0, data_device="cpu")
    apply_to_camera(camera, scaled)
    model = GaussianModel(3)
    with vectorized_gaussian_read(True):
        model.load_ply(str(args.model))
    native_parser = argparse.ArgumentParser()
    group = PipelineParams(native_parser)
    pipeline = group.extract(native_parser.parse_args([]))
    background = torch.zeros(3, device="cuda")
    names = ("_xyz", "_scaling", "_rotation", "_opacity", "_features_dc", "_features_rest")
    outputs = []
    for mode in ("native", "native", "fused", "native", "native", "fused"):
        for name in names:
            getattr(model, name).grad = None
        with fused_radius_adapter(gsplat_adapter, mode == "fused"):
            package = gsplat_adapter.render(camera, model, pipeline, background)
            image = package["render"]
            target = camera.original_image.cuda()
            loss = .8*l1_loss(image, target)+.2*(1-ssim(image, target))
            loss.backward()
            outputs.append({"mode": mode, "pixels": image.detach().cpu(), "loss": float(loss.detach()),
                "radii": package["radii"].detach().cpu(), "gradients": [getattr(model, n).grad.detach().cpu() for n in names],
                "density": package["viewspace_points"].grad.detach().cpu()})
    rows = []
    for left, right in ((0, 1), (0, 2), (3, 4), (3, 5), (1, 4)):
        a, b = outputs[left], outputs[right]
        details = []
        for name, x, y in zip(names, a["gradients"], b["gradients"]):
            difference = (x-y).abs()
            allowed = 2e-6+1e-4*y.abs()
            details.append({"parameter": name, "max_abs": float(difference.max()),
                "count_outside_original_tolerance": int((difference > allowed).sum()),
                "original_guard_passed": bool(torch.allclose(x, y, atol=2e-6, rtol=1e-4))})
        rows.append({"left": left, "right": right, "modes": [a["mode"], b["mode"]],
            "pixels_bit_exact": torch.equal(a["pixels"], b["pixels"]), "loss_exact": a["loss"] == b["loss"],
            "radii_exact": torch.equal(a["radii"], b["radii"]), "parameters": details,
            "density_guard": torch.allclose(a["density"], b["density"], atol=2e-6, rtol=1e-4)})
    result = {"status": "same_source_gradient_repeat_diagnostic_complete", "rows": rows, "sources_inputs": identities,
        "sources_unchanged": identities == [data_identity(p) for p in paths], "new_optimization_count": 0,
        "failed_original_R95_guard_not_overridden": True, "full_training_executed": False}
    atomic_json(args.output/"RESULT.json", result)
    print({"status": result["status"], "rows": rows}, flush=True)


if __name__ == "__main__":
    main()
