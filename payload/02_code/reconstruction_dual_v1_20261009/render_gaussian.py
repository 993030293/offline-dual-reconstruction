"""Read the completed Gaussian model and render six shared capture cameras."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root
    project = Path(__file__).resolve().parents[2]
    v1 = project / "02_code/gaussian_pipeline_optimization_v1_20261005"
    sys.path.insert(0, str(v1))
    from runtime import activate_reference, atomic_json
    runtime = root / "shared/gaussian_runtime"
    activate_reference(runtime)
    import torch
    from arguments import ModelParams, OptimizationParams, PipelineParams
    from scene import Scene, GaussianModel
    from full_intrinsics import camera_loader_adapter
    from gsplat_adapter import activate_gsplat, render
    activate_gsplat(project / "04_outputs/gaussian_pipeline_optimization_v1_20261005/p6_gsplat_build_001")
    torch.set_num_threads(8)
    configuration = json.loads((root / "CONFIG.json").read_text())
    side = root / configuration["case"] / "optimized"
    model_path = side / "training/model"
    native_parser = argparse.ArgumentParser()
    groups = ModelParams(native_parser), OptimizationParams(native_parser), PipelineParams(native_parser)
    namespace = native_parser.parse_args([])
    for key, value in json.loads((runtime / "MAIN_CONFIG.json").read_text())["parameters"].items():
        setattr(namespace, key, value)
    namespace.source_path, namespace.model_path = str(root / "shared/gaussian"), str(model_path)
    dataset, opt, pipe = [group.extract(namespace) for group in groups]
    output = root / "gaussian_diagnostics"
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    rows = []
    shared = json.loads((root / "shared/RESULT.json").read_text())
    capture_names = sorted(shared["registered_names"],
        key=lambda name: int(Path(name).stem.rsplit("_", 1)[1]))
    if len(capture_names) != shared["registered_frames"] or len(set(capture_names)) != len(capture_names):
        raise ValueError("The shared diagnostic camera list is inconsistent")
    indices = np.linspace(0, len(capture_names) - 1, 6).round().astype(int).tolist()
    with torch.no_grad(), camera_loader_adapter(Path(dataset.source_path)) as calibration:
        gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
        scene = Scene(dataset, gaussians, load_iteration=configuration["gaussian_iterations"], shuffle=False)
        cameras = {Path(view.image_name).stem: view for view in scene.getTrainCameras()}
        if set(cameras) != {Path(name).stem for name in capture_names}:
            raise ValueError("Gaussian diagnostic camera denominator changed")
        background = torch.zeros(3, dtype=torch.float32, device="cuda")
        for index in indices:
            view = cameras[Path(capture_names[index]).stem]
            rgb = render(view, gaussians, pipe, background, use_trained_exp=False, separate_sh=False)["render"].clamp(0, 1)
            truth = view.original_image.to(rgb.device)
            if rgb.shape != truth.shape or not bool(torch.isfinite(rgb).all()):
                raise ValueError("Gaussian rendering broke the shared image contract")
            mse, mae = float(torch.mean((rgb - truth)**2)), float(torch.mean(torch.abs(rgb - truth)))
            image = np.rint(rgb.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(image).save(output / f"render_{index:04d}.png")
            rows.append({"index": index, "image_name": view.image_name, "width": view.image_width,
                "height": view.image_height, "training_view_MAE": mae,
                "training_view_PSNR_dB": -10 * np.log10(max(mse, 1e-12)),
                "pixel_standard_deviation": float(image.std()), "calibrated_intrinsics": view.calibrated_intrinsics})
    atomic_json(output / "RESULT.json", {"status": "six_actual_capture_renders_verified", "rows": rows,
        "camera_count": len(cameras), "elapsed_seconds": time.monotonic() - started,
        "strict_heldout_quality_evaluation": False, "independent_geometry_accuracy": False,
        "calibrated_cameras": len(calibration), "camera_selection_source": str(root / "shared/RESULT.json")})


if __name__ == "__main__":
    main()
