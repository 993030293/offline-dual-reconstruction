"""Exercise actual bounded-prefetch order, worker failures and cleanup."""

import argparse
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import activate_reference, atomic_json, data_identity
from prefetch_camera_images import prefetch_camera_images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    activate_reference(args.runtime)
    import torch
    from PIL import Image
    import scene
    import scene.cameras as camera_module
    import utils.camera_utils as camera_utils
    original_load = camera_utils.loadCam
    original_list, original_prepare = scene.cameraList_from_camInfos, camera_module.PILtoTorch
    options = SimpleNamespace(data_device="cpu", resolution=2)
    infos = []
    for index, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255), (73, 111, 203))):
        path = args.output/f"fixture_{index}.png"
        Image.new("RGB", (16, 20), color).save(path)
        infos.append(SimpleNamespace(image_path=str(path), width=16, height=20, depth_path=""))
    def native_camera(options, index, info, scale, synthetic, test):
        with Image.open(info.image_path) as photo:
            return camera_module.PILtoTorch(photo, (round(photo.width/(scale*options.resolution)),
                                                    round(photo.height/(scale*options.resolution))))
    checks = {}
    try:
        camera_utils.loadCam = native_camera
        expected = [native_camera(options, i, info, 1., False, False) for i, info in enumerate(infos)]
        with prefetch_camera_images(2, 2) as receipt:
            actual = scene.cameraList_from_camInfos(infos, 1., options, False, False)
            checks["empty_list"] = scene.cameraList_from_camInfos([], 1., options, False, False) == []
        checks["all_fixture_pixels_and_strides_exact"] = all(torch.equal(a, b) and a.stride() == b.stride() for a, b in zip(expected, actual))
        checks["bounded_ordered_consumption"] = receipt["maximum_pending"] <= 2 and receipt["consumed"] == 4
        cases = [("worker_metadata_failure", [SimpleNamespace(**{**vars(infos[0]), "width": 18})], options, False),
                 ("missing_image", [SimpleNamespace(**{**vars(infos[0]), "image_path": str(args.output/"absent.png")})], options, False),
                 ("unsupported_GPU_residency", infos, SimpleNamespace(data_device="cuda", resolution=2), False),
                 ("unsupported_depth", [SimpleNamespace(**{**vars(infos[0]), "depth_path": "depth.png"})], options, False),
                 ("unsupported_synthetic", infos, options, True)]
        errors = {}
        for label, records, config, synthetic in cases:
            try:
                with prefetch_camera_images(2, 2):
                    scene.cameraList_from_camInfos(records, 1., config, synthetic, False)
            except (ValueError, OSError) as error:
                checks[label] = True
                errors[label] = repr(error)
            else:
                checks[label] = False
            checks[label+"_restored"] = (scene.cameraList_from_camInfos is original_list
                and camera_module.PILtoTorch is original_prepare
                and not any(t.is_alive() and t.name.startswith("camera-decode") for t in threading.enumerate()))
        result = {"status": "lifecycle_passed" if all(checks.values()) else "failed", "checks": checks,
                  "errors": errors, "adapter_source": data_identity(Path(__file__).with_name("prefetch_camera_images.py")), "counted": False}
        atomic_json(args.output/"RESULT.json", result)
        print(checks)
        if not all(checks.values()):
            raise ValueError("A real prefetch lifecycle guard failed")
    finally:
        camera_utils.loadCam = original_load


if __name__ == "__main__":
    main()
