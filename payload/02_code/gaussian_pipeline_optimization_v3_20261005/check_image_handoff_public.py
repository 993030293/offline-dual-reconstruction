"""Check all public development images through native Pillow and CPU preparation."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"), str(ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005")]
from runtime import atomic_json, data_identity
from open_image_handoff import open_image_handoff
from image_source_open_census import image_source_open_census

OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/shf001/g1"


def main():
    OUT.mkdir(exist_ok=False)
    public = ROOT / "04_outputs/public_full_e26_scene_v3_20260929T141800Z/attempt02/01_visual_workspace"
    paths = sorted(path for path in (public / "images").iterdir() if path.is_file())
    if len(paths) != 360:
        raise ValueError("The complete existing public development image pool changed")
    official = ROOT / "04_outputs/box2_3dgs_display_v1_20260930/source/gaussian-splatting/utils/general_utils.py"
    spec = importlib.util.spec_from_file_location("native_public_PILtoTorch", official)
    general = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(general)
    names = ("scene", "scene.cameras", "utils", "utils.camera_utils")
    previous = {name: sys.modules.get(name) for name in names}
    modules = {name: ModuleType(name) for name in names}
    scene, camera_module, utils, camera_utils = [modules[name] for name in names]
    scene.cameras, utils.camera_utils = camera_module, camera_utils
    scene.cameraList_from_camInfos = lambda *args: []
    camera_module.PILtoTorch = general.PILtoTorch
    sources = [data_identity(path) for path in (Path(__file__), CODE / "open_image_handoff.py", CODE / "image_source_open_census.py", official)]
    saved_open = Image.open

    def load_cam(arguments, index, info, scale, synthetic, test):
        image = Image.open(info.image_path)
        if image.mode != "RGB":
            raise ValueError("The native RGB component scope changed on a public frame")
        resolution = (round(image.width / 2), round(image.height / 2))
        tensor = camera_module.PILtoTorch(image, resolution)
        return {"name": info.name, "size": image.size, "dtype": str(tensor.dtype),
                "shape": list(tensor.shape), "stride": list(tensor.stride()), "storage_offset": tensor.storage_offset(),
                "requires_grad": tensor.requires_grad, "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest()}

    camera_utils.loadCam = load_cam
    sys.modules.update(modules)
    sides = []
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for enabled in (False, True):
            leaf = OUT / ("handoff" if enabled else "control")
            leaf.mkdir()
            with image_source_open_census(leaf, public):
                before = [data_identity(path) for path in paths]
                infos = []
                for path in paths:
                    with Image.open(path) as image:
                        infos.append(SimpleNamespace(image_path=str(path), width=image.width, height=image.height, depth_path="", name=path.name))
                with open_image_handoff(leaf, enabled, 4, 8) as receipt:
                    rows = scene.cameraList_from_camInfos(infos, 1., SimpleNamespace(data_device="cpu", resolution=2), False, False)
                after = [data_identity(path) for path in paths]
            sides.append({"rows": rows, "inputs_before": before, "inputs_after": after,
                          "prefetch": receipt, "trace": json.loads((leaf / "IMAGE_HANDOFF_TRACE.json").read_text()),
                          "work": json.loads((leaf / "SOURCE_IMAGE_OPEN_WORK.json").read_text())})
    finally:
        torch.set_num_threads(original_threads)
        for name, native in previous.items():
            if native is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = native
    checks = {
        "all360_full_native_RGB_pixels_layout_dtype_offset_exact": len(sides[0]["rows"]) == len(sides[1]["rows"]) == 360 and sides[0]["rows"] == sides[1]["rows"],
        "all360_public_file_bytes_and_membership_preserved": sides[0]["inputs_before"] == sides[0]["inputs_after"] == sides[1]["inputs_before"] == sides[1]["inputs_after"],
        "original4_workers8_queue_and_all360_native_preparations": all(row["prefetch"]["workers"] == 4 and row["prefetch"]["capacity"] == 8 and row["prefetch"]["maximum_pending"] <= 8 and row["prefetch"]["prepared"] == row["prefetch"]["consumed"] == 360 for row in sides),
        "all_owned_images_threads_file_hooks_and_bindings_restored": all(row["trace"]["original_functions_restored"] and row["trace"]["owned_decode_threads_closed"] and row["trace"]["remaining_images"] == 0 and row["trace"]["recorded"] == row["trace"]["closed_images"] and row["work"]["native_functions_restored"] for row in sides) and Image.open is saved_open,
        "all360_actual_loaded_Image_objects_handed_off_without_fallback": sides[1]["trace"]["handed_off"] == 360 and not sides[1]["trace"]["failed_future_native_header_fallbacks"] and not sides[1]["trace"]["unsupported_native_header_fallbacks"],
        "all_related_own_sources_unchanged": sources == [data_identity(row["path"]) for row in sources],
        "no_actual_owned_decode_threads_retained": not any(t.name.startswith("camera-decode") for t in threading.enumerate()),
    }
    atomic_json(OUT / "RESULT.json", {"status": "public360_native_RGB_preparation_component_exact_not_full_pose_GT_or_training" if all(checks.values()) else "failed_verification",
        "checks": checks, "sources": sources, "sides": sides, "actual_related_source_read_work": sum(row["work"]["successful_native_read_opens"] for row in sides),
        "already_used_development_images_not_blind": True, "CPU_camera_standin_explicit": True, "new_optimization_count": 0})
    print({"checks": checks, "public_frames": 360, "source_read_work": [row["work"]["successful_native_read_opens"] for row in sides], "new": 0})
    if not all(checks.values()):
        raise AssertionError("A full public development preparation component check failed")


if __name__ == "__main__":
    main()
