"""Exercise native Pillow and prefetch semantics with an explicit CPU stand-in."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"), str(ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005")]
from runtime import atomic_json, data_identity
from open_image_handoff import open_image_handoff
from image_source_open_census import image_source_open_census

OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/shf001/g0"


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    fixtures = OUT / "fixtures/images"
    fixtures.mkdir(parents=True)
    official = ROOT / "04_outputs/box2_3dgs_display_v1_20260930/source/gaussian-splatting/utils/general_utils.py"
    spec = importlib.util.spec_from_file_location("native_PILtoTorch_for_cpu_component", official)
    general = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(general)
    values = np.arange(19 * 17 * 3, dtype=np.uint16).reshape(19, 17, 3).astype(np.uint8)
    infos = []
    for index in range(7):
        path = fixtures / (str(index) + (".jpg" if index % 2 else ".png"))
        Image.fromarray(np.roll(values, index, axis=1)).save(path)
        infos.append(SimpleNamespace(image_path=str(path), width=17, height=19, depth_path="", name=str(index)))
    rgba = fixtures / "rgba.png"
    Image.new("RGBA", (17, 19), (11, 23, 37, 89)).save(rgba)
    gif = fixtures / "single.gif"
    Image.new("RGB", (17, 19), (11, 23, 37)).save(gif)
    broken = fixtures / "broken.jpg"
    broken.write_bytes(b"not an image")
    bad_info = lambda path, width=17: SimpleNamespace(image_path=str(path), width=width, height=19, depth_path="", name=path.name)
    args = SimpleNamespace(data_device="cpu", resolution=2)
    module_names = ("scene", "scene.cameras", "utils", "utils.camera_utils")
    previous_modules = {name: sys.modules.get(name) for name in module_names}
    modules = {name: ModuleType(name) for name in module_names}
    scene, camera_module, utils, camera_utils = [modules[name] for name in module_names]
    scene.cameras, utils.camera_utils = camera_module, camera_utils
    scene.cameraList_from_camInfos = lambda *args: []
    camera_module.PILtoTorch = general.PILtoTorch
    prefix = []

    def load_cam(arguments, index, info, scale, synthetic, test):
        image = Image.open(info.image_path)
        if image.mode != "RGB":
            raise ValueError("CPU stand-in requires the native RGB camera scope")
        resolution = (round(image.width / (scale * arguments.resolution)), round(image.height / (scale * arguments.resolution)))
        tensor = camera_module.PILtoTorch(image, resolution)
        value = {"index": index, "name": info.name, "mode": image.mode, "size": image.size,
                 "shape": tuple(tensor.shape), "stride": tuple(tensor.stride()),
                 "pixels": hashlib.sha256(tensor.numpy().tobytes()).hexdigest()}
        prefix.append(value)
        return value

    camera_utils.loadCam = load_cam
    sys.modules.update(modules)
    saved = (scene.cameraList_from_camInfos, camera_module.PILtoTorch, Image.open)
    sources = [Path(__file__), CODE / "open_image_handoff.py", CODE / "image_source_open_census.py",
               ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005/prefetch_camera_images.py", official]
    identities = [data_identity(path) for path in sources]
    cases = [("mixed_formats", [infos]), ("duplicate_same_metadata_object", [[infos[0], infos[0], infos[1], infos[0]]]),
             ("multiple_lists_and_empty", [[], infos[:3], [], infos[3:]]),
             ("dimension_failure", [[*infos[:2], bad_info(Path(infos[2].image_path), 16), *infos[3:]]]),
             ("native_missing_failure", [[*infos[:2], bad_info(fixtures / "missing.jpg"), *infos[3:]]]),
             ("native_corrupt_failure", [[*infos[:2], bad_info(broken), *infos[3:]]]),
             ("native_RGBA_rejection", [[*infos[:2], bad_info(rgba), *infos[3:]]]),
             ("unsupported_GIF_native_rejection", [[*infos[:2], bad_info(gif), *infos[3:]]])]
    checks, results = {}, []
    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for label, lists in cases:
            sides = []
            for enabled in (False, True):
                leaf = OUT / (label + ("_handoff" if enabled else "_control"))
                leaf.mkdir()
                prefix.clear()
                result = {"error": None}
                with image_source_open_census(leaf, fixtures.parent):
                    try:
                        with open_image_handoff(leaf, enabled, 4, 8) as receipt:
                            result["lists"] = [scene.cameraList_from_camInfos(rows, 1., args, False, False) for rows in lists]
                    except Exception as error:
                        result["error"] = {"type": type(error).__name__, "message": str(error)}
                result["prefix"] = list(prefix)
                result["trace"] = json.loads((leaf / "IMAGE_HANDOFF_TRACE.json").read_text())
                result["work"] = json.loads((leaf / "SOURCE_IMAGE_OPEN_WORK.json").read_text())
                result["bindings_restored"] = saved == (scene.cameraList_from_camInfos, camera_module.PILtoTorch, Image.open)
                sides.append(result)
            left, right = sides
            checks[label + "_actual_error_and_completed_prefix_exact"] = left["error"] == right["error"] and left["prefix"] == right["prefix"] and left.get("lists") == right.get("lists")
            checks[label + "_native_bindings_and_workers_and_images_closed"] = all(row["bindings_restored"] and row["trace"]["original_functions_restored"] and row["trace"]["owned_decode_threads_closed"] and row["trace"]["recorded"] == row["trace"]["closed_images"] and row["trace"]["remaining_images"] == 0 and row["trace"]["maximum_owned_images"] <= 8 and row["work"]["native_functions_restored"] for row in sides)
            results.append({"case": label, "sides": sides})
        checks["all_original_component_sources_unchanged"] = identities == [data_identity(path) for path in sources]
        checks["normal_native_read_work_exact_half"] = all(row["sides"][1]["work"]["successful_native_read_opens"] * 2 == row["sides"][0]["work"]["successful_native_read_opens"] for row in results[:3])
        checks["no_owned_decode_worker_remained"] = not any(t.name.startswith("camera-decode") for t in threading.enumerate())
    finally:
        torch.set_num_threads(original_threads)
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    result = {"status": "component_complete_not_full_native_Camera_acceptance" if all(checks.values()) else "failed_verification",
              "checks": checks, "cases": results, "sources": identities, "CPU_camera_standin_explicit": True,
              "actual_native_Pillow_and_PILtoTorch": True, "training_or_full_Camera_validation": "not_run_here", "new_optimization_count": 0}
    atomic_json(OUT / "RESULT.json", result)
    print(json.dumps({"status": result["status"], "checks": checks, "new": 0}))
    if not all(checks.values()):
        raise AssertionError("A native image handoff component contract failed")


if __name__ == "__main__":
    main()
