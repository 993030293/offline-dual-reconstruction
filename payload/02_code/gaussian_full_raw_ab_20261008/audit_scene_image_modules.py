"""Check complete native scene image handoff and digest mechanisms on current data."""

import argparse
from contextlib import ExitStack
import gc
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"
V3 = ROOT / "02_code/gaussian_pipeline_optimization_v3_20261005"
sys.path[:0] = [str(V1), str(V2), str(V3)]
from runtime import activate_reference, atomic_json, data_identity
from full_intrinsics import camera_loader_adapter
from render_evaluation import mounted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    activate_reference(ROOT / "04_outputs/box2_3dgs_display_v1_20260930")
    import torch
    from arguments import ModelParams
    from scene import Scene
    from scene.gaussian_model import GaussianModel
    from camera_execution import camera_execution
    from open_image_handoff import open_image_handoff
    from image_source_open_census import image_source_open_census
    from fused_camera_digest import DigestLibrary, original_digest
    from mapped_image_linux_guard import LinuxRSSGuard
    torch.set_num_threads(1)
    admission = json.loads((args.side / "export/TRAINING_ADMISSION.json").read_text())
    dataset = mounted(admission["dataset"])
    count = admission["training_count"]
    paths = sorted((dataset / "images").glob("*.jpg"))
    if len(paths) != count:
        raise ValueError("Every admitted current camera image must enter the native Scene")
    identities = [data_identity(path) for path in paths]
    library = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/scf001/g0/build/fused_camera_digest.so"
    library_identity = data_identity(library)
    source_paths = [Path(__file__), V3 / "open_image_handoff.py", V3 / "image_source_open_census.py",
        V3 / "fused_camera_digest.py", V2 / "prefetch_camera_images.py", V2 / "camera_execution.py"]
    source_ids = [data_identity(path) for path in source_paths]
    started = time.perf_counter()
    rows, digests = [], []
    state = {"status": "running", "sources": source_ids, "complete_current_image_count": count,
        "native_scene_and_camera_objects": True, "all_input_images": identities, "library": library_identity,
        "no_training_iterations": True, "handoff_rows": rows, "digest_rows": digests}
    atomic_json(args.output / "CONTRACT.json", state)
    try:
        with LinuxRSSGuard(args.output, atomic_json, maximum_seconds=1200, maximum_bytes=24*1024**3):
            for trial in range(3):
                models, cameras, traces, durations = {}, {}, {}, {}
                for label in (("native", "handoff") if trial != 1 else ("handoff", "native")):
                    folder = args.output / (str(trial)+"_"+label)
                    folder.mkdir()
                    (folder / "scene").mkdir()
                    parser_native = argparse.ArgumentParser()
                    group = ModelParams(parser_native)
                    parsed = parser_native.parse_args(["-s", str(dataset), "-m", str(folder / "scene"), "-r", "2", "--data_device", "cpu"])
                    random.seed(0)
                    np.random.seed(0)
                    torch.manual_seed(0)
                    gaussian = GaussianModel(3)
                    before = time.perf_counter()
                    with ExitStack() as stack:
                        loaded = stack.enter_context(camera_loader_adapter(dataset))
                        execution = stack.enter_context(camera_execution(opaque_alpha=True, clampfree=True, calibrated_projection=True))
                        stack.enter_context(image_source_open_census(folder, dataset))
                        prefetch = stack.enter_context(open_image_handoff(folder, label == "handoff", 4, 8))
                        scene = Scene(group.extract(parsed), gaussian, shuffle=True)
                        torch.cuda.synchronize()
                    durations[label] = time.perf_counter()-before
                    cameras[label] = scene.getTrainCameras()
                    models[label] = (scene, gaussian)
                    if len(cameras[label]) != count or len(loaded) != count:
                        raise ValueError("The complete native Scene changed its admitted camera denominator")
                    traces[label] = {"prefetch": prefetch, "camera_execution": execution,
                        "handoff": json.loads((folder / "IMAGE_HANDOFF_TRACE.json").read_text()),
                        "file_reads": json.loads((folder / "SOURCE_IMAGE_OPEN_WORK.json").read_text())}
                checks = []
                tensor_fields = ("original_image", "world_view_transform", "projection_matrix", "full_proj_transform", "camera_center")
                for original, candidate in zip(cameras["native"], cameras["handoff"]):
                    literal = {name: torch.equal(getattr(original, name), getattr(candidate, name)) for name in tensor_fields}
                    alpha = original.alpha_mask is None and candidate.alpha_mask is None
                    metadata = all(getattr(original, name) == getattr(candidate, name) for name in ("image_name", "uid", "colmap_id", "image_width", "image_height", "calibrated_intrinsics"))
                    checks.append({"image_name": original.image_name, "all_tensor_values_literal_exact": literal,
                        "native_metadata_exact": metadata, "opaque_alpha_none_both": alpha})
                    if not all(literal.values()) or not metadata or not alpha:
                        raise ValueError("Native image handoff changed current complete camera state")
                initialization = {name: torch.equal(getattr(models["native"][1], name), getattr(models["handoff"][1], name))
                    for name in ("_xyz", "_features_dc", "_features_rest", "_opacity", "_scaling", "_rotation")}
                rows.append({"trial": trial, "Scene_load_seconds_including_closed_prefetch": durations,
                    "all_complete_camera_checks": checks, "complete_initial_Gaussian_tensor_checks": initialization, "traces": traces})
                if not all(initialization.values()):
                    raise ValueError("The complete native Gaussian initialization changed between handoff sides")
                if trial == 0:
                    pixels = [camera.original_image for camera in cameras["handoff"]]
                    library_object = DigestLibrary(library)
                    for digest_trial in range(3):
                        result, seconds = {}, {}
                        order = ("original", "scalar", "SIMD4") if digest_trial != 1 else ("SIMD4", "scalar", "original")
                        for label in order:
                            before = time.perf_counter()
                            result[label] = [original_digest(image) for image in pixels] if label == "original" else library_object.images(pixels, workers=1 if label == "scalar" else 4, vectorized=label == "SIMD4")
                            seconds[label] = time.perf_counter()-before
                        exact = result["original"] == result["scalar"] == result["SIMD4"]
                        digests.append({"trial": digest_trial, "seconds": seconds, "all_current_camera_pixel_digests_exact": exact,
                            "native_layouts_admitted": all(library_object.eligible(image) for image in pixels), "count": len(pixels)})
                        if not exact:
                            raise ValueError("The full native layout digest differs from original complete pixel bytes")
                    del pixels
                del original, candidate
                cameras.clear()
                models.clear()
                del scene, gaussian
                gc.collect()
                torch.cuda.empty_cache()
                atomic_json(args.output / "STATE.json", state)
        state.update(status="complete_current_native_Scene_and_digest_outcomes_recorded", elapsed_seconds=time.perf_counter()-started,
            source_code_unchanged=source_ids == [data_identity(path) for path in source_paths],
            all_current_images_unchanged=identities == [data_identity(path) for path in paths],
            digest_library_unchanged=library_identity == data_identity(library), maximum_CUDA_reserved_bytes=torch.cuda.max_memory_reserved(),
            primary_training_or_model_quality_not_modified=True)
        atomic_json(args.output / "RESULT.json", state)
    except BaseException as error:
        state.update(status="failed_current_Scene_image_module_audit", error=repr(error), elapsed_seconds=time.perf_counter()-started)
        atomic_json(args.output / "FAILURE.json", state)
        raise


if __name__ == "__main__":
    main()
