"""Build complete owned RGB datasets using a native carrier and cached sampling."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import psutil
import pycolmap

from camera_pixel_remap import SUPPORTED, camera_fields
from complete_carrier_configuration import complete_configuration, OWN
from native_camera_sampling import NativeCameraPixelRemap

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import data_identity, atomic_json


def closed(identity):
    try:
        return psutil.Process(identity["pid"]).create_time() != identity["creation_time"]
    except psutil.NoSuchProcess:
        return True


def standard_raw_jpeg(path, size):
    try:
        with Image.open(path) as image:
            return (image.format == "JPEG" and image.mode == "RGB" and image.size == size
                    and dict(image.getexif()) == {}
                    and dict(image.info) == {"jfif": 257, "jfif_version": (1, 1),
                                             "jfif_unit": 0, "jfif_density": (1, 1)})
    except OSError:
        # A rejected admission read delegates to the original complete reader.
        return False


def sample_stripe(arguments):
    paths, destination, camera_values, option_values, library, sources = arguments
    started = time.perf_counter()
    if os.environ.get("OPENIMAGEIO_THREADS") != "1":
        raise ValueError("The private process did not inherit its bounded image thread setting")
    if sources != [data_identity(row["path"]) for row in sources]:
        raise ValueError("A private sampling process source changed")
    process = psutil.Process()
    owner = {"pid": process.pid, "creation_time": process.create_time()}
    camera = pycolmap.Camera(**camera_values)
    options = pycolmap.UndistortCameraOptions(**option_values)
    operator = NativeCameraPixelRemap(pycolmap.undistort_image, library)
    before_rng = np.random.get_state()
    peak_threads = process.num_threads()
    counts = {"read_seconds": 0., "sample_seconds": 0., "write_seconds": 0.}
    try:
        for path in paths:
            path = Path(path)
            before = time.perf_counter()
            bitmap = pycolmap.Bitmap.read(path, as_rgb=True)
            counts["read_seconds"] += time.perf_counter() - before
            if bitmap is None:
                raise OSError("The original bitmap reader rejected a preadmitted JPEG")
            before = time.perf_counter()
            warped, calibration = operator(options, bitmap, camera)
            counts["sample_seconds"] += time.perf_counter() - before
            if (warped.width, warped.height) != (calibration.width, calibration.height):
                raise ValueError("The original complete warped bitmap dimensions changed")
            before = time.perf_counter()
            warped.set_jpeg_quality(97)
            if not warped.write(Path(destination) / path.name):
                raise OSError("The original JPEG97 writer failed")
            counts["write_seconds"] += time.perf_counter() - before
            peak_threads = max(peak_threads, process.num_threads())
        after_rng = np.random.get_state()
        if not (before_rng[0] == after_rng[0] and np.array_equal(before_rng[1], after_rng[1]) and before_rng[2:] == after_rng[2:]):
            raise ValueError("A private deterministic sampling task changed the NumPy RNG")
        if sources != [data_identity(row["path"]) for row in sources]:
            raise ValueError("A private sampling process source changed while running")
    finally:
        operator.close()
    return {"owner": owner, "image_names": [Path(path).name for path in paths],
            "count": len(paths), "component_task_seconds": counts,
            "whole_stripe_seconds": time.perf_counter() - started,
            "actual_process_thread_peak": peak_threads, "inherited_OPENIMAGEIO_THREADS": "1",
            "sampler_trace": operator.trace, "original_RNG_preserved": True}


def cached_dataset(output, model_path, images, names, library, proof):
    started = time.perf_counter()
    output, proof = Path(output).resolve(), Path(proof).resolve()
    output.relative_to(OWN.resolve())
    proof.relative_to(OWN.resolve())
    names = tuple(names)
    if output.exists() or proof.exists() or not names or len(set(names)) != len(names):
        raise ValueError("A cached dataset requires fresh outputs and complete unique names")
    native = pycolmap.undistort_images
    options = pycolmap.UndistortCameraOptions()
    model = pycolmap.Reconstruction(model_path)
    lookup = {image.name: image for image in model.images.values() if image.has_pose}
    admitted = (pycolmap.__version__ == "4.1.1" and np.__version__ == "2.3.5" and len(model.cameras) == 1
                and set(names) == set(lookup) and list(names) == sorted(names)
                and all(type(name) is str and Path(name).name == name and not any(c in name for c in "\r\n\x00") for name in names))
    camera = next(iter(model.cameras.values()), None)
    admitted = admitted and camera.model_name in SUPPORTED - {"PINHOLE", "SIMPLE_PINHOLE"} and camera.width >= 2 and camera.height >= 2
    admitted = admitted and 4 * camera.width * camera.height * 25 <= 256 * 1024**2
    admitted = admitted and all(standard_raw_jpeg(Path(images) / name, (camera.width, camera.height)) for name in names)
    proof.mkdir(parents=True, exist_ok=False)
    if not admitted:
        native(output, model_path, images, image_names=names, output_type="COLMAP",
               num_patch_match_src_images=0, jpeg_quality=97, num_threads=8)
        result = {"route": "original_before_any_cached_dataset_write", "closed": True,
                  "complete_dataset_seconds": time.perf_counter() - started, "new_optimization_count": 0}
        atomic_json(proof / "RESULT.json", result)
        return result
    native(output, model_path, images, image_names=names[:1], output_type="COLMAP",
           num_patch_match_src_images=0, jpeg_quality=97, num_threads=8)
    configuration = complete_configuration(output, names, names[:1], proof / "configuration")
    source_paths = [Path(__file__), Path(__file__).with_name("native_camera_sampling.py"),
                    Path(__file__).with_name("camera_pixel_remap.py"), Path(__file__).with_name("complete_carrier_configuration.py"),
                    Path(__file__).with_name("native_camera_sampling.cpp"),
                    ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005/runtime.py",
                    Path(library)]
    sources = [data_identity(path) for path in source_paths]
    values = camera_fields(camera)
    stripes = [[str(Path(images) / name) for name in names[1 + index::4]] for index in range(4)]
    previous = os.environ.get("OPENIMAGEIO_THREADS")
    results = []
    try:
        os.environ["OPENIMAGEIO_THREADS"] = "1"
        with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
            arguments = [(paths, str(output / "images"), values, options.todict(), str(library), sources)
                         for paths in stripes if paths]
            results = list(pool.map(sample_stripe, arguments, chunksize=1))
    finally:
        if previous is None:
            os.environ.pop("OPENIMAGEIO_THREADS", None)
        else:
            os.environ["OPENIMAGEIO_THREADS"] = previous
    owners = {(row["owner"]["pid"], row["owner"]["creation_time"]): row["owner"] for row in results}
    emitted = [name for row in results for name in row["image_names"]]
    actual = sorted(path.name for path in (output / "images").iterdir() if path.is_file())
    checks = {"all_complete_original_names_once": actual == sorted(names) and len(emitted) == len(set(emitted)) == len(names) - 1
              and set(emitted) == set(names[1:]),
              "all_actual_sampling_processes_exited_after_shutdown": all(closed(row) for row in owners.values()),
              "all_actual_sampler_DLLs_closed": all(row["sampler_trace"]["closed"] and row["sampler_trace"]["native_DLL_closed"] for row in results),
              "single_build_per_actual_stripe": all(row["sampler_trace"]["builds"] == 1 for row in results),
              "parent_environment_restored": os.environ.get("OPENIMAGEIO_THREADS") == previous,
              "sources_unchanged": sources == [data_identity(path) for path in source_paths]}
    if not all(checks.values()):
        raise AssertionError("A complete cached dataset protection failed")
    result = {"route": "native_full_model_carrier_cached_four_process_RGB_sampling", "checks": checks,
              "closed": True, "configuration": configuration, "workers": results,
              "complete_dataset_seconds": time.perf_counter() - started,
              "component_stripe_times_are_not_whole_wall": True,
              "unrecognized_metadata_or_camera_uses_original_before_mapping": True, "new_optimization_count": 0}
    atomic_json(proof / "RESULT.json", result)
    return result
