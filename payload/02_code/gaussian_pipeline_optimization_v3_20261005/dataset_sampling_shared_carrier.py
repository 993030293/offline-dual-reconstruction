"""Share one complete immutable sampling map across bounded native workers."""

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from multiprocessing import shared_memory
import os
from pathlib import Path
import time

import numpy as np
import psutil
import pycolmap

from dataset_sampling_carrier import ROOT, OWN, closed, standard_raw_jpeg
from camera_pixel_remap import SUPPORTED, camera_fields
from complete_carrier_configuration import complete_configuration
from native_camera_sampling import NativeCameraPixelRemap
from runtime import atomic_json, data_identity


class SharedNativeCameraPixelRemap(NativeCameraPixelRemap):
    def __init__(self, library, camera, options, calibration, headers):
        self.shared_segments = []
        super().__init__(pycolmap.undistort_image, library)
        try:
            maps = []
            for header in headers:
                segment = shared_memory.SharedMemory(name=header["name"], create=False)
                self.shared_segments.append(segment)
                array = np.ndarray(tuple(header["shape"]), dtype=np.dtype(header["dtype"]), buffer=segment.buf)
                if array.nbytes != header["bytes"] or hashlib.sha256(segment.buf[:array.nbytes]).hexdigest() != header["sha256"]:
                    raise ValueError("A complete shared geometry segment changed before admission")
                array.setflags(write=False)
                maps.append(array)
            self.maps = tuple(maps)
            self.calibration = calibration
            self.key = json.dumps({"camera": camera_fields(camera), "options": options.todict()}, sort_keys=True, allow_nan=False)
            self.trace.update(shared_parent_map=True, shared_segments=len(headers),
                map_bytes=sum(array.nbytes for array in self.maps), shared_handles_closed=False)
            del maps, array
        except BaseException:
            self.close()
            raise

    def build(self, camera, calibration):
        raise ValueError("An admitted shared-map worker cannot silently allocate a replacement map")

    def close(self):
        super().close()
        for segment in self.shared_segments:
            segment.close()
        self.trace["shared_handles_closed"] = all(segment.buf is None for segment in self.shared_segments)
        self.shared_segments = []


def sample_stripe(arguments):
    paths, destination, camera_values, option_values, calibration, headers, library, sources = arguments
    started = time.perf_counter()
    if os.environ.get("OPENIMAGEIO_THREADS") != "1" or sources != [data_identity(row["path"]) for row in sources]:
        raise ValueError("A shared-map worker runtime or executed source changed")
    process = psutil.Process()
    owner = {"pid": process.pid, "creation_time": process.create_time()}
    camera = pycolmap.Camera(**camera_values)
    options = pycolmap.UndistortCameraOptions(**option_values)
    before_rng = np.random.get_state()
    operator = SharedNativeCameraPixelRemap(library, camera, options, calibration, headers)
    counts = {"read_seconds": 0., "sample_seconds": 0., "write_seconds": 0.}
    threads = process.num_threads()
    try:
        for path in paths:
            path = Path(path)
            before = time.perf_counter()
            bitmap = pycolmap.Bitmap.read(path, as_rgb=True)
            counts["read_seconds"] += time.perf_counter() - before
            if bitmap is None:
                raise OSError("The original bitmap reader rejected a preadmitted shared-map JPEG")
            before = time.perf_counter()
            warped, result_camera = operator(options, bitmap, camera)
            counts["sample_seconds"] += time.perf_counter() - before
            if (warped.width, warped.height) != (result_camera.width, result_camera.height):
                raise ValueError("A complete original warped bitmap changed its dimensions")
            before = time.perf_counter()
            warped.set_jpeg_quality(97)
            if not warped.write(Path(destination) / path.name):
                raise OSError("The original complete JPEG97 writer failed")
            counts["write_seconds"] += time.perf_counter() - before
            threads = max(threads, process.num_threads())
        after_rng = np.random.get_state()
        if not (before_rng[0] == after_rng[0] and np.array_equal(before_rng[1], after_rng[1]) and before_rng[2:] == after_rng[2:]):
            raise ValueError("A deterministic shared-map task changed the NumPy RNG")
        if sources != [data_identity(row["path"]) for row in sources]:
            raise ValueError("A shared-map worker source changed while executing")
        for segment, header in zip(operator.shared_segments, headers):
            if hashlib.sha256(segment.buf[:header["bytes"]]).hexdigest() != header["sha256"]:
                raise ValueError("A complete shared geometry segment changed during sampling")
    finally:
        operator.close()
    return {"owner": owner, "image_names": [Path(path).name for path in paths], "count": len(paths),
        "component_task_seconds": counts, "whole_stripe_seconds": time.perf_counter() - started,
        "actual_process_thread_peak": threads, "inherited_OPENIMAGEIO_THREADS": "1",
        "sampler_trace": operator.trace, "original_RNG_preserved": True}


def names_unavailable(headers):
    for header in headers:
        try:
            segment = shared_memory.SharedMemory(name=header["name"], create=False)
        except FileNotFoundError:
            continue
        segment.close()
        return False
    return True


def cached_dataset(output, model_path, images, names, library, proof):
    started = time.perf_counter()
    output, proof = Path(output).resolve(), Path(proof).resolve()
    output.relative_to(OWN.resolve())
    proof.relative_to(OWN.resolve())
    names = tuple(names)
    if output.exists() or proof.exists() or not names or len(set(names)) != len(names):
        raise ValueError("A shared dataset requires fresh outputs and complete unique names")
    native = pycolmap.undistort_images
    options = pycolmap.UndistortCameraOptions()
    model = pycolmap.Reconstruction(model_path)
    lookup = {image.name: image for image in model.images.values() if image.has_pose}
    admitted = (pycolmap.__version__ == "4.1.1" and np.__version__ == "2.3.5" and len(model.cameras) == 1
        and set(names) == set(lookup) and list(names) == sorted(names)
        and all(type(name) is str and Path(name).name == name and not any(c in name for c in "\r\n\x00") for name in names))
    camera = next(iter(model.cameras.values()), None)
    admitted = admitted and camera.model_name in SUPPORTED - {"PINHOLE", "SIMPLE_PINHOLE"} and camera.width >= 2 and camera.height >= 2
    admitted = admitted and 2 * camera.width * camera.height * 25 <= 256 * 1024**2
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
    source_paths = [Path(__file__), Path(__file__).with_name("dataset_sampling_carrier.py"),
        Path(__file__).with_name("native_camera_sampling.py"), Path(__file__).with_name("native_camera_sampling.cpp"),
        Path(__file__).with_name("camera_pixel_remap.py"), Path(__file__).with_name("complete_carrier_configuration.py"),
        ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005/runtime.py", Path(library)]
    sources = [data_identity(path) for path in source_paths]
    operator = NativeCameraPixelRemap(pycolmap.undistort_image, library)
    segments, headers, results = [], [], []
    previous = os.environ.get("OPENIMAGEIO_THREADS")
    before_rng = np.random.get_state()
    try:
        operator.build(camera, pycolmap.undistort_camera(options, camera))
        calibration = dict(operator.calibration)
        for array in operator.maps:
            segment = shared_memory.SharedMemory(create=True, size=array.nbytes)
            segments.append(segment)
            view = np.ndarray(array.shape, dtype=array.dtype, buffer=segment.buf)
            np.copyto(view, array)
            view.setflags(write=False)
            headers.append({"name": segment.name, "shape": list(array.shape), "dtype": array.dtype.str,
                "bytes": array.nbytes, "sha256": hashlib.sha256(segment.buf[:array.nbytes]).hexdigest()})
        del array, view
        operator.close()
        atomic_json(proof / "SHARED_SEGMENTS_CREATED.json", {"headers": headers, "owner_pid": os.getpid(),
            "owner_creation_time": psutil.Process().create_time(), "new_optimization_count": 0})
        os.environ["OPENIMAGEIO_THREADS"] = "1"
        stripes = [[str(Path(images) / name) for name in names[1 + index::8]] for index in range(8)]
        with ProcessPoolExecutor(max_workers=8, mp_context=multiprocessing.get_context("spawn")) as pool:
            arguments = [(paths, str(output / "images"), camera_fields(camera), options.todict(), calibration,
                headers, str(library), sources) for paths in stripes if paths]
            results = list(pool.map(sample_stripe, arguments, chunksize=1))
        for segment, header in zip(segments, headers):
            if hashlib.sha256(segment.buf[:header["bytes"]]).hexdigest() != header["sha256"]:
                raise ValueError("The complete parent shared map changed while consumers ran")
    finally:
        operator.close()
        for segment in segments:
            segment.close()
            segment.unlink()
        if previous is None:
            os.environ.pop("OPENIMAGEIO_THREADS", None)
        else:
            os.environ["OPENIMAGEIO_THREADS"] = previous
        atomic_json(proof / "SHARED_SEGMENTS_CLOSED.json", {"headers": headers,
            "all_handles_closed": all(segment.buf is None for segment in segments),
            "all_names_unavailable_after_shutdown": names_unavailable(headers), "new_optimization_count": 0})
    emitted = [name for row in results for name in row["image_names"]]
    after_rng = np.random.get_state()
    checks = {"all_complete_original_names_once": sorted(path.name for path in (output / "images").iterdir() if path.is_file()) == sorted(names)
            and len(emitted) == len(set(emitted)) == len(names) - 1 and set(emitted) == set(names[1:]),
        "all_actual_sampling_processes_exited_after_shutdown": all(closed(row["owner"]) for row in results),
        "all_actual_sampler_DLLs_and_shared_handles_closed": all(row["sampler_trace"]["closed"] and row["sampler_trace"]["native_DLL_closed"]
            and row["sampler_trace"]["shared_handles_closed"] and row["sampler_trace"]["builds"] == 0 for row in results),
        "single_original_parent_build_and_all_shared_names_unlinked": operator.trace["builds"] == 1 and operator.trace["native_DLL_closed"] and names_unavailable(headers),
        "same256MiB_maps_including_copy_peak": 2 * sum(row["bytes"] for row in headers) <= 256 * 1024**2,
        "parent_environment_and_RNG_preserved": os.environ.get("OPENIMAGEIO_THREADS") == previous and before_rng[0] == after_rng[0]
            and np.array_equal(before_rng[1], after_rng[1]) and before_rng[2:] == after_rng[2:],
        "sources_unchanged": sources == [data_identity(path) for path in source_paths]}
    if not all(checks.values()): raise AssertionError("A shared complete dataset protection failed")
    result = {"route": "native_full_model_carrier_shared_eight_process_RGB_sampling", "checks": checks, "closed": True,
        "configuration": configuration, "workers": results, "parent_sampler_trace": operator.trace,
        "shared_headers": headers, "map_copy_peak_bytes": 2 * sum(row["bytes"] for row in headers),
        "complete_dataset_seconds": time.perf_counter() - started, "component_stripe_times_are_not_whole_wall": True,
        "new_optimization_count": 0}
    atomic_json(proof / "RESULT.json", result)
    return result
