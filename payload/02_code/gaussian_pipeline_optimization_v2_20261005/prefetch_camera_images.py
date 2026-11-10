"""Overlap CPU image preparation while constructing calibrated cameras in order."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import time


@contextmanager
def prefetch_camera_images(workers=4, capacity=8):
    if workers <= 0 or capacity < workers:
        raise ValueError("A bounded camera prefetch queue must cover its workers")
    import scene
    import scene.cameras as camera_module
    import utils.camera_utils as camera_utils
    from PIL import Image

    original_list = scene.cameraList_from_camInfos
    native_prepare = camera_module.PILtoTorch
    receipt = {"workers": workers, "capacity": capacity, "maximum_pending": 0,
               "prepared": 0, "consumed": 0, "lists": [], "CPU_only_workers": True}

    def camera_list(infos, resolution_scale, args, is_synthetic, is_test):
        if not infos:
            return []
        if (str(args.data_device) != "cpu" or args.resolution not in (1, 2, 4, 8)
                or resolution_scale <= 0 or is_synthetic or any(info.depth_path for info in infos)):
            raise ValueError("Prefetch requires the registered CPU COLMAP RGB resolution mode")
        started = time.perf_counter()
        jobs = deque()
        cameras = []
        next_submit = 0
        expected_index = 0
        wait_seconds = 0.
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="camera-decode")

        def prepare(info):
            with Image.open(info.image_path) as image:
                if image.size != (info.width, info.height):
                    raise ValueError("Camera metadata and decoded image dimensions differ")
                resolution = (round(image.width/(resolution_scale*args.resolution)),
                              round(image.height/(resolution_scale*args.resolution)))
                prepared = native_prepare(image, resolution)
            if prepared.device.type != "cpu":
                raise ValueError("Image prefetch workers must not allocate CUDA tensors")
            return prepared, resolution

        def fill():
            nonlocal next_submit
            while next_submit < len(infos) and len(jobs) < capacity:
                jobs.append((next_submit, pool.submit(prepare, infos[next_submit])))
                next_submit += 1
                receipt["maximum_pending"] = max(receipt["maximum_pending"], len(jobs))

        def consume(image, resolution):
            nonlocal wait_seconds, expected_index
            if not jobs:
                raise ValueError("The native camera requested an unregistered image")
            index, future = jobs.popleft()
            if index != expected_index or Path(image.filename).resolve() != Path(infos[index].image_path).resolve():
                raise ValueError("Camera image order differs from the bounded prefetch queue")
            before = time.perf_counter()
            tensor, prepared_resolution = future.result()
            wait_seconds += time.perf_counter()-before
            if tuple(resolution) != prepared_resolution:
                raise ValueError("The native camera changed the registered resolution")
            receipt["consumed"] += 1
            expected_index += 1
            return tensor

        try:
            fill()
            camera_module.PILtoTorch = consume
            for index, info in enumerate(infos):
                cameras.append(camera_utils.loadCam(args, index, info, resolution_scale, is_synthetic, is_test))
                if expected_index != index+1:
                    raise ValueError("A camera did not consume exactly one prepared image")
                fill()
            if jobs or expected_index != len(infos):
                raise ValueError("Prefetch left unconsumed camera input")
        finally:
            camera_module.PILtoTorch = native_prepare
            for _, future in jobs:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
        receipt["prepared"] += next_submit
        receipt["lists"].append({"camera_count": len(infos), "seconds": time.perf_counter()-started,
                                 "waiting_for_CPU_preparation_seconds": wait_seconds})
        return cameras

    scene.cameraList_from_camInfos = camera_list
    try:
        yield receipt
    finally:
        scene.cameraList_from_camInfos = original_list
        camera_module.PILtoTorch = native_prepare
