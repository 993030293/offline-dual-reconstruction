"""Separate fresh CPU image preparation from ordered GPU camera construction."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import threading
import time

from runtime import atomic_json


@contextmanager
def phase_isolated_camera_prepare(output, enabled):
    import scene
    import scene.cameras as cameras
    from PIL import Image
    import torch

    if type(enabled) is not bool:
        raise TypeError("The phase route must be explicit")
    native_prepare = cameras.PILtoTorch
    original_list = scene.cameraList_from_camInfos
    active = {}
    lock = threading.Lock()
    bound = False
    trace = {"enabled": enabled, "groups": 0, "prepared": 0, "reused": 0, "native_fallback": 0,
        "workers": 4, "capacity": 8, "maximum_pending": 0, "maximum_entries": 0,
        "maximum_payload_bytes": 8 * 1024**3, "payload_bytes": 0,
        "CPU_preparation_seconds": [], "nonadmitted_groups": [], "stored_original_errors": 0,
        "new_optimization_count": 0, "persistent_cache_or_image_reduction": False}

    def lookup(image, resolution):
        key = str(Path(image.filename).resolve()), tuple(resolution)
        with lock:
            row = active.pop(key, None)
            if row is None:
                trace["native_fallback"] += 1
            else:
                trace["reused"] += 1
        if row is None:
            return native_prepare(image, resolution)
        if row["error"] is not None:
            raise row["error"]
        value = row["value"]
        if value._version != row["version"]:
            raise ValueError("An owned fresh CPU image was mutated before its original consumer")
        return value

    @contextmanager
    def bind():
        nonlocal bound
        if bound or active:
            raise RuntimeError("Fresh preparation phase bindings cannot overlap")
        previous = scene.cameraList_from_camInfos
        bound = True

        def camera_list(infos, resolution_scale, args, is_synthetic, is_test):
            if not enabled or not infos:
                return previous(infos, resolution_scale, args, is_synthetic, is_test)
            reason = None
            if (str(args.data_device) != "cpu" or args.resolution not in (1, 2, 4, 8)
                    or resolution_scale <= 0 or is_synthetic or any(info.depth_path for info in infos)):
                reason = "original_prefetch_scope_not_admitted"
            elif len(infos) > 4096 or torch.get_default_dtype() != torch.float32:
                reason = "owned_count_or_native_float32_scope_not_admitted"
            resolutions = [(round(info.width / (resolution_scale * args.resolution)),
                            round(info.height / (resolution_scale * args.resolution))) for info in infos] if reason is None else []
            keys = [(str(Path(info.image_path).resolve()), resolution) for info, resolution in zip(infos, resolutions)]
            size = sum(width * height * 3 * 4 for width, height in resolutions)
            if reason is None and (len(set(keys)) != len(keys) or min(min(r) for r in resolutions) <= 0 or size > trace["maximum_payload_bytes"]):
                reason = "repeated_binding_invalid_resolution_or_owned_byte_budget"
            if reason is not None:
                trace["nonadmitted_groups"].append({"reason": reason, "count": len(infos)})
                return previous(infos, resolution_scale, args, is_synthetic, is_test)
            jobs = deque()
            pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="phase-prepare")
            next_submit = 0
            started = time.perf_counter()

            def prepare(index):
                info, resolution = infos[index], resolutions[index]
                try:
                    with Image.open(info.image_path) as image:
                        if image.size != (info.width, info.height):
                            raise ValueError("Camera metadata and decoded image dimensions differ")
                        value = native_prepare(image, resolution)
                    if (value.device.type != "cpu" or value.dtype != torch.float32 or value.requires_grad
                            or tuple(value.shape) != (3, resolution[1], resolution[0])
                            or value.stride() != (1, 3 * resolution[0], 3) or value.storage_offset() != 0):
                        raise ValueError("The original fresh preparation left its admitted opaque RGB layout")
                    return {"value": value, "version": value._version, "error": None}
                except BaseException as error:
                    # The original ordered future consumer raises at this frame.
                    return {"value": None, "version": None, "error": error}

            def fill():
                nonlocal next_submit
                while next_submit < len(infos) and len(jobs) < 8:
                    jobs.append((next_submit, pool.submit(prepare, next_submit)))
                    next_submit += 1
                    trace["maximum_pending"] = max(trace["maximum_pending"], len(jobs))

            try:
                fill()
                while jobs:
                    index, future = jobs.popleft()
                    row = future.result()
                    active[keys[index]] = row
                    trace["prepared"] += 1
                    trace["stored_original_errors"] += row["error"] is not None
                    fill()
            finally:
                for _, future in jobs:
                    future.cancel()
                pool.shutdown(wait=True, cancel_futures=True)
            trace["groups"] += 1
            trace["payload_bytes"] += size
            trace["maximum_entries"] = max(trace["maximum_entries"], len(active))
            trace["CPU_preparation_seconds"].append(time.perf_counter() - started)
            if any(t.name.startswith("phase-prepare") for t in threading.enumerate()):
                raise RuntimeError("A fresh CPU worker overlapped GPU camera construction")
            try:
                result = previous(infos, resolution_scale, args, is_synthetic, is_test)
                if active:
                    raise ValueError("The original full camera consumer left fresh prepared images unused")
                return result
            finally:
                active.clear()

        if enabled:
            scene.cameraList_from_camInfos = camera_list
        try:
            yield
        finally:
            scene.cameraList_from_camInfos = previous
            active.clear()
            bound = False

    if enabled:
        cameras.PILtoTorch = lookup
    succeeded = False
    try:
        yield bind
        succeeded = True
    finally:
        active.clear()
        cameras.PILtoTorch = native_prepare
        scene.cameraList_from_camInfos = original_list
        trace.update(active_entries_after_close=len(active), original_functions_restored=cameras.PILtoTorch is native_prepare and scene.cameraList_from_camInfos is original_list,
            original_CPU_workers_closed=not any(t.name.startswith("phase-prepare") for t in threading.enumerate()),
            status="complete_phase_isolated_preparation_trace0" if succeeded else "failed_phase_isolated_preparation_trace0")
        atomic_json(Path(output) / "PHASE_PREPARATION_TRACE.json", trace)
