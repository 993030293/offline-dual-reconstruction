"""Pass an owned loaded Pillow image across the existing bounded prefetch queue."""

from contextlib import contextmanager
import difflib
import inspect
from pathlib import Path
import threading
import time


class HeaderOwner:
    """Own images by queue index, including repeated camera metadata objects."""

    def __init__(self, native_open, receipt, capacity):
        self.native_open = native_open
        self.receipt = receipt
        self.capacity = capacity
        self.main_thread = threading.get_ident()
        self.lock = threading.Lock()
        self.images = {}
        self.active = None

    def record(self, index, image, tensor, resolution):
        with self.lock:
            if index in self.images:
                raise ValueError("A prepared image already has an owner")
            self.images[index] = image
            self.receipt["recorded"] += 1
            self.receipt["maximum_owned_images"] = max(self.receipt["maximum_owned_images"], len(self.images))
            if len(self.images) > self.capacity:
                raise ValueError("Owned source images exceeded the original queue capacity")
        return tensor, resolution

    def activate(self, info, jobs, index):
        if self.active is not None or not jobs or jobs[0][0] != index:
            raise ValueError("The original ordered header queue changed")
        self.active = (info.image_path, index, jobs[0][1])

    def open(self, fp, *args, **kwargs):
        active = self.active
        if threading.get_ident() != self.main_thread or active is None or type(fp) is not str or fp != active[0] or args or kwargs:
            return self.native_open(fp, *args, **kwargs)
        _, index, future = active
        started = time.perf_counter()
        try:
            future.result()
        except BaseException:
            # The original consume still observes this failed future after native header checks.
            self.receipt["failed_future_native_header_fallbacks"] += 1
            return self.native_open(fp, *args, **kwargs)
        finally:
            self.receipt["header_future_wait_seconds"] += time.perf_counter() - started
        with self.lock:
            image = self.images.get(index)
        if image is None:
            raise ValueError("A completed native preparation lost its source image")
        if image.format not in ("JPEG", "PNG") or image.mode != "RGB" or image.__dict__.get("is_animated", False):
            self.receipt["unsupported_native_header_fallbacks"] += 1
            return self.native_open(fp, *args, **kwargs)
        if image.fp is not None or image.filename != fp:
            raise ValueError("An owned prepared image retained a file or changed its native filename")
        image.getpixel((0, 0))
        self.receipt["handed_off"] += 1
        self.receipt["concrete_image_types"][type(image).__module__ + "." + type(image).__name__] = True
        return image

    def release(self, index):
        if self.active is None or self.active[1] != index:
            raise ValueError("An original camera released a different source image")
        self.active = None
        with self.lock:
            image = self.images.pop(index, None)
        if image is not None:
            image.close()
            self.receipt["closed_images"] += 1

    def drain(self):
        self.active = None
        with self.lock:
            images = list(self.images.values())
            self.images.clear()
        for image in images:
            image.close()
            self.receipt["closed_images"] += 1
        self.receipt["remaining_images"] = 0


def transform_factory(before):
    replacements = (
        ("        def prepare(info):", "        def prepare(index, info):"),
        ("            return prepared, resolution", "            return _header_owner.record(index, image, prepared, resolution)"),
        ("pool.submit(prepare, infos[next_submit])", "pool.submit(prepare, next_submit, infos[next_submit])"),
        ("                cameras.append(camera_utils.loadCam", "                _header_owner.activate(info, jobs, index)\n                cameras.append(camera_utils.loadCam"),
        ('                    raise ValueError("A camera did not consume exactly one prepared image")\n                fill()',
         '                    raise ValueError("A camera did not consume exactly one prepared image")\n                _header_owner.release(index)\n                fill()'),
        ("            pool.shutdown(wait=True, cancel_futures=True)", "            pool.shutdown(wait=True, cancel_futures=True)\n            _header_owner.drain()"),
    )
    after = before
    for old, new in replacements:
        if after.count(old) != 1:
            raise ValueError("The original native prefetch boundary changed: " + old)
        after = after.replace(old, new)
    return after


@contextmanager
def open_image_handoff(output, enabled, workers=4, capacity=8):
    import prefetch_camera_images as parent
    import scene
    import scene.cameras as cameras
    from PIL import Image
    from runtime import atomic_json

    factory = parent.prefetch_camera_images
    original_list, original_prepare, original_open = scene.cameraList_from_camInfos, cameras.PILtoTorch, Image.open
    receipt = {"enabled": enabled, "recorded": 0, "handed_off": 0, "closed_images": 0,
               "remaining_images": 0, "maximum_owned_images": 0, "header_future_wait_seconds": 0.,
               "failed_future_native_header_fallbacks": 0, "unsupported_native_header_fallbacks": 0,
               "concrete_image_types": {}, "no_persistent_cache": True, "original_capacity": capacity}
    before = inspect.getsource(factory)
    after = transform_factory(before) if enabled else before
    archive = Path(output).with_name(Path(output).name + "_handoff_sources")
    archive.mkdir(parents=True, exist_ok=False)
    (archive / "EXECUTED_PREFETCH_FACTORY.py").write_text(after, encoding="utf-8")
    (archive / "PREFETCH_FACTORY.diff").write_text("".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile="original_prefetch", tofile="loaded_image_handoff")), encoding="utf-8")
    native = inspect.unwrap(factory)
    namespace = dict(native.__globals__)
    if native.__module__ != parent.__name__ or any(namespace.get(name) is not parent.__dict__.get(name) for name in ("time", "deque", "ThreadPoolExecutor")):
        raise ValueError("The native contextmanager globals changed")
    owner = HeaderOwner(original_open, receipt, capacity)
    namespace["_header_owner"] = owner
    exec(compile(after, "<owned_loaded_image_handoff>", "exec"), namespace)
    succeeded = False
    try:
        if enabled:
            Image.open = owner.open
        with namespace["prefetch_camera_images"](workers, capacity) as prefetch:
            yield prefetch
        succeeded = True
    finally:
        Image.open = original_open
        owner.drain()
        receipt["original_functions_restored"] = scene.cameraList_from_camInfos is original_list and cameras.PILtoTorch is original_prepare and parent.prefetch_camera_images is factory and Image.open is original_open
        receipt["owned_decode_threads_closed"] = not any(t.name.startswith("camera-decode") for t in threading.enumerate())
        receipt["status"] = "completed" if succeeded else "original_failure_preserved"
        atomic_json(Path(output) / "IMAGE_HANDOFF_TRACE.json", receipt)
