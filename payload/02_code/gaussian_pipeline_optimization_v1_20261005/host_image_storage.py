"""Bound CPU image storage while retaining exact float32 inputs and strides."""

from contextlib import contextmanager
import time

import numpy as np
import torch
import cv2


class HostImageStorage:
    def __init__(self, maximum_bytes=4*1024**3):
        if maximum_bytes < 1024:
            raise ValueError("The image store requires space for the exact normalization table")
        self.maximum_bytes = maximum_bytes
        self.lut = (torch.arange(256, dtype=torch.float32)/255.).numpy()
        self.entries = {}
        self.names = set()
        self.active = None
        self.payload_bytes = self.lut.nbytes
        self.original_bytes = 0
        self.samples = 0
        self.restore_seconds = 0.
        self.encoding_seconds = 0.
        self.maximum_active_float_bytes = 0
        self.layout_counts = {"packed_permutation": 0, "strided_copy": 0, "constant": 0}

    def _encode(self, tensor, constant_allowed=False):
        if tensor is None:
            return None
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32 or tensor.requires_grad:
            raise ValueError("Only constant CPU float32 camera inputs are supported")
        array = tensor.numpy()
        if not array.size or not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
            raise ValueError("Image values must be finite, nonempty and in[0,1]")
        shape, stride = tuple(tensor.shape), tuple(tensor.stride())
        span = 1+sum((n-1)*s for n, s in zip(shape, stride))
        if any(s <= 0 for s in stride) or span > 4*array.size:
            raise ValueError("Unsupported overlapping or excessively sparse image layout")
        first = array.flat[0]
        constant = constant_allowed and np.all(array.view(np.uint32) == np.asarray(first).view(np.uint32))
        if constant:
            payload = np.array([first], dtype=np.float32)
            permutation = inverse = None
            layout = "constant"
        else:
            permutation = tuple(sorted(range(array.ndim), key=lambda axis: -stride[axis]))
            inverse = tuple(np.argsort(permutation).tolist())
            physical = array.transpose(permutation)
            layout = "packed_permutation" if physical.flags.c_contiguous else "strided_copy"
            if layout == "strided_copy":
                physical = array
                permutation = inverse = None
            payload = np.ascontiguousarray(np.rint(physical*255).astype(np.uint8))
            decoded = self.lut[payload]
            if inverse is not None:
                decoded = decoded.transpose(inverse)
            if not np.array_equal(decoded.view(np.uint32), array.view(np.uint32)):
                raise ValueError("Input is not exactly representable by the native normalization table")
        return {"shape": shape, "stride": stride, "payload": payload, "constant": bool(constant),
                "layout": layout, "inverse_permutation": inverse,
                "logical_float_bytes": array.size*4, "restored_storage_bytes": span*4}

    def add(self, camera):
        started = time.monotonic()
        if id(camera) in self.entries or camera.image_name in self.names:
            raise ValueError("Camera objects and image names must be unique")
        rgb, alpha = camera.original_image, camera.alpha_mask
        if rgb is None or rgb.ndim != 3 or rgb.shape[0] != 3:
            raise ValueError("RGB input must have shape[3,H,W]")
        if alpha is not None and tuple(alpha.shape) != (1, rgb.shape[1], rgb.shape[2]):
            raise ValueError("Alpha dimensions must match RGB")
        if getattr(camera, "depth_reliable", False) or getattr(camera, "invdepthmap", None) is not None:
            raise ValueError("Depth inputs require a separately verified storage contract")
        encoded = (self._encode(rgb), self._encode(alpha, constant_allowed=True))
        added = sum(row["payload"].nbytes for row in encoded if row is not None)
        if self.payload_bytes+added > self.maximum_bytes:
            raise MemoryError("The fixed CPU image store capacity would be exceeded")
        self.entries[id(camera)] = {"camera": camera, "encoded": encoded}
        self.names.add(camera.image_name)
        self.payload_bytes += added
        self.original_bytes += sum(row["logical_float_bytes"] for row in encoded if row is not None)
        for row in encoded:
            if row is not None:
                self.layout_counts[row["layout"]] += 1
        camera.original_image = None
        camera.alpha_mask = None
        self.encoding_seconds += time.monotonic()-started

    def _restore(self, encoded):
        if encoded is None:
            return None
        if encoded["layout"] == "packed_permutation":
            # LUT copies the original CPU division bit patterns without arithmetic.
            physical = cv2.LUT(encoded["payload"].reshape(-1, 1), self.lut).reshape(encoded["payload"].shape)
            tensor = torch.from_numpy(physical.transpose(encoded["inverse_permutation"]))
            if tuple(tensor.stride()) != encoded["stride"]:
                # Unit axes may have different equivalent strides in array views.
                tensor = tensor.as_strided(encoded["shape"], encoded["stride"])
            return tensor
        tensor = torch.empty_strided(encoded["shape"], encoded["stride"], dtype=torch.float32, device="cpu")
        array = tensor.numpy()
        if encoded["constant"]:
            array[...] = encoded["payload"][0]
        else:
            array[...] = self.lut[encoded["payload"]]
        return tensor

    def activate(self, camera):
        if id(camera) not in self.entries:
            raise ValueError("The requested camera is outside the frozen image store")
        if self.active is not None:
            raise RuntimeError("Release the active camera before restoring another input")
        started = time.monotonic()
        encoded = self.entries[id(camera)]["encoded"]
        camera.original_image, camera.alpha_mask = [self._restore(row) for row in encoded]
        self.active = camera
        self.samples += 1
        self.restore_seconds += time.monotonic()-started
        self.maximum_active_float_bytes = max(self.maximum_active_float_bytes,
            sum(row["restored_storage_bytes"] for row in encoded if row is not None))

    def release(self):
        if self.active is not None:
            self.active.original_image = None
            self.active.alpha_mask = None
            self.active = None

    def receipt(self):
        return {"camera_count": len(self.entries), "samples": self.samples,
                "encoded_payload_bytes": self.payload_bytes, "original_logical_float_bytes": self.original_bytes,
                "payload_reduction": 1-self.payload_bytes/self.original_bytes if self.original_bytes else None,
                "maximum_active_float_bytes": self.maximum_active_float_bytes,
                "maximum_encoded_bytes": self.maximum_bytes, "encoding_seconds": self.encoding_seconds,
                "restore_seconds": self.restore_seconds, "active_camera": self.active.image_name if self.active else None,
                "layout_counts": dict(self.layout_counts), "lookup_backend": "OpenCV_CPU_float32_LUT",
                "CPU_original_float_copies_retained": False, "GPU_encoded_cache_used": False,
                "only_active_camera_has_float_inputs": all(
                    (entry["camera"].original_image is not None) == (entry["camera"] is self.active)
                    for entry in self.entries.values())}


@contextmanager
def host_image_loader(store):
    import utils.camera_utils as camera_utils

    original = camera_utils.loadCam
    original_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)

    def compact_load(*args, **kwargs):
        camera = original(*args, **kwargs)
        store.add(camera)
        return camera

    camera_utils.loadCam = compact_load
    try:
        yield store
    finally:
        store.release()
        camera_utils.loadCam = original
        cv2.setNumThreads(original_threads)
