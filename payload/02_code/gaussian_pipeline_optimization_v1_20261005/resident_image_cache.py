"""Bounded CUDA storage with exact reconstruction of normalized byte images."""

import time

import numpy as np
import torch


class ResidentImageCache:
    """Keep immutable compact inputs and expose only the currently sampled floats."""

    def __init__(self, cameras, maximum_bytes, gpu_budget_bytes):
        started = time.monotonic()
        self.cameras = list(cameras)
        self.active = None
        self.samples = 0
        self.host = {}
        self.encoded = {}
        constant_alpha = {}
        self.maximum_bytes = int(maximum_bytes)
        self.payload_bytes = 1024
        self.decoded_bytes = 0
        if not self.cameras or self.maximum_bytes <= 1024:
            raise ValueError("A nonempty camera set and positive cache capacity are required")
        if len({id(camera) for camera in self.cameras}) != len(self.cameras):
            raise ValueError("Duplicate camera objects are forbidden")
        if len({camera.image_name for camera in self.cameras}) != len(self.cameras):
            raise ValueError("Duplicate camera names are forbidden")
        for camera in self.cameras:
            rgb, alpha = camera.original_image, camera.alpha_mask
            for value in (rgb, alpha):
                if value is None:
                    continue
                if (value.device.type != "cpu" or value.dtype != torch.float32
                        or value.requires_grad):
                    raise ValueError("Cache inputs must be finite CPU float32 constants in [0,1]")
                array = value.numpy()
                if (not np.isfinite(array).all() or array.min() < 0 or array.max() > 1):
                    raise ValueError("Cache inputs must be finite CPU float32 constants in [0,1]")
            if rgb.ndim != 3 or rgb.shape[0] != 3:
                raise ValueError("RGB input must have shape [3,H,W]")
            if alpha is not None and alpha.shape != (1, rgb.shape[1], rgb.shape[2]):
                raise ValueError("Alpha dimensions must match RGB")
            self.host[id(camera)] = (rgb, alpha)
            self.payload_bytes += rgb.numel()
            self.decoded_bytes += rgb.numel()*4
            if alpha is not None:
                bits = alpha.numpy().view(np.int32)
                constant_alpha[id(camera)] = bool(np.all(bits == bits.flat[0]))
                self.payload_bytes += 4 if constant_alpha[id(camera)] else alpha.numel()
                self.decoded_bytes += alpha.numel()*4
        if self.payload_bytes > self.maximum_bytes:
            raise MemoryError("The complete encoded camera set exceeds its registered capacity")
        if self.payload_bytes + 4*1024**3 > gpu_budget_bytes:
            raise MemoryError("Insufficient registered model and densification headroom")
        free, _ = torch.cuda.mem_get_info()
        if self.payload_bytes + 2*1024**3 > free:
            raise MemoryError("Insufficient current GPU headroom")
        # Copy the CPU division result exactly, avoiding platform-dependent division.
        self.lut = (torch.arange(256, dtype=torch.float32)/255).cuda()
        for camera in self.cameras:
            rgb, alpha = self.host[id(camera)]
            encoded_rgb = self._encode(rgb)
            if alpha is None:
                encoded_alpha = None
            elif constant_alpha[id(camera)]:
                encoded_alpha = ("constant", alpha.flatten()[0:1].cuda(), tuple(alpha.shape))
            else:
                encoded_alpha = ("byte", self._encode(alpha), None)
            self.encoded[id(camera)] = (encoded_rgb, encoded_alpha)
        torch.cuda.synchronize()
        self.preparation_seconds = time.monotonic()-started

    def _encode(self, source):
        reference = source.cuda()
        code = torch.round(reference*255).to(torch.uint8)
        decoded = self.lut[code.long()]
        if not torch.equal(decoded.view(torch.int32), reference.contiguous().view(torch.int32)):
            raise ValueError("Input cannot be reconstructed bit-for-bit from normalized bytes")
        return code

    def activate(self, camera):
        if id(camera) not in self.encoded:
            raise ValueError("The sampled camera is not part of the frozen cache")
        if self.active is not None:
            raise RuntimeError("Finish the current iteration before another camera is activated")
        rgb, alpha = self.encoded[id(camera)]
        camera.original_image = self.lut[rgb.long()]
        if alpha is not None:
            kind, data, shape = alpha
            camera.alpha_mask = data.expand(shape) if kind == "constant" else self.lut[data.long()]
        self.active = camera
        self.samples += 1

    def release(self):
        if self.active is not None:
            camera = self.active
            camera.original_image, camera.alpha_mask = self.host[id(camera)]
            self.active = None

    def finish(self):
        self.release()
        restored = all(camera.original_image is self.host[id(camera)][0]
                       and camera.alpha_mask is self.host[id(camera)][1] for camera in self.cameras)
        if not restored:
            raise RuntimeError("Original input references were not restored")
        return {"camera_count": len(self.cameras), "samples": self.samples,
                "encoded_payload_bytes": self.payload_bytes,
                "original_tensor_bytes": self.decoded_bytes,
                "maximum_cache_bytes": self.maximum_bytes,
                "preparation_seconds": self.preparation_seconds,
                "all_initial_inputs_bit_exact": True,
                "original_references_restored": restored,
                "CPU_originals_retained": True}
