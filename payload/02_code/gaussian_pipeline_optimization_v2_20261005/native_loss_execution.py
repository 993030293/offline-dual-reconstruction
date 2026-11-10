"""Reuse immutable SSIM kernels and optionally replay the unchanged native CUDA graph."""

from contextlib import contextmanager
import time


class CachedWindowSSIM:
    """Keep native convolution arithmetic and gradients with bounded constant kernels."""

    def __init__(self, module):
        self.module, self.entries = module, {}
        self.trace = {"calls": 0, "kernel_builds": 0, "kernel_hits": 0, "kernel_bytes": 0,
            "kernel_prepare_seconds": 0., "maximum_entries": 16, "native_formula_unchanged": True}

    def __call__(self, a, b, window_size=11, size_average=True):
        if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
            raise ValueError("Native SSIM requires matching pixel shapes, types and devices")
        channel = a.size(-3)
        key = window_size, channel, a.dtype, a.device
        before = time.perf_counter()
        if key not in self.entries:
            if len(self.entries) >= 16:
                raise ValueError("The native SSIM constant kernel cache exceeded its bound")
            window = self.module.create_window(window_size, channel)
            if a.is_cuda:
                window = window.cuda(a.get_device())
            window = window.type_as(a)
            self.entries[key] = window, window._version
            self.trace["kernel_bytes"] += window.numel()*window.element_size()
            self.trace["kernel_builds"] += 1
        else:
            self.trace["kernel_hits"] += 1
        window, version = self.entries[key]
        if window._version != version:
            raise ValueError("An immutable native SSIM kernel was modified")
        self.trace["kernel_prepare_seconds"] += time.perf_counter()-before
        self.trace["calls"] += 1
        return self.module._ssim(a, b, window, window_size, channel, size_average)


class GraphSSIM:
    """Capture native SSIM forward and backward for fixed first-order training shapes."""

    def __init__(self, module):
        self.native = CachedWindowSSIM(module)
        self.entries = {}
        self.trace = {"calls": 0, "captures": 0, "capture_seconds": 0., "maximum_entries": 2,
            "native_reference_kernels_unchanged": True, "mixed_precision": False, "higher_order_gradients_supported": False}

    def __call__(self, a, b, window_size=11, size_average=True):
        import torch
        if (not a.is_cuda or not a.requires_grad or b.requires_grad or window_size != 11 or not size_average
                or a.shape != b.shape or a.size(-3) != 3):
            raise ValueError("The SSIM graph requires the native first-order CUDA RGB training contract")
        key = a.shape, a.stride(), b.stride(), a.dtype, b.dtype, a.device
        if key not in self.entries:
            if len(self.entries) >= 2:
                raise ValueError("The native SSIM graph exceeded its declared static-shape bound")
            before = time.perf_counter()
            sample_a = a.detach().clone(memory_format=torch.preserve_format).requires_grad_(True)
            sample_b = b.detach().clone(memory_format=torch.preserve_format)
            self.native(sample_a, sample_b)
            self.entries[key] = torch.cuda.make_graphed_callables(self.native, (sample_a, sample_b), num_warmup_iters=3)
            torch.cuda.synchronize()
            self.trace["captures"] += 1
            self.trace["capture_seconds"] += time.perf_counter()-before
        self.trace["calls"] += 1
        return self.entries[key](a, b)


@contextmanager
def native_loss_execution(mode):
    import utils.loss_utils as losses
    if mode not in ("reference", "kernel", "graph"):
        raise ValueError("An explicit native SSIM execution mechanism is required")
    original = losses.ssim
    adapted = None if mode == "reference" else CachedWindowSSIM(losses) if mode == "kernel" else GraphSSIM(losses)
    if adapted is not None:
        losses.ssim = adapted
    try:
        yield adapted
    finally:
        losses.ssim = original

