"""Quantize final display artifacts on their device without changing any metrics."""

from contextlib import contextmanager
import hashlib
import inspect
import time

import numpy as np
from PIL import Image
import torch


def native_picture(tensor):
    pixels = (tensor.detach().cpu().permute(1, 2, 0).numpy()*255).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(pixels)


class DeviceArtifactPicture:
    def __init__(self, enabled=True, maximum_elements=64*1024**2):
        if type(enabled) is not bool or type(maximum_elements) is not int or not 1 <= maximum_elements <= 64*1024**2:
            raise ValueError("Artifact quantization requires explicit bounded admission")
        self.enabled, self.maximum_elements = enabled, maximum_elements
        self.calls = []

    def __call__(self, tensor):
        started = time.perf_counter()
        route = (self.enabled and torch.__version__ == "2.7.1+cu128" and np.__version__ == "1.26.4" and tensor.device.type == "cuda"
            and tensor.dtype == torch.float32 and tensor.ndim == 3 and tensor.shape[0] == 3
            and min(tensor.shape) > 0 and tensor.numel() <= self.maximum_elements)
        if route:
            route = bool(torch.isfinite(tensor).all().item())
        if route:
            quantized = (tensor.detach()*255).round().clamp_(0, 255).to(torch.uint8)
            host = quantized.permute(1, 2, 0).contiguous().cpu()
            result = Image.fromarray(host.numpy())
            transferred = host.numel()*host.element_size()
            dtype = str(host.dtype)
        else:
            host = tensor.detach().cpu()
            pixels = (host.permute(1, 2, 0).numpy()*255).round().clip(0, 255).astype(np.uint8)
            result = Image.fromarray(pixels)
            transferred = host.numel()*host.element_size() if tensor.device.type != "cpu" else 0
            dtype = str(host.dtype)
        elapsed = time.perf_counter()-started
        # Artifact fingerprints validate pixels; metadata is not counted as transfer savings.
        self.calls.append({"shape": list(tensor.shape), "input_dtype": str(tensor.dtype),
            "input_device": str(tensor.device), "route": "device_uint8" if route else "native",
            "actual_returned_CPU_tensor_dtype": dtype, "device_to_host_tensor_bytes": transferred,
            "original_float_tensor_bytes": tensor.numel()*tensor.element_size(), "seconds": elapsed,
            "pixel_sha256": hashlib.sha256(result.tobytes()).hexdigest(), "mode": result.mode,
            "size": list(result.size)})
        return result


@contextmanager
def device_artifact_evaluation(enabled=True):
    started = time.perf_counter()
    import evaluate_pair
    original = evaluate_pair.main
    source = inspect.getsource(original)
    anchor = '''    def picture(tensor):
        pixels = (tensor.detach().cpu().permute(1, 2, 0).numpy()*255).round().clip(0, 255).astype(np.uint8)
        return Image.fromarray(pixels)
'''
    if source.count(anchor) != 1:
        raise ValueError("The fixed native final artifact conversion changed")
    adapter = DeviceArtifactPicture(enabled)
    namespace = dict(evaluate_pair.__dict__)
    namespace["_artifact_picture"] = adapter
    # The preserved outer evaluator binds the already activated runtime afterward.
    namespace["activate_reference"] = lambda runtime: evaluate_pair.activate_reference(runtime)
    source = source.replace(anchor, "    picture = _artifact_picture\n")
    exec(compile(source, "<unchanged_evaluation_with_exact_artifact_picture>", "exec"), namespace)
    evaluate_pair.main = namespace["main"]
    try:
        yield {"calls": adapter.calls, "composition_seconds": time.perf_counter()-started,
            "maximum_elements": adapter.maximum_elements, "metric_and_training_math_changed": False}
    finally:
        evaluate_pair.main = original
