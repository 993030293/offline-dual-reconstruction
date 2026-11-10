"""Replay only the unchanged Q30 prediction tail in one owned CUDA graph."""

import inspect
import textwrap
import time

import torch

from reference_ssim import ReferenceSSIM
from interior_ssim_reuse import InteriorSSIM


def formula_sources():
    source = textwrap.dedent(inspect.getsource(ReferenceSSIM.__call__))
    begin = "    mu_pred = F.conv2d(prediction, window, padding=window_size//2, groups=channels)"
    end = "    return result.mean() if size_average else result.mean(1).mean(1).mean(1)"
    if source.count(begin) != 1 or source.count(end) != 1:
        raise ValueError("The frozen Q30 prediction-tail source changed")
    start, finish = source.index(begin), source.index(end)
    body = source[start:finish]
    tail = ("def original_tail(prediction, reference, window, mu_ref, mu_ref_squared, "
            "variance_ref, window_size, channels):\n" + body + "    return result\n")
    call = source[:start] + (
        "    return self.graph_tail(prediction, reference, window, mu_ref, mu_ref_squared, "
        "variance_ref, window_size, channels)\n")
    return tail, call


_TAIL_SOURCE, _CALL_SOURCE = formula_sources()
_FORMULA_NAMESPACE = dict(ReferenceSSIM.__call__.__globals__)
exec(compile(_TAIL_SOURCE, "<original-Q30-prediction-tail>", "exec"), _FORMULA_NAMESPACE)
ORIGINAL_TAIL = _FORMULA_NAMESPACE["original_tail"]


class CapturedSSIMTail:
    """Own fixed tensor addresses and return independent maps to the crop consumer."""

    def __init__(self):
        self.entry = None
        self.reference_key = None
        self.closed = False
        self.trace = {"captures": 0, "replays": 0, "native_calls": 0, "capture_seconds": 0.0,
                      "copy_input_bytes": 0, "maximum_entries": 1, "close_seconds": 0.0,
                      "closed": False, "reference_forgets": 0}

    @staticmethod
    def signature(value):
        return (value.data_ptr(), value.storage_offset(), tuple(value.shape), tuple(value.stride()),
                value.dtype, value.device, value._version)

    def forget_reference(self):
        self.reference_key = None
        self.trace["reference_forgets"] += 1

    def __call__(self, prediction, reference, window, mu_ref, mu_ref_squared, variance_ref,
                 window_size, channels):
        if self.closed:
            raise RuntimeError("The owned evaluation graph is closed")
        values = (prediction, reference, window, mu_ref, mu_ref_squared, variance_ref)
        admitted = (prediction.device.type == "cuda" and prediction.dtype == torch.float32
                    and tuple(prediction.shape[:2]) == (1, 3) and window_size == 11
                    and min(prediction.shape[-2:]) > 10
                    and all(not value.requires_grad for value in values))
        key = tuple((tuple(value.shape), tuple(value.stride()), value.dtype, value.device) for value in values)
        if not admitted or (self.entry is not None and key != self.entry["key"]):
            self.trace["native_calls"] += 1
            return ORIGINAL_TAIL(*values, window_size, channels)
        reference_key = tuple(self.signature(value) for value in values[1:])
        if self.entry is None:
            before = time.perf_counter()
            static = tuple(value.detach().clone(memory_format=torch.preserve_format) for value in values)
            charged = sum(value.numel() * value.element_size() for value in static)
            if charged > 128 * 1024**2:
                raise MemoryError("Owned graph static inputs exceed the registered byte budget")
            stream = torch.cuda.Stream(device=prediction.device)
            stream.wait_stream(torch.cuda.current_stream(prediction.device))
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                warm = ORIGINAL_TAIL(*static, window_size, channels)
                del warm
            torch.cuda.current_stream(prediction.device).wait_stream(stream)
            torch.cuda.synchronize(prediction.device)
            with torch.cuda.graph(graph, stream=stream, capture_error_mode="global"):
                output = ORIGINAL_TAIL(*static, window_size, channels)
            torch.cuda.current_stream(prediction.device).wait_stream(stream)
            self.entry = {"key": key, "static": static, "graph": graph, "output": output,
                          "static_input_bytes": charged}
            self.reference_key = reference_key
            self.trace["captures"] += 1
            self.trace["capture_seconds"] += time.perf_counter() - before
            self.trace["copy_input_bytes"] += charged
        else:
            static = self.entry["static"]
            static[0].copy_(prediction)
            self.trace["copy_input_bytes"] += prediction.numel() * prediction.element_size()
            if reference_key != self.reference_key:
                for destination, value in zip(static[1:], values[1:]):
                    destination.copy_(value)
                    self.trace["copy_input_bytes"] += value.numel() * value.element_size()
                self.reference_key = reference_key
        self.entry["graph"].replay()
        self.trace["replays"] += 1
        return self.entry["output"].clone(memory_format=torch.preserve_format)

    def clear(self):
        if self.closed:
            return
        before = time.perf_counter()
        if self.entry is not None:
            torch.cuda.synchronize()
            self.entry["graph"].reset()
            self.entry = None
        self.reference_key = None
        self.closed = True
        self.trace.update(close_seconds=time.perf_counter() - before, closed=True)


class GraphMapReferenceSSIM(ReferenceSSIM):
    def __init__(self, window_factory, maximum_bytes):
        super().__init__(window_factory, maximum_bytes=maximum_bytes)
        self.graph_tail = CapturedSSIMTail()

    def clear(self):
        self.graph_tail.clear()
        super().clear()


_CALL_NAMESPACE = dict(ReferenceSSIM.__call__.__globals__)
exec(compile(_CALL_SOURCE, "<original-Q30-reference-prefix-and-owned-graph>", "exec"), _CALL_NAMESPACE)
GraphMapReferenceSSIM.__call__ = _CALL_NAMESPACE["__call__"]


class GraphInteriorSSIM(InteriorSSIM):
    def __init__(self, window_factory, maximum_bytes=1024**3):
        super().__init__(window_factory, maximum_bytes=maximum_bytes)
        self.cache = GraphMapReferenceSSIM(window_factory, maximum_bytes)

    def __call__(self, *args, **kwargs):
        if self.cache.graph_tail.closed:
            raise RuntimeError("The owned evaluation graph is closed")
        return super().__call__(*args, **kwargs)

    def receipt(self):
        result = super().receipt()
        result["owned_graph"] = dict(self.cache.graph_tail.trace,
            live_graph_entries=int(self.cache.graph_tail.entry is not None))
        return result

    def finish_reference(self):
        self.cache.graph_tail.forget_reference()
