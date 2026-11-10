"""Count successful new CUDA tensor API outputs within the SSIM consumer only."""

from contextlib import contextmanager

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


class OutputCensus(TorchDispatchMode):
    def __init__(self, trace):
        super().__init__()
        self.trace = trace

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        inputs, _ = tree_flatten((args, kwargs))
        aliases = {(x.device, x.untyped_storage().data_ptr()) for x in inputs
                   if isinstance(x, torch.Tensor) and x.device.type == "cuda"}
        result = func(*args, **kwargs)
        outputs, _ = tree_flatten(result)
        seen, rows = set(), []
        for x in outputs:
            if isinstance(x, torch.Tensor) and x.device.type == "cuda":
                identity = (x.device, x.untyped_storage().data_ptr())
                if identity not in aliases and identity not in seen:
                    seen.add(identity)
                    rows.append({"shape":list(x.shape), "dtype":str(x.dtype), "stride":list(x.stride()),
                                 "elements":x.numel(), "bytes":x.numel()*x.element_size()})
        if rows:
            self.trace["rows"].append({"operation":str(func), "outputs":rows})
            self.trace["successful_new_CUDA_output_elements"] += sum(x["elements"] for x in rows)
            self.trace["successful_new_CUDA_output_bytes"] += sum(x["bytes"] for x in rows)
        return result


def new_trace():
    return {"rows":[], "successful_new_CUDA_output_elements":0, "successful_new_CUDA_output_bytes":0,
            "unit":"successful_new_tensor_API_logical_outputs_views_and_input_aliases_excluded",
            "hardware_memory_bandwidth_or_peak_claim":False}


@contextmanager
def output_census(trace):
    with OutputCensus(trace):
        yield trace
