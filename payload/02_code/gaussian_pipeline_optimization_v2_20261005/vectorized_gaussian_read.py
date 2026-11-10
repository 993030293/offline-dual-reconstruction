"""Read native float32 Gaussian PLY blocks without intermediate float64 feature tables."""

from contextlib import contextmanager
import time

import numpy as np


def native_names(degree):
    return (["x", "y", "z", "nx", "ny", "nz"]+["f_dc_"+str(i) for i in range(3)]+
        ["f_rest_"+str(i) for i in range(3*(degree+1)**2-3)]+["opacity"]+
        ["scale_"+str(i) for i in range(3)]+["rot_"+str(i) for i in range(4)])


@contextmanager
def vectorized_gaussian_read(enabled=True):
    import torch
    from torch import nn
    from plyfile import PlyData
    from scene.gaussian_model import GaussianModel
    original = GaussianModel.load_ply
    trace = []

    def load(model, path, use_train_test_exp=False):
        if use_train_test_exp or model.max_sh_degree != 3:
            raise ValueError("The exact native Gaussian reader requires the declared degree3 RGB model without exposure split")
        started = time.perf_counter()
        ply = PlyData.read(path)
        if len(ply.elements) != 1 or ply.elements[0].name != "vertex" or ply.byte_order not in ("<", "="):
            raise ValueError("The native Gaussian PLY element or byte order changed")
        data, names = ply.elements[0].data, native_names(model.max_sh_degree)
        if list(data.dtype.names) != names or data.dtype.itemsize != 4*len(names) or len(data) == 0:
            raise ValueError("The native Gaussian PLY no longer has its exact packed float32 field layout")
        for index, name in enumerate(names):
            if data.dtype.fields[name] != (np.dtype("<f4"), 4*index):
                raise ValueError("A native Gaussian PLY field changed its type, offset or padding")
        matrix = data.view(np.float32).reshape(len(data), len(names))
        if not np.isfinite(matrix).all():
            raise FloatingPointError("The native Gaussian reader found a nonfinite field")
        positions = {name: i for i, name in enumerate(names)}

        def block(begin, width):
            return np.ascontiguousarray(matrix[:, begin:begin+width])

        def parameter(value):
            return nn.Parameter(torch.tensor(value, dtype=torch.float, device="cuda").requires_grad_(True))

        model._xyz = parameter(block(0, 3))
        dc = block(positions["f_dc_0"], 3).reshape(len(data), 3, 1)
        model._features_dc = nn.Parameter(parameter(dc).transpose(1, 2).contiguous().requires_grad_(True))
        rest = block(positions["f_rest_0"], 45).reshape(len(data), 3, 15)
        model._features_rest = nn.Parameter(parameter(rest).transpose(1, 2).contiguous().requires_grad_(True))
        model._opacity = parameter(block(positions["opacity"], 1))
        model._scaling = parameter(block(positions["scale_0"], 3))
        model._rotation = parameter(block(positions["rot_0"], 4))
        model.active_sh_degree = model.max_sh_degree
        torch.cuda.synchronize()
        trace.append({"path": str(path), "vertices": len(data), "load_seconds": time.perf_counter()-started,
            "float64_intermediate_feature_table_bytes": 0, "native_degree": model.max_sh_degree,
            "all_source_fields_and_values_preserved": True})

    if enabled:
        GaussianModel.load_ply = load
    try:
        yield trace
    finally:
        GaussianModel.load_ply = original

