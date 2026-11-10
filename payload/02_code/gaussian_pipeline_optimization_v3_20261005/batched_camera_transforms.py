"""Prepare private camera transform tensors from unchanged native host matrices."""

import numpy as np


def prepare_camera_transforms(host_matrices):
    import torch
    if not 0 < len(host_matrices) <= 4096:
        raise ValueError("The private camera transform batch is empty or exceeds its bound")
    for matrix in host_matrices:
        if (not isinstance(matrix, np.ndarray) or matrix.dtype != np.float32 or matrix.shape != (4, 4)
                or not matrix.flags.c_contiguous or not np.isfinite(matrix).all()):
            raise ValueError("The private batch requires unchanged dense finite native float32 matrices")
    host = torch.tensor(np.stack(host_matrices)).transpose(1, 2)
    views = host.cuda()
    inverses = views.inverse()
    result = []
    for index in range(len(host_matrices)):
        view = views[index].clone(memory_format=torch.preserve_format)
        inverse = inverses[index].clone(memory_format=torch.preserve_format)
        result.append((view, inverse[3, :3]))
    return result
