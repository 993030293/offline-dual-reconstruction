"""Sample owned RGB arrays with the original double-to-float interpolation order."""

import torch


def sample_pixels(raw, maps):
    if raw.dtype != torch.uint8 or raw.ndim != 3 or raw.shape[-1] != 3 or raw.device.type != "cuda":
        raise ValueError("The pixel sampler requires owned CUDA HWC RGB8 input")
    x0, y0, dx, dy, valid = maps
    if any(array.shape != raw.shape[:2] or array.device != raw.device for array in maps):
        raise ValueError("Complete sampling maps must bind to the same input dimensions and device")
    if x0.dtype != torch.int64 or y0.dtype != torch.int64 or dx.dtype != torch.float64 or dy.dtype != torch.float64 or valid.dtype != torch.bool:
        raise ValueError("Sampling map types changed before original arithmetic")
    result = torch.empty_like(raw)
    with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
        for first in range(0, raw.shape[0], 128):
            stop = min(first + 128, raw.shape[0])
            xs, ys = x0[first:stop], y0[first:stop]
            wx, wy = dx[first:stop, :, None], dy[first:stop, :, None]
            upper = (1.0 - wx) * raw[ys, xs].to(torch.float64) + wx * raw[ys, xs + 1].to(torch.float64)
            lower = (1.0 - wx) * raw[ys + 1, xs].to(torch.float64) + wx * raw[ys + 1, xs + 1].to(torch.float64)
            colors = ((1.0 - wy) * upper + wy * lower).to(torch.float32)
            rounded = torch.clamp(torch.floor(colors.to(torch.float64) + 0.5), 0, 255).to(torch.uint8)
            rounded[~valid[first:stop]] = 0
            result[first:stop] = rounded
    return result
