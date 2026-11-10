"""Reuse reference statistics in evaluation using the fixed INRIA SSIM formula.

The official window factory and constants remain unchanged. This adapter is
evaluation-only; it must never replace a differentiable training loss.
"""

from collections import OrderedDict

import torch
import torch.nn.functional as F


class ReferenceSSIM:
    def __init__(self, window_factory, maximum_bytes=1024**3):
        if maximum_bytes <= 0:
            raise ValueError("Reference-statistic memory budget must be positive")
        self.window_factory = window_factory
        self.maximum_bytes = int(maximum_bytes)
        self.windows, self.entries = OrderedDict(), OrderedDict()
        self.hits = self.misses = self.evictions = self.oversize = 0
        self.resident_bytes = self.peak_bytes = 0

    def __call__(self, prediction, reference, window_size=11, size_average=True):
        if prediction.requires_grad or reference.requires_grad:
            raise ValueError("Reference statistics reuse is evaluation-only")
        if (prediction.ndim != 4 or prediction.shape != reference.shape
                or prediction.dtype != reference.dtype or prediction.device != reference.device
                or not prediction.is_floating_point() or window_size <= 0 or window_size % 2 == 0):
            raise ValueError("SSIM requires matching NCHW floating tensors and an odd positive window")
        channels = prediction.shape[1]
        window_key = (window_size, channels, prediction.dtype, prediction.device)
        if window_key not in self.windows:
            if len(self.windows) == 16:
                self.windows.popitem(last=False)
            self.windows[window_key] = self.window_factory(window_size, channels).to(
                device=prediction.device, dtype=prediction.dtype)
        self.windows.move_to_end(window_key)
        window = self.windows[window_key]
        # A retained tensor prevents allocator address reuse; its version detects mutation.
        key = (reference.data_ptr(), reference.storage_offset(), tuple(reference.shape),
               tuple(reference.stride()), reference.dtype, reference.device, reference._version,
               window_size)
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            _, mu_ref, mu_ref_squared, variance_ref, charge = self.entries[key]
        else:
            self.misses += 1
            mu_ref = F.conv2d(reference, window, padding=window_size//2, groups=channels)
            mu_ref_squared = mu_ref.pow(2)
            variance_ref = F.conv2d(reference*reference, window, padding=window_size//2, groups=channels)-mu_ref_squared
            charge = sum(t.numel()*t.element_size() for t in (reference, mu_ref, mu_ref_squared, variance_ref))
            if charge <= self.maximum_bytes:
                while self.entries and self.resident_bytes+charge > self.maximum_bytes:
                    _, old = self.entries.popitem(last=False)
                    self.resident_bytes -= old[-1]
                    self.evictions += 1
                self.entries[key] = (reference, mu_ref, mu_ref_squared, variance_ref, charge)
                self.resident_bytes += charge
                self.peak_bytes = max(self.peak_bytes, self.resident_bytes)
            else:
                self.oversize += 1
        mu_pred = F.conv2d(prediction, window, padding=window_size//2, groups=channels)
        mu_pred_squared = mu_pred.pow(2)
        mean_product = mu_pred*mu_ref
        variance_pred = F.conv2d(prediction*prediction, window, padding=window_size//2, groups=channels)-mu_pred_squared
        covariance = F.conv2d(prediction*reference, window, padding=window_size//2, groups=channels)-mean_product
        result = ((2*mean_product+.01**2)*(2*covariance+.03**2))/(
            (mu_pred_squared+mu_ref_squared+.01**2)*(variance_pred+variance_ref+.03**2))
        return result.mean() if size_average else result.mean(1).mean(1).mean(1)

    def receipt(self):
        return {"hits": self.hits, "misses": self.misses, "evictions": self.evictions,
                "oversize_uncached": self.oversize, "resident_accounted_bytes": self.resident_bytes,
                "peak_accounted_bytes": self.peak_bytes, "maximum_bytes": self.maximum_bytes,
                "cached_reference_views": len(self.entries), "cached_windows": len(self.windows),
                "training_use_allowed": False}

    def clear(self):
        self.entries.clear()
        self.windows.clear()
        self.resident_bytes = 0
