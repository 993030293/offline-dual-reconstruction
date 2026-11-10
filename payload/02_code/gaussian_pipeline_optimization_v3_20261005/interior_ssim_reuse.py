"""Reuse SSIM crop interiors while recomputing the original zero-padded edges."""

import inspect

import torch

import reference_ssim


def map_reference_factory(window_factory, maximum_bytes):
    source = inspect.getsource(reference_ssim.ReferenceSSIM)
    class_anchor = "class ReferenceSSIM:"
    return_anchor = "return result.mean() if size_average else result.mean(1).mean(1).mean(1)"
    if source.count(class_anchor) != 1 or source.count(return_anchor) != 1:
        raise ValueError("The existing exact reference SSIM source changed")
    source = source.replace(class_anchor, "class _MapReferenceSSIM:").replace(return_anchor, "return result")
    namespace = dict(vars(reference_ssim))
    exec(compile(source, "<private-original-Q30-map>", "exec"), namespace)
    return namespace["_MapReferenceSSIM"](window_factory, maximum_bytes=maximum_bytes)


class InteriorSSIM:
    def __init__(self, window_factory, maximum_bytes=1024**3):
        self.maximum_bytes = maximum_bytes
        self.cache = map_reference_factory(window_factory, maximum_bytes)
        self.last = None
        self.calls = self.reused = self.native = self.invalidated = 0

    @staticmethod
    def signature(tensor):
        return (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tuple(tensor.shape),
                tuple(tensor.stride()), tensor.dtype, tensor.device, tensor._version)

    @staticmethod
    def scope(prediction, reference, window_size):
        return (prediction.ndim == reference.ndim == 4 and prediction.shape == reference.shape
                and prediction.shape[:2] == (1, 3) and prediction.device.type == "cuda"
                and prediction.device == reference.device and prediction.dtype == reference.dtype == torch.float32
                and not prediction.requires_grad and not reference.requires_grad and window_size == 11)

    @staticmethod
    def coordinates(tensor, parent):
        if (tensor.untyped_storage().data_ptr() != parent.untyped_storage().data_ptr()
                or tensor.stride() != parent.stride() or tensor.dtype != parent.dtype or tensor.device != parent.device):
            return None
        row_stride, column_stride = parent.stride()[2:]
        delta = tensor.storage_offset() - parent.storage_offset()
        if row_stride <= 0 or column_stride <= 0 or delta < 0:
            return None
        y, remainder = divmod(delta, row_stride)
        x, remaining = divmod(remainder, column_stride)
        h, w = tensor.shape[2:]
        if remaining or y + h > parent.shape[2] or x + w > parent.shape[3]:
            return None
        if (parent[:, :, y:y+h, x:x+w].storage_offset() != tensor.storage_offset()
                or parent[:, :, y:y+h, x:x+w].stride() != tensor.stride()):
            return None
        return y, x, h, w

    def reserve_last_map(self, amount):
        self.cache.maximum_bytes = self.maximum_bytes - amount
        while self.cache.entries and self.cache.resident_bytes > self.cache.maximum_bytes:
            _, old = self.cache.entries.popitem(last=False)
            self.cache.resident_bytes -= old[-1]
            self.cache.evictions += 1

    def __call__(self, prediction, reference, window_size=11, size_average=True):
        self.calls += 1
        coords = None
        supported = self.scope(prediction, reference, window_size)
        if supported and self.last is not None:
            old_prediction, old_reference, pred_signature, ref_signature, full_map = self.last
            if self.signature(old_prediction) != pred_signature or self.signature(old_reference) != ref_signature:
                self.last = None
                self.invalidated += 1
                self.reserve_last_map(0)
            else:
                pred_coords = self.coordinates(prediction, old_prediction)
                ref_coords = self.coordinates(reference, old_reference)
                if pred_coords is not None and pred_coords == ref_coords and min(pred_coords[2:]) > 10:
                    coords = pred_coords
        if coords is not None:
            y, x, h, w = coords
            result = full_map[:, :, y:y+h, x:x+w].clone()
            radius = window_size // 2
            top = self.cache(prediction[:, :, :2*radius], reference[:, :, :2*radius], window_size)
            result[:, :, :radius] = top[:, :, :radius]
            bottom = self.cache(prediction[:, :, -2*radius:], reference[:, :, -2*radius:], window_size)
            result[:, :, -radius:] = bottom[:, :, -radius:]
            left = self.cache(prediction[:, :, :, :2*radius], reference[:, :, :, :2*radius], window_size)
            result[:, :, radius:-radius, :radius] = left[:, :, radius:-radius, :radius]
            right = self.cache(prediction[:, :, :, -2*radius:], reference[:, :, :, -2*radius:], window_size)
            result[:, :, radius:-radius, -radius:] = right[:, :, radius:-radius, -radius:]
            self.reused += 1
        else:
            self.native += 1
            self.last = None
            self.reserve_last_map(0)
            result = self.cache(prediction, reference, window_size, size_average)
            amount = result.numel() * result.element_size()
            if supported and amount < self.maximum_bytes:
                self.reserve_last_map(amount)
                self.last = (prediction, reference, self.signature(prediction), self.signature(reference), result)
        return result.mean() if size_average else result.mean(1).mean(1).mean(1)

    def receipt(self):
        retained = 0 if self.last is None else self.last[-1].numel() * self.last[-1].element_size()
        return {**self.cache.receipt(), "calls": self.calls, "interior_reused": self.reused,
                "original_full_map_calls": self.native, "invalidated": self.invalidated,
                "retained_last_map_bytes": retained,
                "total_accounted_bytes": self.cache.resident_bytes + retained,
                "maximum_total_bytes": self.maximum_bytes, "training_use_allowed": False}

    def clear(self):
        self.last = None
        self.cache.clear()
