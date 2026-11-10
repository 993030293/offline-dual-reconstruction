"""Observe the unchanged original seed creation and retain metadata only."""

from contextlib import contextmanager
import time


@contextmanager
def observe_seed_initialization(model_class, torch):
    original = model_class.create_from_pcd
    trace = {"calls": [], "observer_only_no_initialization_skipped": True,
        "logical_tensor_bytes_are_not_unique_allocation_or_RSS": True,
        "wall_API_bounds_are_not_CUDA_kernel_times": True}

    def observed(self, pcd, cam_infos, spatial_lr_scale):
        before_sync = time.perf_counter()
        torch.cuda.synchronize()
        sync_start = time.perf_counter() - before_sync
        started = time.perf_counter()
        value = original(self, pcd, cam_infos, spatial_lr_scale)
        API_seconds = time.perf_counter() - started
        before_sync = time.perf_counter()
        torch.cuda.synchronize()
        after_sync = time.perf_counter() - before_sync
        fields = {}
        for name in ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity", "max_radii2D", "_exposure"):
            tensor = getattr(self, name)
            fields[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "device": str(tensor.device),
                "numel": tensor.numel(), "logical_bytes": tensor.numel() * tensor.element_size(),
                "storage_pointer": tensor.untyped_storage().data_ptr(), "requires_grad": tensor.requires_grad}
        trace["calls"].append({"seed_points": len(pcd.points), "cam_infos": len(cam_infos),
            "spatial_lr_scale_argument": float(spatial_lr_scale), "original_API_seconds": API_seconds,
            "before_API_actual_synchronize_seconds": sync_start, "after_API_actual_synchronize_seconds": after_sync,
            "original_API_and_after_sync_seconds": API_seconds + after_sync,
            "fields": fields, "sum_retained_logical_tensor_bytes": sum(row["logical_bytes"] for row in fields.values()),
            "exposure_mapping_count": len(self.exposure_mapping), "pretrained_exposures_is_None": self.pretrained_exposures is None})
        return value

    model_class.create_from_pcd = observed
    try:
        yield trace
    finally:
        model_class.create_from_pcd = original
        trace["original_create_from_pcd_binding_restored"] = model_class.create_from_pcd is original
