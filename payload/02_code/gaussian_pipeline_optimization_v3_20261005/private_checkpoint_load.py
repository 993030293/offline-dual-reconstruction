"""Apply explicit private mapping only to one already verified native checkpoint."""

from contextlib import contextmanager
import mmap
import os
from pathlib import Path
import time


def admission(path):
    import torch
    if not isinstance(path, (str, os.PathLike)):
        return False, "stream_input_selects_original_loader"
    if torch.__version__ != "2.7.1+cu128" or torch.serialization.get_default_mmap_options() != mmap.MAP_PRIVATE:
        return False, "version_or_private_mapping_contract"
    with Path(path).open("rb") as stream:
        if not torch.serialization._is_zipfile(stream):
            return False, "non_ZIP_selects_original_loader"
    return True, "fixed_native_ZIP_and_MAP_PRIVATE"


@contextmanager
def private_checkpoint_load(checkpoint, enabled=True):
    import torch
    selected = Path(checkpoint).resolve()
    compatible, reason = admission(selected)
    if enabled and not compatible:
        raise ValueError("Choose the original checkpoint loader before execution: "+reason)
    original = torch.load
    trace = {"selected_checkpoint": str(selected), "enabled": enabled, "reason": reason,
        "calls": [], "saved_device_labels_unchanged": True, "MAP_SHARED_not_allowed": True, "counted": False}

    def load(path, *args, **kwargs):
        applies = isinstance(path, (str, os.PathLike)) and Path(path).resolve() == selected
        if not applies:
            return original(path, *args, **kwargs)
        if kwargs.get("mmap") is not None:
            raise ValueError("The complete native recovery already selected a mapping policy")
        if enabled and torch.serialization.get_default_mmap_options() != mmap.MAP_PRIVATE:
            raise ValueError("A global setting changed the selected checkpoint to shared mapping")
        start = time.perf_counter()
        result = original(path, *args, **kwargs, mmap=enabled)
        trace["calls"].append({"seconds": time.perf_counter()-start, "mmap": enabled,
            "map_location_argument_unchanged": str(kwargs.get("map_location")), "weights_only_argument_unchanged": kwargs.get("weights_only")})
        return result

    torch.load = load
    try:
        yield trace
    finally:
        torch.load = original
        trace["original_global_loader_restored"] = torch.load is original
