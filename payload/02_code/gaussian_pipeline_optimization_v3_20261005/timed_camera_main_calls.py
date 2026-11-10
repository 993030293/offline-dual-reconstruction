"""Attribute original path, image-open and tensor calls without replacement."""

from contextlib import contextmanager
from pathlib import Path
import threading
import time

from runtime import atomic_json


@contextmanager
def timed_camera_main_calls(output):
    import torch
    from PIL import Image

    originals = {"Path.resolve": Path.resolve, "Image.open": Image.open,
                 "Tensor.cuda": torch.Tensor.cuda, "Tensor.inverse": torch.Tensor.inverse}
    rows = {key: [] for key in originals}
    lock = threading.Lock()

    def measured(key):
        native = originals[key]
        def invoke(*args, **kwargs):
            started = time.perf_counter()
            result = native(*args, **kwargs)
            elapsed = time.perf_counter() - started
            first = args[0] if args else kwargs.get("fp")
            if key.startswith("Tensor."):
                identity = {"shape": list(first.shape), "device": str(first.device)}
            else:
                identity = {"path": str(first)}
            row = {"thread": threading.current_thread().name, "seconds": elapsed, **identity}
            with lock:
                rows[key].append(row)
            return result
        return invoke

    Path.resolve = measured("Path.resolve")
    Image.open = measured("Image.open")
    torch.Tensor.cuda = measured("Tensor.cuda")
    torch.Tensor.inverse = measured("Tensor.inverse")
    succeeded = False
    try:
        yield
        succeeded = True
    finally:
        Path.resolve = originals["Path.resolve"]
        Image.open = originals["Image.open"]
        torch.Tensor.cuda = originals["Tensor.cuda"]
        torch.Tensor.inverse = originals["Tensor.inverse"]
        checks = {
            "all_original_four_functions_restored": Path.resolve is originals["Path.resolve"] and Image.open is originals["Image.open"] and torch.Tensor.cuda is originals["Tensor.cuda"] and torch.Tensor.inverse is originals["Tensor.inverse"],
            "all1016_native_camera_inverse_calls_on_main": sum(row["thread"] == "MainThread" and row["shape"] == [4, 4] for row in rows["Tensor.inverse"]) == 1016,
            "original_native_path_guard_called_at_least_twice_per_camera": sum(row["thread"] == "MainThread" for row in rows["Path.resolve"]) >= 2032,
            "original_image_open_on_main_and_CPU_workers": sum(row["thread"] == "MainThread" for row in rows["Image.open"]) >= 1016 and sum(row["thread"].startswith("camera-decode") for row in rows["Image.open"]) == 1016,
        }
        summary = {}
        for key, values in rows.items():
            groups = {}
            for row in values:
                group = "main" if row["thread"] == "MainThread" else "worker"
                entry = groups.setdefault(group, {"calls": 0, "cumulative_seconds": 0.})
                entry["calls"] += 1
                entry["cumulative_seconds"] += row["seconds"]
            summary[key] = groups
        atomic_json(Path(output) / "MAIN_CALL_COST.json", {"status": "complete_original_main_call_cost0" if succeeded and all(checks.values()) else "failed_verification",
            "checks": checks, "groups": summary, "all_calls": rows,
            "new_optimization_count": 0, "no_native_algorithm_or_path_check_skipped": True,
            "worker_sums_not_scene_wall_or_formal_speed_claim": True})
        if succeeded and not all(checks.values()):
            raise AssertionError("The complete native call diagnostic missed an original binding")
