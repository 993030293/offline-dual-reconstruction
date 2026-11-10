"""Bind segmented support to three explicit immutable states of a private export."""

from contextlib import contextmanager
import inspect
from pathlib import Path
import time

from segmented_camera_support import native_support_many


BEFORE = '''        support_before = {im.name: camera_geometry_metrics(model, im, policy["camera_geometry_admission"])
                          for im in model.images.values() if im.has_pose}'''
AFTER = '''        support_after = {im.name: camera_geometry_metrics(model, im, policy["camera_geometry_admission"])
                         for im in model.images.values() if im.has_pose}'''
FINAL = '''        final_support = [camera_geometry_metrics(pinhole, im, policy["camera_geometry_admission"])
                         for im in pinhole.images.values() if im.has_pose]'''


@contextmanager
def segmented_export_execution(native, output, enabled):
    started = time.perf_counter()
    original = native.main
    source = inspect.getsource(original)
    if any(source.count(anchor) != 1 for anchor in (BEFORE, AFTER, FINAL)):
        raise ValueError("The three original full camera support state boundaries changed")
    trace = {"enabled": enabled, "stages": [], "setup_seconds": None, "restore_seconds": None,
             "original_main_restored": False, "no_cross_filter_or_undistort_reuse": True}

    def support(model, policy, function, label, dictionary):
        begin = time.perf_counter()
        images = [image for image in model.images.values() if image.has_pose]
        record = {"stage": label, "images": len(images), "status": "running"}
        trace["stages"].append(record)
        try:
            if enabled:
                rows, details = native_support_many(model, images, policy, Path(output)/label, function)
            else:
                rows = [function(model, image, policy) for image in images]
                details = {"route": "original_scalar_with_R112", "images": len(images), "closed": True}
            record.update(status="complete_original_camera_support_fields", details=details)
            return {row["image_name"]: row for row in rows} if dictionary else rows
        except BaseException as error:
            record.update(status="failed", error=repr(error))
            raise
        finally:
            record["complete_seconds"] = time.perf_counter()-begin

    modified = source.replace(BEFORE,
        '        support_before = _r132_support(model, policy["camera_geometry_admission"], camera_geometry_metrics, "before", True)')
    modified = modified.replace(AFTER,
        '        support_after = _r132_support(model, policy["camera_geometry_admission"], camera_geometry_metrics, "after", True)')
    modified = modified.replace(FINAL,
        '        final_support = _r132_support(pinhole, policy["camera_geometry_admission"], camera_geometry_metrics, "final", False)')
    namespace = dict(original.__globals__)
    namespace["_r132_support"] = support
    exec(compile(modified, "<private_original_segmented_camera_support_export>", "exec"), namespace)
    try:
        native.main = namespace[original.__name__]
        trace["setup_seconds"] = time.perf_counter()-started
        yield trace
    finally:
        begin = time.perf_counter()
        native.main = original
        trace["restore_seconds"] = time.perf_counter()-begin
        trace["original_main_restored"] = native.main is original
