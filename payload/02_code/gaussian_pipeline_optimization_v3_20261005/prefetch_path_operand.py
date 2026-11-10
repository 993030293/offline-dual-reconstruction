"""Reuse a resolved operand only within one owned static path comparison."""

from contextlib import contextmanager
import difflib
import inspect
from pathlib import Path


def same_image_path(left, right, receipt):
    """Preserve native alias handling and the first native resolve exception."""
    left_path = Path(left)
    resolved = left_path.resolve()
    receipt["native_resolve_calls"] += 1
    if type(left) is str and type(right) is str and left == right and left_path.is_absolute():
        receipt["same_absolute_operand"] += 1
        return True
    receipt["native_fallback_comparisons"] += 1
    result = resolved == Path(right).resolve()
    receipt["native_resolve_calls"] += 1
    return result


@contextmanager
def prefetch_path_operand(output, enabled, workers=4, capacity=8):
    import prefetch_camera_images as parent
    import scene
    import scene.cameras as cameras
    from runtime import atomic_json

    original_factory = parent.prefetch_camera_images
    original_list, original_prepare = scene.cameraList_from_camInfos, cameras.PILtoTorch
    receipt = {"enabled": enabled, "same_absolute_operand": 0, "native_fallback_comparisons": 0,
               "native_resolve_calls": 0, "no_cross_comparison_cache": True,
               "static_registered_namespace_required": True}
    before = inspect.getsource(original_factory)
    anchor = "Path(image.filename).resolve() != Path(infos[index].image_path).resolve()"
    if before.count(anchor) != 1:
        raise ValueError("The original prefetch path comparison boundary changed")
    after = before.replace(anchor, "not _same_image_path(image.filename, infos[index].image_path)") if enabled else before
    archive = Path(output).with_name(Path(output).name + "_operand_sources")
    archive.mkdir(parents=True, exist_ok=False)
    (archive / "EXECUTED_PREFETCH_FACTORY.py").write_text(after, encoding="utf-8")
    (archive / "PREFETCH_FACTORY.diff").write_text("".join(difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile="original_prefetch", tofile="same_operand_reuse")), encoding="utf-8")
    native_function = inspect.unwrap(original_factory)
    namespace = dict(native_function.__globals__)
    if native_function.__module__ != parent.__name__ or any(namespace.get(name) is not parent.__dict__.get(name) for name in ("time", "deque", "ThreadPoolExecutor")):
        raise ValueError("The original contextmanager function globals changed")
    namespace["_same_image_path"] = lambda left, right: same_image_path(left, right, receipt)
    exec(compile(after, "<owned_same_path_operand>", "exec"), namespace)
    succeeded = False
    try:
        with namespace["prefetch_camera_images"](workers, capacity) as prefetch:
            yield prefetch
        succeeded = True
    finally:
        receipt["original_functions_restored"] = scene.cameraList_from_camInfos is original_list and cameras.PILtoTorch is original_prepare and parent.prefetch_camera_images is original_factory
        receipt["status"] = "completed" if succeeded else "original_failure_preserved"
        atomic_json(Path(output) / "PATH_OPERAND_TRACE.json", receipt)

