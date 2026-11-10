"""Consume private batched transforms within the unchanged calibrated camera path."""

from contextlib import contextmanager
import difflib
import inspect
import linecache
from pathlib import Path
import textwrap

import numpy as np

from batched_camera_transforms import prepare_camera_transforms


@contextmanager
def batched_camera_execution(output, trace):
    import torch
    import scene
    import scene.cameras as cameras
    original_init = cameras.Camera.__init__
    original_list = scene.cameraList_from_camInfos
    consumer = original_list
    list_bound = False
    native_world = original_init.__globals__["getWorld2View2"]
    before = textwrap.dedent(inspect.getsource(original_init))
    anchors = (
        ("    self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()",
         "    self.world_view_transform, _private_center = _private_camera_transform(R, T, trans, scale)"),
        ("    self.camera_center = self.world_view_transform.inverse()[3, :3]",
         "    self.camera_center = _private_center"),
    )
    after = before
    for anchor, replacement in anchors:
        if after.count(anchor) != 1:
            raise ValueError("The original camera transform source changed")
        after = after.replace(anchor, replacement)
    active = {}
    trace.update(batch_groups=0, batch_prepared=0, batch_consumed=0, native_calls=0,
                 empty_groups=0, nonadmitted_groups=[], native_reasons=[], maximum_active_entries=0)

    def native(R, T, trans, scale, reason):
        trace["native_calls"] += 1
        trace["native_reasons"].append(reason)
        view = torch.tensor(native_world(R, T, trans, scale)).transpose(0, 1).cuda()
        return view, view.inverse()[3, :3]

    def lookup(R, T, trans, scale):
        key = id(R), id(T)
        row = active.get(key)
        if row is None:
            return native(R, T, trans, scale, "no_admitted_unconsumed_binding")
        if (not isinstance(trans, np.ndarray) or trans.dtype != np.float64 or trans.shape != (3,)
                or trans.tobytes() != np.zeros(3).tobytes() or scale != 1.0
                or R.tobytes() != row["R_bytes"] or T.tobytes() != row["T_bytes"]):
            return native(R, T, trans, scale, "changed_binding_or_nondefault_normalization")
        active.pop(key)
        trace["batch_consumed"] += 1
        return row["view"], row["center"]

    def camera_list(cam_infos, *args, **kwargs):
        if active:
            raise RuntimeError("Camera transform batch scopes cannot overlap")
        if not cam_infos:
            trace["empty_groups"] += 1
            return consumer(cam_infos, *args, **kwargs)
        reason = None
        if not 0 < len(cam_infos) <= 4096:
            reason = "count_outside_private_bound"
        elif any(not isinstance(c.R, np.ndarray) or not isinstance(c.T, np.ndarray)
                 or c.R.dtype != np.float64 or c.T.dtype != np.float64 or c.R.shape != (3, 3) or c.T.shape != (3,)
                 or not np.isfinite(c.R).all() or not np.isfinite(c.T).all() for c in cam_infos):
            reason = "unsupported_or_nonfinite_pose"
        elif len({(id(c.R), id(c.T)) for c in cam_infos}) != len(cam_infos):
            reason = "repeated_mutable_pose_binding"
        elif any(np.linalg.det(c.R) == 0 for c in cam_infos):
            reason = "singular_original_rotation"
        if reason is not None:
            trace["nonadmitted_groups"].append({"reason": reason, "count": len(cam_infos)})
            return consumer(cam_infos, *args, **kwargs)
        hosts = []
        try:
            for c in cam_infos:
                hosts.append(native_world(c.R, c.T))
        except np.linalg.LinAlgError as error:
            trace["nonadmitted_groups"].append({"reason": "original_host_inverse_nonadmission", "error": repr(error)})
            return consumer(cam_infos, *args, **kwargs)
        if any(not np.isfinite(matrix).all() for matrix in hosts):
            trace["nonadmitted_groups"].append({"reason": "nonfinite_native_float32_host_matrix"})
            return consumer(cam_infos, *args, **kwargs)
        try:
            pairs = prepare_camera_transforms(hosts)
        except torch.linalg.LinAlgError as error:
            trace["nonadmitted_groups"].append({"reason": "original_batch_inverse_nonadmission", "error": repr(error)})
            return consumer(cam_infos, *args, **kwargs)
        for c, (view, center) in zip(cam_infos, pairs):
            active[(id(c.R), id(c.T))] = {"R_bytes": c.R.tobytes(), "T_bytes": c.T.tobytes(), "view": view, "center": center}
        trace["batch_groups"] += 1
        trace["batch_prepared"] += len(cam_infos)
        trace["maximum_active_entries"] = max(trace["maximum_active_entries"], len(active))
        try:
            return consumer(cam_infos, *args, **kwargs)
        finally:
            active.clear()

    @contextmanager
    def bind_current_camera_list():
        nonlocal consumer, list_bound
        if list_bound or active:
            raise RuntimeError("Private camera-list bindings cannot overlap")
        previous = scene.cameraList_from_camInfos
        consumer, list_bound = previous, True
        scene.cameraList_from_camInfos = camera_list
        trace["list_bound_after_prefetch"] = True
        try:
            yield
        finally:
            active.clear()
            scene.cameraList_from_camInfos = previous
            consumer, list_bound = original_list, False

    namespace = dict(original_init.__globals__)
    namespace["_private_camera_transform"] = lookup
    filename = str(Path(output) / "EXECUTED_BATCH_CAMERA_INIT.py")
    prior_linecache = linecache.cache.get(filename)
    linecache.cache[filename] = (len(after), None, after.splitlines(keepends=True), filename)
    Path(filename).write_text(after, encoding="utf-8")
    (Path(output) / "BATCH_CAMERA_INIT.diff").write_text("".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True), fromfile="original", tofile="private_batch")), encoding="utf-8")
    exec(compile(after, filename, "exec"), namespace)
    cameras.Camera.__init__ = namespace["__init__"]
    try:
        yield bind_current_camera_list
    finally:
        active.clear()
        cameras.Camera.__init__, scene.cameraList_from_camInfos = original_init, original_list
        if prior_linecache is None:
            linecache.cache.pop(filename, None)
        else:
            linecache.cache[filename] = prior_linecache
        trace["original_functions_and_linecache_restored"] = cameras.Camera.__init__ is original_init and scene.cameraList_from_camInfos is original_list
        trace["active_entries_after_close"] = len(active)
