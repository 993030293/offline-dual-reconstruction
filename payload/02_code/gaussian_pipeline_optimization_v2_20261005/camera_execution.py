"""Combine independently profiled native RGB camera allocation mechanisms."""

from contextlib import contextmanager
from dataclasses import asdict
import inspect
import math
import textwrap


@contextmanager
def camera_execution(opaque_alpha=False, clampfree=False, calibrated_projection=False):
    import torch
    import scene.cameras as cameras
    import full_intrinsics
    original_init, original_apply = cameras.Camera.__init__, full_intrinsics.apply_to_camera
    source = textwrap.dedent(inspect.getsource(original_init))
    edits = [
        (opaque_alpha,
         "        self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))",
         "        self.alpha_mask = None"),
        (clampfree,
         "    self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)",
         "    self.original_image = gt_image.to(self.data_device)"),
        (calibrated_projection,
         "    self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()\n    self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)",
         "    self.projection_matrix = None\n    self.full_proj_transform = None")]
    for enabled, before, after in edits:
        if type(enabled) is not bool or source.count(before) != 1:
            raise ValueError("The exact native RGB camera implementation or explicit mechanism flags changed")
        if enabled:
            source = source.replace(before, after)
    namespace = dict(original_init.__globals__)
    exec(compile(source, "<combined_exact_native_RGB_camera_execution>", "exec"), namespace)
    compiled = namespace["__init__"]
    signature = inspect.signature(original_init)
    entries = {}
    trace = {"native_camera_calls": 0, "opaque_alpha": opaque_alpha, "clampfree": clampfree,
        "calibrated_projection": calibrated_projection, "projection_builds": 0, "projection_hits": 0,
        "maximum_projection_entries": 16, "no_metadata_seek_or_input_precision_change": True}

    def initialize(camera, *args, **kwargs):
        bound = signature.bind(camera, *args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        if (values["image"].mode != "RGB" or values["invdepthmap"] is not None or values["train_test_exp"] or
                values["is_test_view"] or values["is_test_dataset"] or values["data_device"] != "cpu"):
            raise ValueError("The exact camera execution requires opaque CPU RGB without depth or exposure split")
        # The bounded prefetch context patches this binding after camera setup.
        compiled.__globals__["PILtoTorch"] = cameras.PILtoTorch
        compiled(camera, *args, **kwargs)
        trace["native_camera_calls"] += 1

    def apply(camera, intrinsic):
        if (camera.image_width, camera.image_height) != (intrinsic.width, intrinsic.height):
            raise ValueError("The actual RGB pixels and calibrated camera dimensions differ")
        camera.FoVx = 2*math.atan(intrinsic.width/(2*intrinsic.fx))
        camera.FoVy = 2*math.atan(intrinsic.height/(2*intrinsic.fy))
        view = camera.world_view_transform
        key = intrinsic, camera.znear, camera.zfar, view.device, view.dtype
        if key not in entries:
            if len(entries) >= 16:
                raise ValueError("The calibrated constant projection cache exceeded its declared bound")
            projection = torch.as_tensor(intrinsic.projection(camera.znear, camera.zfar), device=view.device, dtype=view.dtype).T.contiguous()
            entries[key] = projection, projection._version
            trace["projection_builds"] += 1
        else:
            trace["projection_hits"] += 1
        projection, version = entries[key]
        if projection._version != version:
            raise ValueError("A shared native calibrated projection was mutated")
        camera.projection_matrix = projection
        camera.full_proj_transform = view@projection
        camera.calibrated_intrinsics = asdict(intrinsic)
        camera.pixel_convention = "colmap_corner_first_center_0.5"
        return camera

    cameras.Camera.__init__ = initialize
    if calibrated_projection:
        full_intrinsics.apply_to_camera = apply
    try:
        yield trace
        if any(value._version != version for value, version in entries.values()):
            raise ValueError("A shared projection was modified before complete camera execution cleanup")
    finally:
        cameras.Camera.__init__, full_intrinsics.apply_to_camera = original_init, original_apply
