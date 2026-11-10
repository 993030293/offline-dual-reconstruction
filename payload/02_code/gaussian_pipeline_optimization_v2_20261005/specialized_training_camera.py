"""Specialize only opaque RGB training inputs and immediately calibrated camera transforms."""

from contextlib import contextmanager
from dataclasses import asdict
import inspect
import math
import textwrap
import time


@contextmanager
def specialized_training_camera(mode):
    import torch
    import scene.cameras as cameras
    import full_intrinsics
    if mode not in ("baseline", "opaque", "clampfree", "projection"):
        raise ValueError("An explicit individually measured camera specialization is required")
    original_init, original_apply = cameras.Camera.__init__, full_intrinsics.apply_to_camera
    source = textwrap.dedent(inspect.getsource(original_init))
    alpha = "        self.alpha_mask = torch.ones_like(resized_image_rgb[0:1, ...].to(self.data_device))"
    image = "    self.original_image = gt_image.clamp(0.0, 1.0).to(self.data_device)"
    projection = "    self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()\n    self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)"
    for anchor in (alpha, image, projection):
        if source.count(anchor) != 1:
            raise ValueError("The pinned native camera implementation changed its guarded input contract")
    if mode == "opaque":
        source = source.replace(alpha, "        self.alpha_mask = None")
    if mode == "clampfree":
        source = source.replace(image, "    self.original_image = gt_image.to(self.data_device)")
    if mode == "projection":
        source = source.replace(projection, "    self.projection_matrix = None\n    self.full_proj_transform = None")
    boundaries = [("    self.alpha_mask = None", "    if train_test_exp and is_test_view:", "alpha"),
        ("    self.original_image =", "    self.image_width =", "clamp"),
        ("    self.projection_matrix =", "    self.camera_center =", "legacy_projection")]
    for begin, end, key in boundaries:
        lines = source.splitlines(keepends=True)
        starts = [i for i, line in enumerate(lines) if line.startswith(begin)]
        ends = [i for i, line in enumerate(lines) if line.startswith(end)]
        if len(starts) != 1 or len(ends) != 1 or starts[0] >= ends[0]:
            raise ValueError("A native camera measurement phase no longer has unique boundaries")
        lines.insert(ends[0], "    camera_phase_trace['"+key+"'] += camera_phase_clock()-camera_phase_started\n")
        lines.insert(starts[0], "    camera_phase_started = camera_phase_clock()\n")
        source = "".join(lines)
    trace = {"mode": mode, "native_calls": 0, "RGB_uint8_PIL_only": True, "alpha": 0., "clamp": 0.,
        "legacy_projection": 0., "calibrated_projection": 0., "projection_builds": 0, "projection_hits": 0,
        "opaque_alpha_bytes_not_allocated": 0, "clamp_image_bytes_not_copied": 0,
        "real_alpha_allocated_bytes": 0, "projection_cache_max_entries": 16,
        "data_semantics_or_loss_formula_changed": False}
    namespace = dict(original_init.__globals__)
    namespace.update(camera_phase_trace=trace, camera_phase_clock=time.perf_counter)
    exec(compile(source, "<guarded_RGB_training_camera_specialization>", "exec"), namespace)
    compiled = namespace["__init__"]
    signature = inspect.signature(original_init)
    cache = {}

    def initialize(camera, *args, **kwargs):
        bound = signature.bind(camera, *args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        if (values["image"].mode != "RGB" or values["invdepthmap"] is not None or values["train_test_exp"]
                or values["is_test_view"] or values["is_test_dataset"] or values["data_device"] != "cpu"):
            raise ValueError("The camera specialization excludes alpha, depth, exposure split and non-CPU input variants")
        compiled.__globals__["PILtoTorch"] = cameras.PILtoTorch
        compiled(camera, *args, **kwargs)
        trace["native_calls"] += 1
        count = camera.image_width*camera.image_height
        if mode == "opaque":
            trace["opaque_alpha_bytes_not_allocated"] += count*4
        else:
            trace["real_alpha_allocated_bytes"] += camera.alpha_mask.numel()*camera.alpha_mask.element_size()
        if mode == "clampfree":
            trace["clamp_image_bytes_not_copied"] += 3*count*4

    def calibrated(camera, intrinsic):
        begin = time.perf_counter()
        if mode != "projection":
            try:
                return original_apply(camera, intrinsic)
            finally:
                trace["calibrated_projection"] += time.perf_counter()-begin
        if (camera.image_width, camera.image_height) != (intrinsic.width, intrinsic.height):
            raise ValueError("The calibrated camera and image dimensions differ")
        camera.FoVx = 2.*math.atan(intrinsic.width/(2.*intrinsic.fx))
        camera.FoVy = 2.*math.atan(intrinsic.height/(2.*intrinsic.fy))
        view = camera.world_view_transform
        key = (intrinsic, camera.znear, camera.zfar, view.device, view.dtype)
        if key not in cache:
            if len(cache) >= 16:
                raise ValueError("The fixed calibrated projection cache exceeded its declared bound")
            value = torch.as_tensor(intrinsic.projection(camera.znear, camera.zfar), device=view.device, dtype=view.dtype).T.contiguous()
            cache[key] = (value, value._version)
            trace["projection_builds"] += 1
        else:
            trace["projection_hits"] += 1
        value, version = cache[key]
        if value._version != version:
            raise ValueError("A shared calibrated projection was mutated")
        camera.projection_matrix = value
        camera.full_proj_transform = view@value
        camera.calibrated_intrinsics = asdict(intrinsic)
        camera.pixel_convention = "colmap_corner_first_center_0.5"
        trace["calibrated_projection"] += time.perf_counter()-begin
        return camera

    cameras.Camera.__init__, full_intrinsics.apply_to_camera = initialize, calibrated
    try:
        yield trace
        if any(value._version != version for value, version in cache.values()):
            raise ValueError("An immutable calibrated projection changed before camera specialization cleanup")
    finally:
        cameras.Camera.__init__, full_intrinsics.apply_to_camera = original_init, original_apply

