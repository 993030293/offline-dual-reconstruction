"""Check the installed plane CUDA extension against one known off-center plane."""

import math
from types import SimpleNamespace

from pgsr_calibration import apply_camera, calibrated_render


def verify(model_class, native_render):
    import torch
    from scene.cameras import Camera
    camera = Camera.__new__(Camera)
    torch.nn.Module.__init__(camera)
    camera.image_width, camera.image_height = 64, 48
    camera.znear, camera.zfar = .01, 100.
    camera.world_view_transform = torch.eye(4, device="cuda")
    camera.camera_center = torch.zeros(3, device="cuda")
    apply_camera(camera, (60., 61., 24.1, 29.7))
    model = model_class(0)
    model._xyz = torch.nn.Parameter(torch.tensor([[0., 0., 2.]], device="cuda"))
    model._scaling = torch.nn.Parameter(torch.log(torch.tensor([[.4, .35, .003]], device="cuda")))
    angle = .4
    model._rotation = torch.nn.Parameter(torch.tensor([[math.cos(angle/2), 0., math.sin(angle/2), 0.]], device="cuda"))
    model._opacity = torch.nn.Parameter(torch.tensor([[3.]], device="cuda"))
    model._features_dc = torch.nn.Parameter(torch.full((1, 1, 3), .3, device="cuda"))
    model._features_rest = torch.nn.Parameter(torch.empty((1, 0, 3), device="cuda"))
    pipeline = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    trace = {}
    result = calibrated_render(native_render, trace)(camera, model, pipeline, torch.zeros(3, device="cuda"),
        return_plane=True, return_depth_normal=True)
    supported = result["rendered_alpha"].squeeze() > .25
    normal = model.get_normal(camera).detach()[0]
    distance = torch.abs(torch.dot(normal, model.get_xyz.detach()[0]))
    expected = distance / -(torch.sum(camera.get_rays() * normal, -1))
    error = float(torch.abs(result["plane_depth"].squeeze()[supported]-expected[supported]).max().detach())
    if not supported.any() or error > .0002 or not torch.isfinite(result["depth_normal"]).all():
        raise ValueError("The installed plane kernel broke the off-center geometric contract")
    loss = result["plane_depth"].squeeze()[supported].mean() + result["render"].mean()
    loss.backward()
    checked = {}
    for name in ("_xyz", "_scaling", "_rotation", "_opacity", "_features_dc"):
        gradient = getattr(model, name).grad
        if gradient is None or not torch.isfinite(gradient).all():
            raise ValueError("The actual plane CUDA gradient failed: " + name)
        checked[name] = float(gradient.abs().max())
    return {"status": "native_plane_CUDA_forward_backward_verified", "supported_pixels": int(supported.sum()),
        "maximum_known_plane_depth_error_m": error, "gradient_absolute_maximum": checked, "trace": trace,
        "evidence_level": "installed_extension_geometric_diagnostic_not_scene_GT"}
