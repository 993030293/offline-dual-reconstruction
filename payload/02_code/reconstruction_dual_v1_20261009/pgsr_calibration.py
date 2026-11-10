"""Apply complete corner-domain intrinsics to native PGSR cameras and rays."""

from contextlib import contextmanager
import math

import numpy as np


def scaled_intrinsics(params, source_size, target_size):
    fx, fy, cx, cy = map(float, params)
    sx, sy = target_size[0] / source_size[0], target_size[1] / source_size[1]
    return fx * sx, fy * sy, cx * sx, cy * sy


def projection_matrix(params, width, height, near, far):
    fx, fy, cx, cy = params
    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[0, 0], matrix[1, 1] = 2 * fx / width, 2 * fy / height
    matrix[0, 2], matrix[1, 2] = 2 * cx / width - 1, 2 * cy / height - 1
    matrix[2, 2], matrix[2, 3], matrix[3, 2] = far / (far - near), -far * near / (far - near), 1
    return matrix


def camera_K(camera, scale=1.0):
    width, height = int(camera.image_width / scale), int(camera.image_height / scale)
    fx, fy, cx, cy = scaled_intrinsics(camera.corner_K, (camera.image_width, camera.image_height), (width, height))
    return np.array([[fx, 0, cx - .5], [0, fy, cy - .5], [0, 0, 1]], dtype=np.float32)


def apply_camera(camera, params):
    import torch
    camera.corner_K = tuple(map(float, params))
    fx, fy, cx, cy = camera.corner_K
    camera.Fx, camera.Fy, camera.Cx, camera.Cy = fx, fy, cx - .5, cy - .5
    camera.FoVx, camera.FoVy = 2 * math.atan(camera.image_width / (2 * fx)), 2 * math.atan(camera.image_height / (2 * fy))
    matrix = projection_matrix(params, camera.image_width, camera.image_height, camera.znear, camera.zfar)
    camera.projection_matrix = torch.as_tensor(matrix, dtype=camera.world_view_transform.dtype,
        device=camera.world_view_transform.device).T.contiguous()
    camera.full_proj_transform = camera.world_view_transform @ camera.projection_matrix
    return camera


def calibrated_render(native_render, trace):
    def render(view, model, pipeline, background, *arguments, **keywords):
        import torch
        normal_requested = keywords.get("return_depth_normal", True)
        plane_requested = keywords.get("return_plane", True)
        keywords["return_depth_normal"] = False
        result = native_render(view, model, pipeline, background, *arguments, **keywords)
        if plane_requested:
            if not hasattr(view, "corner_K"):
                raise ValueError("Plane-depth correction requires an admitted complete K")
            rays = view.get_rays().permute(2, 0, 1)
            denominator = -(torch.sum(result["rendered_normal"] * rays, dim=0, keepdim=True) + 1e-8)
            corrected = result["rendered_distance"] / denominator
            if trace.get("plane_depth_calls", 0) == 0:
                valid = torch.isfinite(corrected) & torch.isfinite(result["plane_depth"]) & (result["rendered_alpha"] > .2)
                trace["first_supported_depth_pixels"] = int(valid.sum())
                trace["first_mean_native_centered_depth_delta_m"] = float(torch.abs(corrected[valid] - result["plane_depth"][valid]).mean()) if valid.any() else None
            trace["plane_depth_calls"] = trace.get("plane_depth_calls", 0) + 1
            result["plane_depth"] = corrected
            if normal_requested:
                from gaussian_renderer import render_normal
                result["depth_normal"] = render_normal(view, corrected.squeeze()) * result["rendered_alpha"].detach()
        return result
    return render


@contextmanager
def calibrated_cameras(dataset):
    import torch
    import utils.camera_utils as camera_utils
    from scene.cameras import Camera
    from scene.colmap_loader import read_intrinsics_binary
    intrinsics = read_intrinsics_binary(dataset / "sparse/cameras.bin")
    original_loader = camera_utils.loadCam
    methods = {name: getattr(Camera, name) for name in ("get_k", "get_inv_k", "get_rays", "get_calib_matrix_nerf")}
    trace = {"loaded": 0, "maximum_ray_round_trip_error_px": 0.0, "full_principal_point": True,
             "COLMAP_pixel_domain": "corner", "PGSR_tensor_pixel_domain": "integer_center"}

    def load(args, index, info, resolution_scale):
        camera = original_loader(args, index, info, resolution_scale)
        source = intrinsics[info.uid]
        if source.model != "PINHOLE":
            raise ValueError("PGSR calibration adapter requires rectified PINHOLE")
        params = scaled_intrinsics(source.params, (source.width, source.height),
                                   (camera.image_width, camera.image_height))
        apply_camera(camera, params)
        trace["loaded"] += 1
        if trace["loaded"] == 1:
            rays = camera.get_rays().cpu().numpy()
            y, x = np.indices((camera.image_height, camera.image_width))
            error = float(max(np.abs(rays[..., 0] * camera.Fx + camera.Cx - x).max(),
                              np.abs(rays[..., 1] * camera.Fy + camera.Cy - y).max()))
            if error > .0002:
                raise ValueError("Actual PGSR ray calibration failed")
            trace["maximum_ray_round_trip_error_px"] = error
        return camera

    def get_k(camera, scale=1.0):
        return torch.as_tensor(camera_K(camera, scale), device=camera.world_view_transform.device)

    def get_inv_k(camera, scale=1.0):
        return torch.linalg.inv(get_k(camera, scale))

    def get_rays(camera, scale=1.0):
        width, height = int(camera.image_width / scale), int(camera.image_height / scale)
        matrix = camera_K(camera, scale)
        y, x = torch.meshgrid(torch.arange(height, device=camera.world_view_transform.device),
                             torch.arange(width, device=camera.world_view_transform.device), indexing="ij")
        return torch.stack([(x - float(matrix[0, 2])) / float(matrix[0, 0]),
                            (y - float(matrix[1, 2])) / float(matrix[1, 1]), torch.ones_like(x)], -1).float()

    def get_calib(camera, scale=1.0):
        return get_k(camera, scale).cpu(), camera.world_view_transform.T.contiguous()

    camera_utils.loadCam = load
    Camera.get_k, Camera.get_inv_k, Camera.get_rays, Camera.get_calib_matrix_nerf = get_k, get_inv_k, get_rays, get_calib
    try:
        yield trace
    finally:
        camera_utils.loadCam = original_loader
        for name, method in methods.items():
            setattr(Camera, name, method)
