"""Train calibrated PGSR and fuse camera-z depth into a real surface point cloud."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import psutil

from pgsr_calibration import calibrated_cameras, camera_K, calibrated_render

SOURCE = Path("/opt/e26_thin_surface_v1/pgsr")


def publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def prepare(root):
    sys.path.insert(0, str(SOURCE))
    os.chdir(SOURCE)
    import torch
    torch.set_num_threads(8)
    from arguments import ModelParams, OptimizationParams, PipelineParams
    parser = argparse.ArgumentParser()
    groups = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    config = json.loads((root / "CONFIG.json").read_text())
    namespace = parser.parse_args(["-s", str(root / "shared/pgsr"), "-m", str(root / "pgsr/model"),
        "-r", "1", "--data_device", "cpu", "--iterations", str(config["pgsr_iterations"]),
        "--multi_view_sample_num", "16384", "--max_abs_split_points", "20000", "--max_all_points", "2500000"])
    dataset, opt, pipe = [group.extract(namespace) for group in groups]
    dataset.preload_img = False
    return config, dataset, opt, pipe


def train(root):
    import torch
    config, dataset, opt, pipe = prepare(root)
    expected = json.loads((root / "shared/RESULT.json").read_text())["registered_frames"]
    output = root / "pgsr"
    output.mkdir(exist_ok=False)
    if not torch.cuda.is_available():
        raise RuntimeError("PGSR requires the installed CUDA runtime")
    spec = importlib.util.spec_from_file_location("native_delivery_pgsr_train", SOURCE / "train.py")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    native.setup_seed(0)
    models = []
    original_model = native.GaussianModel
    original_backward = torch.autograd.backward
    original_scene = native.Scene
    original_render = native.render
    render_trace = {"complete_K_plane_depth": True, "native_centered_depth_kernel_output_recomputed": True,
                    "differentiable_plane_ratio_from_native_normal_distance": True}
    started = time.monotonic()
    state = {"status": "running", "iteration": 0, "target_iterations": opt.iterations,
             "source_commit": subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip(),
             "dataset": dataset.source_path, "full_calibration_adapter": True, "seed": 0,
             "preload_img": False, "data_device": "cpu", "max_all_points": opt.max_all_points,
             "multi_view_sample_num": opt.multi_view_sample_num, "GT_used": False}

    class TrackedModel(original_model):
        def __init__(self, *arguments, **keywords):
            super().__init__(*arguments, **keywords)
            models.append(self)

    def scene(*arguments, **keywords):
        result = original_scene(*arguments, **keywords)
        if len(result.getTrainCameras()) != expected or result.getTestCameras():
            raise ValueError("PGSR changed the complete training camera denominator")
        return result

    def backward(tensors, *arguments, **keywords):
        values = (tensors,) if isinstance(tensors, torch.Tensor) else tensors
        if not all(bool(torch.isfinite(value).all()) for value in values):
            raise ValueError("PGSR produced a nonfinite training loss")
        original_backward(tensors, *arguments, **keywords)
        model = models[0]
        parameters = {name: getattr(model, name) for name in
            ('_xyz', '_scaling', '_rotation', '_opacity', '_features_dc', '_features_rest')}
        checks = [torch.isfinite(value).all() for value in parameters.values()]
        checks += [torch.isfinite(value.grad).all() for value in parameters.values() if value.grad is not None]
        if not bool(torch.stack(checks).all()):
            publish(output / 'NONFINITE_UPDATE_REJECTION.json', {
                'status': 'nonfinite_parameters_or_gradients_rejected_before_optimizer_update',
                'iteration': state['iteration'] + 1,
                'parameter_nonfinite_counts': {name: int((~torch.isfinite(value)).sum())
                    for name, value in parameters.items()},
                'gradient_nonfinite_counts': {name: int((~torch.isfinite(value.grad)).sum())
                    for name, value in parameters.items() if value.grad is not None},
                'gradients_sanitized_or_points_removed': False})
            raise ValueError('PGSR produced nonfinite parameters or gradients before optimizer update')
        state["iteration"] += 1
        if state["iteration"] % 100 == 0:
            state.update(elapsed_seconds=time.monotonic() - started,
                loss=float(values[0].detach().mean()), gaussian_count=int(models[0].get_xyz.shape[0]),
                rss_bytes=psutil.Process().memory_info().rss,
                peak_cuda_reserved_bytes=int(torch.cuda.max_memory_reserved()))
            if state["elapsed_seconds"] > 7200:
                raise RuntimeError("PGSR exceeded the frozen training time budget")
            if state["peak_cuda_reserved_bytes"] > 10 * 1024**3:
                raise RuntimeError("PGSR exceeded its frozen CUDA allocation budget")
            if state["rss_bytes"] > 32 * 1024**3:
                raise RuntimeError("PGSR exceeded its Linux resident-memory budget")
            publish(output / "STATE.json", state)

    try:
        native.GaussianModel, native.Scene = TrackedModel, scene
        native.render = calibrated_render(original_render, render_trace)
        with calibrated_cameras(Path(dataset.source_path)) as calibration:
            from check_native_pgsr import verify
            state["native_plane_kernel_diagnostic"] = verify(original_model, original_render)
            publish(output / "NATIVE_PLANE_DIAGNOSTIC.json", state["native_plane_kernel_diagnostic"])
            # The diagnostic differentiates once and must not advance the training counter.
            torch.autograd.backward = backward
            native.training(dataset, opt, pipe, [], [opt.iterations], [opt.iterations], None, -1)
        if state["iteration"] != opt.iterations or len(models) != 1:
            raise ValueError("PGSR did not finish every requested iteration")
        model = models[0]
        for tensor in (model.get_xyz, model.get_opacity, model.get_scaling, model.get_rotation, model.get_features):
            for chunk in torch.split(tensor, 65536):
                if not bool(torch.isfinite(chunk).all()):
                    raise ValueError("PGSR trained parameters are nonfinite")
        ply = output / f"model/point_cloud/iteration_{opt.iterations}/point_cloud.ply"
        if not ply.is_file():
            raise ValueError("PGSR final parameter PLY was not saved")
        state.update(status="completed", elapsed_seconds=time.monotonic() - started,
            gaussian_count=int(model.get_xyz.shape[0]), registered_frames=expected, calibration=calibration,
            parameter_ply=str(ply), sampled_surface_cloud_generated=False)
        state["render_calibration"] = render_trace
        publish(output / "STATE.json", state)
        publish(output / "TRAINING_RESULT.json", state)
    except BaseException:
        state.update(status="failed_training", elapsed_seconds=time.monotonic() - started,
                     traceback=traceback.format_exc())
        publish(output / "STATE.json", state)
        publish(output / "TRAINING_FAILURE.json", state)
        raise
    finally:
        native.GaussianModel, native.Scene, torch.autograd.backward = original_model, original_scene, original_backward
        native.render = original_render


def surface_legacy(root):
    import torch
    import open3d as o3d
    from PIL import Image
    config, dataset, opt, pipe = prepare(root)
    from scene import Scene, GaussianModel
    from gaussian_renderer import render as native_render
    render_trace = {"complete_K_plane_depth": True}
    render = calibrated_render(native_render, render_trace)
    trained = json.loads((root / "pgsr/TRAINING_RESULT.json").read_text())
    if trained["status"] != "completed" or trained["iteration"] != config["pgsr_iterations"]:
        raise ValueError("Surface requires the complete trained PGSR branch")
    output = root / "pgsr/surface"
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    state = {"status": "running", "depth_unit": "meters", "depth_kind": "camera_z",
             "depth_filter": "finite positive range and grazing angle under 80 degrees", "frames_integrated": 0}
    volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=config["surface_voxel_m"],
        sdf_trunc=config["surface_truncation_m"], color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    try:
        with torch.no_grad(), calibrated_cameras(Path(dataset.source_path)) as calibration:
            from check_native_pgsr import verify
            with torch.enable_grad():
                state["native_plane_kernel_diagnostic"] = verify(GaussianModel, native_render)
            publish(output / "NATIVE_PLANE_DIAGNOSTIC.json", state["native_plane_kernel_diagnostic"])
            model = GaussianModel(dataset.sh_degree)
            scene = Scene(dataset, model, load_iteration=opt.iterations, shuffle=False)
            views = scene.getTrainCameras()
            if len(views) != trained["registered_frames"]:
                raise ValueError("Surface camera denominator changed")
            diagnostics = set(np.linspace(0, len(views) - 1, 6).round().astype(int).tolist())
            background = torch.zeros(3, dtype=torch.float32, device="cuda")
            rows = []
            for index, view in enumerate(views):
                rendered = render(view, model, pipe, background, return_plane=True, return_depth_normal=True)
                depth = rendered["plane_depth"].squeeze().detach().cpu().numpy().astype(np.float32)
                rays = view.get_rays()
                normals = rendered["depth_normal"].permute(1, 2, 0)
                norm = torch.linalg.vector_norm(normals, dim=-1)
                cosine = torch.sum(torch.nn.functional.normalize(rays, dim=-1) *
                                   torch.nn.functional.normalize(normals, dim=-1), dim=-1).abs()
                angle_valid = ((norm > 1e-6) & (cosine >= np.cos(np.deg2rad(80)))).cpu().numpy()
                valid = np.isfinite(depth) & (depth > 0) & (depth <= config["surface_depth_max_m"]) & angle_valid
                depth[~valid] = 0
                color = np.rint(rendered["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                intrinsic = camera_K(view)
                camera = o3d.camera.PinholeCameraIntrinsic(view.image_width, view.image_height,
                    float(intrinsic[0, 0]), float(intrinsic[1, 1]), float(intrinsic[0, 2]), float(intrinsic[1, 2]))
                extrinsic = view.world_view_transform.T.detach().cpu().numpy().astype(np.float64)
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(np.ascontiguousarray(color)), o3d.geometry.Image(np.ascontiguousarray(depth)),
                    depth_scale=1.0, depth_trunc=config["surface_depth_max_m"], convert_rgb_to_intensity=False)
                volume.integrate(rgbd, camera, extrinsic)
                row = {"image_name": view.image_name, "valid_depth_pixels": int(valid.sum()),
                       "total_pixels": int(valid.size), "depth_median_m": float(np.median(depth[valid])) if valid.any() else None}
                rows.append(row)
                if index in diagnostics:
                    Image.fromarray(color).save(output / f"render_{index:04d}.png")
                    truth, _ = view.get_image()
                    rgb = np.rint(truth.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(rgb).save(output / f"reference_{index:04d}.png")
                    np.save(output / f"depth_{index:04d}.npy", depth)
                if (index + 1) % 20 == 0 or index + 1 == len(views):
                    state.update(frames_integrated=index + 1, elapsed_seconds=time.monotonic() - started,
                                 last_frame=row, rss_bytes=psutil.Process().memory_info().rss)
                    if state["elapsed_seconds"] > 7200 or state["rss_bytes"] > 32 * 1024**3:
                        raise RuntimeError("Surface fusion exceeded its registered time or resident-memory budget")
                    publish(output / "STATE.json", state)
                    print(json.dumps({"surface_frames": index + 1, "total": len(views)}), flush=True)
            mesh = volume.extract_triangle_mesh()
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_unreferenced_vertices()
            mesh.compute_vertex_normals()
            vertices, triangles = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
            if len(vertices) < 100 or len(triangles) < 100 or not np.isfinite(vertices).all():
                raise ValueError("PGSR TSDF did not yield a valid surface")
            if not o3d.io.write_triangle_mesh(str(output / "surface_mesh.ply"), mesh):
                raise IOError("Failed to save the surface mesh")
            full_cloud = o3d.geometry.PointCloud()
            full_cloud.points = mesh.vertices
            full_cloud.normals = mesh.vertex_normals
            full_cloud.colors = mesh.vertex_colors
            if not o3d.io.write_point_cloud(str(output / "surface_full_points.ply"), full_cloud):
                raise IOError("Failed to save all fused surface vertices")
            full_readback = o3d.io.read_point_cloud(str(output / "surface_full_points.ply"))
            full_points = np.asarray(full_readback.points)
            if full_points.shape != vertices.shape or not np.array_equal(full_points, vertices):
                raise ValueError("Full surface vertex cloud readback changed actual geometry")
            o3d.utility.random.seed(0)
            cloud = mesh.sample_points_uniformly(number_of_points=config["surface_sample_count"])
            if not o3d.io.write_point_cloud(str(output / "surface_points.ply"), cloud):
                raise IOError("Failed to save surface samples")
            loaded = o3d.io.read_point_cloud(str(output / "surface_points.ply"))
            points = np.asarray(loaded.points)
            if len(points) != config["surface_sample_count"] or not np.isfinite(points).all():
                raise ValueError("Actual surface sample readback failed")
            state.update(status="completed", elapsed_seconds=time.monotonic() - started,
                vertices=len(vertices), triangles=len(triangles), sampled_points=len(points),
                full_surface_points=len(full_points), full_surface_pointcloud_is_all_mesh_vertices=True,
                minimum=points.min(0).tolist(), maximum=points.max(0).tolist(), calibration=calibration,
                frames_integrated=len(views), points_are_mesh_surface_samples=True,
                largest_component_filter_applied=False, independent_GT_accuracy_measured=False)
            state["render_calibration"] = render_trace
            publish(output / "FRAME_DEPTH_QA.json", {"rows": rows})
            publish(output / "STATE.json", state)
            publish(root / "pgsr/SURFACE_RESULT.json", state)
    except BaseException:
        state.update(status="failed_surface", traceback=traceback.format_exc())
        publish(output / "STATE.json", state)
        publish(root / "pgsr/SURFACE_FAILURE.json", state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "surface"))
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    if args.mode == "train":
        train(args.root)
    else:
        from surface_tiled import surface
        surface(args.root)


if __name__ == "__main__":
    main()
