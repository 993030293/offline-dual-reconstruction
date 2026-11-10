"""Package actual dual outputs, common cameras, and both independently movable viewers."""

import argparse
import hashlib
import html
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import numpy as np
from PIL import Image
from ply_io import read_ply
import pycolmap

from common import HERE, OLD, ROOT, SFM_PYTHON, DISTRO, atomic_json, read_json
from start_viewer import start


def identity(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(block)
    return {"bytes": Path(path).stat().st_size, "sha256": digest.hexdigest()}


def validate_cloud(path, expected, memory_map=False):
    vertices, faces = read_ply(path, memory_map=memory_map)
    if faces is not None:
        raise ValueError("A cloud output unexpectedly contains triangles")
    names = vertices.dtype.names
    if len(vertices) != expected:
        raise ValueError("Actual point payload has another denominator")
    for start in range(0, len(vertices), 65536):
        for name in names:
            if np.issubdtype(vertices.dtype[name], np.number) and not np.isfinite(vertices[name][start:start + 65536]).all():
                raise ValueError("Nonfinite point property: " + name)
    return vertices


def browser_dependencies():
    node_override, package_override = os.environ.get("RECON_NODE_EXECUTABLE"), os.environ.get("RECON_PLAYWRIGHT_PACKAGE")
    if bool(node_override) != bool(package_override):
        raise ValueError("Both explicit browser runtime overrides must be supplied together")
    if node_override:
        node, package = Path(node_override), Path(package_override)
    else:
        binding = read_json(HERE / "OFFLINE_RUNTIME.local.json")
        node, package = Path(binding["node"]), Path(binding["playwright_package"])
    if not node.is_file() or not (package / "index.js").is_file():
        raise FileNotFoundError("The explicit installed offline browser runtime is unavailable; downloads are forbidden")
    return node, package


def all_scene_diagnostic_main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    configuration = read_json(root / "CONFIG.json")
    shared = read_json(root / "shared/RESULT.json")
    side = root / configuration["case"] / "optimized"
    gaussian = read_json(side / "training/STATE.json")
    pgsr = read_json(root / "pgsr/TRAINING_RESULT.json")
    surface = read_json(root / "pgsr/SURFACE_RESULT.json")
    diagnostic = read_json(root / "gaussian_diagnostics/RESULT.json")
    if gaussian["status"] != "completed" or gaussian["iteration"] != configuration["gaussian_iterations"]:
        raise ValueError("The Gaussian branch did not complete")
    if any(value["status"] != "completed" for value in (pgsr, surface)):
        raise ValueError("The PGSR training and fused surface must both complete")
    if pgsr["registered_frames"] != shared["registered_frames"] or surface["frames_integrated"] != shared["registered_frames"]:
        raise ValueError("The two branches used different admitted camera sets")
    gaussian_path = side / f"training/model/point_cloud/iteration_{gaussian['iteration']}/point_cloud.ply"
    splats = validate_cloud(gaussian_path, gaussian["gaussian_count"])
    if len([name for name in splats.dtype.names if name.startswith("f_rest_")]) != 45:
        raise ValueError("The complete SH3 Gaussian representation changed")
    points_path, mesh_path = root / "pgsr/surface/surface_points.ply", root / "pgsr/surface/surface_mesh.ply"
    full_points_path = root / "pgsr/surface/surface_full_points.ply"
    full_points = validate_cloud(full_points_path, surface["full_surface_points"], memory_map=True)
    points = validate_cloud(points_path, surface["sampled_points"])
    mesh_vertices, mesh_faces = read_ply(mesh_path, memory_map=True)
    if len(mesh_vertices) != surface["vertices"] or mesh_faces is None or len(mesh_faces) != surface["triangles"]:
        raise ValueError("The fused mesh readback changed")
    for first_vertex in range(0, len(full_points), 65536):
        if not np.array_equal(full_points[first_vertex:first_vertex + 65536], mesh_vertices[first_vertex:first_vertex + 65536]):
            raise ValueError("The complete surface cloud differs from actual mesh vertices")
    if not surface["points_are_mesh_surface_samples"]:
        raise ValueError("A Gaussian-center cloud is not a PGSR surface deliverable")
    delivery = root / "delivery"
    delivery.mkdir(exist_ok=False)
    for source, name in ((gaussian_path, "gaussian_scene.ply"), (points_path, "pgsr_surface_points.ply"),
                         (full_points_path, "pgsr_surface_full_points.ply"),
                         (mesh_path, "pgsr_surface_mesh.ply")):
        shutil.copyfile(source, delivery / name)
        if identity(source) != identity(delivery / name):
            raise ValueError("A delivered binary changed during copying")
    model = pycolmap.Reconstruction(root / "shared/rectified/sparse")
    views = []
    for image in sorted(model.images.values(), key=lambda value: int(value.name.split("_")[2].split(".")[0])):
        camera = model.cameras[image.camera_id]
        pose = image.cam_from_world().inverse()
        fx, fy, cx, cy = map(float, camera.params)
        views.append({"image_name": image.name, "position": pose.translation.tolist(),
            "rotation_c2w": pose.rotation.matrix().tolist(), "width": camera.width, "height": camera.height,
            "raw_header_timestamp_ns": int(image.name.split("_")[2].split(".")[0]),
            "fx": fx, "fy": fy, "cx": cx, "cy": cy})
    if len(views) != shared["registered_frames"] or sorted(view["image_name"] for view in views) != shared["registered_names"]:
        raise ValueError("Delivered cameras changed the common registration")
    atomic_json(delivery / "cameras.json", {"pixel_domain": "COLMAP_corner_first_center_0.5",
        "units": shared["world_units"], "views": views, "intrinsics": shared["cameras"]})
    frontend = read_json(side / "PIPELINE_STATE.json")
    pure_similarity = frontend["metric_selected"] == "pure_sim3_alignment"
    if pure_similarity:
        alignment = read_json(side / "metric_sim3_renderer_recovery/CONTRACT.json")["new_alignment"]
    else:
        metric = read_json(side / "metric/RESULT.json")
        if metric["status"] != "metric_interface_verified" or metric["selected"] != frontend["metric_selected"]:
            raise ValueError("The admitted metric branch changed before scale packaging")
        alignment = read_json(side / "metric/ALIGNMENT.json")
    atomic_json(delivery / "scale.json", {"status": "metric_interface_from_recorded_coarse_motion",
        "source": alignment["source"], "initial_similarity_alignment": alignment["new_Sim3"],
        "camera_position_fit_residuals_m": alignment["residuals_m"],
        "raw_odometry_producer_confirmed": False, "original_metric_BA_passed": not pure_similarity,
        "admitted_metric_branch": frontend["metric_selected"],
        "independent_GT_scale_or_object_dimension_accuracy_verified": False,
        "fit_residuals_are_not_geometry_accuracy": True,
        "common_world_point_component_error": shared["maximum_point_component_error"],
        "common_world_camera_component_error": shared["maximum_pose_component_error"]})
    with (root / "viewer_staging.log").open("x", encoding="utf-8") as stream:
        subprocess.run([str(SFM_PYTHON), str(HERE / "build_viewer.py"), "--campaign", str(root),
            "--case", configuration["case"], "--mode", "optimized"], stdout=stream, stderr=subprocess.STDOUT,
            check=True, timeout=240)
    viewer = root / "viewer"
    branch = configuration["case"] + "-optimized"
    folder = viewer / branch / "cases/packed"
    atomic_json(folder / "camera_path.json", views)
    metadata = read_json(folder / "display_manifest.json")
    metadata["shared_rectified_camera_input"] = str(root / "shared/RESULT.json")
    atomic_json(folder / "display_manifest.json", metadata)
    point_folder = viewer / "surface"
    point_folder.mkdir()
    focus = np.asarray(metadata["focus"])
    radius = float(np.asarray(metadata["roi_maximum"])[0] - focus[0])
    packed = np.empty((len(points), 6), dtype="<f4")
    for axis, name in enumerate(("x", "y", "z")):
        packed[:, axis] = points[name]
    for axis, name in enumerate(("red", "green", "blue")):
        packed[:, axis + 3] = points[name] / 255.
    if not np.isfinite(packed).all():
        raise ValueError("The point viewer payload is nonfinite")
    packed.tofile(point_folder / "points.bin")
    if np.fromfile(point_folder / "points.bin", dtype="<f4").tobytes() != packed.tobytes():
        raise ValueError("The complete point preview changed bytes")
    atomic_json(point_folder / "manifest.json", {"points": len(points), "focus": focus.tolist(),
        "focusRadius": radius, "defaultView": len(views)//4, "world_coordinates_changed": False,
        "source_cloud": identity(points_path), "point_asset": identity(point_folder / "points.bin"),
        "binary_layout": "little_endian_float32_interleaved_xyz_rgb_01", "display_crop_only": True})
    atomic_json(point_folder / "camera_path.json", views)
    shutil.copyfile(HERE / "point_viewer.html", point_folder / "index.html")
    shutil.copyfile(HERE / "point_viewer.js", point_folder / "point_viewer.js")
    shutil.copyfile(folder / "calibrated_camera.js", point_folder / "calibrated_camera.js")
    page = (point_folder / "index.html").read_text(encoding="utf-8")
    (point_folder / "index.html").write_text(page.replace("box1-optimized", branch), encoding="utf-8")
    identity_record = {"run_root": str(root), "case": configuration["case"], "id": str(uuid.uuid4())}
    atomic_json(viewer / "DELIVERY_IDENTITY.json", identity_record)
    label = html.escape(configuration["case"])
    landing = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>双输出重建交付</title><style>body{font:17px system-ui;background:#0f1824;color:#eaf3ff;margin:8vh auto;max-width:800px;padding:24px;line-height:1.8}a{color:#92d2ff}li{margin:20px 0}</style><h1>'+label+' · 双输出重建</h1><p>本轮从原始录制重新解码与重建，两路共用 '+str(len(views))+' 个相机和同一米制坐标。</p><ul><li><a href="./'+branch+'/cases/packed/index.html">打开 3D 高斯自由飞行预览</a> · '+str(len(splats))+' 个高斯</li><li><a href="./surface/index.html">打开 PGSR 表面点云自由移动预览</a> · '+str(len(points))+' 个表面采样点</li></ul><p>支持环绕、平移、缩放、自由飞行和采集路径。自动聚焦只裁剪显示；完整模型保留。尺度来自录制粗位姿对齐，尚未通过独立 GT 验证。点数和 TSDF 体素大小不代表测量精度。</p></html>'
    (viewer / "index.html").write_text(landing, encoding="utf-8")
    server = start(root)
    node, package = browser_dependencies()
    packing = root / "native_packing"
    packing.mkdir()
    with (packing / "run.log").open("x", encoding="utf-8") as stream:
        subprocess.run([str(node), str(HERE / "pack_preview.cjs"),
            str(package), server["url"] + branch + "/cases/packed/export.html", str(packing)],
            stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=360,
            creationflags=subprocess.CREATE_NO_WINDOW)
    packing_result = read_json(packing / "RESULT.json")
    if packing_result["status"] != "native_assets_and_viewer_ready":
        raise ValueError("The complete native Gaussian preview did not load")
    behavior = root / "browser_review"
    behavior.mkdir()
    with (behavior / "run.log").open("x", encoding="utf-8") as stream:
        subprocess.run([str(node), str(HERE / "review_viewers.cjs"), str(package), server["url"], str(behavior), branch],
            stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=300,
            creationflags=subprocess.CREATE_NO_WINDOW)
    behavior_result = read_json(behavior / "RESULT.json")
    if behavior_result["status"] != "both_viewers_actual_interactions_verified":
        raise ValueError("Both actual interactive viewers must pass")
    pgsr_diagnostics = []
    for row in diagnostic["rows"]:
        index = row["index"]
        rendered = np.asarray(Image.open(root / f"pgsr/surface/render_{index:04d}.png"), dtype=np.float64)/255
        reference = np.asarray(Image.open(root / f"pgsr/surface/reference_{index:04d}.png"), dtype=np.float64)/255
        if rendered.shape != reference.shape or rendered.shape[:2] != (row["height"], row["width"]):
            raise ValueError("The two branches rendered different calibrated dimensions")
        mse = float(np.mean((rendered-reference)**2))
        pgsr_diagnostics.append({"index": index, "training_view_MAE": float(np.mean(np.abs(rendered-reference))),
            "training_view_PSNR_dB": float(-10*np.log10(max(mse,1e-12))), "pixel_standard_deviation": float(rendered.std())})
    result = {"status": "both_delivered_binaries_and_actual_viewers_verified", "raw_input": configuration["raw"],
        "historical_derived_inputs_used": False, "shared_camera_count": len(views), "gaussian_count": len(splats),
        "pgsr_parameter_gaussian_count": pgsr["gaussian_count"], "surface_points": len(points),
        "full_surface_points": len(full_points), "preview_sample_count_is_not_full_geometry_denominator": True,
        "mesh_vertices": surface["vertices"], "mesh_triangles": surface["triangles"],
        "files": {path.name: identity(path) for path in delivery.glob("*.ply")}, "viewer": server,
        "binary_pointcloud_is_PGSR_depth_TSDF_surface": True, "independent_GT_accuracy_verified": False,
        "units": shared["world_units"], "training_view_diagnostics_not_heldout": {"gaussian": diagnostic["rows"], "pgsr": pgsr_diagnostics},
        "Gaussian_native_asset_arrays_verified": True, "point_asset_complete_bytes_verified": True,
        "actual_browser_behavior": str(behavior / "RESULT.json"),
        "browser_dependency": {"node": str(node), "playwright": str(package)}}
    atomic_json(root / "DELIVERY_RESULT.json", result)
    licenses = delivery / "licenses"
    licenses.mkdir()
    shutil.copyfile(ROOT / "04_outputs/box2_3dgs_display_v1_20260930/source/gaussian-splatting/LICENSE.md",
                    licenses / "INRIA_3DGS_LICENSE.md")
    license_text = subprocess.check_output(["wsl.exe", "-d", DISTRO, "--exec", "/bin/cat",
                                          "/opt/e26_thin_surface_v1/pgsr/LICENSE.md"], text=True, encoding="utf-8")
    (licenses / "PGSR_LICENSE.md").write_text(license_text, encoding="utf-8")
    for name in ("SPARK-LICENSE.txt", "THREE-LICENSE.txt"):
        shutil.copyfile(viewer / "shared-build/vendor" / name, licenses / name)
    atomic_json(delivery / "THIRD_PARTY.json", {"sources": {"COLMAP": "COLMAP 4.1.1 installed native runtime",
        "INRIA_3DGS": "54c035f7834b564019656c3e3fcc3646292f727d", "PGSR": pgsr["source_commit"],
        "gsplat": "installed isolated p6 build", "Open3D": "installed PGSR WSL runtime"},
        "commercial_distribution_clearance_verified": False, "license_inventory_exhaustive": False,
        "PGSR_license_has_research_nonprofit_and_noncommercial_terms": True})
    restart = "param([int]$Port = 0)\n$taskViewerArguments = @('--run-root', '"+str(root).replace("'", "''")+"')\n"
    restart += "if ($Port -gt 0) { $taskViewerArguments += @('--port', [string]$Port) }\n"
    restart += "& '"+str(SFM_PYTHON).replace("'", "''")+"' '"+str(HERE / "start_viewer.py").replace("'", "''")+"' @taskViewerArguments\n"
    (root / "START_VIEWER.ps1").write_text(restart, encoding="utf-8")
    print(result["status"], server["url"], flush=True)


def main():
    parser = argparse.ArgumentParser(description='Deliver the established target surface and common Gaussian branch')
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--target-root', type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve(strict=True)
    target = args.target_root.resolve(strict=True) if args.target_root else root / 'target'
    if not (target / '09_ground_target_m3/TARGET_SUPPORT_AUDIT.json').is_file():
        raise ValueError('The rejected all-scene surface cannot substitute for the required target branch')
    from package_target import package
    package(root, target)


if __name__ == "__main__":
    main()
