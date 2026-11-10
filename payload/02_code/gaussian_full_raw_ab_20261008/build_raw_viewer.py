"""Bind fresh models and camera paths to the preserved generic free-flight viewer."""

import argparse
import html
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pycolmap

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
sys.path.insert(0, str(V1))
from runtime import data_identity
from atomic_json_windows import atomic_json, read_json

SOURCE_ROOT = ROOT / "04_outputs/gaussian_pipeline_optimization_v2_20261005/viewer_execution_001/release"
SOURCE = SOURCE_ROOT / "baseline/cases/packed"
CASES = {"box2": "8月20日 Box2", "box1": "8月20日 Box1", "c1": "8月18日绕箱", "c0_july": "7月原始绕箱"}
MODES = {"baseline": "原生基准", "optimized": "优化模型"}


def read(path):
    return read_json(path)


def read_vertices(path):
    with Path(path).open("rb") as stream:
        lines = []
        for _ in range(256):
            line = stream.readline(256)
            if not line.endswith(b"\n"):
                raise ValueError("The bounded float32 PLY header is incomplete")
            lines.append(line.decode("ascii").strip())
            if lines[-1] == "end_header":
                break
        if lines[:2] != ["ply", "format binary_little_endian 1.0"] or lines[-1] != "end_header":
            raise ValueError("Only the fresh native float32 binary Gaussian PLY is admitted")
        elements = [line for line in lines if line.startswith("element ")]
        if len(elements) != 1 or not elements[0].startswith("element vertex "):
            raise ValueError("Unexpected extra PLY elements")
        count = int(elements[0].split()[-1])
        properties = [line.split() for line in lines if line.startswith("property ")]
        if not properties or not all(len(row) == 3 and row[1] == "float" for row in properties):
            raise ValueError("The native Gaussian property contract changed")
        dtype = np.dtype([(row[2], "<f4") for row in properties])
        vertices = np.fromfile(stream, dtype=dtype, count=count)
        if len(vertices) != count or stream.read(1):
            raise ValueError("The complete PLY payload is truncated or has trailing elements")
    return lines, vertices


def focus_from_rays(views):
    centers = np.asarray([row["position"] for row in views])
    directions = np.asarray([row["rotation_c2w"] for row in views])[:, :, 2]
    matrices = np.eye(3)[None]-directions[:, :, None]*directions[:, None, :]
    weights = np.ones(len(centers))
    for _ in range(6):
        system = np.einsum("n,nij->ij", weights, matrices)
        if np.linalg.cond(system) > 1e8:
            raise ValueError("Capture rays do not support a stable automatic focus")
        rhs = np.einsum("n,nij,nj->i", weights, matrices, centers)
        focus = np.linalg.solve(system, rhs)
        errors = np.linalg.norm(np.einsum("nij,nj->ni", matrices, focus-centers), axis=1)
        scale = max(float(np.median(errors)), 1e-8)
        weights = np.minimum(1., 1.5*scale/np.maximum(errors, 1e-8))
    positive = np.einsum("ni,ni->n", focus-centers, directions)
    radius = float(np.median(np.linalg.norm(centers-focus, axis=1)))*.65
    if radius <= 0 or np.mean(positive > 0) < .5:
        raise ValueError("An automatic display focus is not in front of most captured cameras")
    return focus, radius, {"method": "robust_least_squares_camera_optical_ray_intersection",
        "crop_radius_to_median_capture_distance_ratio": .65, "object_GT_or_class_used": False,
        "crop_is_display_only_not_object_segmentation": True, "ray_distance_p50_m": float(np.median(errors))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--case", choices=tuple(CASES), required=True)
    parser.add_argument("--mode", choices=tuple(MODES), required=True)
    args = parser.parse_args()
    side = args.campaign / args.case / args.mode
    trained = read(side / "training/STATE.json")
    if trained["status"] != "completed" or trained["iteration"] != trained["target_iterations"]:
        raise ValueError("Only a completed fresh training model may be staged")
    root = args.campaign / "viewer"
    root.mkdir(exist_ok=True)
    shared = root / "shared-build/vendor"
    if not shared.exists():
        shutil.copytree(SOURCE_ROOT / "shared-build/vendor", shared)
    original_vendor = SOURCE_ROOT / "shared-build/vendor"
    source_vendor_paths = {path.relative_to(original_vendor) for path in original_vendor.rglob("*") if path.is_file()}
    target_vendor_paths = {path.relative_to(shared) for path in shared.rglob("*") if path.is_file()}
    if source_vendor_paths != target_vendor_paths:
        raise ValueError("The complete shared vendor file set differs from the preserved renderer")
    for relative in sorted(source_vendor_paths):
        if (original_vendor / relative).read_bytes() != (shared / relative).read_bytes():
            raise ValueError("The complete shared vendor source differs from the preserved renderer")
    branch = args.case+"-"+args.mode
    branch_root = root / branch
    folder = branch_root / "cases/packed"
    folder.mkdir(parents=True, exist_ok=False)
    files = ("index.html", "viewer.js", "calibrated_camera.js", "lazy_models.js", "motion_resolution.js",
             "packed_splat_asset.js", "persistent_asset_cache.js", "render_filter_contract.js", "render_schedule.js")
    for name in files:
        shutil.copyfile(SOURCE / name, folder / name)
    # The new diagnostic budget preserves all SH orders and Gaussian rows for larger fresh inputs.
    packed = (folder / "packed_splat_asset.js").read_text()
    anchor = "const LIMIT = 64 * 1024 * 1024;"
    if packed.count(anchor) != 1:
        raise ValueError("The preserved packed asset budget anchor changed")
    (folder / "packed_splat_asset.js").write_text(packed.replace(anchor, "const LIMIT = 256 * 1024 * 1024;"))
    admission = read(side / "export/TRAINING_ADMISSION.json")
    source_map = Path(admission["source_model"]).resolve()
    source_map.relative_to(side.resolve())
    full_map = pycolmap.Reconstruction(source_map)
    if len([image for image in full_map.images.values() if image.has_pose]) != admission["full_registered_denominator"]:
        raise ValueError("The admitted full camera denominator changed before staging")
    pinhole = next(iter(pycolmap.Reconstruction(Path(admission["dataset"]) / "sparse/0").cameras.values()))
    if pinhole.model_name != "PINHOLE" or len(pinhole.params) != 4:
        raise ValueError("The fresh display requires a calibrated undistorted sensor")
    fx, fy, cx, cy = map(float, pinhole.params)
    views = []
    for image in sorted((value for value in full_map.images.values() if value.has_pose),
                        key=lambda value: int(value.name.split("_")[2].split(".")[0])):
        pose = image.cam_from_world().inverse()
        views.append({"image_name": image.name, "position": pose.translation.tolist(),
            "rotation_c2w": pose.rotation.matrix().tolist(), "width": int(pinhole.width), "height": int(pinhole.height),
            "fx": fx, "fy": fy, "cx": cx, "cy": cy})
    focus, radius, focus_detail = focus_from_rays(views)
    model_path = side / f"training/model/point_cloud/iteration_{trained['iteration']}/point_cloud.ply"
    header, vertices = read_vertices(model_path)
    if len(vertices) != trained["gaussian_count"] or len([name for name in vertices.dtype.names if name.startswith("f_rest_")]) != 45:
        raise ValueError("The full Gaussian population or SH3 coefficients changed")
    if not all(np.isfinite(vertices[name]).all() for name in vertices.dtype.names):
        raise ValueError("Nonfinite attributes cannot be displayed")
    xyz = np.column_stack([vertices[name] for name in ("x", "y", "z")])
    keep = np.linalg.norm(xyz-focus, axis=1) <= radius
    if not keep.any():
        raise ValueError("The generic automatic focus crop has no Gaussian support; full source remains untouched")
    shutil.copyfile(model_path, folder / "scene.ply")
    cropped = vertices[keep].copy()
    cropped_header = ["element vertex "+str(len(cropped)) if line.startswith("element vertex ") else line for line in header]
    with (folder / "box.ply").open("wb") as stream:
        stream.write(("\n".join(cropped_header)+"\n").encode("ascii"))
        cropped.tofile(stream)
    if read_vertices(folder / "box.ply")[1].tobytes() != cropped.tobytes():
        raise ValueError("The complete native cropped Gaussian attribute rows changed")
    atomic_json(folder / "camera_path.json", views)
    title = CASES[args.case]+" · "+MODES[args.mode]
    metadata = {"case": args.case, "mode": args.mode, "label": title, "training_image_count": admission["training_count"],
        "full_calibrated_camera_count": len(views), "default_view": len(views)//4,
        "focus": focus.tolist(), "roi_minimum": (focus-radius).tolist(), "roi_maximum": (focus+radius).tolist(),
        "full_gaussian_count": len(vertices), "box_gaussian_count": int(keep.sum()), "focus_derivation": focus_detail,
        "source_model": data_identity(model_path), "fresh_camera_map": str(source_map),
        "full_scene_unfiltered_for_display": True, "metric_coordinate_unit": "coarse_sensor_motion_aligned_meters",
        "independent_GT_dimensions_unverified": True, "old_camera_path_or_ROI_used": False}
    atomic_json(folder / "display_manifest.json", metadata)
    assets = {}
    vendor = data_identity(shared / "spark.module.js")
    for label, count in (("scene", len(vertices)), ("box", int(keep.sum()))):
        identity = data_identity(folder / (label+".ply"))
        assets[label] = {"spark_sha256": vendor["sha256"], "source_ply_sha256": identity["sha256"],
                         "source_bytes": identity["bytes"], "numSplats": count}
    atomic_json(branch_root / "ASSET_SOURCES.json", assets)
    atomic_json(folder / "render_filter.json", {"schema": 1, "rasterizeMode": "classic", "eps2d": .3,
        "opacityCompensation": False, "antialiasing": False, "trainingConfigSha256": data_identity(side / "training/CONFIG.json")["sha256"],
        "models": {key: value["source_ply_sha256"] for key, value in assets.items()}})
    page = (folder / "index.html").read_text(encoding="utf-8")
    page = page.replace("Box2 · 自由飞行三维预览", title+" · 自由飞行预览").replace("可自由移动的 Box2 高斯三维场景", "可自由移动的高斯三维场景")
    page = page.replace("新地图 · 优化组合 · 1,016 帧", title+" · "+str(admission["training_count"])+" 帧")
    page = page.replace("箱体范围", "自动聚焦范围").replace("箱顶和远离采集路径的位置", "遮挡处和远离采集路径的位置")
    start = page.index('<select id="case-select">')
    end = page.index("</select>", start)+len("</select>")
    options = "".join('<option value="'+case+"-"+mode+'"'+(" selected" if case+"-"+mode == branch else "")+'>'
                      +html.escape(label+" · "+MODES[mode])+"</option>" for case, label in CASES.items() for mode in MODES)
    page = page[:start]+'<select id="case-select">'+options+'</select>'+page[end:]
    (folder / "index.html").write_text(page, encoding="utf-8")
    code = (folder / "viewer.js").read_text(encoding="utf-8").replace("箱体范围", "自动聚焦范围")
    code += '\nfetch("../../../available.json").then(reply => reply.json()).then(ready => { for (const option of document.getElementById("case-select").options) option.disabled = !ready.branches.includes(option.value); }).catch(showError);\n'
    (folder / "viewer.js").write_text(code, encoding="utf-8")
    shutil.copyfile(Path(__file__).with_name("viewer_validation.js"), folder / "viewer_validation.js")
    (folder / "index.html").write_text(page.replace('</body>', '<script type="module" src="./viewer_validation.js"></script></body>'), encoding="utf-8")
    shutil.copyfile(Path(__file__).with_name("pack_export.js"), folder / "pack_export.js")
    (folder / "export.html").write_text('''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><link rel="icon" href="data:,">
<title>本轮显示资产准备</title><script type="importmap">{"imports":{"three":"../../../shared-build/vendor/three.module.js","three/addons/":"../../../shared-build/vendor/","@sparkjsdev/spark":"../../../shared-build/vendor/spark.module.js"}}</script>
</head><body><button id="pack">生成两份显示资产</button><pre id="status" role="status">等待生成</pre><script>addEventListener("error", event => { document.getElementById("status").textContent = event.message; }); addEventListener("unhandledrejection", event => { document.getElementById("status").textContent = String(event.reason); });</script><script type="module" src="./pack_export.js"></script></body></html>''', encoding="utf-8")
    atomic_json(branch_root / "STAGING_RECEIPT.json", {"status": "fresh_models_staged_native_packing_and_browser_verification_pending",
        "source": data_identity(Path(__file__)), "sources": [data_identity(SOURCE / name) for name in files],
        "assets": assets, "metadata": metadata, "shared_vendor_complete_bytes_equal": True, "packed_asset_budget_bytes": 256*1024**2})
    print(json.dumps({"branch": branch, "full_gaussians": len(vertices), "focus_gaussians": int(keep.sum()), "capture_cameras": len(views)}))


if __name__ == "__main__":
    main()
