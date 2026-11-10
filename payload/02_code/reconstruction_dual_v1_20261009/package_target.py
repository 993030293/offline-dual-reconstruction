"""Deliver the genuine ground-removed target surface and its existing common 3DGS."""

import argparse
import html
from pathlib import Path
import shutil
import subprocess
import uuid

import numpy as np
import pycolmap

from common import ROOT, HERE, SFM_PYTHON, DISTRO, atomic_json, read_json
from package_delivery import browser_dependencies, identity, validate_cloud
from ply_io import read_ply
from start_viewer import start
from viewer_review import run_review
from native_packing import run_packing


def package(root, target):
    config, shared = read_json(root / 'CONFIG.json'), read_json(root / 'shared/RESULT.json')
    m3 = target / '09_ground_target_m3'
    extraction = read_json(m3 / 'EXTRACTION_RECEIPT.json')
    audit = read_json(m3 / 'FINAL_AUDIT.json')
    support = read_json(m3 / 'TARGET_SUPPORT_AUDIT.json')
    tsdf = read_json(Path(extraction['input']['tsdf_receipt']))
    if (extraction['status'] != 'complete_ground_removed_target_roi_observed_surface'
            or audit['status'] != 'pass' or support['status'] != 'pass'):
        raise ValueError('The point deliverable requires real M3 and target support acceptance')
    if tsdf['frame_count'] != shared['registered_frames']:
        raise ValueError('The target branch and Gaussian branch used different camera denominators')
    output = extraction['output']
    cloud_path, mesh_path = Path(output['point_cloud']), Path(output['mesh'])
    points = validate_cloud(cloud_path, output['point_count'])
    vertices, faces = read_ply(mesh_path, memory_map=True)
    if len(vertices) != output['mesh_vertices'] or faces is None or len(faces) != output['mesh_triangles']:
        raise ValueError('The actual target mesh changed')
    for name in vertices.dtype.names:
        if not np.isfinite(vertices[name]).all():
            raise ValueError('A target mesh property is nonfinite')
    side = root / config['case'] / 'optimized'
    gaussian = read_json(side / 'training/STATE.json')
    gaussian_path = side / f"training/model/point_cloud/iteration_{gaussian['iteration']}/point_cloud.ply"
    if gaussian['status'] != 'completed' or gaussian['iteration'] != config['gaussian_iterations']:
        raise ValueError('The common Gaussian branch is incomplete')
    splats = validate_cloud(gaussian_path, gaussian['gaussian_count'], memory_map=True)
    pgsr = read_json(root / 'pgsr/TRAINING_RESULT.json')
    if pgsr['status'] != 'completed' or pgsr['registered_frames'] != shared['registered_frames']:
        raise ValueError('The common PGSR model is incomplete')
    delivery = target / 'delivery'
    delivery.mkdir(exist_ok=False)
    files = {}
    for source, name in ((gaussian_path, 'gaussian_scene.ply'), (cloud_path, 'pgsr_target_points.ply'),
                         (mesh_path, 'pgsr_target_surface_mesh.ply')):
        destination = delivery / name
        shutil.copyfile(source, destination)
        files[name] = identity(destination)
        if files[name] != identity(source):
            raise ValueError('A delivered binary changed in transport')
    model = pycolmap.Reconstruction(root / 'shared/rectified/sparse')
    views = []
    for image in sorted(model.images.values(), key=lambda row: int(Path(row.name).stem.rsplit('_', 1)[1])):
        camera, pose = model.cameras[image.camera_id], image.cam_from_world().inverse()
        fx, fy, cx, cy = map(float, camera.params)
        views.append({'image_name': image.name, 'position': pose.translation.tolist(),
            'rotation_c2w': pose.rotation.matrix().tolist(), 'width': camera.width, 'height': camera.height,
            'raw_header_timestamp_ns': int(Path(image.name).stem.rsplit('_', 1)[1]),
            'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy})
    if sorted(row['image_name'] for row in views) != shared['registered_names']:
        raise ValueError('Delivered cameras differ from the shared geometric input')
    atomic_json(delivery / 'cameras.json', {'views': views, 'pixel_domain': 'COLMAP_corner_first_center_0.5',
        'units': shared['world_units'], 'intrinsics': shared['cameras']})
    frontend = read_json(side / 'PIPELINE_STATE.json')
    pure = frontend['metric_selected'] == 'pure_sim3_alignment'
    alignment = (read_json(side / 'metric_sim3_renderer_recovery/CONTRACT.json')['new_alignment']
                 if pure else read_json(side / 'metric/ALIGNMENT.json'))
    atomic_json(delivery / 'scale.json', {'status': 'coarse_odometry_aligned_metric_interface',
        'initial_similarity_alignment': alignment['new_Sim3'], 'camera_fit_residuals_m': alignment['residuals_m'],
        'raw_odometry_producer_confirmed': False, 'original_metric_BA_passed': not pure,
        'independent_GT_scale_or_object_dimension_accuracy_verified': False})
    viewer = root / 'viewer'
    if not viewer.is_dir():
        subprocess.run([str(SFM_PYTHON), str(HERE / 'build_viewer.py'), '--campaign', str(root),
            '--case', config['case'], '--mode', 'optimized'], check=True, timeout=300,
            creationflags=subprocess.CREATE_NO_WINDOW)
    branch = config['case'] + '-optimized'
    gaussian_folder = viewer / branch / 'cases/packed'
    atomic_json(gaussian_folder / 'camera_path.json', views)
    metadata_path = gaussian_folder / 'display_manifest.json'
    metadata = read_json(metadata_path)
    metadata['source_full_map_calibrated_camera_count'] = metadata.get(
        'source_full_map_calibrated_camera_count', metadata['full_calibrated_camera_count'])
    metadata['full_calibrated_camera_count'] = len(views)
    metadata['default_view'] = len(views) // 4
    metadata['shared_rectified_camera_input'] = str(root / 'shared/RESULT.json')
    metadata['camera_path_uses_all_admitted_training_frames'] = True
    atomic_json(metadata_path, metadata)
    adaptive = bool(tsdf.get('adaptive_sensor_contract_completed', False))
    strict = bool(tsdf['strict_ReliableLidar_profile_completed'])
    diagnostic = tsdf['profile'] == 'VisualTargetDiagnostic'
    if not (adaptive or strict or diagnostic):
        raise ValueError('The surface has no explicit completed evidence profile')
    point_name = 'target_adaptive' if adaptive else 'target'
    point_folder = viewer / point_name
    point_folder.mkdir(exist_ok=False)
    packed = np.empty((len(points), 6), '<f4')
    for axis, name in enumerate(('x', 'y', 'z')):
        packed[:, axis] = points[name]
    for axis, name in enumerate(('red', 'green', 'blue')):
        packed[:, axis + 3] = points[name] / 255.
    if not np.isfinite(packed).all() or (packed[:, 3:] < 0).any() or (packed[:, 3:] > 1).any():
        raise ValueError('The target preview payload is invalid')
    packed.tofile(point_folder / 'points.bin')
    if not np.array_equal(np.fromfile(point_folder / 'points.bin', dtype='<f4').reshape(-1, 6), packed):
        raise ValueError('The target point preview changed')
    lower, upper = packed[:, :3].min(0), packed[:, :3].max(0)
    focus = (lower.astype(np.float64) + upper) / 2
    radius = float(np.linalg.norm(upper - lower) / 2 * 1.01)
    atomic_json(point_folder / 'manifest.json', {'points': len(points), 'focus': focus.tolist(),
        'focusRadius': radius, 'defaultView': len(views) // 4, 'defaultNavigation': 'overview', 'world_coordinates_changed': False,
        'geometry_scope': 'ground_removed_target_roi', 'display_crop_only': False,
        'source_cloud': identity(cloud_path), 'point_asset': identity(point_folder / 'points.bin'),
        'binary_layout': 'little_endian_float32_interleaved_xyz_rgb_01',
        'strict_lidar_profile_completed': strict, 'adaptive_sensor_contract_completed': adaptive,
        'diagnostic_visual_target': diagnostic})
    atomic_json(point_folder / 'camera_path.json', views)
    shutil.copyfile(HERE / 'point_viewer.js', point_folder / 'point_viewer.js')
    shutil.copyfile(gaussian_folder / 'calibrated_camera.js', point_folder / 'calibrated_camera.js')
    page = (HERE / 'point_viewer.html').read_text(encoding='utf-8')
    page = page.replace('box1-optimized', branch).replace('PGSR 表面点云', 'PGSR 目标表面 · 已去地面')
    page = page.replace('完整表面', '全部目标表面').replace('自动聚焦', '目标聚焦')
    page = page.replace('<a href="../">双输出交付</a>', '<a href="../">双输出交付</a><a href="./reference_style.html">原版点云预览</a>')
    if diagnostic:
        page = page.replace('坐标单位 m；', '视觉目标诊断；严格雷达支撑未通过。坐标单位 m；')
    elif adaptive:
        page = page.replace('坐标单位 m；', '新选帧保留；有效观测参与雷达约束。坐标单位 m；')
    (point_folder / 'index.html').write_text(page, encoding='utf-8')
    shutil.copyfile(m3 / 'ground_removed_target_preview.html', point_folder / 'reference_style.html')
    if not (viewer / 'DELIVERY_IDENTITY.json').exists():
        atomic_json(viewer / 'DELIVERY_IDENTITY.json', {'run_root': str(root), 'case': config['case'], 'id': str(uuid.uuid4())})
    licenses = delivery / 'licenses'
    licenses.mkdir()
    shutil.copyfile(ROOT / '04_outputs/box2_3dgs_display_v1_20260930/source/gaussian-splatting/LICENSE.md',
        licenses / 'INRIA_3DGS_LICENSE.md')
    license_text = subprocess.check_output(['wsl.exe', '-d', DISTRO, '--exec', '/bin/cat',
        '/opt/e26_thin_surface_v1/pgsr/LICENSE.md'], text=True, encoding='utf-8')
    (licenses / 'PGSR_LICENSE.md').write_text(license_text, encoding='utf-8')
    for name in ('SPARK-LICENSE.txt', 'THREE-LICENSE.txt'):
        shutil.copyfile(viewer / 'shared-build/vendor' / name, licenses / name)
    atomic_json(delivery / 'THIRD_PARTY.json', {'COLMAP': 'installed COLMAP 4.1.1',
        'PGSR': pgsr['source_commit'], 'INRIA_3DGS_gsplat_Open3D_Spark_Three': 'original authors',
        'original_target_algorithms': 'installed local v1r23 package',
        'license_inventory_exhaustive': False, 'commercial_distribution_clearance_verified': False,
        'PGSR_license_has_research_nonprofit_and_noncommercial_terms': True})
    landing_old = viewer / 'index.html'
    if landing_old.exists():
        shutil.copyfile(landing_old, target / 'PRIOR_VIEWER_LANDING.html')
    label = html.escape(config['case'])
    note = ('本次为显式视觉目标诊断，严格每帧雷达覆盖路线未验收。' if diagnostic else
        '保留新选帧集合；雷达只使用通过时间和稳定支持检查的真实观测。三候选、M3 去地面和支持检查已运行；旧每帧覆盖契约不宣称通过。'
        if adaptive else '数值目标区域、雷达支撑、三候选选择、M3 去地面及目标支持检查已经运行。')
    landing = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>目标表面双输出</title>'
        '<style>body{font:17px system-ui;background:#0f1824;color:#eaf3ff;margin:8vh auto;max-width:850px;padding:24px;line-height:1.8}a{color:#92d2ff}li{margin:20px 0}</style>'
        '<h1>' + label + ' · 高斯与目标表面</h1><p>两路共用 ' + str(len(views)) + ' 个相机及原有米制坐标。</p>'
        '<ul><li><a href="./' + point_name + '/index.html">打开去地面的 PGSR 目标点云 · 自由飞行</a> · '
        + str(len(points)) + ' 个目标表面采样点</li><li><a href="./' + point_name + '/reference_style.html">打开与旧流程相同的点云预览</a></li>'
        '<li><a href="./' + branch + '/cases/packed/index.html">打开 3D 高斯预览</a></li></ul><p>' + note + '</p>'
        '<p>点云来自本轮 PGSR 观测表面及目标提取。点数与体素尺寸不代表测量精度；没有独立几何 GT。</p></html>')
    landing_old.write_text(landing, encoding='utf-8')
    server = start(root)
    node, playwright = browser_dependencies()
    if not (root / 'native_packing/RESULT.json').exists():
        packing = root / 'native_packing'
        packing.mkdir(exist_ok=False)
        run_packing(node, playwright, server['url'] + branch + '/cases/packed/export.html', packing, root)
    if read_json(root / 'native_packing/RESULT.json')['status'] != 'native_assets_and_viewer_ready':
        raise ValueError('The Gaussian preview asset was not verified')
    behavior = target / 'browser_review'
    behavior.mkdir()
    interaction = run_review(node, playwright, server['url'], behavior, branch, point_name + '/index.html')
    if interaction['status'] != 'both_viewers_actual_interactions_verified':
        raise ValueError('The actual target and Gaussian interactions failed')
    result = {'status': ('visual_target_diagnostic_and_dual_viewers_verified' if diagnostic else
            'adaptive_sensor_dual_target_outputs_and_viewers_verified' if adaptive else 'dual_target_outputs_and_viewers_verified'),
        'run_root': str(root), 'target_root': str(target), 'profile': tsdf['profile'],
        'point_branch_scope': 'target_roi_after_M3_ground_removal_and_component_filter',
        'strict_ReliableLidar_profile_completed': strict, 'adaptive_sensor_contract_completed': adaptive,
        'shared_camera_count': len(views), 'gaussian_count': len(splats),
        'PGSR_parameter_gaussian_count': pgsr['gaussian_count'], 'surface_points': len(points),
        'mesh_vertices': len(vertices), 'mesh_triangles': len(faces), 'files': files,
        'delivery_directory': str(delivery), 'viewer': server,
        'target_url': server['url'] + point_name + '/index.html',
        'reference_style_url': server['url'] + point_name + '/reference_style.html',
        'actual_browser_behavior': str(behavior / 'RESULT.json'), 'target_support_audit': str(m3 / 'TARGET_SUPPORT_AUDIT.json'),
        'independent_geometry_accuracy_verified': False, 'new_unassisted_raw_to_outputs_run_verified': False,
        'runtime_language_model_used': False, 'prior_all_scene_point_delivery_accepted': False}
    atomic_json(target / 'DELIVERY_RESULT.json', result)
    print(result['status'], result['target_url'], flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--target-root', required=True, type=Path)
    args = parser.parse_args()
    package(args.run_root.resolve(strict=True), args.target_root.resolve(strict=True))
