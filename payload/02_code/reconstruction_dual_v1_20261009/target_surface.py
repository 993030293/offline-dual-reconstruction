"""Restore the established three-candidate target surface after calibrated PGSR."""

import argparse
from collections import OrderedDict
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import psutil

from pgsr_branch import prepare, publish, SOURCE
from pgsr_calibration import calibrated_cameras, calibrated_render
from surface_candidate_policy import select_surface_candidate
from surface_roi_policy import trim_observed_surface

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / '05_packages/e26_video_to_target_m3_nollm_v1r23_20260927_installed_candidate'
sys.path.insert(0, str(PACKAGE / 'code'))
import run_reliable_lidar_tsdf as reference


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def native_path(path):
    value = str(path).replace('\\', '/')
    if len(value) > 2 and value[1:3] == ':/':
        return Path('/mnt/' + value[0].lower() + value[2:])
    return Path(value)


def canonical_path(path):
    value = str(path)
    if value.startswith('/mnt/') and len(value) > 7 and value[6] == '/':
        return value[5].upper() + ':' + value[6:]
    return value


class FrameCache:
    def __init__(self, directory, maximum=16):
        self.directory, self.maximum, self.entries = directory, maximum, OrderedDict()

    def get(self, index):
        if index in self.entries:
            self.entries.move_to_end(index)
            return self.entries[index]
        with np.load(self.directory / f'{index:05d}.npz') as archive:
            frame = {key: archive[key] for key in ('depth', 'base', 'broad', 'color')}
        self.entries[index] = frame
        while len(self.entries) > self.maximum:
            self.entries.popitem(last=False)
        return frame


def sparse_support(points_path, roi):
    from audit_target_preservation import read_points
    points = read_points(points_path)
    floor = roi['roi_derivation']['spatial_sparse_floor_plane']
    minimum, maximum = [np.asarray(roi['roi_world_lio_m'][key]) for key in ('minimum', 'maximum')]
    height = points @ np.asarray(floor['normal']) + float(floor['offset_m'])
    support = points[(points >= minimum).all(1) & (points <= maximum).all(1)
                     & (height > .10) & (height < 2.20)]
    if len(support) < 1000:
        raise ValueError('Too few elevated metric sparse points for the original surface gate')
    return support


def run(root, target, profile='AdaptiveSensor'):
    import torch
    import open3d as o3d
    visual_diagnostic = profile == 'VisualTargetDiagnostic'
    adaptive_sensor = profile == 'AdaptiveSensor'
    output = target / ('07_visual_target_tsdf' if visual_diagnostic else '07_reliable_lidar_tsdf')
    output.mkdir(exist_ok=False)
    cache_dir = output / 'render_cache'
    cache_dir.mkdir()
    started = time.monotonic()
    state = {'status': 'rendering_target_depth', 'frames_rendered': 0, 'profile': profile,
             'reference_code': canonical_path(PACKAGE / 'code/run_reliable_lidar_tsdf.py')}
    try:
        config, dataset, opt, pipe = prepare(root)
        trained = read(root / 'pgsr/TRAINING_RESULT.json')
        shared = read(root / 'shared/RESULT.json')
        roi = read(target / 'vehicle_roi_receipt.json')
        lidar_path = target / 'stable_lidar/stable_lidar_receipt.json'
        lidar = None if visual_diagnostic else read(lidar_path)
        if trained['status'] != 'completed' or roi['status'] != 'complete':
            raise ValueError('A prerequisite is incomplete')
        if not visual_diagnostic and (lidar['status'] != 'complete' or lidar['image_count'] != shared['registered_frames']):
            raise ValueError('The registered LiDAR evidence contract is incomplete')
        if not visual_diagnostic and lidar.get('profile') != profile:
            raise ValueError('Surface fusion and LiDAR admission profiles disagree')
        if trained['registered_frames'] != shared['registered_frames']:
            raise ValueError('The target branch changed the common image set')
        lidar_rows = {} if visual_diagnostic else {Path(row['image_name']).stem: row for row in lidar['per_frame']}
        minimum, maximum = [np.asarray(roi['roi_world_lio_m'][key]) for key in ('minimum', 'maximum')]
        focus = np.asarray(roi['roi_derivation']['focus_world_lio_m'])
        from scene import Scene, GaussianModel
        from gaussian_renderer import render as native_render
        trace = {'complete_K_plane_depth': True}
        render = calibrated_render(native_render, trace)
        geometries, views, rows = [], [], []
        with torch.no_grad(), calibrated_cameras(Path(dataset.source_path)) as calibration:
            model = GaussianModel(dataset.sh_degree)
            scene = Scene(dataset, model, load_iteration=opt.iterations, shuffle=False)
            cameras = scene.getTrainCameras()
            if len(cameras) != shared['registered_frames']:
                raise ValueError('PGSR loaded another camera denominator')
            if not visual_diagnostic and {Path(view.image_name).stem for view in cameras} != set(lidar_rows):
                raise ValueError('Registered anchors and PGSR views disagree')
            background = torch.zeros(3, device='cuda')
            for index, camera in enumerate(cameras):
                view = SimpleNamespace(image_name=camera.image_name, R=np.asarray(camera.R), T=np.asarray(camera.T),
                    Fx=float(camera.Fx), Fy=float(camera.Fy), Cx=float(camera.Cx), Cy=float(camera.Cy))
                geometry = reference.camera_geometry(view)
                rendered = render(camera, model, pipe, background)
                depth = rendered['plane_depth'].squeeze().cpu().numpy().astype(np.float32)
                alpha = rendered['rendered_alpha'].squeeze().cpu().numpy()
                normal = torch.nn.functional.normalize(rendered['depth_normal'].permute(1, 2, 0), p=2, dim=-1)
                rays = torch.nn.functional.normalize(camera.get_rays(), p=2, dim=-1)
                cosine = torch.sum(normal * rays, dim=-1).abs().clamp(0., 1.).cpu().numpy()
                camera_points = reference.camera_points_from_depth(view, depth)
                world = camera_points @ geometry[1][:3, :3].T + geometry[1][:3, 3]
                valid = (np.isfinite(depth) & (depth > 0) & (depth < reference.MAXIMUM_DEPTH_M)
                         & (world >= minimum).all(-1) & (world <= maximum).all(-1))
                base = valid & (alpha >= reference.MINIMUM_RENDERED_ALPHA) & (cosine >= np.cos(np.deg2rad(60)))
                broad = valid & (cosine >= np.cos(np.deg2rad(80)))
                color = (rendered['render'].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                np.savez_compressed(cache_dir / f'{index:05d}.npz', depth=depth, base=base, broad=broad, color=color)
                views.append(view)
                geometries.append(geometry)
                rows.append({'image_name': view.image_name, 'base_valid_pixel_count': int(base.sum()),
                             'broad_valid_pixel_count': int(broad.sum())})
                del rendered, normal, rays, camera_points, world
                if (index + 1) % 20 == 0 or index + 1 == len(cameras):
                    state.update(frames_rendered=index + 1, elapsed_seconds=time.monotonic() - started,
                        linux_RSS_bytes=psutil.Process().memory_info().rss,
                        peak_CUDA_reserved_bytes=int(torch.cuda.max_memory_reserved()))
                    publish(output / 'STATE.json', state)
                    print('rendered target', index + 1, '/', len(cameras), flush=True)
        state['calibration'], state['render_calibration'] = calibration, trace
        del scene, model, cameras, camera, background
        torch.cuda.empty_cache()
        centers = np.stack([geometry[2] for geometry in geometries])
        neighbors = None if visual_diagnostic else reference.choose_neighbors(centers, focus)
        cache = FrameCache(cache_dir)
        total_base = total_trusted = total_lidar = total_overrides = 0
        mesh_by_mode = {}
        roi_box = o3d.geometry.AxisAlignedBoundingBox(minimum, maximum)
        modes = ('pgsr_depth_only', 'pgsr_depth_broad') if visual_diagnostic else (
            'registered_lidar_fused', 'pgsr_depth_only', 'pgsr_depth_broad')
        for mode in modes:
            volume = o3d.pipelines.integration.ScalableTSDFVolume(
                voxel_length=reference.VOXEL_SIZE_M, sdf_trunc=reference.SDF_TRUNCATION_M,
                color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
            state.update(status='integrating_target_candidates', current_candidate=mode, frames_integrated=0)
            for index, view in enumerate(views):
                frame = cache.get(index)
                depth, base = frame['depth'], frame['base']
                if mode == 'registered_lidar_fused':
                    flat = np.flatnonzero(base.reshape(-1))
                    local = reference.camera_points_from_depth(view, depth).reshape(-1, 3)[flat]
                    world = local @ geometries[index][1][:3, :3].T + geometries[index][1][:3, 3]
                    support_count = np.zeros(len(flat), np.uint8)
                    for neighbor_index in neighbors[index]:
                        other = views[neighbor_index]
                        transform = geometries[neighbor_index][0]
                        camera_points = world @ transform[:3, :3].T + transform[:3, 3]
                        projected_z = np.maximum(camera_points[:, 2], 1e-9)
                        x = np.rint(camera_points[:, 0] / projected_z * other.Fx + other.Cx).astype(np.int64)
                        y = np.rint(camera_points[:, 1] / projected_z * other.Fy + other.Cy).astype(np.int64)
                        neighbor = cache.get(neighbor_index)
                        height, width = neighbor['depth'].shape
                        inside = (camera_points[:, 2] > .05) & (x >= 0) & (x < width) & (y >= 0) & (y < height)
                        candidate = np.flatnonzero(inside)
                        residual = np.abs(neighbor['depth'][y[candidate], x[candidate]] - camera_points[candidate, 2])
                        tolerance = np.maximum(reference.DEPTH_ABSOLUTE_TOLERANCE_M,
                            reference.DEPTH_RELATIVE_TOLERANCE * camera_points[candidate, 2])
                        consistent = neighbor['base'][y[candidate], x[candidate]] & (residual <= tolerance)
                        support_count[candidate[consistent]] += 1
                    trusted = np.zeros(depth.size, bool)
                    trusted[flat] = support_count >= reference.MINIMUM_NEIGHBOR_SUPPORT
                    trusted = trusted.reshape(depth.shape)
                    anchor_row = lidar_rows[Path(view.image_name).stem]
                    if adaptive_sensor and anchor_row['sensor_role'] != 'visual_and_stable_lidar':
                        lidar_depth = np.zeros_like(depth)
                    else:
                        with np.load(lidar_path.parent / anchor_row['path']) as archive:
                            lidar_depth = reference.project_lidar_depth(archive['points_world_lio_m'],
                                geometries[index][0], view, *depth.shape)
                    lidar_valid = lidar_depth > 0
                    fused = np.where(trusted, depth, 0.).astype(np.float32)
                    overrides = lidar_valid & (fused > 0) & (np.abs(fused - lidar_depth) > .05)
                    fused[lidar_valid] = lidar_depth[lidar_valid]
                    reference.integrate_depth(volume, frame['color'], fused, view, geometries[index][0])
                    lidar_only = np.where(lidar_valid, lidar_depth, 0.).astype(np.float32)
                    for _ in range(reference.LIDAR_EXTRA_INTEGRATIONS):
                        reference.integrate_depth(volume, frame['color'], lidar_only, view, geometries[index][0])
                    total_base += int(base.sum())
                    total_trusted += int(trusted.sum())
                    total_lidar += int(lidar_valid.sum())
                    total_overrides += int(overrides.sum())
                    rows[index].update(neighbor_indices=neighbors[index], multi_view_trusted_pixel_count=int(trusted.sum()),
                        lidar_anchor_pixel_count=int(lidar_valid.sum()), lidar_override_disagreement_count=int(overrides.sum()),
                        sensor_role=anchor_row.get('sensor_role', 'legacy_per_frame_lidar_gate'))
                else:
                    mask = frame['base'] if mode == 'pgsr_depth_only' else frame['broad']
                    visual_depth = np.where(mask, depth, 0.).astype(np.float32)
                    reference.integrate_depth(volume, frame['color'], visual_depth, view, geometries[index][0])
                if (index + 1) % 20 == 0 or index + 1 == len(views):
                    state.update(frames_integrated=index + 1, elapsed_seconds=time.monotonic() - started,
                        linux_RSS_bytes=psutil.Process().memory_info().rss,
                        total_multi_view_trusted_pixels=total_trusted, total_lidar_anchor_pixels=total_lidar)
                    publish(output / 'STATE.json', state)
                    print('integrated', mode, index + 1, '/', len(views), flush=True)
                    if state['linux_RSS_bytes'] > 24 * 1024**3:
                        raise RuntimeError('Target TSDF exceeded the 24 GiB Linux RSS guard')
            mesh = volume.extract_triangle_mesh()
            del volume
            mesh = mesh.crop(roi_box)
            mesh.compute_vertex_normals()
            if not mesh.has_vertices() or not mesh.has_triangles():
                raise ValueError('The target TSDF candidate is empty: ' + mode)
            mesh_by_mode[mode] = mesh
        if not visual_diagnostic and (total_trusted / max(total_base, 1) < .05 or total_lidar < 10000):
            raise ValueError('The original multi-view or projected LiDAR support gate failed')
        support = sparse_support(target / 'metric_sparse/points3D.txt', roi)
        metrics = {mode: reference.surface_candidate_metrics(mesh, support) for mode, mesh in mesh_by_mode.items()}
        if visual_diagnostic:
            # This explicit diagnostic is the established broad visual path, not a passed LiDAR gate.
            selected = 'pgsr_depth_broad'
            selection = {'method': 'explicit_broad_visual_diagnostic_fixed_before_surface_results',
                'selected_mode': selected, 'candidates': metrics, 'strict_three_candidate_gate_passed': False,
                'strict_lidar_pairing_failure': canonical_path(target / 'stable_lidar/PAIRING_FAILURE.json')}
        else:
            selected, selection = select_surface_candidate(
                *[metrics[mode] for mode in modes], reference.select_surface_candidate)
            if metrics[selected]['sparse_support_fraction_within_5cm'] < reference.MINIMUM_VISUAL_SPARSE_SUPPORT_FRACTION:
                raise ValueError('The selected surface does not meet the existing 95 percent sparse support threshold')
        selection['candidate_meshes'] = {}
        for mode, mesh in mesh_by_mode.items():
            path = output / ('candidate_' + mode + '_mesh.ply')
            if not o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False):
                raise RuntimeError('Could not save a target TSDF candidate')
            selection['candidate_meshes'][mode] = {'path': canonical_path(path), 'sha256': reference.sha256_file(path)}
        ground_policy = read(PACKAGE / 'ground/config.json')['ground_classification']
        mesh, trim = trim_observed_surface(mesh_by_mode[selected], support, roi, ground_policy, reference)
        raw_mesh = output / 'target_reliable_lidar_pgsr_5mm_raw_mesh.ply'
        if not o3d.io.write_triangle_mesh(str(raw_mesh), mesh, write_ascii=False):
            raise RuntimeError('Could not write the selected target surface')
        receipt = {'schema_version': ('e26_visual_target_diagnostic_tsdf_v1' if visual_diagnostic else
                    'e26_adaptive_sensor_pgsr_tsdf_v1' if adaptive_sensor else 'e26_reliable_lidar_pgsr_tsdf_v1'),
            'status': 'raw_s08_mesh_generated', 'profile': profile,
            'source_model': canonical_path(root / 'pgsr/model'), 'source_iteration': config['pgsr_iterations'],
            'source_training_receipt': canonical_path(root / 'pgsr/TRAINING_RESULT.json'),
            'source_roi_receipt': canonical_path(target / 'vehicle_roi_receipt.json'),
            'source_lidar_receipt': None if visual_diagnostic else canonical_path(lidar_path),
            'lidar_anchor_coverage': {'status': 'not_admitted_diagnostic_visual_only'} if visual_diagnostic else lidar['anchor_coverage'],
            'image_alignment': {'status': 'complete_common_training_set', 'trained_view_count': len(views),
                'selected_view_count': shared['registered_frames'], 'unused_anchor_image_names': []},
            'language_model_used': False, 'semantic_model_used': False, 'object_mask_used': False,
            'known_object_dimensions_used': False, 'roi_derivation': roi['roi_derivation'],
            'roi_world_lio_m': roi['roi_world_lio_m'], 'roi_boundary_trim': trim,
            'surface_candidate_selection': selection,
            'multi_view_gate': {'neighbor_count': 8, 'minimum_neighbor_support': 3,
                'base_pixel_count': total_base, 'trusted_pixel_count': total_trusted,
                'trusted_fraction': total_trusted / max(total_base, 1), 'minimum_rendered_alpha': .60,
                'maximum_normal_view_angle_degrees': 60., 'absolute_depth_tolerance_m': .03,
                'relative_depth_tolerance': .012},
            'lidar_anchor_fusion': {'applied_to_selected_mesh': selected == 'registered_lidar_fused',
                'projected_anchor_pixel_count': total_lidar, 'pgsr_disagreement_override_count': total_overrides,
                'dilation_pixels': 1, 'extra_tsdf_integrations': 2},
            'broad_visual_gate': {'maximum_normal_view_angle_degrees': 80., 'rendered_alpha_filter': 'none',
                'valid_pixel_count': sum(row['broad_valid_pixel_count'] for row in rows)},
            'voxel_size_m': reference.VOXEL_SIZE_M, 'sdf_truncation_m': reference.SDF_TRUNCATION_M,
            'frame_count': len(views), 'mesh': {'path': canonical_path(raw_mesh),
                'sha256': reference.sha256_file(raw_mesh), 'vertices': len(mesh.vertices), 'triangles': len(mesh.triangles)},
            'runtime_seconds': time.monotonic() - started, 'per_frame': rows,
            'calibration': state['calibration'], 'render_calibration': state['render_calibration'],
            'reference_code_sha256': reference.sha256_file(PACKAGE / 'code/run_reliable_lidar_tsdf.py'),
            'strict_ReliableLidar_profile_completed': profile == 'ReliableLidar',
            'adaptive_sensor_contract_completed': adaptive_sensor,
            'visual_frame_set_preserved': len(views) == shared['registered_frames'],
            'all_three_surface_candidates_evaluated': not visual_diagnostic,
            'scientific_boundary': ('Explicit visual target diagnostic after strict LiDAR pairing failure; not an accepted ReliableLidar run.'
                if visual_diagnostic else 'The visual set is preserved; LiDAR contributes only where genuine sensor evidence passes admission. Legacy all-frame coverage is not claimed; no independent geometry ground truth.'
                if adaptive_sensor else 'Observed target ROI surface selected by the established gates; no independent geometry ground truth.')}
        publish(output / 'reliable_lidar_tsdf_receipt.json', receipt)
        state.update(status='completed', selected_mode=selected, mesh=receipt['mesh'],
                     elapsed_seconds=receipt['runtime_seconds'])
        publish(output / 'STATE.json', state)
        print('selected target surface', selected, len(mesh.vertices), len(mesh.triangles), flush=True)
    except BaseException:
        state.update(status='failed_target_surface', traceback=traceback.format_exc(), elapsed_seconds=time.monotonic() - started)
        publish(output / 'FAILURE.json', state)
        publish(output / 'STATE.json', state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--profile', choices=('AdaptiveSensor', 'ReliableLidar', 'VisualTargetDiagnostic'), default='AdaptiveSensor')
    args = parser.parse_args()
    run(args.run_root.resolve(strict=True), args.output_root.resolve(strict=True), args.profile)
