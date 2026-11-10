"""Fuse full calibrated RGBD into bounded spatial tiles without changing voxel size."""

import itertools
import json
from pathlib import Path
import time
import traceback

import numpy as np
import psutil

from pgsr_branch import prepare, publish
from pgsr_calibration import calibrated_cameras, calibrated_render, camera_K
from ply_io import read_ply

BLOCK = 16
CORE = 32
HALO = 1
MEMORY_LIMIT = 16 * 1024**3
VERTEX = np.dtype([(name, '<f4') for name in ('x', 'y', 'z', 'nx', 'ny', 'nz')]
                  + [(name, 'u1') for name in ('red', 'green', 'blue')])
FACE = np.dtype([('count', 'u1'), ('indices', '<i4', (3,))])


def make_grid(voxel, blocks):
    import open3d as o3d
    return o3d.t.geometry.VoxelBlockGrid(attr_names=('tsdf', 'weight', 'color'),
        attr_dtypes=(o3d.core.float32,) * 3, attr_channels=((1,), (1,), (3,)),
        voxel_size=voxel, block_resolution=BLOCK, block_count=max(1, int(blocks)),
        device=o3d.core.Device('CPU:0'))


def image(array):
    import open3d as o3d
    return o3d.t.geometry.Image(o3d.core.Tensor(np.ascontiguousarray(array)))


def matrix(array):
    import open3d as o3d
    return o3d.core.Tensor(np.asarray(array, np.float64), o3d.core.float64)


def membership(coordinates, tile):
    origin = np.asarray(tile, np.int64) * CORE
    return ((coordinates >= origin - HALO) & (coordinates < origin + CORE + HALO)).all(1)


def relevant_tiles(coordinates):
    # A shared block grid and halo preserve interpolation across core boundaries.
    # Include corner halos as well as face halos.
    bases = np.unique(coordinates // CORE, axis=0)
    neighbors = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
    candidates = np.unique((bases[:, None, :] + neighbors).reshape(-1, 3), axis=0)
    return [tuple(map(int, tile)) for tile in candidates if membership(coordinates, tile).any()]


def fuse_frame(grid, coordinates, depth, color, intrinsic, extrinsic, config):
    import open3d as o3d
    # Float depth requires float color in Open3D 0.18; its float color domain is 0..1.
    grid.integrate(o3d.core.Tensor(np.ascontiguousarray(coordinates, dtype=np.int32)),
        image(depth), image(color.astype(np.float32) / 255.), matrix(intrinsic), matrix(extrinsic),
        1., config['surface_depth_max_m'], config['surface_truncation_m'] / config['surface_voxel_m'])


def core_mesh(grid, tile, voxel):
    mesh = grid.extract_triangle_mesh(weight_threshold=0.).to_legacy()
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.triangles)
    if not len(faces):
        return mesh
    origin = np.asarray(tile) * CORE * BLOCK * voxel
    centroid = vertices[faces].mean(1)
    retained = ((centroid >= origin) & (centroid < origin + CORE * BLOCK * voxel)).all(1)
    mesh.remove_triangles_by_mask(~retained)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


def header(stream, vertices, faces=None):
    rows = ['ply', 'format binary_little_endian 1.0', 'element vertex ' + str(vertices)]
    for name in VERTEX.names:
        rows.append('property ' + ('uchar' if VERTEX[name].kind == 'u' else 'float') + ' ' + name)
    if faces is not None:
        rows.extend(['element face ' + str(faces), 'property list uchar int vertex_indices'])
    stream.write(('\n'.join(rows + ['end_header']) + '\n').encode('ascii'))


def save_mesh(path, mesh):
    positions, normals = np.asarray(mesh.vertices), np.asarray(mesh.vertex_normals)
    colors, triangles = np.asarray(mesh.vertex_colors), np.asarray(mesh.triangles)
    if (colors.shape != positions.shape or normals.shape != positions.shape
            or not np.isfinite(positions).all() or not np.isfinite(colors).all() or not np.isfinite(normals).all()):
        raise ValueError('Invalid tiled mesh geometry or colors')
    if colors.size and (colors.min() < -1e-5 or colors.max() > 1.00001):
        raise ValueError('Float RGB integration changed its normalized color domain')
    vertices = np.empty(len(positions), VERTEX)
    for axis, name in enumerate(('x', 'y', 'z')):
        vertices[name] = positions[:, axis]
    for axis, name in enumerate(('nx', 'ny', 'nz')):
        vertices[name] = normals[:, axis]
    for axis, name in enumerate(('red', 'green', 'blue')):
        vertices[name] = np.rint(np.clip(colors[:, axis], 0, 1) * 255).astype(np.uint8)
    faces = np.empty(len(triangles), FACE)
    faces['count'], faces['indices'] = 3, triangles
    with path.open('xb') as stream:
        header(stream, len(vertices), len(faces))
        vertices.tofile(stream)
        faces.tofile(stream)
    return len(vertices), len(faces)


def triangle_data(vertices, faces):
    positions = np.column_stack([vertices[name] for name in ('x', 'y', 'z')])
    triangles = positions[faces['indices']].astype(np.float64)
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1) / 2
    if not np.isfinite(area).all() or (area < 0).any():
        raise ValueError('Nonfinite triangle area')
    return triangles, area


def merge_and_sample(output, pieces, number):
    total_vertices = sum(row['vertices'] for row in pieces)
    total_faces = sum(row['triangles'] for row in pieces)
    if total_vertices < 100 or total_faces < 100 or total_vertices >= 2**31:
        raise ValueError('Full mesh is empty or exceeds its declared index format')
    with (output / 'surface_mesh.ply').open('xb') as mesh, (output / 'surface_full_points.ply').open('xb') as cloud:
        header(mesh, total_vertices, total_faces)
        header(cloud, total_vertices)
        for row in pieces:
            vertices, _ = read_ply(output / row['path'])
            if vertices.dtype != VERTEX or not all(np.isfinite(vertices[name]).all() for name in VERTEX.names):
                raise ValueError('Tiled vertex readback failed')
            vertices.tofile(mesh)
            vertices.tofile(cloud)
        offset = 0
        for row in pieces:
            vertices, faces = read_ply(output / row['path'])
            faces['indices'] += offset
            faces.tofile(mesh)
            offset += len(vertices)
    areas = []
    for row in pieces:
        vertices, faces = read_ply(output / row['path'])
        _, area = triangle_data(vertices, faces)
        areas.append(float(area.sum()))
    total_area = sum(areas)
    if not total_area > 0:
        raise ValueError('Cannot sample a zero-area mesh')
    rng = np.random.default_rng(0)
    picks = np.sort(rng.random(number) * total_area)
    samples = np.empty(number, VERTEX)
    offset, written = 0., 0
    for row, local_area in zip(pieces, areas):
        right = int(np.searchsorted(picks, offset + local_area, side='left'))
        count = right - written
        if count:
            vertices, faces = read_ply(output / row['path'])
            triangles, area = triangle_data(vertices, faces)
            indices = np.searchsorted(np.cumsum(area), picks[written:right] - offset, side='right')
            if len(indices) and indices.max() >= len(faces):
                raise ValueError('Area sampling exceeded its triangle denominator')
            root = np.sqrt(rng.random(count))
            fraction = rng.random(count)
            barycentric = np.column_stack((1 - root, root * (1 - fraction), root * fraction))
            positions = (triangles[indices] * barycentric[:, :, None]).sum(1)
            vertex_indices = faces['indices'][indices]
            normals = np.stack([vertices[name][vertex_indices] for name in ('nx', 'ny', 'nz')], axis=2)
            normals = (normals * barycentric[:, :, None]).sum(1)
            length = np.linalg.norm(normals, axis=1)
            normals /= np.maximum(length[:, None], 1e-12)
            colors = np.stack([vertices[name][vertex_indices] for name in ('red', 'green', 'blue')], axis=2)
            colors = np.rint((colors * barycentric[:, :, None]).sum(1)).astype(np.uint8)
            for axis, name in enumerate(('x', 'y', 'z')):
                samples[name][written:right] = positions[:, axis]
            for axis, name in enumerate(('nx', 'ny', 'nz')):
                samples[name][written:right] = normals[:, axis]
            for axis, name in enumerate(('red', 'green', 'blue')):
                samples[name][written:right] = colors[:, axis]
        written, offset = right, offset + local_area
    if written != number or not all(np.isfinite(samples[name]).all() for name in VERTEX.names):
        raise ValueError('Complete surface sampling failed')
    with (output / 'surface_points.ply').open('xb') as stream:
        header(stream, number)
        samples.tofile(stream)
    loaded, faces = read_ply(output / 'surface_points.ply')
    if faces is not None or not np.array_equal(loaded, samples):
        raise ValueError('Surface sample binary readback changed geometry')
    return total_vertices, total_faces, np.column_stack([samples[name] for name in ('x', 'y', 'z')])


def diagnostic():
    config = {'surface_voxel_m': .004, 'surface_truncation_m': .016, 'surface_depth_max_m': 20.}
    depth = np.full((48, 64), .3, np.float32)
    color = np.broadcast_to(np.array([64, 128, 192], np.uint8), (48, 64, 3)).copy()
    intrinsic = np.array([[80., 0, 31.5], [0, 81., 23.5], [0, 0, 1.]])
    extrinsic = np.eye(4)
    grid = make_grid(.004, 64)
    coordinates = grid.compute_unique_block_coordinates(image(depth), matrix(intrinsic), matrix(extrinsic), 1., 20., 4.).numpy()
    fuse_frame(grid, coordinates, depth, color, intrinsic, extrinsic, config)
    reference = grid.extract_triangle_mesh(weight_threshold=0.).to_legacy()
    faces = np.asarray(reference.triangles)
    reference_centers = np.asarray(reference.vertices)[faces].mean(1)
    parts = []
    error = 0.
    colors_error = 0.
    for tile in relevant_tiles(coordinates):
        local = make_grid(.004, len(coordinates) + 1)
        selected = coordinates[membership(coordinates, tile)]
        fuse_frame(local, selected, depth, color, intrinsic, extrinsic, config)
        mesh = core_mesh(local, tile, .004)
        if len(mesh.triangles):
            points = np.asarray(mesh.vertices)
            parts.append(points[np.asarray(mesh.triangles)].mean(1))
            error = max(error, float(np.max(np.abs(points[:, 2] - .3))))
            colors_error = max(colors_error, float(np.max(np.abs(np.asarray(mesh.vertex_colors) - np.array([64, 128, 192]) / 255.))))
    actual = np.concatenate(parts)
    canonical = lambda points: sorted(map(tuple, np.round(points, 7)))
    if len(faces) < 100 or canonical(actual) != canonical(reference_centers) or error > 1e-6 or colors_error > 1e-5:
        raise ValueError('Known plane or tiled core boundary equivalence failed')
    return {'status': 'known_plane_color_and_tile_boundary_verified', 'triangles': len(faces),
        'maximum_known_depth_error_m': error, 'maximum_normalized_color_error': colors_error,
        'tile_triangle_centroids_equal_global_fixture': True,
        'positive_weight_threshold': 0., 'actual_scene_accuracy_measured': False,
        'legacy_TSDF_scene_equivalence_claimed': False}


def surface(root):
    import torch
    from PIL import Image
    config, dataset, opt, pipe = prepare(root)
    from scene import Scene, GaussianModel
    from gaussian_renderer import render as native_render
    render_trace = {'complete_K_plane_depth': True}
    render = calibrated_render(native_render, render_trace)
    trained = json.loads((root / 'pgsr/TRAINING_RESULT.json').read_text())
    if trained['status'] != 'completed' or trained['iteration'] != config['pgsr_iterations']:
        raise ValueError('Surface requires the complete trained PGSR branch')
    output = root / 'pgsr/surface'
    output.mkdir(exist_ok=False)
    cache = output / 'rgbd_cache'
    cache.mkdir()
    (output / 'tiles').mkdir()
    started = time.monotonic()
    state = {'status': 'running', 'phase': 'render_cache', 'depth_unit': 'meters', 'depth_kind': 'camera_z',
        'depth_filter': 'finite positive range and grazing angle under 80 degrees', 'frames_integrated': 0,
        'backend': 'Open3D_tensor_CPU_spatial_tiles', 'block_resolution': BLOCK,
        'core_blocks_per_axis': CORE, 'halo_blocks': HALO, 'Linux_RSS_limit_bytes': MEMORY_LIMIT,
        'voxel_m': config['surface_voxel_m'], 'truncation_m': config['surface_truncation_m'],
        'original_failed_legacy_attempt_retained': True, 'legacy_scene_equivalence_claimed': False,
        'mesh_seam_policy': 'shared_world_grid_halo_half_open_triangle_centroid_ownership_no_welding',
        'positive_weight_threshold': 0., 'global_scene_ROI_filter_applied': False}

    def heartbeat(**updates):
        state.update(updates, elapsed_seconds=time.monotonic() - started, rss_bytes=psutil.Process().memory_info().rss)
        if state['elapsed_seconds'] > 7200 or state['rss_bytes'] > MEMORY_LIMIT:
            raise RuntimeError('Tiled fusion exceeded its explicit time or Linux memory budget')
        publish(output / 'STATE.json', state)
        print(json.dumps({key: state[key] for key in ('phase', 'elapsed_seconds', 'rss_bytes')}), flush=True)

    try:
        state['fusion_diagnostic'] = diagnostic()
        publish(output / 'TILE_BOUNDARY_DIAGNOSTIC.json', state['fusion_diagnostic'])
        rows, blocks, tile_frames = [], [], {}
        coordinate_grid = make_grid(config['surface_voxel_m'], 1)
        with torch.no_grad(), calibrated_cameras(Path(dataset.source_path)) as calibration:
            from check_native_pgsr import verify
            with torch.enable_grad():
                state['native_plane_kernel_diagnostic'] = verify(GaussianModel, native_render)
            publish(output / 'NATIVE_PLANE_DIAGNOSTIC.json', state['native_plane_kernel_diagnostic'])
            model = GaussianModel(dataset.sh_degree)
            scene = Scene(dataset, model, load_iteration=opt.iterations, shuffle=False)
            views = scene.getTrainCameras()
            if len(views) != trained['registered_frames']:
                raise ValueError('Surface camera denominator changed')
            diagnostics = set(np.linspace(0, len(views) - 1, 6).round().astype(int).tolist())
            background = torch.zeros(3, dtype=torch.float32, device='cuda')
            for index, view in enumerate(views):
                rendered = render(view, model, pipe, background, return_plane=True, return_depth_normal=True)
                depth = rendered['plane_depth'].squeeze().detach().cpu().numpy().astype(np.float32)
                rays, normals = view.get_rays(), rendered['depth_normal'].permute(1, 2, 0)
                norm = torch.linalg.vector_norm(normals, dim=-1)
                cosine = torch.sum(torch.nn.functional.normalize(rays, dim=-1) *
                    torch.nn.functional.normalize(normals, dim=-1), dim=-1).abs()
                angle = ((norm > 1e-6) & (cosine >= np.cos(np.deg2rad(80)))).cpu().numpy()
                valid = np.isfinite(depth) & (depth > 0) & (depth <= config['surface_depth_max_m']) & angle
                depth[~valid] = 0
                color = np.rint(rendered['render'].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                intrinsic = camera_K(view)
                extrinsic = view.world_view_transform.T.detach().cpu().numpy().astype(np.float64)
                coordinates = coordinate_grid.compute_unique_block_coordinates(image(depth), matrix(intrinsic),
                    matrix(extrinsic), 1., config['surface_depth_max_m'],
                    config['surface_truncation_m'] / config['surface_voxel_m']).numpy().copy()
                if not len(coordinates):
                    raise ValueError('An admitted frame has no valid depth blocks')
                blocks.append(coordinates)
                for tile in relevant_tiles(coordinates):
                    tile_frames.setdefault(tile, []).append(index)
                np.savez(cache / f'{index:04d}.npz', depth=depth, color=color, intrinsic=intrinsic, extrinsic=extrinsic)
                row = {'image_name': view.image_name, 'valid_depth_pixels': int(valid.sum()), 'total_pixels': int(valid.size),
                    'depth_median_m': float(np.median(depth[valid])), 'active_blocks': len(coordinates)}
                rows.append(row)
                if index in diagnostics:
                    Image.fromarray(color).save(output / f'render_{index:04d}.png')
                    truth, _ = view.get_image()
                    rgb = np.rint(truth.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(rgb).save(output / f'reference_{index:04d}.png')
                    np.save(output / f'depth_{index:04d}.npy', depth)
                if (index + 1) % 20 == 0 or index + 1 == len(views):
                    heartbeat(frames_rendered=index + 1, total_frames=len(views), last_frame=row, planned_tiles=len(tile_frames))
            state['calibration'] = calibration
        del model, scene, views, rendered, rays, normals, background
        torch.cuda.empty_cache()
        publish(output / 'FRAME_DEPTH_QA.json', {'rows': rows})
        tiles = sorted(tile_frames)
        publish(output / 'TILE_PLAN.json', {'tile_count': len(tiles), 'all_frames': len(rows),
            'tile_core_size_m': CORE * BLOCK * config['surface_voxel_m'],
            'rows': [{'tile': list(tile), 'frames': tile_frames[tile]} for tile in tiles]})
        if len(tiles) > 2048:
            raise RuntimeError('Full scene exceeds the declared tile planning budget; no crop was applied')
        state.update(phase='tile_fusion', total_tiles=len(tiles))
        pieces, integrated = [], set()
        for ordinal, tile in enumerate(tiles):
            selected = {index: blocks[index][membership(blocks[index], tile)] for index in tile_frames[tile]}
            unique = np.unique(np.concatenate(list(selected.values())), axis=0)
            maximum = (CORE + HALO * 2)**3
            if len(unique) > maximum:
                raise ValueError('Tile active block bounds were violated')
            grid = make_grid(config['surface_voxel_m'], len(unique))
            for index in tile_frames[tile]:
                with np.load(cache / f'{index:04d}.npz') as frame:
                    fuse_frame(grid, selected[index], frame['depth'], frame['color'],
                        frame['intrinsic'], frame['extrinsic'], config)
                integrated.add(index)
            mesh = core_mesh(grid, tile, config['surface_voxel_m'])
            if len(mesh.triangles):
                relative = 'tiles/' + f'{ordinal:04d}.ply'
                vertices, triangles = save_mesh(output / relative, mesh)
                pieces.append({'path': relative, 'tile': list(tile), 'vertices': vertices, 'triangles': triangles,
                    'blocks': len(unique), 'frames': len(tile_frames[tile])})
            del grid, mesh, unique, selected
            heartbeat(tiles_completed=ordinal + 1, frames_integrated=len(integrated),
                mesh_vertices_so_far=sum(row['vertices'] for row in pieces),
                last_tile=list(tile), last_tile_frames=len(tile_frames[tile]))
        if len(integrated) != trained['registered_frames']:
            raise ValueError('Not every admitted frame contributed to tiled integration')
        publish(output / 'TILE_RESULTS.json', {'rows': pieces})
        heartbeat(phase='mesh_merge_and_sampling')
        vertices, triangles, points = merge_and_sample(output, pieces, config['surface_sample_count'])
        heartbeat(status='completed', vertices=vertices, triangles=triangles, sampled_points=len(points),
            full_surface_points=vertices, full_surface_pointcloud_is_all_mesh_vertices=True,
            minimum=points.min(0).tolist(), maximum=points.max(0).tolist(),
            points_are_mesh_surface_samples=True, largest_component_filter_applied=False,
            independent_GT_accuracy_measured=False, render_calibration=render_trace)
        publish(root / 'pgsr/SURFACE_RESULT.json', state)
    except BaseException:
        state.update(status='failed_surface', elapsed_seconds=time.monotonic() - started, traceback=traceback.format_exc())
        publish(output / 'STATE.json', state)
        publish(root / 'pgsr/SURFACE_FAILURE.json', state)
        raise
