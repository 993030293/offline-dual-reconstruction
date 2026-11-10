"""Share the existing point policy between seed qualification and BA preparation."""

import ast
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def native_point_policy():
    policy_path = ROOT / '05_packages/e26_video_to_target_m3_nollm_v1r23_20260927_installed_candidate/config/METRIC_ALIGNMENT_POLICY.json'
    policy = json.loads(policy_path.read_text())['point_filter']
    camera_source = ROOT / '04_outputs/box2_3dgs_display_v1_20260930/source/gaussian-splatting/scene/cameras.py'
    tree = ast.parse(camera_source.read_text())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Camera']
    if len(classes) != 1:
        raise ValueError('The native renderer Camera class changed')
    constants = [node.value.value for node in ast.walk(classes[0]) if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant) and any(isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name) and target.value.id == 'self' and target.attr == 'znear'
        for target in node.targets)]
    if len(constants) != 1 or not isinstance(constants[0], (int, float)) or constants[0] <= 0:
        raise ValueError('The native renderer near plane contract changed')
    return policy_path, policy, camera_source, float(constants[0])


def point_policy_removals(model, scale, minimum_track, near_m, scope='full-training-policy'):
    """Keep the original global point filtering order and rejection behavior."""
    if scope not in ('full-training-policy', 'renderer-only'):
        raise ValueError('Unknown point policy scope')
    if not np.isfinite(scale) or scale <= 0 or not np.isfinite(near_m) or near_m <= 0:
        raise ValueError('Point policy requires a positive finite scale and renderer near plane')
    removals = {}
    for key, point in model.points3D.items():
        if scope == 'full-training-policy' and point.track.length() < minimum_track:
            removals[int(key)] = {'point3D_id': int(key), 'track_length': point.track.length(),
                'reason': 'existing_training_minimum_point_track_length'}
    for image in model.images.values():
        if not image.has_pose:
            continue
        entries = [point for point in image.points2D if point.has_point3D() and int(point.point3D_id) not in removals]
        xyz = np.asarray([model.points3D[point.point3D_id].xyz for point in entries]).reshape(-1, 3)
        pose = image.cam_from_world()
        depths = (xyz @ pose.rotation.matrix().T + pose.translation)[:, 2] * scale
        if not np.isfinite(depths).all() or np.any(depths <= 0):
            raise ValueError('The supported source already has invalid observed depths')
        for index in np.flatnonzero(depths <= near_m):
            point_id = int(entries[index].point3D_id)
            removals[point_id] = {'point3D_id': point_id, 'track_length': model.points3D[point_id].track.length(),
                'reason': 'existing_native_renderer_near_plane', 'source_depth_m': float(depths[index])}
    return removals


def qualified_seed_observations(model, removals):
    """Count actual surviving observations without changing poses, points, or tracks."""
    return {image.name: sum(point.has_point3D() and int(point.point3D_id) not in removals
        for point in image.points2D) for image in model.images.values() if image.has_pose}
