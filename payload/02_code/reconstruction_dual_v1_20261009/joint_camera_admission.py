"""Stabilize the camera-point admission graph with unchanged geometry thresholds."""

from contextlib import contextmanager
import inspect
import linecache
from pathlib import Path

import numpy as np

from atomic_json_windows import atomic_json
from destination_camera_admission import finalize_working_subset

AFTER = '''        support_after = {im.name: camera_geometry_metrics(model, im, policy["camera_geometry_admission"])
                         for im in model.images.values() if im.has_pose}'''

EXTENT = '''        centers = np.asarray([im.projection_center() for im in model.images.values() if im.has_pose])
        pp = policy["point_filter"]
        if not np.isfinite(centers).all() or np.linalg.norm(centers, axis=1).max() > pp["maximum_camera_radius_m"]:
            raise ValueError("The metric camera extent is invalid")
        lo, hi = centers.min(0)-pp["camera_aabb_margin_m"], centers.max(0)+pp["camera_aabb_margin_m"]'''


def training_candidate_extent(model, camera_policy, point_policy, support_function, output):
    """Apply metric extent only after the unchanged geometry-support partition."""
    registered = {image.name: image for image in model.images.values() if image.has_pose}
    supports = {
        name: support_function(model, image, camera_policy)
        for name, image in registered.items()
    }
    candidates = {name for name, row in supports.items() if row['passed']}
    witnesses = set(registered) - candidates
    minimum_fraction = camera_policy['minimum_retained_fraction_of_sim3_inliers']
    if (len(candidates) < camera_policy['minimum_retained_cameras']
            or len(candidates) / len(registered) < minimum_fraction):
        raise ValueError('The unchanged geometry support leaves too few training candidates')
    candidate_centers = np.asarray([
        registered[name].projection_center() for name in sorted(candidates)
    ])
    candidate_radii = np.linalg.norm(candidate_centers, axis=1)
    full_radii = {
        name: float(np.linalg.norm(image.projection_center()))
        for name, image in registered.items()
    }
    if (not np.isfinite(candidate_centers).all()
            or not np.isfinite(candidate_radii).all()
            or candidate_radii.max() > point_policy['maximum_camera_radius_m']):
        raise ValueError('A geometry-supported training candidate has an invalid metric extent')
    atomic_json(Path(output) / 'PRE_FILTER_TRAINING_EXTENT.json', {
        'status': 'training_candidate_extent_verified',
        'full_registered_denominator': len(registered),
        'training_support_candidate_count': len(candidates),
        'training_support_candidate_fraction': len(candidates) / len(registered),
        'nontraining_witness_names': sorted(witnesses),
        'full_map_maximum_camera_radius_m': max(full_radii.values()),
        'training_candidate_maximum_camera_radius_m': float(candidate_radii.max()),
        'maximum_camera_radius_m': point_policy['maximum_camera_radius_m'],
        'minimum_retained_fraction': minimum_fraction,
        'thresholds_unchanged': True,
        'full_map_camera_registrations_preserved': True,
        'nontraining_witnesses_not_used_for_training_AABB': True,
    })
    return candidate_centers


def stabilize(model, point_policy, camera_policy, lo, hi, filter_points, support_function, full_count, output):
    support = {image.name: support_function(model, image, camera_policy)
        for image in model.images.values() if image.has_pose}
    original_failure = any(not row['passed'] for row in support.values())
    removed_points, excluded_support, rounds = [], {}, []
    for index in range(full_count + 1):
        rejected = {name for name, row in support.items() if not row['passed']}
        if not rejected:
            break
        before = model.num_reg_images()
        frames = {image.frame_id for image in model.images.values() if image.has_pose and image.name in rejected}
        if any(image.has_pose and image.frame_id in frames and image.name not in rejected
                for image in model.images.values()):
            raise ValueError('A rig frame would remove a qualified camera')
        excluded_support.update({name: support[name] for name in rejected})
        for frame in sorted(frames):
            model.deregister_frame(frame)
        removed = filter_points(model, point_policy, lo, hi)
        removed_points.extend(removed)
        after = model.num_reg_images()
        rounds.append({'round': index, 'before_cameras': before, 'after_cameras': after,
            'new_excluded_names': sorted(rejected), 'removed_points': len(removed),
            'remaining_points': model.num_points3D()})
        atomic_json(Path(output) / 'JOINT_ADMISSION_PROGRESS.json', {'rounds': rounds,
            'full_registered_denominator': full_count, 'single_pass_support_guard_passed': not original_failure})
        if after >= before:
            raise ValueError('Joint admission did not reduce the dependency graph')
        if (after < camera_policy['minimum_retained_cameras']
                or after / full_count < camera_policy['minimum_retained_fraction_of_sim3_inliers']
                or model.num_points3D() < point_policy['minimum_retained_sparse_points']):
            raise ValueError('Joint admission failed the unchanged minimum geometry or full-map coverage bound')
        support = {image.name: support_function(model, image, camera_policy)
            for image in model.images.values() if image.has_pose}
    else:
        raise ValueError('Joint admission exceeded its deterministic camera-count bound')
    if (model.num_reg_images() < camera_policy['minimum_retained_cameras']
            or model.num_reg_images() / full_count < camera_policy['minimum_retained_fraction_of_sim3_inliers']):
        raise ValueError('Joint admission cannot satisfy the frozen full-map retention bound')
    result = {'status': 'joint_camera_point_admission_stable', 'protocol': 'camera_point_admission_v2',
        'full_registered_denominator': full_count, 'retained_cameras': model.num_reg_images(),
        'retained_fraction_of_full_map': model.num_reg_images() / full_count,
        'original_single_pass_support_guard_passed': not original_failure,
        'original_single_pass_failure_not_relabelled': original_failure,
        'thresholds_unchanged': True, 'fallback_track_length_used': False,
        'minimum_full_map_retained_fraction': camera_policy['minimum_retained_fraction_of_sim3_inliers'],
        'rounds': rounds, 'new_excluded_support': excluded_support,
        'new_excluded_names': sorted(excluded_support), 'retained_points': model.num_points3D()}
    atomic_json(Path(output) / 'JOINT_CAMERA_ADMISSION.json', result)
    return support, excluded_support, removed_points


def adapt(module):
    original = module.main
    source = inspect.getsource(original)
    if source.count(AFTER) != 1:
        raise ValueError('The original post-subset support boundary changed')
    addition = '''
        support_after, _joint_excluded, _joint_removed = _joint_stabilize(model, pp,
            policy["camera_geometry_admission"], lo, hi, filter_points,
            camera_geometry_metrics, len(registered), args.output)
        support_before.update(_joint_excluded)
        excluded.update(_joint_excluded)
        keep = set(registered.values()) - excluded
        removed_after.extend(_joint_removed)
'''
    modified = source.replace(AFTER, AFTER + addition)
    if modified.count(EXTENT) != 1:
        raise ValueError('The original full-map metric extent boundary changed')
    extent_replacement = '''        pp = policy["point_filter"]
        centers = _training_candidate_extent(model, policy["camera_geometry_admission"],
            pp, camera_geometry_metrics, args.output)
        lo, hi = centers.min(0)-pp["camera_aabb_margin_m"], centers.max(0)+pp["camera_aabb_margin_m"]'''
    modified = modified.replace(EXTENT, extent_replacement)
    destination_anchor = '        pose_error, point_error = 0., 0.'
    if modified.count(destination_anchor) != 1:
        raise ValueError('The original delivered-image coordinate boundary changed')
    destination_addition = '''        keep = _destination_finalize(model, pinhole, pp,
            policy["camera_geometry_admission"], lo, hi, filter_points,
            camera_geometry_metrics, len(registered), args.output, dataset, filtered,
            rows, excluded, removed_after, receipt)
        if ({im.name for im in pinhole.images.values() if im.has_pose} != keep
                or {p.name for p in (dataset/"images").iterdir() if p.is_file()} != keep
                or set(model.points3D) != set(pinhole.points3D)):
            raise ValueError("Destination admission changed the final paired membership")
'''
    modified = modified.replace(destination_anchor, destination_addition + destination_anchor)
    source_name = '<explicit_joint_camera_point_admission_export>'
    linecache.cache[source_name] = (len(modified), None, modified.splitlines(True), source_name)
    namespace = dict(original.__globals__)
    namespace['_joint_stabilize'] = stabilize
    namespace['_destination_finalize'] = finalize_working_subset
    namespace['_training_candidate_extent'] = training_candidate_extent
    exec(compile(modified, source_name, 'exec'), namespace)
    module.main = namespace['main']
    return original


@contextmanager
def installed(stage):
    original_import = stage.importlib.import_module
    changed = []

    def load(name, *args, **kwargs):
        module = original_import(name, *args, **kwargs)
        if name == 'export_admitted_map' and not changed:
            changed.append((module, adapt(module)))
        return module

    stage.importlib.import_module = load
    try:
        yield
    finally:
        stage.importlib.import_module = original_import
        for module, original in changed:
            module.main = original
