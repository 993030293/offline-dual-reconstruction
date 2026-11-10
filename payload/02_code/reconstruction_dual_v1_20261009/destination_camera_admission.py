"""Stabilize original camera support in both distorted and delivered image spaces."""

from pathlib import Path
import json
import shutil

import numpy as np

from atomic_json_windows import atomic_json


def registered(model):
    return {image.name: image for image in model.images.values() if image.has_pose}


def coordinate_snapshot(model):
    return ({name: image.cam_from_world().matrix().copy() for name, image in registered(model).items()},
            {key: point.xyz.copy() for key, point in model.points3D.items()},
            {key: (camera.model_name, camera.width, camera.height, camera.params.copy())
             for key, camera in model.cameras.items()})


def unchanged_coordinates(model, snapshot):
    poses, points, cameras = snapshot
    for name, image in registered(model).items():
        if name not in poses or np.max(np.abs(image.cam_from_world().matrix() - poses[name])) > 1e-9:
            raise ValueError('Destination admission changed an original camera pose')
    for key, point in model.points3D.items():
        if key not in points or np.linalg.norm(point.xyz - points[key]) > 1e-9:
            raise ValueError('Destination admission changed or invented a three-dimensional point')
    if set(model.cameras) != set(cameras):
        raise ValueError('Destination admission changed the calibration identities')
    for key, camera in model.cameras.items():
        name, width, height, params = cameras[key]
        if (camera.model_name != name or camera.width != width or camera.height != height
                or not np.array_equal(camera.params, params)):
            raise ValueError('Destination admission changed the delivered calibration')


def paired_membership(source, destination):
    left, right = registered(source), registered(destination)
    if set(left) != set(right) or set(source.points3D) != set(destination.points3D):
        raise ValueError('Paired destination admission requires equal camera and point identities')
    for name, image in left.items():
        peer = right[name]
        if (image.image_id != peer.image_id or image.frame_id != peer.frame_id
                or image.camera_id != peer.camera_id or len(image.points2D) != len(peer.points2D)
                or np.max(np.abs(image.cam_from_world().matrix() - peer.cam_from_world().matrix())) > 1e-9
                or any(point.point3D_id != other.point3D_id for point, other in zip(image.points2D, peer.points2D))):
            raise ValueError('Paired destination admission changed pose or observation identities')
    if any(np.linalg.norm(point.xyz - destination.points3D[key].xyz) > 1e-9
           for key, point in source.points3D.items()):
        raise ValueError('Undistortion changed original three-dimensional coordinates')


def bounds_valid(model, point_policy, camera_policy, full_count):
    if (not isinstance(full_count, int) or full_count <= 0 or full_count < model.num_reg_images()
            or model.num_reg_images() < camera_policy['minimum_retained_cameras']
            or model.num_reg_images() / full_count < camera_policy['minimum_retained_fraction_of_sim3_inliers']
            or model.num_points3D() < point_policy['minimum_retained_sparse_points']):
        raise ValueError('Destination admission failed original full-map retention or point bounds')


def stabilize(source, destination, point_policy, camera_policy, lo, hi, filter_points,
              support_function, full_count, output):
    """Prune a paired working subset monotonically without changing original thresholds."""
    output = Path(output)
    paired_membership(source, destination)
    snapshots = [coordinate_snapshot(model) for model in (source, destination)]
    bounds_valid(source, point_policy, camera_policy, full_count)
    original_count = source.num_reg_images()
    excluded, removed_points, rounds = {}, [], []
    initial_rejection = None
    for index in range(full_count + 1):
        supports = [{name: support_function(model, image, camera_policy)
            for name, image in registered(model).items()} for model in (source, destination)]
        rejected = {name for name in supports[0]
                    if not supports[0][name]['passed'] or not supports[1][name]['passed']}
        if initial_rejection is None:
            initial_rejection = {name: {'source': supports[0][name], 'destination': supports[1][name]}
                                 for name in sorted(rejected)}
            if rejected:
                atomic_json(output / 'ORIGINAL_DESTINATION_SUPPORT_REJECTION.json', {
                    'status': 'original_final_image_support_rejected_not_relabelled',
                    'full_registered_denominator': full_count, 'actual_candidate_cameras': original_count,
                    'original_camera_policy': camera_policy, 'rejected_support': initial_rejection,
                    'original_point_policy': point_policy, 'thresholds_relaxed': False})
                source_path = output / 'source_before_destination_admission'
                destination_path = output / 'pinhole_before_destination_admission'
                source_path.mkdir(exist_ok=False)
                destination_path.mkdir(exist_ok=False)
                source.write(source_path)
                destination.write(destination_path)
        if not rejected:
            break
        before = source.num_reg_images()
        for model in (source, destination):
            images = registered(model)
            frames = {images[name].frame_id for name in rejected}
            if any(image.frame_id in frames and name not in rejected for name, image in images.items()):
                raise ValueError('A rig frame would remove a camera that passed both image-space gates')
            for frame in sorted(frames):
                model.deregister_frame(frame)
        excluded.update({name: {'source': supports[0][name], 'destination': supports[1][name]}
                         for name in sorted(rejected)})
        removed = filter_points(source, point_policy, lo, hi)
        removed_points.extend(removed)
        for key in sorted(set(destination.points3D) - set(source.points3D)):
            destination.delete_point3D(key)
        paired_membership(source, destination)
        after = source.num_reg_images()
        if after >= before:
            raise ValueError('Destination admission did not reduce the original dependency graph')
        rounds.append({'round': index, 'before_cameras': before, 'after_cameras': after,
            'new_excluded_names': sorted(rejected), 'removed_points': len(removed),
            'remaining_points': source.num_points3D()})
        atomic_json(output / 'DESTINATION_ADMISSION_PROGRESS.json', {'rounds': rounds,
            'full_registered_denominator': full_count, 'original_candidate_rejection': initial_rejection})
        bounds_valid(source, point_policy, camera_policy, full_count)
    else:
        raise ValueError('Destination admission exceeded its deterministic camera-count bound')
    for model, snapshot in zip((source, destination), snapshots):
        unchanged_coordinates(model, snapshot)
    paired_membership(source, destination)
    result = {'status': 'source_and_destination_camera_point_admission_stable',
        'protocol': 'camera_point_destination_admission_v3', 'used': bool(initial_rejection),
        'full_registered_denominator': full_count, 'candidate_before_destination_cameras': original_count,
        'retained_cameras': source.num_reg_images(), 'retained_points': source.num_points3D(),
        'retained_observations': source.compute_num_observations(),
        'retained_fraction_of_full_map': source.num_reg_images() / full_count,
        'minimum_full_map_retained_fraction': camera_policy['minimum_retained_fraction_of_sim3_inliers'],
        'original_camera_policy': camera_policy, 'original_point_policy': point_policy,
        'original_final_support_guard_passed': not initial_rejection,
        'original_support_failure_not_relabelled': bool(initial_rejection),
        'original_candidate_rejection': initial_rejection, 'new_excluded_support': excluded,
        'new_excluded_names': sorted(excluded), 'rounds': rounds,
        'thresholds_unchanged': True, 'complete_registered_denominator_unchanged': True,
        'full_metric_model_modified': False, 'camera_poses_or_point_coordinates_modified': False,
        'source_or_delivered_calibration_modified': False, 'new_observations_invented': False,
        'source_and_destination_support_passed': True, 'paired_observation_identity_verified': True,
        'training_admission_granted': False, 'runtime_language_model_used': False}
    atomic_json(output / 'DESTINATION_JOINT_ADMISSION.json', result)
    return supports, excluded, removed_points, result


def finalize_working_subset(source, destination, point_policy, camera_policy, lo, hi,
                            filter_points, support_function, full_count, output, dataset, filtered,
                            rows, excluded_names, removed_after, receipt):
    """Persist only a verified paired working subset and retain rejected image artifacts."""
    supports, rejected, removed, result = stabilize(source, destination, point_policy, camera_policy,
        lo, hi, filter_points, support_function, full_count, output)
    excluded_names.update(rejected)
    removed_after.extend(removed)
    keep = set(registered(source))
    for row in rows:
        name = row['image_name']
        row['training_role'] = 'admitted' if name in keep else 'excluded_insufficient_support'
        row['support_in_training_subset'] = supports[0].get(name)
        row['support_in_delivered_pinhole_subset'] = supports[1].get(name)
        if name in rejected:
            row['destination_representation_support_rejection'] = rejected[name]
    if rejected:
        dataset, output = Path(dataset).resolve(strict=True), Path(output).resolve(strict=True)
        dataset.relative_to(output)
        image_root = (dataset / 'images').resolve(strict=True)
        retained = output / 'excluded_undistorted_images'
        retained.mkdir(exist_ok=False)
        for name in sorted(rejected):
            image = (image_root / name).resolve(strict=True)
            image.relative_to(image_root)
            if image.parent != image_root or Path(name).name != name:
                raise ValueError('Excluded image is outside the owned export image directory')
            target = retained / name
            if target.exists():
                raise FileExistsError(target)
            shutil.move(str(image), str(target))
        source.write(Path(filtered))
        destination.write(dataset / 'sparse')
        filter_path = Path(output) / 'POINT_FILTER.json'
        point_filter = json.loads(filter_path.read_text(encoding='utf-8'))
        atomic_json(Path(output) / 'POINT_FILTER_BEFORE_DESTINATION_ADMISSION.json', point_filter)
        point_filter['after_subset'] = removed_after
        point_filter['destination_admission_removed_points'] = removed
        atomic_json(filter_path, point_filter)
    receipt.update(training_count=len(keep), excluded_names=sorted(excluded_names), rows=rows,
        training_points=source.num_points3D(), training_observations=source.compute_num_observations(),
        destination_joint_admission=result)
    atomic_json(Path(output) / 'TRAINING_ADMISSION.json', receipt)
    return keep
