"""Retain verified same-run PnP updates when another required camera fails."""

import numpy as np
import pycolmap


PARTIAL_PNP_PROTOCOL = 'same_run_verified_PnP_updates_retained_unverified_camera_roles_v1'


def preserve_verified_updates(original, model, decisions):
    """Restore failed source cameras as unaccepted seeds without undoing successes."""
    by_name = {row['image_name']: row for row in decisions}
    if len(by_name) != len(decisions):
        raise ValueError('Partial PnP decisions contain duplicate image names')
    original_images = {image.name: image for image in original.images.values() if image.has_pose}
    actual = {image.name: image for image in model.images.values()}
    accepted = {name for name, row in by_name.items() if row['status'] == 'genuine_PnP_accepted'}
    failed = set(by_name) - accepted
    if not accepted or not failed or not set(by_name) <= set(actual):
        raise ValueError('Partial PnP recovery needs genuine successes and unresolved actual cameras')
    minimum = pycolmap.IncrementalMapperOptions().abs_pose_min_num_inliers
    for name in accepted:
        image = actual[name]
        if not image.has_pose or image.num_points3D < minimum:
            raise ValueError('A retained PnP success lacks actual pose or native visual support')
    frozen = {name: actual[name].cam_from_world().matrix().copy() for name in accepted}
    for name in sorted(failed & set(original_images)):
        old, image = original_images[name], actual[name]
        if image.has_pose or image.num_points3D:
            raise ValueError('An unresolved PnP camera still has allegedly verified observations')
        model.frames[image.frame_id].set_cam_from_world(image.camera_id, old.cam_from_world())
        model.register_frame(image.frame_id)
        for index, point in enumerate(old.points2D):
            if point.has_point3D() and point.point3D_id in model.points3D:
                model.add_observation(point.point3D_id, pycolmap.TrackElement(image.image_id, index))
    names = {image.name for image in model.images.values() if image.has_pose}
    if not set(original_images) <= names:
        raise ValueError('Partial recovery lost an originally registered required camera')
    if any(not np.array_equal(actual[name].cam_from_world().matrix(), matrix)
           for name, matrix in frozen.items()):
        raise ValueError('Restoring unresolved camera roles undid a verified PnP pose')
    if any(not np.array_equal(camera.params, model.cameras[key].params)
           for key, camera in original.cameras.items()):
        raise ValueError('Partial recovery changed fixed calibration')
    return {'protocol': PARTIAL_PNP_PROTOCOL, 'verified_PnP_names': sorted(accepted),
        'unverified_registered_names': sorted(failed & set(original_images)),
        'unregistered_required_initial_names': sorted(failed - set(original_images)),
        'original_registered_names_all_retained': True,
        'all_unverified_cameras_require_genuine_triangulation_and_joint_BA': True,
        'coarse_pose_fabricated_as_verified_pose': False,
        'runtime_language_model_used': False, 'training_admission_granted': False}
