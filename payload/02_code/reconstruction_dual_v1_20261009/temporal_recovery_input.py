"""Check partial augmentation as a recovery input without accepting its full gate."""

from pathlib import Path

import numpy as np
import pycolmap

from common import read_json

SCOPE = 'partial_temporal_candidates_final_initial_registration_still_required'


def classify(result, contract, initial_names, before, after):
    initial_names, before, after = set(initial_names), set(before), set(after)
    if result['status'] != 'augmentation_evaluated' or result['purpose'] != 'registration':
        raise ValueError('Only a completed registration-candidate evaluation can supply a recovery input')
    if (contract['purpose'] != 'registration' or set(contract['original_images']) != initial_names
            or len(contract['original_images']) != len(initial_names)
            or contract['original_count'] != len(initial_names) or result['original_count'] != len(initial_names)
            or set(contract['registered_before']) != before or contract['old_frames_removed']
            or contract['GT_used'] or contract['old_SfM_used'] or not contract['unchanged_initial_quality_gates']
            or not result['old_input_denominator_unchanged']):
        raise ValueError('The original candidate denominator or provenance changed')
    additions = contract['additions']
    optional = {row['image_name'] for row in additions}
    if (len(optional) != len(additions) or len(optional) != result['added_count']
            or optional & (initial_names | before) or not before <= initial_names
            or any(row['decision'] != 'eligible_not_selected' or not row['coarse_pose_valid']
                or row.get('excluded_role') is not None for row in additions)):
        raise ValueError('An optional registration candidate violated its existing selection role')
    if not before <= after:
        raise ValueError('Partial augmentation lost an existing registered camera')
    if not after <= initial_names | optional:
        raise ValueError('Partial augmentation inserted an undeclared camera')
    best = max(result['models'], key=lambda row: row['registered_count'])
    missing, rejected = initial_names - after, optional - after
    registered_additions = optional & after
    if (best['registered_count'] != len(after) or best['original_registered_count'] != len(before)
            or best['original_registered_frames_lost']
            or set(best['additions_registered']) != registered_additions
            or len(best['additions_registered']) != len(registered_additions)
            or set(best['original_failures_remaining']) != missing
            or set(best['remaining_failures']) != missing | rejected
            or result['original_registration_guard_passed'] != (not missing)
            or (missing and result['counted'])):
        raise ValueError('Native registration accounting differs from the actual model')
    return {'status': 'partial_registered_candidate_input_full_original_gate_failed' if missing
            else 'all_initial_registered_candidate_input_not_training_admission',
        'scope': SCOPE if missing else 'complete_original_registration',
        'original_initial_count': len(initial_names), 'original_registered_before': len(before),
        'actual_registered_input_count': len(after), 'registered_optional_names': sorted(registered_additions),
        'unregistered_optional_names': sorted(rejected), 'required_initial_still_unregistered': sorted(missing),
        'original_registration_guard_passed': not missing, 'original_failed_gate_relabelled_as_passed': False,
        'all_original_registered_cameras_retained': True, 'initial_names_removed': [],
        'final_complete_initial_registration_and_geometry_checks_still_required': True,
        'training_or_delivery_authorized': False, 'LLM_or_manual_recovery_used': False}


def verify(initial, recovery, output, result):
    initial, recovery, output = (Path(path).resolve(strict=True) for path in (initial, recovery, output))
    side = initial.parent
    if recovery.parent != side or output.parent != side or output.name != 'temporal':
        raise ValueError('A recovery input must belong to the same temporal stage and run')
    config = read_json(side.parent.parent / 'CONFIG.json')
    if config['schema'] != 3 or config['case'] != side.parent.name:
        raise ValueError('Only the current dual-delivery DAG may defer candidate evaluation to final recovery')
    manifest = read_json(initial / 'INITIAL_FRAME_MANIFEST.json')
    initial_names = {row['image_name'] for row in manifest['selected_frames']}
    if len(initial_names) != manifest['receipt']['selected_count']:
        raise ValueError('The frozen initial manifest has duplicate or inconsistent subjects')
    contract = read_json(output / 'CONTRACT.json')
    if Path(contract['source_recovery_receipt']['path']).resolve() != recovery / 'RESULT.json':
        raise ValueError('Temporal evaluation used a different recovery source')
    source_result = read_json(recovery / 'RESULT.json')
    original_row = max(source_result['models'], key=lambda row: row['registered_count'])
    best = max(result['models'], key=lambda row: row['registered_count'])
    source_path, model_path = (Path(row['path']).resolve(strict=True) for row in (original_row, best))
    source_path.relative_to(recovery)
    model_path.relative_to(output)
    original, actual = pycolmap.Reconstruction(source_path), pycolmap.Reconstruction(model_path)
    before = {image.name for image in original.images.values() if image.has_pose}
    after = {image.name for image in actual.images.values() if image.has_pose}
    if original_row['registered_count'] != len(before):
        raise ValueError('The retained recovery model differs from its native receipt')
    if set(original.cameras) != set(actual.cameras) or any(
            not np.array_equal(camera.params, actual.cameras[key].params)
            or camera.model_name != actual.cameras[key].model_name
            or (camera.width, camera.height) != (actual.cameras[key].width, actual.cameras[key].height)
            for key, camera in original.cameras.items()):
        raise ValueError('Temporal recovery changed the existing fixed optical cameras')
    value = classify(result, contract, initial_names, before, after)
    images = Path(result['image_root']).resolve(strict=True)
    images.relative_to(output)
    for image in actual.images.values():
        if image.has_pose and (not np.isfinite(image.cam_from_world().matrix()).all()
                or not (images / image.name).is_file()
                or (image.name in value['registered_optional_names'] and image.num_points3D == 0)):
            raise ValueError('A registered input camera has no actual image, finite pose, or optional visual support')
    if not actual.num_points3D() or any(not np.isfinite(point.xyz).all() for point in actual.points3D.values()):
        raise ValueError('Temporal recovery has no finite genuine map support')
    value.update(source_model=str(source_path), input_model=str(model_path),
        actual_points=actual.num_points3D(), fixed_intrinsics_unchanged=True,
        initial_manifest=str(initial / 'INITIAL_FRAME_MANIFEST.json'),
        native_result=str(output / 'RESULT.json'))
    return value
