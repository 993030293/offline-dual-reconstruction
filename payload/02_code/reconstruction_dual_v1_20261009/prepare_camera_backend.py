"""Prepare supported camera geometry, preserving expected recovery failures for audit."""

import argparse
import hashlib
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import pycolmap

from common import HERE, atomic_json, read_json
from camera_support_recovery import model_identity, coarse_consistency
from camera_initialization_roles import SENSOR_INITIALIZATION_PROTOCOL
from camera_visual_recovery import qualified_coarse_names


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_cache_config(stored, current):
    """Accept path aliases only when they identify the same filesystem object."""
    if stored.keys() != current.keys():
        raise ValueError('Prepared camera geometry cache configuration fields changed')
    aliases = []
    for key, value in current.items():
        if stored[key] == value:
            continue
        same_file = False
        if key in ('source_model', 'raw'):
            try:
                same_file = Path(stored[key]).samefile(value)
            except OSError:
                pass
        if not same_file:
            raise ValueError('Prepared camera geometry cache input changed: ' + key)
        aliases.append({'field': key, 'stored_path': stored[key], 'current_path': value,
            'same_filesystem_object_verified': True})
    return aliases


def require_automatic_geometry(value):
    """Keep developer-adopted results outside the production recovery path."""
    if ('adopted' in value.get('method', '')
            or any(row.get('assistant_intervention') or row.get('not_an_unassisted_cold_start')
                for row in value.get('stages', []))):
        raise ValueError('Production recovery cannot consume developer-adopted diagnostic geometry')


def prepare(source, initial, raw, output, adopted=None, verified_metric=None):
    # Reject development adoption before consulting any completed stage cache.
    if adopted is not None or verified_metric is not None:
        raise ValueError('Production camera recovery rejects manually adopted diagnostic geometry')
    source, initial, raw = (Path(value).resolve(strict=True) for value in (source, initial, raw))
    output = Path(output).resolve()
    config = {'source_model': str(source), 'source_identity': model_identity(source),
        'camera_initialization_protocol': SENSOR_INITIALIZATION_PROTOCOL,
        'raw': str(raw), 'raw_bytes': raw.stat().st_size, 'raw_mtime_ns': raw.stat().st_mtime_ns,
        'initial_manifest_identity': fingerprint(initial / 'INITIAL_FRAME_MANIFEST.json'),
        'raw_frame_decisions_identity': fingerprint(initial / 'frame_decisions.jsonl')}
    if (output / 'RESULT.json').is_file():
        value = read_json(output / 'RESULT.json')
        require_automatic_geometry(value)
        aliases = validate_cache_config(read_json(output / 'CONFIG.json'), config)
        if (value['status'] != 'same_run_camera_geometry_prepared_BA_required'
                or model_identity(Path(value['model'])) != value['output_identity']):
            raise ValueError('Prepared camera geometry cache is incomplete or changed')
        if aliases:
            atomic_json(output / ('CACHE_RESUME_' + str(time.time_ns()) + '.json'),
                {'status': 'unchanged_input_cache_path_alias_verified', 'path_aliases': aliases,
                    'all_nonpath_input_identities_unchanged': True, 'output_identity_verified': True,
                    'original_config_and_result_preserved': True})
        return value
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'CONFIG.json', config)
    started = time.monotonic()
    history = []

    def stage(label, script, arguments, budget, result, expected_failure=None):
        command = [__import__('sys').executable, str(HERE / script), *map(str, arguments)]
        before = time.monotonic()
        with (output / (label + '.log')).open('x', encoding='utf-8') as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=budget,
                creationflags=subprocess.CREATE_NO_WINDOW)
        row = {'label': label, 'command': command, 'returncode': completed.returncode,
            'seconds': time.monotonic() - before, 'expected_failure_recovery': False}
        history.append(row)
        if completed.returncode:
            if expected_failure and Path(expected_failure).is_file():
                failure = read_json(expected_failure)
                if failure['status'] == 'genuine_PnP_support_incomplete' and failure['source_identity'] == config['source_identity']:
                    row['expected_failure_recovery'] = True
                    row['failure_receipt'] = str(expected_failure)
                    atomic_json(output / 'STAGES.json', {'rows': history})
                    return None
            raise RuntimeError('Camera backend stage failed: ' + label)
        atomic_json(output / 'STAGES.json', {'rows': history})
        return read_json(result)

    try:
        original = pycolmap.Reconstruction(source)
        original_names = {image.name for image in original.images.values() if image.has_pose}
        initial_names = {row['image_name'] for row in read_json(initial / 'INITIAL_FRAME_MANIFEST.json')['selected_frames']}
        missing_initial = initial_names - original_names
        requires_real_BA, method = False, 'existing_visual_map_with_pre_BA_point_policy'
        image_root = source.parent / 'images'
        if not image_root.is_dir():
            image_root = initial / 'images'
        if adopted is None:
            role_arguments = []
            errors = coarse_consistency(original, initial)
            minimum = pycolmap.IncrementalMapperOptions().abs_pose_min_num_inliers
            weak = [image.name for image in original.images.values() if image.has_pose
                and (image.num_points3D < minimum or errors.get(image.name, 0) > .6)]
            unbound = original_names - qualified_coarse_names(initial)
            current = source
            if weak or missing_initial or unbound:
                matches = output / 'missing_matches'
                matching = stage('missing_matches', 'camera_support_recovery.py', ['--source-model', source,
                    '--initial-frames', initial, '--output', matches, '--matching-only'], 2000, matches / 'RESULT.json')
                visual_arguments = ['--database', matching['database'], '--images', matching['images']]
                use_native_recovery = bool(unbound)
                if weak or missing_initial:
                    pnp = output / 'genuine_pnp'
                    localized = stage('genuine_pnp', 'camera_support_pnp.py', ['--source-model', source,
                        '--initial-frames', initial, '--database', matching['database'], '--output', pnp], 180,
                        pnp / 'RESULT.json', pnp / 'UNVERIFIED_FRAMES.json')
                    if localized and localized.get('required_initial_names_all_registered') is True:
                        current, method = Path(localized['model']), 'genuine_fixed_map_PnP'
                    else:
                        use_native_recovery = True
                        if (pnp / 'RECOVERY_INPUT.json').is_file():
                            partial = read_json(pnp / 'RECOVERY_INPUT.json')
                            if (partial['status'] != 'same_run_partial_PnP_recovery_input_BA_required'
                                    or partial['source_identity'] != config['source_identity']
                                    or partial['initial_manifest_identity'] != config['initial_manifest_identity']
                                    or partial['training_admission_granted'] is not False
                                    or model_identity(Path(partial['model'])) != partial['output_identity']):
                                raise ValueError('Partial PnP recovery input lacks same-run identity or role evidence')
                            current = Path(partial['model'])
                        bridge = output / 'raw_quality_bridge'
                        matched_bridge = stage('raw_quality_bridge', 'camera_support_bridge.py',
                            ['--source-model', current, '--initial-frames', initial,
                                '--database', matching['database'], '--decisions', pnp / 'FRAME_DECISIONS.json',
                                '--raw', raw, '--output', bridge, '--matching-only'], 2600, bridge / 'RESULT.json')
                        visual_arguments = ['--database', matched_bridge['database'],
                            '--images', matched_bridge['images'], '--bridge', bridge]
                if use_native_recovery:
                    initialized = output / 'native_visual_recovery'
                    recovered = stage('native_visual_recovery', 'camera_visual_recovery.py',
                        ['--source-model', current, '--initial-frames', initial, '--output', initialized,
                            *visual_arguments], 630, initialized / 'RESULT.json')
                    rebuilt_path = output / 'all_point_rebuilding'
                    rebuilt = stage('all_point_rebuilding', 'camera_point_rebuild.py',
                        ['--source-model', recovered['model'], '--initial-frames', initial,
                            '--database', recovered['database'], '--images', recovered['images'],
                            '--camera-role-result', initialized / 'RESULT.json', '--output', rebuilt_path],
                        400, rebuilt_path / 'RESULT.json')
                    current, image_root = Path(rebuilt['model']), Path(rebuilt['images'])
                    role_arguments = ['--camera-role-result', rebuilt_path / 'RESULT.json']
                    requires_real_BA, method = True, 'genuine_native_visual_recovery_all_points_and_joint_BA_required'
            prepared_path = output / 'point_preparation'
            prepared = stage('point_preparation', 'prepare_metric_support.py', ['--source-model', current,
                '--initial-frames', initial, '--output', prepared_path, '--scope', 'full-training-policy', *role_arguments],
                120, prepared_path / 'RESULT.json')
            conditioned_path = output / 'joint_visual_conditioning'
            conditioned = stage('joint_visual_conditioning', 'camera_joint_conditioning.py',
                ['--source-model', prepared['model'], '--initial-frames', initial,
                    '--camera-role-result', prepared_path / 'RESULT.json', '--output', conditioned_path],
                360, conditioned_path / 'RESULT.json')
            final_path = output / 'final_point_preparation'
            prepared = stage('final_point_preparation', 'prepare_metric_support.py',
                ['--source-model', conditioned['model'], '--initial-frames', initial, '--output', final_path,
                    '--scope', 'full-training-policy', '--camera-role-result', conditioned_path / 'RESULT.json'],
                120, final_path / 'RESULT.json')
        prepared_model = Path(prepared['model'])
        actual = pycolmap.Reconstruction(prepared_model)
        actual_names = {image.name for image in actual.images.values() if image.has_pose}
        if not original_names <= actual_names or any(not (image_root / name).is_file() for name in actual_names):
            raise ValueError('Camera backend lost an original camera or its actual image')
        if not initial_names <= actual_names:
            raise ValueError('Camera backend did not recover every required initially selected frame')
        if any(not np.array_equal(camera.params, actual.cameras[key].params) for key, camera in original.cameras.items()):
            raise ValueError('Camera backend changed the fixed optical intrinsics')
        value = {'status': 'same_run_camera_geometry_prepared_BA_required', 'method': method,
            'source_model': str(source), 'source_identity': config['source_identity'], 'model': str(prepared_model),
            'output_identity': model_identity(prepared_model), 'images': str(image_root),
            'registered_frames': actual.num_reg_images(), 'original_registered_frames': original.num_reg_images(),
            'original_registered_names_all_retained': True, 'added_registered_frames': len(actual_names - original_names),
            'requires_native_BA_coarse_initialization': requires_real_BA,
            'pure_similarity_must_not_accept_unverified_coarse_initialization': requires_real_BA,
            'training_camera_admission_still_required': True, 'stages': history,
            'actual_joint_visual_conditioning_completed': True,
            'actual_variable_camera_blocks': conditioned['actual_variable_camera_blocks'],
            'metric_comparison_reference_frozen_only_after_native_visual_conditioning': True,
            'elapsed_seconds': time.monotonic() - started, 'independent_geometry_accuracy_verified': False,
            'runtime_language_model_used': False, 'manual_geometry_adoption_allowed': False,
            'recovery_decisions': 'preencoded_real_PnP_native_registration_all_point_rebuild_joint_BA_and_original_metric_gates'}
        require_automatic_geometry(value)
        atomic_json(output / 'RESULT.json', value)
        return value
    except BaseException:
        atomic_json(output / 'FAILURE.json', {'status': 'failed_camera_backend_preparation',
            'traceback': traceback.format_exc(), 'stages': history, 'elapsed_seconds': time.monotonic() - started})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-model', 'initial-frames', 'raw', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--adopt-prepared-result', type=Path)
    parser.add_argument('--verified-metric-root', type=Path)
    args = parser.parse_args()
    value = prepare(args.source_model, args.initial_frames, args.raw, args.output,
        args.adopt_prepared_result, args.verified_metric_root)
    print(value['status'], value['registered_frames'], value['method'], flush=True)
