"""Bind the established target branch to this run's shared metric geometry."""

import argparse
from pathlib import Path
import re

import numpy as np
import pycolmap

from common import ROOT, HERE, atomic_json, read_json
from package_delivery import identity

PACKAGE = ROOT / '05_packages/e26_video_to_target_m3_nollm_v1r23_20260927_installed_candidate'


def prepare(run, output, before_training=False):
    config = read_json(run / 'CONFIG.json')
    shared = read_json(run / 'shared/RESULT.json')
    if not before_training:
        trained = read_json(run / 'pgsr/TRAINING_RESULT.json')
        if trained['status'] != 'completed' or trained['registered_frames'] != shared['registered_frames']:
            raise ValueError('The target branch requires this run\'s completed common PGSR input')
    model = pycolmap.Reconstruction(run / 'shared/rectified/sparse')
    names = sorted(image.name for image in model.images.values() if image.has_pose)
    if names != shared['registered_names']:
        raise ValueError('The target branch camera set changed')
    raw = Path(config['raw']).resolve(strict=True)
    if (raw.stat().st_size, raw.stat().st_mtime_ns) != (config['raw_bytes'], config['raw_mtime_ns']):
        raise ValueError('The source recording changed')
    sparse = output / 'metric_sparse'
    sparse.mkdir(parents=True, exist_ok=False)
    model.write_text(sparse)
    manifest = []
    for name in names:
        match = re.fullmatch(r'frame_(\d+)_(\d+)\.jpg', name)
        if match is None:
            raise ValueError('A common frame lacks its raw header timestamp')
        manifest.append({'image_name': name, 'camera_header_ns': int(match[2]),
                         'raw_frame_index': int(match[1])})
    manifest.sort(key=lambda row: row['camera_header_ns'])
    timestamps = np.asarray([row['camera_header_ns'] for row in manifest], np.int64)
    if not (np.diff(timestamps) > 0).all():
        raise ValueError('Selected raw timestamps are not unique and increasing')
    atomic_json(output / 'registered_image_manifest.json', manifest)
    atomic_json(output / 'CONTRACT_CORRECTION.json', {
        'status': 'same_run_target_geometry_contract_bound',
        'reason': 'The target branch requires numerical ROI, surface candidate selection, M3 ground removal and target support gates.',
        'delivery_contract': str(HERE / 'DELIVERY_CONTRACT.md'),
        'historical_Box2_output_required_at_runtime': False,
        'reference_package': str(PACKAGE),
        'reference_outputs_used_as_reconstruction_input': False,
        'same_run_geometry_and_trained_model_reused': True,
        'raw_to_shared_geometry_stage_completed': bool((run / 'stages/front/COMPLETED.json').is_file()
            and config.get('schema') == 3 and not config['historical_derived_inputs_used']),
        'trained_model_already_completed_when_bound': not before_training,
        'same_run_metric_sparse': str(sparse),
        'unchanged_registered_frames': len(names),
        'raw': str(raw), 'raw_bytes': raw.stat().st_size,
        'original_metric_BA_failure_preserved': True,
        'runtime_language_model_used': False,
        'independent_geometry_accuracy_verified': False,
    })
    model_path = run / f"pgsr/model/point_cloud/iteration_{config['pgsr_iterations']}/point_cloud.ply"
    atomic_json(output / 'INPUT_RESULT.json', {
        'status': 'same_run_target_inputs_bound', 'metric_sparse': str(sparse),
        'manifest': str(output / 'registered_image_manifest.json'),
        'frames': len(names), 'points': model.num_points3D(),
        'PGSR_model': str(model_path), 'PGSR_model_identity': None if before_training else identity(model_path),
        'training_pending_when_bound': before_training,
        'package': str(PACKAGE), 'world_frame': 'coarse_odometry_aligned_odom_meters',
        'known_object_dimensions_used': False, 'old_scene_outputs_used': False,
    })
    print('same_run_target_inputs_bound', len(names), model.num_points3D(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--before-training', action='store_true')
    args = parser.parse_args()
    prepare(args.run_root.resolve(strict=True), args.output_root.resolve(), args.before_training)
