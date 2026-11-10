"""Add raw-quality temporal bridge views around measured camera support deficits."""

import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import time
import traceback

import cv2
import numpy as np
import pycolmap

from common import ROOT, COLMAP, atomic_json, importable_path, read_json
from camera_support_recovery import model_identity, coarse_consistency
from camera_bridge_selection import BINS, select_bridges
from native_command import run_native


def run(source, initial, database_source, decisions_path, raw, output, matching_only=False):
    source, initial, database_source, decisions_path, raw = (Path(value).resolve(strict=True)
        for value in (source, initial, database_source, decisions_path, raw))
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        original = pycolmap.Reconstruction(source)
        rows = [json.loads(line) for line in (initial / 'frame_decisions.jsonl').read_text().splitlines()]
        recorded = read_json(decisions_path)
        triggers = {row['image_name'] for row in recorded['rows'] if row['status'] != 'genuine_PnP_accepted'}
        suspects = set(triggers)
        original_names = {image.name for image in original.images.values() if image.has_pose}
        initial_names = {row['image_name'] for row in read_json(initial / 'INITIAL_FRAME_MANIFEST.json')['selected_frames']}
        missing_initial = initial_names - original_names
        critical = missing_initial | {image.name for image in original.images.values()
            if image.has_pose and image.num_points3D == 0}
        triggers.update(critical)
        additions, selection, coverage = select_bridges(rows, original_names, triggers, critical_triggers=critical)
        atomic_json(output / 'CONTRACT.json', {'source_model': str(source), 'source_identity': model_identity(source),
            'raw': str(raw), 'raw_bytes': raw.stat().st_size, 'raw_mtime_ns': raw.stat().st_mtime_ns,
            'measured_deficit_decisions': str(decisions_path), 'triggers': sorted(triggers),
            'temporarily_excluded_weak_cameras': sorted(suspects), 'original_images_removed_from_final_delivery': [],
            'required_initial_recovery_names': sorted(missing_initial),
            'required_camera_support_priority_names': sorted(critical),
            'required_recovery_bins_prioritized_within_unchanged_global_budget': True,
            'selection_rule': coverage['algorithm'], 'coverage_budget_decision': coverage,
            'bins_seconds': BINS, 'maximum_new_images': 64, 'selection': selection,
            'additions': list(additions.values()), 'unchanged_raw_quality_policy': True,
            'all_original_frames_required_after_subsequent_genuine_PnP': True,
            'GT_used': False, 'coarse_poses_inserted_as_visual_poses': False})
        if not additions or len(additions) > 64:
            raise ValueError('Temporal support recovery requires 1..64 actual unchanged-quality raw additions')
        private = pycolmap.Reconstruction(original)
        for image in original.images.values():
            if image.has_pose and image.name in suspects:
                private.deregister_frame(image.frame_id)
        private_path = output / 'trusted_source'
        private_path.mkdir()
        private.write(private_path)
        images = output / 'images'
        images.mkdir()
        source_images = source.parent / 'images'
        if not source_images.is_dir():
            source_images = initial / 'images'
        for name in sorted(original_names):
            os.link(source_images / name, images / name)
        for name in sorted(missing_initial):
            os.link(initial / 'images' / name, images / name)
        sys.path.insert(0, str(ROOT / '02_code/box2_initial_pose_quality_v2_20261002'))
        baseline = importlib.import_module('initial_selection')
        baseline.TOOLS = Path(importable_path(baseline.TOOLS))
        sys.path.insert(0, str(baseline.TOOLS))
        from mcap_audit import McapIndex
        from camera_motion import decode_bayer
        from color_contract import demosaic_rggb8_to_bgr
        cv2.setNumThreads(1)
        wanted = {row['camera_frame']: row for row in additions.values()}
        for index, message in McapIndex(raw).iter_selected_messages(baseline.CAMERA_TOPIC, wanted):
            stamp, _, pixels = decode_bayer(message['payload'])
            row = wanted[index]
            if int(stamp) != row['camera_header_ns']:
                raise ValueError('Raw temporal support timestamp changed')
            ok, encoded = cv2.imencode('.jpg', demosaic_rggb8_to_bgr(pixels),
                [cv2.IMWRITE_JPEG_QUALITY, baseline.POLICY['jpeg_quality']])
            if not ok:
                raise RuntimeError('Raw support image encoding failed')
            (images / row['image_name']).write_bytes(encoded.tobytes())
        if {path.name for path in images.iterdir()} != original_names | missing_initial | additions.keys():
            raise ValueError('Raw support image export is incomplete')
        database = output / 'database.db'
        with sqlite3.connect('file:' + database_source.as_posix() + '?mode=ro', uri=True) as incoming:
            with sqlite3.connect(database) as destination:
                incoming.backup(destination)
            cameras = [row[0] for row in incoming.execute('SELECT camera_id FROM cameras')]
        if len(cameras) != 1:
            raise ValueError('The support bridge requires its verified single fixed optical camera')
        new_list = output / 'new_images.txt'
        new_list.write_text('\n'.join(sorted(additions)) + '\n', encoding='utf-8')
        all_names = sorted(original_names | missing_initial | additions.keys())
        pairs = output / 'new_pairs.txt'
        pairs.write_text(''.join(a + ' ' + b + '\n' for i, a in enumerate(all_names) for b in all_names[i+1:]
            if a in additions or b in additions), encoding='utf-8')
        allowed = output / 'mapper_images.txt'
        allowed.write_text('\n'.join(sorted((original_names - suspects) | missing_initial | additions.keys())) + '\n', encoding='utf-8')
        timings = {}

        def execute(label, command, gpu=False):
            budget = 1800 if label == 'matching' else 600
            timings[label] = run_native(command, output, label, budget, gpu,
                lease_filename=label + '_GPU_LEASE.json', lease_label='camera_support_bridge_' + label)

        execute('features', [COLMAP, 'feature_extractor', '--database_path', database, '--image_path', images,
            '--image_list_path', new_list, '--ImageReader.existing_camera_id', cameras[0],
            '--FeatureExtraction.use_gpu', '1', '--FeatureExtraction.gpu_index', '0',
            '--FeatureExtraction.num_threads', '8', '--FeatureExtraction.max_image_size', '1600',
            '--SiftExtraction.max_num_features', '8192', '--default_random_seed', '0'], True)
        execute('matching', [COLMAP, 'matches_importer', '--database_path', database,
            '--match_list_path', pairs, '--match_type', 'pairs', '--FeatureMatching.use_gpu', '1',
            '--FeatureMatching.gpu_index', '0', '--FeatureMatching.num_threads', '8',
            '--TwoViewGeometry.random_seed', '0', '--default_random_seed', '0'], True)
        if matching_only:
            result = {'status': 'raw_quality_bridges_matched_geometry_pending', 'source_model': str(source),
                'source_identity': model_identity(source), 'database': str(database), 'images': str(images),
                'support_candidates': len(additions), 'accepted_support_frames': 0,
                'coarse_or_visual_geometry_not_yet_verified': True,
                'timings': timings, 'elapsed_seconds': time.monotonic() - started}
            atomic_json(output / 'RESULT.json', result)
            return result
        models = output / 'native_models'
        models.mkdir()
        execute('mapping', [COLMAP, 'mapper', '--database_path', database, '--image_path', images,
            '--input_path', private_path, '--output_path', models, '--Mapper.image_list_path', allowed,
            '--Mapper.fix_existing_frames', '1', '--Mapper.num_threads', '8', '--Mapper.random_seed', '0',
            '--default_random_seed', '0', '--Mapper.ba_refine_focal_length', '0',
            '--Mapper.ba_refine_principal_point', '0', '--Mapper.ba_refine_extra_params', '0',
            '--Mapper.ba_refine_sensor_from_rig', '0', '--Mapper.max_runtime_seconds', '590'])
        supported = pycolmap.Reconstruction(models)
        registered = {image.name for image in supported.images.values() if image.has_pose}
        if not original_names - suspects <= registered:
            raise ValueError('Native support bridge lost a trusted original camera')
        changed = max((float(np.max(np.abs(image.cam_from_world().matrix()
            - supported.images[image.image_id].cam_from_world().matrix()))) for image in original.images.values()
            if image.name not in suspects and image.has_pose), default=0)
        if changed > 1e-9 or any(not np.array_equal(camera.params, supported.cameras[key].params)
                for key, camera in original.cameras.items()):
            raise ValueError('Temporal support bridge moved a trusted camera or intrinsics')
        errors = coarse_consistency(supported, initial)
        mapper = pycolmap.IncrementalMapperOptions()
        augmentation_decisions = []
        for name in sorted(additions):
            image = next((image for image in supported.images.values() if image.name == name), None)
            accepted = bool(image and image.has_pose and image.num_points3D >= mapper.abs_pose_min_num_inliers
                and errors.get(name, float('inf')) <= .6)
            augmentation_decisions.append({'name': name, 'status': 'genuine_visual_support' if accepted
                else 'abstain_support_candidate', 'observations': image.num_points3D if image else 0,
                'coarse_center_delta_m': errors.get(name)})
            if not accepted and image and image.has_pose:
                supported.deregister_frame(image.frame_id)
        destination = output / 'models'
        destination.mkdir()
        supported.write(destination)
        result = {'status': 'temporal_bridge_support_ready', 'source_model': str(source),
            'source_identity': model_identity(source), 'model': str(destination), 'output_identity': model_identity(destination),
            'database': str(database), 'images': str(images), 'original_required_names': sorted(original_names),
            'suspect_names': sorted(suspects), 'support_candidates': len(additions),
            'accepted_support_frames': sum(row['status'] == 'genuine_visual_support' for row in augmentation_decisions),
            'support_decisions': augmentation_decisions, 'trusted_pose_maximum_component_delta': changed,
            'timings': timings, 'elapsed_seconds': time.monotonic() - started,
            'original_camera_repair_still_pending': True}
        atomic_json(output / 'RESULT.json', result)
        return result
    except BaseException:
        atomic_json(output / 'FAILURE.json', {'status': 'failed_temporal_bridge_support',
            'traceback': traceback.format_exc(), 'elapsed_seconds': time.monotonic() - started})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-model', 'initial-frames', 'database', 'decisions', 'raw', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--matching-only', action='store_true')
    args = parser.parse_args()
    value = run(args.source_model, args.initial_frames, args.database, args.decisions, args.raw, args.output, args.matching_only)
    print(value['status'], value['support_candidates'], value['accepted_support_frames'], flush=True)
