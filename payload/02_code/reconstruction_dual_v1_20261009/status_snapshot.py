"""Read compact live stage states with Windows delete-sharing enabled."""

import argparse
import json
from pathlib import Path
import time

import psutil

from common import read_json


def snapshot(root):
    config = read_json(root / 'CONFIG.json')
    side = root / config['case'] / 'optimized'
    paths = {'controller': root / 'STATE.json', 'front': side / 'PIPELINE_STATE.json',
        'initial': side / 'initial/STATE.json', 'sfm': side / 'sfm/STATE.json',
        'gaussian': side / 'training/STATE.json', 'pgsr': root / 'pgsr/STATE.json',
        'target': root / 'target/CONTROLLER_STATE.json',
        'surface': root / 'target/07_reliable_lidar_tsdf/STATE.json'}
    keys = ('status', 'current_stage', 'iteration', 'target_iterations', 'elapsed_seconds',
        'wall_seconds', 'completed', 'total', 'selected_count', 'selected', 'eligible',
        'raw_frame_count', 'registered_frames', 'gaussian_count', 'peak_cuda_reserved_bytes',
        'rss_bytes', 'frames_rendered', 'frames_integrated', 'candidate', 'controller_elapsed_seconds')
    rows = {}
    for label, path in paths.items():
        if not path.is_file():
            continue
        value = read_json(path)
        row = {key: value[key] for key in keys if key in value}
        if label == 'sfm':
            row['stages'] = [{key: stage[key] for key in ('name', 'status', 'seconds', 'elapsed_seconds')
                if key in stage} for stage in value.get('stages', [])]
        if label == 'controller':
            row['live_total_wall_seconds_since_first_start'] = time.time() - value['first_started_unix']
            current = root / 'stages' / value['current_stage'] / 'PROCESS.json'
            if current.is_file():
                process_state = read_json(current)
                row['current_stage_process_status'] = process_state['status']
                row['current_stage_process_seconds'] = process_state.get('elapsed_seconds')
            try:
                row['owned_controller_alive'] = psutil.Process(value['pid']).create_time() == value['process_birth']
            except psutil.NoSuchProcess:
                row['owned_controller_alive'] = False
        rows[label] = row
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.run_root.resolve(strict=True)), ensure_ascii=False))
