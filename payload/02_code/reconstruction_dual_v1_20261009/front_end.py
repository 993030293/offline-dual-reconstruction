"""Bind tested global registration and sparse augmentation to the raw workflow."""

import importlib.util
import os
import subprocess
import sys

from common import HERE, OLD, ROOT


def initial_budget(command):
    values = list(map(str, command))
    if values.count('--maximum-seconds') != 1:
        raise ValueError('The initial stage must have one owned process budget')
    index = values.index('--maximum-seconds') + 1
    if values[index] != '1200':
        raise ValueError('The preserved initial stage budget anchor changed')
    values[index] = '3000'
    return values


def temporal_scope(command):
    values = list(map(str, command))
    if ('--purpose' in values and values[values.index('--purpose') + 1] == 'registration'
            and '--mode' in values and values[values.index('--mode') + 1] == 'optimized'):
        values.append('--recovery-input-only')
    return values


def main():
    spec = importlib.util.spec_from_file_location("preserved_raw_frontend", OLD / "run_side.py")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    from pose_gap_support import bind
    native.select_pose_gap_candidates = bind(native.select_pose_gap_candidates)
    original = subprocess.run

    def dispatch(command, *positional, **keywords):
        values = list(map(str, command))
        fresh_root = os.environ.get('RECON_FRESH_RAW_SENSOR_RUN_ROOT')
        if fresh_root and str(OLD / 'run_coarse_c0.py') in values:
            values[values.index(str(OLD / 'run_coarse_c0.py'))] = str(HERE / 'coarse_handoff.py')
            values.extend(['--run-root', fresh_root])
        source = str(OLD / "run_initial.py")
        if source in values:
            values = initial_budget(values)
            values[values.index(source)] = str(HERE / "run_initial_safe.py")
            if fresh_root and '--coarse-trajectory' not in values:
                from pathlib import Path
                values.extend(['--coarse-trajectory', str(Path(fresh_root) / 'raw_sensor_backend/coarse/coarse_body_odometry.npz')])
        source = str(OLD / "run_sfm.py")
        if source in values:
            values[values.index(source)] = str(HERE / "run_sfm_global.py")
        source = str(OLD / "run_native_stage.py")
        if source in values and "--kind" in values and values[values.index("--kind") + 1] == "temporal":
            values[values.index(source)] = str(HERE / "run_supported_sparse.py")
            values = temporal_scope(values)
        elif source in values:
            values[values.index(source)] = str(HERE / "run_native_supported.py")
            if "--kind" in values and values[values.index("--kind") + 1] == "metric":
                index = values.index("--maximum-seconds") + 1
                if values[index] != "600":
                    raise ValueError("The preserved native metric stage budget anchor changed")
                values[index] = "9200"
        source = str(OLD / "metric_sim3_recovery.py")
        if source in values:
            values[values.index(source)] = str(HERE / "metric_sim3_checked.py")
        return original(values, *positional, **keywords)

    try:
        subprocess.run = dispatch
        native.main()
    finally:
        subprocess.run = original


if __name__ == "__main__":
    main()
