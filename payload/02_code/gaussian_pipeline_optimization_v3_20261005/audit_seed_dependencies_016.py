"""Record native source dependencies and actual point metadata without CUDA."""

import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil

import numpy as np
from plyfile import PlyData

ROOT = Path('/mnt/c/lidar-camera-fusion-mask-robust-downsampling')
CODE = ROOT / '02_code/gaussian_pipeline_optimization_v3_20261005'
OUT = ROOT / '04_outputs/gaussian_pipeline_optimization_v3_20261005/oup016/source0'


def identity(path):
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    runtime = ROOT / '04_outputs/box2_3dgs_display_v1_20260930/source/gaussian-splatting'
    knn = Path('/opt/e26_thin_surface_v1/2dgs/submodules/simple-knn')
    paths = [Path(__file__), runtime / 'train.py', runtime / 'scene/gaussian_model.py',
             runtime / 'scene/__init__.py', runtime / 'scene/dataset_readers.py',
             ROOT / '02_code/gaussian_pipeline_optimization_v1_20261005/checkpoint_adapter.py',
             *[knn / name for name in ('spatial.cu', 'simple_knn.cu', 'ext.cpp', 'LICENSE.md')]]
    ids = [identity(path) for path in paths]
    archive = OUT / 'source_snapshot'
    archive.mkdir()
    for index, path in enumerate(paths):
        shutil.copyfile(path, archive / (f'{index:03d}_' + path.name))
    symbols = {}
    for path in paths:
        if path.suffix == '.py':
            tree = ast.parse(path.read_text(encoding='utf-8'))
            symbols[str(path)] = [{'name': node.name, 'line': node.lineno, 'end_line': node.end_lineno}
                                 for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                                 and node.name in ('training', 'create_from_pcd', 'restore', 'training_setup',
                                                   'capture', '__init__', 'prepare', 'restore_loop', 'fetchPly')]
    ply = ROOT / '04_outputs/gaussian_pipeline_optimization_v1_20261005/new_map_export_003/dataset/sparse/0/points3D.ply'
    before = identity(ply)
    vertices = PlyData.read(ply)['vertex']
    points = np.vstack([vertices[key] for key in ('x', 'y', 'z')]).T
    colors = np.vstack([vertices[key] for key in ('red', 'green', 'blue')]).T / 255.0
    metadata = {name: {'shape': list(value.shape), 'dtype': str(value.dtype), 'strides': list(value.strides),
                       'all_finite': bool(np.isfinite(value).all())}
                for name, value in (('points', points), ('colors', colors))}
    train = (runtime / 'train.py').read_text(encoding='utf-8')
    adapter = paths[5].read_text(encoding='utf-8')
    model = (runtime / 'scene/gaussian_model.py').read_text(encoding='utf-8')
    checks = {'original_training_setup_occurs_between_Scene_and_restore':
              train.index('scene = Scene(dataset, gaussians)') < train.index('gaussians.training_setup(opt)')
              < train.index('gaussians.restore(model_params, opt)'),
              'prepare_sets_exposure_before_restore_and_keeps_independent_load':
              adapter.index('gaussians._exposure = state["exposure"]') < adapter.index('gaussians.restore(state["gaussian_capture"], options)')
              < adapter.index('independent = torch.load'),
              'native_initializer_contains_host_zero_feature_upload':
              model.count('features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()') == 1,
              'existing_R157_is_same_zero_feature_family': (CODE / 'native_feature_zero_v2.py').is_file(),
              'original_points_finite_metadata_confirmed': points.shape == (75767, 3) and metadata['points']['all_finite'] and metadata['colors']['all_finite'],
              'all_sources_and_point_file_unchanged': ids == [identity(path) for path in paths] and before == identity(ply)}
    result = {'status': 'source_audit_passed' if all(checks.values()) else 'failed', 'checks': checks,
              'sources': ids, 'symbols': symbols, 'point_file': before, 'point_metadata': metadata,
              'installed_knn_direct_url': json.loads(importlib.metadata.distribution('simple_knn').read_text('direct_url.json')),
              'no_Torch_or_CUDA_import_or_reconstruction': True, 'new_optimization_count': 0,
              'next_action': 'R157_V7_complete_missing_validation_no_R169_duplicate'}
    (OUT / 'RESULT.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'status': result['status'], 'checks': checks, 'point_metadata': metadata}))
    if not all(checks.values()):
        raise AssertionError('A native source dependency assertion failed')


if __name__ == '__main__':
    main()
