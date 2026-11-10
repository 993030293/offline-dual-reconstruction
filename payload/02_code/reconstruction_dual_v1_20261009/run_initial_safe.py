"""Publish safe receipts and bind per-frame provenance to the actual sensor source."""

from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import sys

from common import OLD, ROOT, atomic_json, importable_path, read_json
from sensor_backend import checked_generated_source
from coarse_baseline_diversity import candidate_pair_extension


def load_native():
    reference_tools = ROOT / ('05_packages/e26_video_to_target_m3_nollm_v1r23_20260927_installed_candidate/'
        'code/reference_sources/e26_c2_support/frontend/tools')
    extended_tools = Path(importable_path(reference_tools))
    if not (extended_tools / 'camera_motion.py').is_file():
        raise FileNotFoundError('The installed raw-camera decoder is unavailable')
    sys.path.insert(0, str(extended_tools))
    spec = importlib.util.spec_from_file_location('preserved_initial_with_safe_receipts', OLD / 'run_initial.py')
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    native.baseline.TOOLS = extended_tools
    return native


@contextmanager
def pose_provenance(native, args):
    root = os.environ.get('RECON_FRESH_RAW_SENSOR_RUN_ROOT')
    if bool(root) != bool(args.coarse_trajectory):
        raise ValueError('A generated coarse trajectory requires its same-attempt sensor binding')
    if not root:
        yield
        return
    backend, _ = checked_generated_source(root, args.raw)
    expected = (backend / 'coarse/coarse_body_odometry.npz').resolve(strict=True)
    bound_root = Path(root).resolve(strict=True)
    trajectory = Path(args.coarse_trajectory).resolve(strict=True)
    trajectory.relative_to(bound_root)
    if trajectory != expected:
        handoff_path = trajectory.parent / 'SAME_ATTEMPT_HANDOFF.json'
        if not handoff_path.is_file():
            raise ValueError('Initial selection cannot use another attempt\'s or an unverified coarse trajectory')
        handoff = read_json(handoff_path)
        if (handoff['status'] != 'fresh_motion_handoff_verified'
                or Path(handoff['source']).resolve(strict=True) != expected.parent
                or Path(handoff['run_root']).resolve(strict=True) != bound_root
                or Path(handoff['source_sensor_receipt']).resolve(strict=True) != backend / 'RESULT.json'
                or Path(handoff['source_raw']).resolve(strict=True) != Path(args.raw).resolve(strict=True)
                or handoff['cross_attempt_inputs_used'] is not False
                or handoff['runtime_language_model_used'] is not False
                or handoff['coarse_solver_rerun_during_handoff'] is not False):
            raise ValueError('The generated coarse handoff has no verified same-attempt sensor lineage')
        if trajectory.read_bytes() != expected.read_bytes():
            raise ValueError('The generated coarse handoff payload differs from its actual same-attempt source')
    Path(args.output).resolve().relative_to(bound_root)
    original_at = native.baseline.CoarseCameraCurve.at
    original_many = native.coarse_camera_at_many

    def annotate(value):
        if 'C_W_m' in value:
            value.update(source_role='fresh_raw_LiDAR_IMU_coarse_estimate',
                pose_producer='raw_IMU_deskew_and_local_map_translation_ICP',
                coarse_source_run_root=str(bound_root))
        return value

    def at(curve, *positional, **keywords):
        return annotate(original_at(curve, *positional, **keywords))

    def many(*positional, **keywords):
        return [annotate(value) for value in original_many(*positional, **keywords)]

    try:
        native.baseline.CoarseCameraCurve.at = at
        native.coarse_camera_at_many = many
        yield
    finally:
        native.baseline.CoarseCameraCurve.at = original_at
        native.coarse_camera_at_many = original_many


def main():
    native = load_native()
    native.atomic_json = atomic_json
    original_run = native.run
    from initial_trackability import trackability_selection

    def run(args):
        with pose_provenance(native, args):
            with candidate_pair_extension(native, args.mode == 'optimized', native.baseline.POLICY):
                with trackability_selection(native, args):
                    return original_run(args)

    native.run = run
    native.main()


if __name__ == '__main__':
    main()
