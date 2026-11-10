"""Check physical derivatives, native metric lifecycle, and strict rejection."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pycolmap

from metric_geometry_recovery import (_RECEIPT_PUBLISHER, _solve, _write_json,
    maximum_camera_radius, recover_metric_geometry)
from native_positive_depth import PositiveDepthCost
from positive_depth_point_initialization import (check_pixel_gradient, initialize_observed_point,
    verify_native_pixel_formula)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'02_code/gaussian_pipeline_optimization_v2_20261005'))
from native_observation_checks import observation_identity_native, projection_validity_native


def helpers():
    path = ROOT/'02_code/box2_lio_guided_sfm_v1_20261002/common.py'
    spec = importlib.util.spec_from_file_location('native_geometry_recovery_test_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(scale=1., translation=None, frames=4):
    options = pycolmap.SyntheticDatasetOptions()
    options.num_rigs, options.num_cameras_per_rig, options.num_frames_per_rig = 1, 1, frames
    options.num_points3D = 20
    options.camera_model_id = pycolmap.CameraModelId.OPENCV
    options.camera_params = [1280., 1270., 512., 384., .05, .001, .0003, -.0002]
    model = pycolmap.synthesize_dataset(options)
    model.transform(pycolmap.Sim3d(scale, pycolmap.Rotation3d(),
        np.zeros(3) if translation is None else np.asarray(translation)))
    rows = [{'image_id': image.image_id, 'image_name': image.name,
        'C_W_m': image.projection_center(), 'position_covariance_m2': np.eye(3)*.15**2}
        for image in model.images.values()]
    return model, rows


def metrics(model):
    errors = []
    for image in model.images.values():
        pose = image.cam_from_world()
        for point in image.points2D:
            if point.has_point3D():
                prediction = model.cameras[image.camera_id].img_from_cam(
                    pose.rotation.matrix() @ model.points3D[point.point3D_id].xyz+pose.translation)
                errors.append(float(np.linalg.norm(prediction-point.xy)))
    return {'reprojection_px': {'p95': float(np.quantile(errors, .95))}}


CHECKS = SimpleNamespace(observation_identity=observation_identity_native,
    projection_validity=projection_validity_native)


class ReceiptPublicationTest(unittest.TestCase):
    def test_existing_unique_receipt_cannot_be_overwritten_without_owned_update(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'STATE.json'
            _write_json(helpers(), path, {'status': 'first'})
            with self.assertRaises(FileExistsError):
                _write_json(helpers(), path, {'status': 'wrong'})
            self.assertEqual(json.loads(path.read_text())['status'], 'first')
            _write_json(helpers(), path, {'status': 'second'}, update=True)
            self.assertEqual(json.loads(path.read_text())['status'], 'second')

    @unittest.skipUnless(sys.platform == 'win32', 'Windows sharing semantics')
    def test_transient_windows_sharing_violation_publishes_complete_new_state(self):
        original_replace = _RECEIPT_PUBLISHER.os.replace
        attempts = []

        def replace(source, destination):
            attempts.append(destination)
            if len(attempts) == 1:
                error = PermissionError(13, 'Simulated Windows sharing violation')
                error.winerror = 32
                raise error
            return original_replace(source, destination)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'STATE.json'
            with patch.object(_RECEIPT_PUBLISHER.os, 'replace', side_effect=replace):
                _write_json(helpers(), path, {'status': 'completed'})
            self.assertEqual(len(attempts), 2)
            self.assertEqual(json.loads(path.read_text())['status'], 'completed')

    def test_unknown_publication_error_is_not_swallowed_or_old_state_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'STATE.json'
            _write_json(helpers(), path, {'status': 'old'})
            error = OSError(28, 'Simulated full disk')
            error.winerror = 112
            with patch.object(_RECEIPT_PUBLISHER.os, 'replace', side_effect=error), self.assertRaises(OSError):
                _write_json(helpers(), path, {'status': 'new'}, update=True)
            self.assertEqual(json.loads(path.read_text())['status'], 'old')
            self.assertEqual(list(Path(directory).glob('*.tmp')), [])


class PhysicalCostTest(unittest.TestCase):
    def test_nontrivial_sensor_pose_derivatives_match_finite_difference(self):
        quaternion = np.array([.2, -.1, .3, .8])
        quaternion /= np.linalg.norm(quaternion)
        sensor = pycolmap.Rigid3d(pycolmap.Rotation3d(quaternion), np.array([.2, -.3, .4]))
        cost = PositiveDepthCost(sensor, .01)
        pose = np.r_[quaternion, [.1, -.2, -10.]]
        xyz = np.array([.3, -.4, .7])
        value, dp, dx = cost.value_and_jacobians(pose, xyz)
        actual = sensor.rotation.matrix() @ (pycolmap.Rotation3d(quaternion).matrix() @ xyz+pose[4:])
        self.assertAlmostEqual(value, (.01-actual[2]-sensor.translation[2])/.01, places=9)
        variables, analytic = np.r_[pose, xyz], np.r_[dp, dx]
        numerical = []
        for axis in range(10):
            delta = np.eye(10)[axis]*1e-6
            plus, minus = variables+delta, variables-delta
            numerical.append((cost.value_and_jacobians(plus[:7], plus[7:])[0]
                - cost.value_and_jacobians(minus[:7], minus[7:])[0])/(2e-6))
        np.testing.assert_allclose(analytic, numerical, atol=1e-6, rtol=1e-6)

    def test_visible_point_has_zero_residual_and_jacobians(self):
        value, dp, dx = PositiveDepthCost(pycolmap.Rigid3d(), .01).value_and_jacobians(
            np.r_[0., 0., 0., 1., 0., 0., 0.], np.array([1., 2., 3.]))
        self.assertEqual(value, 0.)
        self.assertFalse(np.any(dp))
        self.assertFalse(np.any(dx))

    def test_invalid_physical_scale_rejected(self):
        for near in (0., -1., np.nan, np.inf):
            with self.subTest(near=near), self.assertRaises(ValueError):
                PositiveDepthCost(pycolmap.Rigid3d(), near)


class NativeRecoveryTest(unittest.TestCase):
    def test_complete_native_entry_restores_nonzero_metric_origin_and_actual_factors(self):
        model, rows = fixture(1.4, [7., -3., 2.])
        centers = {image.image_id: image.projection_center().copy() for image in model.images.values()}
        original = observation_identity_native(model)
        with tempfile.TemporaryDirectory() as directory:
            result, validity = _solve(model, rows, helpers(), CHECKS, Path(directory)/'native',
                pycolmap.LossFunctionType.CAUCHY, set(model.points3D), .01, 8)
            self.assertTrue(validity['valid'])
            self.assertTrue(result['factor_count_verified'])
            self.assertEqual(result['active_position_priors'], 4)
            self.assertEqual(result['separate_physical_depth_residuals'], 80)
            self.assertEqual(result['solution']['actual_threads_used'], 1)
            self.assertEqual(original, observation_identity_native(model))
            for key, image in model.images.items():
                np.testing.assert_allclose(image.projection_center(), centers[key], atol=1e-8, rtol=0)

    def test_point_initializer_recovers_known_genuine_geometry_without_changing_tracks(self):
        model, _ = fixture()
        key = next(iter(model.points3D))
        original_xyz = model.points3D[key].xyz.copy()
        identity = observation_identity_native(model)
        model.points3D[key].xyz = original_xyz+np.array([.1, -.2, .3])
        xyz, result = initialize_observed_point(model, key, .01)
        self.assertTrue(result['optimizer_success'])
        self.assertTrue(result['finite_positive_depth'])
        np.testing.assert_allclose(xyz, original_xyz, atol=1e-4, rtol=0)
        self.assertEqual(identity, observation_identity_native(model))

    def test_wrong_camera_model_is_explicitly_rejected(self):
        model, _ = fixture()
        camera = next(iter(model.cameras.values()))
        camera.model = pycolmap.CameraModelId.PINHOLE
        camera.params = camera.params[:4]
        with self.assertRaisesRegex(ValueError, 'OPENCV'):
            initialize_observed_point(model, next(iter(model.points3D)), .01)

    def test_far_parallax_optimum_obeys_the_tiny_positive_variable_bound(self):
        model, _ = fixture()
        key = next(iter(model.points3D))
        point = model.points3D[key]
        identity = observation_identity_native(model)
        for index, entry in enumerate(point.track.elements):
            image = model.images[entry.image_id]
            center = np.array([10.+index*.2, 0., 0.])
            model.frames[image.frame_id].rig_from_world = pycolmap.Rigid3d(
                pycolmap.Rotation3d(), -center)
            camera = model.cameras[image.camera_id]
            image.points2D[entry.point2D_idx].xy = camera.img_from_cam(
                np.array([.1+index*.01, .02, 1.]))
        point.xyz = np.array([10., 0., -4.])
        xyz, result = initialize_observed_point(model, key, .01)
        self.assertTrue(result['optimizer_success'], result['optimizer_message'])
        self.assertTrue(result['finite_positive_depth'])
        self.assertTrue(result['final_native_formula_equivalence_checked'])
        self.assertGreaterEqual(result['optimized_chart_variables'][2], result['inverse_depth_minimum'])
        self.assertLess(result['objective_value'], result['actual_physical_seed_pixel_cost'])
        self.assertEqual(identity, observation_identity_native(model))

    def test_large_distorted_seed_gradient_is_conditioned_without_dropping_observations(self):
        model, _ = fixture(frames=3)
        key = next(iter(model.points3D))
        point = model.points3D[key]
        identity = observation_identity_native(model)
        for angle, entry in zip((-1.55, 0., 1.55), point.track.elements):
            image = model.images[entry.image_id]
            rotation = pycolmap.Rotation3d(np.array([0., angle, 0.]))
            model.frames[image.frame_id].rig_from_world = pycolmap.Rigid3d(rotation, np.zeros(3))
            camera = model.cameras[image.camera_id]
            image.points2D[entry.point2D_idx].xy = camera.img_from_cam(np.array([0., 0., 1.]))
        point.xyz = np.array([0., 0., -6.])
        xyz, result = initialize_observed_point(model, key, .01)
        self.assertTrue(result['optimizer_success'], result['optimizer_message'])
        self.assertTrue(result['finite_positive_depth'])
        self.assertGreater(result['actual_physical_seed_pixel_cost'], 1e6)
        self.assertEqual(result['absolute_true_pixel_solver_tolerance'], 1e-12)
        self.assertTrue(result['final_native_formula_equivalence_checked'])
        self.assertEqual(identity, observation_identity_native(model))

    def test_native_formula_check_uses_the_same_homogeneous_camera_vectors(self):
        model, _ = fixture(frames=3)
        key = next(iter(model.points3D))
        identity = observation_identity_native(model)
        coordinates = []
        for entry, scale in zip(model.points3D[key].track.elements, (1e-9, 1., 1e9)):
            image = model.images[entry.image_id]
            ray = model.cameras[image.camera_id].cam_from_img(image.points2D[entry.point2D_idx].xy)
            coordinates.append(np.r_[ray, 1.]*scale)
        cost = verify_native_pixel_formula(model, key, coordinates, 0.)
        self.assertLess(cost, 1e-8)
        self.assertEqual(identity, observation_identity_native(model))
        with self.assertRaisesRegex(ValueError, 'OPENCV formula'):
            verify_native_pixel_formula(model, key, coordinates, 1.)

    def test_native_formula_check_rejects_missing_camera_vectors(self):
        model, _ = fixture(frames=3)
        key = next(iter(model.points3D))
        with self.assertRaisesRegex(ValueError, 'every actual observation'):
            verify_native_pixel_formula(model, key, np.ones((2, 3)), 0.)

    def test_complex_gradient_check_recovers_derivatives_hidden_by_a_large_constant(self):
        variables = np.array([1., 2., 3.])
        derivative = np.array([1000., 1., 5.])
        cost = lambda value: 1e24+np.dot(derivative, value)
        error, checks = check_pixel_gradient(variables, derivative, cost, cost)
        self.assertLess(error, 1e-10)
        self.assertEqual(len(checks), 3)
        self.assertTrue(all(row['absolute_gradient_error'] > row['unchanged_gradient_error_limit']
                            for row in checks[:2]))
        self.assertEqual(checks[-1]['method'], 'complex_step_same_pixel_formula_and_huber_branches')

    def test_complex_gradient_check_rejects_an_incorrect_analytic_derivative(self):
        variables = np.array([1., 2., 3.])
        derivative = np.array([1000., 1., 5.])
        cost = lambda value: 1e24+np.dot(derivative, value)
        with self.assertRaisesRegex(ValueError, 'analytic pixel derivative'):
            check_pixel_gradient(variables, derivative+np.array([1., 0., 0.]), cost, cost)

    def test_recovery_updates_owned_state_and_preserves_input_model(self):
        model, rows = fixture(.5, [1., -2., 3.])
        centers = {image.image_id: image.projection_center().copy() for image in model.images.values()}
        identity = observation_identity_native(model)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'recovery'
            recovered, result = recover_metric_geometry(model, rows, metrics(model)['reprojection_px']['p95'],
                output, helpers(), CHECKS, metrics, 8)
            self.assertEqual(result['status'], 'completed_original_metric_guards')
            self.assertTrue((output/'accepted_model/images.bin').is_file())
            self.assertEqual(identity, observation_identity_native(recovered))
            self.assertEqual(json.loads((output/'STATE.json').read_text())['status'], result['status'])
            for key, image in model.images.items():
                np.testing.assert_array_equal(image.projection_center(), centers[key])

    def test_world_frame_radius_guard_cannot_be_hidden_by_normalization(self):
        model, rows = fixture(.5, [100., 0., 0.])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'rejected'
            with self.assertRaisesRegex(ValueError, 'metric extent'):
                recover_metric_geometry(model, rows, 0., output, helpers(), CHECKS, metrics, 8)
            self.assertTrue((output/'FAILURE.json').is_file())
            self.assertFalse((output/'accepted_model').exists())

    def test_nonfinite_radius_is_rejected(self):
        model = SimpleNamespace(images={1: SimpleNamespace(has_pose=True,
            projection_center=lambda: np.array([np.nan, 0., 1.]))})
        with self.assertRaises(ValueError):
            maximum_camera_radius(model)

    def test_valid_projection_does_not_end_recovery_before_the_pixel_guard_passes(self):
        model, rows = fixture()
        states = iter(({'reprojection_px': {'p95': .08}}, {'reprojection_px': {'p95': .01}},
            {'reprojection_px': {'p95': .01}}))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'deferred_pixel_admission'
            recovered, result = recover_metric_geometry(model, rows, 0., output, helpers(), CHECKS,
                lambda unused: next(states), 8)
            self.assertEqual(len(result['rounds']), 2)
            self.assertFalse(result['rounds'][0]['original_admission_guards_at_this_round']['passed'])
            self.assertTrue(result['rounds'][1]['original_admission_guards_at_this_round']['passed'])
            self.assertEqual(observation_identity_native(model), observation_identity_native(recovered))

    def test_persistent_pixel_guard_failure_stops_at_the_original_three_round_budget(self):
        model, rows = fixture()
        calls = []
        def persistent_quality(unused):
            calls.append(True)
            return {'reprojection_px': {'p95': .1}}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'bounded_pixel_rejection'
            with self.assertRaisesRegex(ValueError, 'metric extent guard'):
                recover_metric_geometry(model, rows, 0., output, helpers(), CHECKS, persistent_quality, 8)
            self.assertEqual(len(list(output.glob('physical_joint_BA_*_ADMISSION_GUARDS.json'))), 3)
            self.assertTrue((output/'FAILURE.json').is_file())
            self.assertFalse((output/'accepted_model').exists())

    def test_shared_policy_import_does_not_use_the_wrong_common_module(self):
        code = ("import sys; from pathlib import Path; "
            f"sys.path.insert(0,{str(ROOT/'02_code/box2_lio_guided_sfm_v1_20261002')!r}); import common; "
            f"sys.path.insert(0,{str(Path(__file__).resolve().parent)!r}); "
            "from metric_point_policy import native_point_policy; print(native_point_policy()[3])")
        result = subprocess.run([sys.executable, '-c', code], text=True,
            capture_output=True, timeout=10, check=True)
        self.assertEqual(float(result.stdout.strip()), .01)


if __name__ == '__main__':
    unittest.main()
