"""Check directed focus geometry independently of object shape and image identity."""

import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from target_focus import forward_focus
import target_focus


class TargetFocusTests(unittest.TestCase):
    def orbit(self, count=32):
        angle = np.arange(count) * (2 * np.pi / count)
        target = np.array([.4, -.8, .3])
        centers = target + np.column_stack((2 * np.cos(angle), 1.5 * np.sin(angle), .8 + .2 * np.sin(2 * angle)))
        directions = target - centers
        directions /= np.linalg.norm(directions, axis=1)[:, None]
        return target, centers, directions

    def test_arbitrary_orbit_recovers_the_actual_common_focus_without_mutation(self):
        target, centers, directions = self.orbit()
        before = centers.copy(), directions.copy()
        focus, _, _, witness = forward_focus(centers, directions, .01)
        np.testing.assert_allclose(focus, target, atol=1e-7)
        np.testing.assert_array_equal(centers, before[0])
        np.testing.assert_array_equal(directions, before[1])
        self.assertEqual(witness['actual_positive_count'], len(centers))

    def test_world_rotation_translation_and_camera_permutation_preserve_focus(self):
        target, centers, directions = self.orbit()
        rotation = Rotation.from_rotvec([.3, -.2, .4]).as_matrix()
        translation = np.array([2., -3., .8])
        order = np.random.default_rng(43).permutation(len(centers))
        focus, _, _, _ = forward_focus((centers @ rotation.T + translation)[order], (directions @ rotation.T)[order], .01)
        np.testing.assert_allclose(focus, rotation @ target + translation, atol=1e-7)

    def test_forward_constraint_is_active_without_reducing_the_camera_denominator(self):
        angle = np.arange(18) * (2 * np.pi / 18)
        centers = np.column_stack((np.cos(angle), np.sin(angle), np.full(18, .6)))
        directions = -centers / np.linalg.norm(centers, axis=1)[:, None]
        centers = np.vstack((centers, [-.02, 0, 0], [-.03, 0, 0]))
        directions = np.vstack((directions, [-1., 0, 0], [-1., 0, 0]))
        _, _, _, witness = forward_focus(centers, directions, .01)
        self.assertTrue(witness['constrained_refinement_used'])
        self.assertEqual(witness['camera_count'], 20)
        self.assertEqual(witness['required_positive_count'], 19)
        self.assertGreaterEqual(witness['actual_positive_count'], 19)
        self.assertGreaterEqual(witness['minimum_constrained_forward_depth_m'], .01 - 1e-12)

    def test_outward_rays_cannot_be_relabelled_as_a_valid_target(self):
        angle = np.arange(20) * (2 * np.pi / 20)
        centers = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(20)))
        with self.assertRaisesRegex(ValueError, 'intersection'):
            forward_focus(centers, centers.copy(), .01)

    def test_parallel_rays_fail_instead_of_inventing_unobservable_depth(self):
        centers = np.column_stack((np.arange(8), np.zeros(8), np.zeros(8)))
        with self.assertRaisesRegex(ValueError, 'identify'):
            forward_focus(centers, np.tile([0., 0., 1.], (8, 1)), .01)

    def test_nonfinite_or_nonunit_camera_directions_are_rejected(self):
        _, centers, directions = self.orbit()
        with self.assertRaisesRegex(ValueError, 'unit'):
            forward_focus(centers, 2 * directions, .01)
        directions[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            forward_focus(centers, directions, .01)

    def boundary_orbit(self):
        angle = np.arange(18) * (2 * np.pi / 18)
        centers = np.column_stack((np.cos(angle), np.sin(angle), np.full(18, .6)))
        directions = -centers / np.linalg.norm(centers, axis=1)[:, None]
        return np.vstack((centers, [-.02, 0, 0], [-.03, 0, 0])), np.vstack((directions, [-1., 0, 0], [-1., 0, 0]))

    def delayed_constraint_result(self, deficit):
        original = target_focus.minimize

        def numerical_result(*args, **kwargs):
            result = original(*args, **kwargs)
            if kwargs.get('method') == 'SLSQP':
                constraint = kwargs['constraints'][0]
                slack = constraint.A @ result.x - constraint.lb
                row = int(np.argmin(slack))
                result.x -= constraint.A[row] * (slack[row] + deficit)
            return result
        return numerical_result

    def test_completed_solver_roundoff_is_corrected_without_relaxing_original_guards(self):
        centers, directions = self.boundary_orbit()
        before = centers.copy(), directions.copy()
        with patch.object(target_focus, 'minimize', side_effect=self.delayed_constraint_result(1e-12)):
            _, _, _, witness = forward_focus(centers, directions, .01)
        correction = witness['numerical_constraint_correction']
        self.assertTrue(correction['used'])
        self.assertFalse(correction['original_guard_tolerance_relaxed'])
        self.assertGreaterEqual(witness['actual_positive_count'], 19)
        self.assertGreaterEqual(witness['minimum_constrained_forward_depth_m'],
            .01 - correction['original_guard_tolerance_m'])
        self.assertLessEqual(correction['actual_maximum_focus_component_shift_m'], correction['maximum_numerical_shift_m'])
        np.testing.assert_array_equal(centers, before[0])
        np.testing.assert_array_equal(directions, before[1])

    def test_actual_geometric_violation_is_not_disguised_as_numerical_roundoff(self):
        centers, directions = self.boundary_orbit()
        with patch.object(target_focus, 'minimize', side_effect=self.delayed_constraint_result(.001)):
            with self.assertRaisesRegex(ValueError, 'exceeds bounded numerical correction'):
                forward_focus(centers, directions, .01)

    def test_numerical_correction_keeps_rigid_transform_and_order_independence(self):
        centers, directions = self.boundary_orbit()
        rotation = Rotation.from_rotvec([.7, -.4, .2]).as_matrix()
        order = np.random.default_rng(81).permutation(len(centers))
        for actual_centers, actual_directions in [(centers, directions),
            ((centers @ rotation.T + [3., -2., .4])[order], (directions @ rotation.T)[order])]:
            with patch.object(target_focus, 'minimize', side_effect=self.delayed_constraint_result(1e-12)):
                _, _, _, witness = forward_focus(actual_centers, actual_directions, .01)
            self.assertTrue(witness['numerical_constraint_correction']['used'])
            self.assertEqual(witness['camera_count'], 20)
            self.assertEqual(witness['required_positive_count'], 19)
            self.assertGreaterEqual(witness['actual_positive_count'], 19)

    def test_infeasible_numerical_polish_is_not_swallowed(self):
        centers, directions = self.boundary_orbit()
        original = target_focus.linprog

        def reject_correction(objective, *args, **kwargs):
            if len(objective) == 4:
                return SimpleNamespace(success=False)
            return original(objective, *args, **kwargs)
        with patch.object(target_focus, 'minimize', side_effect=self.delayed_constraint_result(1e-12)):
            with patch.object(target_focus, 'linprog', side_effect=reject_correction):
                with self.assertRaisesRegex(ValueError, 'not feasible'):
                    forward_focus(centers, directions, .01)

    def test_ordinary_geometric_focus_does_not_claim_numerical_recovery(self):
        _, centers, directions = self.orbit()
        _, _, _, witness = forward_focus(centers, directions, .01)
        self.assertFalse(witness['numerical_constraint_correction']['used'])


if __name__ == '__main__':
    unittest.main()
