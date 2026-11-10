"""Regress sparse-invisible corner loss and class-independent surface protection."""

import unittest

import numpy as np

from surface_roi_policy import choose_preserving_candidate, protected_triangles

POLICY = {"maximum_signed_height_m": .04, "minimum_normal_cosine": .65,
          "unconditional_low_artifact_maximum_height_m": .015}


class ObservedSurfacePreservationTests(unittest.TestCase):
    def protect(self, vertices, normal=(0., 0., 1.), offset=0.):
        return protected_triangles(vertices, [[0, 1, 2]], normal, offset, POLICY)[0][0]

    def test_elevated_textureless_surface_is_protected(self):
        self.assertTrue(self.protect([[0, 0, .2], [1, 0, .2], [0, 1, .2]]))

    def test_contact_side_below_ground_height_ceiling_is_protected(self):
        self.assertTrue(self.protect([[0, 0, 0], [1, 0, 0], [0, 0, .03]]))

    def test_ground_parallel_low_surface_can_be_trimmed(self):
        self.assertFalse(self.protect([[0, 0, .02], [1, 0, .02], [0, 1, .02]]))

    def test_unconditionally_low_artifact_is_not_protected(self):
        self.assertFalse(self.protect([[0, 0, 0], [1, 0, 0], [0, 0, .01]]))

    def test_floor_rotation_and_offset_preserve_the_same_decision(self):
        self.assertTrue(self.protect([[0, 2.2, 0], [1, 2.2, 0], [0, 2.2, 1]], (0., 1., 0.), -2.))

    def test_nonfinite_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            self.protect([[0, 0, .2], [float("nan"), 0, .2], [0, 1, .2]])

    def test_nonunit_floor_is_rejected(self):
        with self.assertRaises(ValueError):
            self.protect([[0, 0, .2], [1, 0, .2], [0, 1, .2]], (0., 0., 2.))

    def test_invalid_triangle_indices_are_rejected(self):
        with self.assertRaises(ValueError):
            protected_triangles([[0, 0, 0]], [[0, 1, 2]], [0, 0, 1], 0, POLICY)

    def test_sparse_support_alone_cannot_admit_a_missing_corner(self):
        rows = [{"padding_m": .03, "protected_triangles_lost": 1, "absolute_support_drop": 0.},
                {"padding_m": .06, "protected_triangles_lost": 0, "absolute_support_drop": 0.}]
        self.assertEqual(choose_preserving_candidate(rows, .002)["padding_m"], .06)

    def test_original_sparse_support_loss_bound_is_retained(self):
        rows = [{"padding_m": .03, "protected_triangles_lost": 0, "absolute_support_drop": .003},
                {"padding_m": .06, "protected_triangles_lost": 0, "absolute_support_drop": 0.}]
        self.assertEqual(choose_preserving_candidate(rows, .002)["padding_m"], .06)

    def test_absence_of_an_eligible_candidate_is_a_failure(self):
        with self.assertRaises(ValueError):
            choose_preserving_candidate([{"protected_triangles_lost": 1, "absolute_support_drop": 0.}], .002)

    def test_invalid_support_evidence_is_rejected(self):
        with self.assertRaises(ValueError):
            choose_preserving_candidate([{"protected_triangles_lost": 0, "absolute_support_drop": float("nan")}], .002)


if __name__ == "__main__":
    unittest.main()
