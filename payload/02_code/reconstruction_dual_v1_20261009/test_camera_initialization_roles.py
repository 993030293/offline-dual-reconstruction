"""Regression checks for preserving visual evidence and required recovery scope."""

import unittest

from camera_initialization_roles import initialization_roles, observation_decisions


class InitializerRoleTests(unittest.TestCase):
    def test_unverified_metric_conflict_can_be_reinitialized_without_dropping_camera(self):
        roles = initialization_roles({'verified': 40, 'conflicted': 80}, {'verified', 'conflicted'},
            {'bridge'}, {'verified', 'conflicted', 'bridge'}, unverified_conflicted_names={'conflicted'})
        self.assertEqual(roles['visual_initializer_names'], ['verified'])
        self.assertEqual(roles['unverified_conflicted_original_names'], ['conflicted'])
        self.assertEqual(set(roles['coarse_initializer_names']), {'bridge', 'conflicted'})
        self.assertEqual(set(roles['required_names']), {'verified', 'conflicted'})

    def test_conflicted_seed_without_raw_source_or_actual_camera_is_rejected(self):
        for coarse, names in ((set(), {'view_a'}), ({'view_a'}, {'foreign'})):
            with self.subTest(names=names), self.assertRaises(ValueError):
                initialization_roles({'view_a': 40}, {'view_a'}, set(), coarse,
                    unverified_conflicted_names=names)

    def test_weak_observed_visual_pose_is_not_replaced_by_coarse_pose(self):
        roles = initialization_roles({'view_a': 15, 'view_b': 40}, {'view_a', 'view_b'}, {'bridge'},
                                     {'view_a', 'view_b', 'bridge'})
        self.assertEqual(roles['visual_initializer_names'], ['view_a', 'view_b'])
        self.assertEqual(roles['coarse_initializer_names'], ['bridge'])

    def test_initially_selected_missing_frame_is_required(self):
        roles = initialization_roles({'view_a': 40}, {'view_a', 'pending'}, {'bridge'},
                                     {'view_a', 'pending', 'bridge'})
        self.assertEqual(roles['missing_initial_names'], ['pending'])
        self.assertEqual(set(roles['required_names']), {'view_a', 'pending'})
        self.assertEqual(set(roles['coarse_initializer_names']), {'pending', 'bridge'})

    def test_observed_seed_with_zero_physical_support_is_an_unverified_initializer(self):
        roles = initialization_roles({'view_a': 15, 'view_b': 40}, {'view_a', 'view_b'}, {'bridge'},
            {'view_a', 'view_b', 'bridge'}, {'view_a': 0, 'view_b': 40})
        self.assertEqual(roles['degenerate_observed_original_names'], ['view_a'])
        self.assertEqual(roles['visual_initializer_names'], ['view_b'])
        self.assertEqual(set(roles['coarse_initializer_names']), {'view_a', 'bridge'})
        self.assertEqual(set(roles['required_names']), {'view_a', 'view_b'})

    def test_one_qualified_observation_does_not_trigger_arbitrary_support_threshold(self):
        roles = initialization_roles({'view_a': 15}, {'view_a'}, set(), {'view_a'}, {'view_a': 1})
        self.assertEqual(roles['visual_initializer_names'], ['view_a'])
        self.assertEqual(roles['coarse_initializer_names'], [])

    def test_qualified_support_cannot_invent_counts_or_omit_camera(self):
        for counts in ({}, {'view_a': -1}, {'view_a': 16}, {'view_a': float('nan')}, {'view_a': True}):
            with self.subTest(counts=counts), self.assertRaisesRegex(ValueError, 'Qualified seed support'):
                initialization_roles({'view_a': 15}, {'view_a'}, set(), {'view_a'}, counts)

    def test_degenerate_observed_seed_cannot_receive_an_invented_coarse_pose(self):
        with self.assertRaisesRegex(ValueError, 'qualified raw coarse pose'):
            initialization_roles({'view_a': 15}, {'view_a'}, set(), set(), {'view_a': 0})

    def test_missing_required_frame_cannot_receive_an_invented_pose(self):
        with self.assertRaisesRegex(ValueError, 'qualified raw coarse pose'):
            initialization_roles({'view_a': 40}, {'view_a', 'pending'}, {'bridge'}, {'view_a', 'bridge'})

    def test_zero_observation_source_needs_explicit_initializer(self):
        roles = initialization_roles({'view_a': 0, 'view_b': 40}, {'view_a'}, set(), {'view_a'})
        self.assertEqual(roles['unobserved_original_names'], ['view_a'])
        self.assertEqual(roles['coarse_initializer_names'], ['view_a'])
        with self.assertRaises(ValueError):
            initialization_roles({'view_a': 0}, {'view_a'}, set(), set())

    def test_bridge_cannot_replace_original_or_pending_initial_frame(self):
        for existing in ('view_a', 'pending'):
            with self.subTest(existing=existing), self.assertRaisesRegex(ValueError, 'cannot replace'):
                initialization_roles({'view_a': 40}, {'view_a', 'pending'}, {existing}, {'view_a', 'pending'})

    def test_zero_optional_witness_is_rejected_without_dropping_required_camera(self):
        counts = {'view_a': 15, 'pending': 1, 'bridge_a': 0, 'bridge_b': 1}
        decisions = observation_decisions(counts, {'view_a', 'pending'}, {'bridge_a', 'bridge_b'})
        self.assertEqual(decisions['rejected_optional_names'], ['bridge_a'])
        self.assertEqual(decisions['unsupported_required_names'], [])
        self.assertEqual(counts['view_a'], 15)

    def test_zero_required_camera_remains_a_failure(self):
        decisions = observation_decisions({'view_a': 0, 'bridge': 0}, {'view_a', 'pending'}, {'bridge'})
        self.assertEqual(decisions['unsupported_required_names'], ['pending', 'view_a'])
        self.assertFalse(decisions['required_frames_may_be_silently_dropped'])
        self.assertTrue(decisions['zero_observation_guard_unchanged'])

    def test_unknown_camera_or_role_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            observation_decisions({'foreign': 100}, {'view_a'}, {'bridge'})
        with self.assertRaises(ValueError):
            observation_decisions({'view_a': 100}, {'view_a'}, {'view_a'})


if __name__ == '__main__':
    unittest.main()
