"""Check explicit native registration routing without weakening support admission."""

import unittest
from unittest import mock

from camera_required_feature_recovery import register_with_existing_geometry


class RequiredRegistrationPolicyTests(unittest.TestCase):
    def test_native_PnP_success_does_not_replace_the_pose_with_structureless_registration(self):
        mapper, options = mock.Mock(), object()
        mapper.register_next_image.return_value = True
        ok, method, attempts = register_with_existing_geometry(mapper, options, 17)
        self.assertTrue(ok)
        self.assertEqual(method, 'native_PnP')
        self.assertEqual(attempts, [{'method': 'native_PnP', 'native_registered': True}])
        mapper.register_next_image.assert_called_once_with(options, 17)
        mapper.register_next_structure_less_image.assert_not_called()

    def test_structureless_registration_requires_a_recorded_native_PnP_rejection(self):
        mapper, options = mock.Mock(), object()
        mapper.register_next_image.return_value = False
        mapper.register_next_structure_less_image.return_value = True
        ok, method, attempts = register_with_existing_geometry(mapper, options, 19)
        self.assertTrue(ok)
        self.assertEqual(method, 'native_structureless')
        self.assertEqual(attempts, [{'method': 'native_PnP', 'native_registered': False},
                                  {'method': 'native_structureless', 'native_registered': True}])
        self.assertEqual(mapper.method_calls, [mock.call.register_next_image(options, 19),
                                             mock.call.register_next_structure_less_image(options, 19)])

    def test_both_native_rejections_are_retained_without_fabricating_a_pose(self):
        mapper, options = mock.Mock(), object()
        mapper.register_next_image.return_value = False
        mapper.register_next_structure_less_image.return_value = False
        ok, method, attempts = register_with_existing_geometry(mapper, options, 23)
        self.assertFalse(ok)
        self.assertIsNone(method)
        self.assertEqual([row['native_registered'] for row in attempts], [False, False])


if __name__ == '__main__':
    unittest.main()
