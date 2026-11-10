"""Keep partial original registration failures distinct from pose-gap completion."""

import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / '02_code/gaussian_box1_time20_v1_20261009'))
from run_native_sparse import validate_registration_guards


class SparseRecoveryScopeTests(unittest.TestCase):
    def setUp(self):
        self.result = {'purpose': 'pose_gap', 'original_registration_guard_passed': False,
            'pose_gap_registration_guard_passed': True, 'added_count': 1, 'counted': False,
            'models': [{'registered_count': 101, 'original_registered_count': 100,
                'original_registered_frames_lost': [], 'additions_registered': ['new.jpg']}]}

    def test_complete_pose_gap_does_not_relabel_original_failures(self):
        original = copy.deepcopy(self.result)
        self.assertEqual(validate_registration_guards(self.result),
            'pose_gap_only_with_existing_registered_cameras_preserved')
        self.assertEqual(self.result, original)

    def test_failed_pose_gap_is_rejected(self):
        self.result['original_registration_guard_passed'] = True
        self.result['pose_gap_registration_guard_passed'] = False
        with self.assertRaisesRegex(RuntimeError, 'every selected pose-gap'):
            validate_registration_guards(self.result)

    def test_lost_existing_camera_is_rejected(self):
        self.result['models'][0]['original_registered_frames_lost'] = ['old.jpg']
        with self.assertRaisesRegex(RuntimeError, 'lost a registered camera'):
            validate_registration_guards(self.result)

    def test_missing_additional_camera_is_rejected(self):
        self.result['models'][0]['additions_registered'] = []
        with self.assertRaisesRegex(RuntimeError, 'selected additional image'):
            validate_registration_guards(self.result)

    def test_other_augmentation_keeps_the_complete_original_gate(self):
        self.result['purpose'] = 'temporal'
        with self.assertRaisesRegex(RuntimeError, 'complete original'):
            validate_registration_guards(self.result)
        self.result['original_registration_guard_passed'] = True
        self.assertEqual(validate_registration_guards(self.result), 'complete_original_registration')


if __name__ == '__main__':
    unittest.main()
