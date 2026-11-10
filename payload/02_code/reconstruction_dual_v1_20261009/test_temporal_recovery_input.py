"""Keep candidate recovery inputs separate from final complete registration."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from common import atomic_json, read_json
from front_end import temporal_scope
import run_supported_sparse
from temporal_recovery_input import classify, SCOPE


class TemporalRecoveryInputTests(unittest.TestCase):
    def setUp(self):
        self.initial, self.before, self.after = set('abcde'), set('abc'), set('abcf')
        self.contract = {'purpose': 'registration', 'original_images': list('abcde'), 'original_count': 5,
            'registered_before': list('abc'), 'old_frames_removed': [], 'GT_used': False, 'old_SfM_used': False,
            'unchanged_initial_quality_gates': True, 'additions': [{'image_name': name,
                'decision': 'eligible_not_selected', 'coarse_pose_valid': True, 'excluded_role': None} for name in 'fg']}
        self.result = {'status': 'augmentation_evaluated', 'purpose': 'registration', 'original_count': 5,
            'old_input_denominator_unchanged': True, 'added_count': 2,
            'original_registration_guard_passed': False, 'counted': False,
            'models': [{'registered_count': 4, 'original_registered_count': 3,
                'original_registered_frames_lost': [], 'additions_registered': ['f'],
                'original_failures_remaining': list('de'), 'remaining_failures': list('deg')}]}

    def evaluate(self):
        return classify(self.result, self.contract, self.initial, self.before, self.after)

    def test_partial_input_keeps_required_failures_and_optional_abstentions(self):
        original = copy.deepcopy((self.result, self.contract))
        value = self.evaluate()
        self.assertEqual(value['scope'], SCOPE)
        self.assertEqual(value['required_initial_still_unregistered'], list('de'))
        self.assertEqual(value['unregistered_optional_names'], ['g'])
        self.assertEqual(value['registered_optional_names'], ['f'])
        self.assertFalse(value['original_registration_guard_passed'])
        self.assertFalse(value['training_or_delivery_authorized'])
        self.assertEqual((self.result, self.contract), original)

    def test_existing_registered_camera_cannot_be_lost(self):
        self.after.remove('a')
        with self.assertRaisesRegex(ValueError, 'lost an existing'):
            self.evaluate()

    def test_undeclared_camera_cannot_be_inserted(self):
        self.after.add('h')
        with self.assertRaisesRegex(ValueError, 'undeclared'):
            self.evaluate()

    def test_partial_original_gate_cannot_be_relabelled_as_passed(self):
        self.result['original_registration_guard_passed'] = True
        with self.assertRaisesRegex(ValueError, 'accounting'):
            self.evaluate()

    def test_partial_result_cannot_be_counted_as_accepted(self):
        self.result['counted'] = True
        with self.assertRaisesRegex(ValueError, 'accounting'):
            self.evaluate()

    def test_required_initial_cannot_be_disguised_as_optional(self):
        self.contract['additions'][0]['image_name'] = 'd'
        with self.assertRaisesRegex(ValueError, 'selection role'):
            self.evaluate()

    def test_original_denominator_and_eligibility_remain_required(self):
        self.contract['original_count'] = 3
        with self.assertRaisesRegex(ValueError, 'denominator'):
            self.evaluate()
        self.contract['original_count'] = 5
        self.contract['additions'][1]['excluded_role'] = 'evaluation'
        with self.assertRaisesRegex(ValueError, 'selection role'):
            self.evaluate()

    def test_actual_full_registration_still_does_not_authorize_training(self):
        self.after = set('abcdef')
        self.result['original_registration_guard_passed'] = True
        self.result['models'][0].update(registered_count=6, original_failures_remaining=[], remaining_failures=['g'])
        value = self.evaluate()
        self.assertTrue(value['original_registration_guard_passed'])
        self.assertEqual(value['required_initial_still_unregistered'], [])
        self.assertEqual(value['scope'], 'complete_original_registration')
        self.assertFalse(value['training_or_delivery_authorized'])

    def test_pose_gap_cannot_use_this_scope(self):
        self.result['purpose'] = 'pose_gap'
        with self.assertRaisesRegex(ValueError, 'registration-candidate'):
            self.evaluate()

    def test_only_optimized_registration_is_dispatched_as_recovery_input(self):
        command = ['python', 'stage.py', '--kind', 'temporal', '--mode', 'optimized', '--purpose', 'registration']
        self.assertEqual(temporal_scope(command), command + ['--recovery-input-only'])
        baseline = command.copy()
        baseline[baseline.index('optimized')] = 'baseline'
        self.assertEqual(temporal_scope(baseline), baseline)
        gap = command.copy()
        gap[gap.index('registration')] = 'pose_gap'
        self.assertEqual(temporal_scope(gap), gap)

    def test_real_native_body_compilation_keeps_partial_verification_false(self):
        import run_native_stage
        import run_temporal_augmentation as native

        original_body = native.main
        with tempfile.TemporaryDirectory() as scratch:
            root = Path(scratch)
            output = root / 'temporal'
            command = ['run_supported_sparse.py', '--kind', 'temporal', '--mode', 'optimized',
                '--purpose', 'registration', '--initial', str(root / 'initial'), '--recovery', str(root / 'recovery'),
                '--output', str(output), '--recovery-input-only']

            def completed_evaluation():
                output.mkdir()
                atomic_json(output / 'RESULT.json', self.result)

            with patch.object(sys, 'argv', command), patch.object(run_supported_sparse, 'install'), \
                    patch.object(run_native_stage, 'main', completed_evaluation), \
                    patch('temporal_recovery_input.verify', return_value=self.evaluate()) as verification:
                run_supported_sparse.main()
                self.assertNotIn('--recovery-input-only', sys.argv)
                verification.assert_called_once()
            value = read_json(output / 'RESULT.json')
            self.assertFalse(value['sparse_graph_registration_verified'])
            self.assertEqual(value['sparse_graph_registration_guard_scope'], SCOPE)
            self.assertFalse(value['original_registration_guard_passed'])
            self.assertFalse(value['counted'])
            self.assertTrue(value['original_full_registration_failure_preserved'])
            self.assertTrue((output / 'executed_native_body.py').is_file())
            self.assertIs(native.main, original_body)

    def test_cli_cannot_apply_partial_scope_to_pose_gap(self):
        arguments = ['wrapper.py', '--purpose', 'pose_gap', '--mode', 'optimized', '--recovery-input-only']
        with patch.object(sys, 'argv', arguments), patch.object(run_supported_sparse, 'install') as install:
            with self.assertRaisesRegex(ValueError, 'pose-gap or baseline'):
                run_supported_sparse.main()
            install.assert_not_called()


if __name__ == '__main__':
    unittest.main()
