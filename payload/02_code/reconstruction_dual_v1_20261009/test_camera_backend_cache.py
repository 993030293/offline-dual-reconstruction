"""Regression checks for same-file cache aliases and immutable input identities."""

from pathlib import Path
import unittest
from unittest.mock import patch

from prepare_camera_backend import validate_cache_config


class CameraBackendCacheTests(unittest.TestCase):
    def setUp(self):
        self.config = {'source_model': 'source', 'raw': 'recording',
            'source_identity': {'images.bin': 'model-hash'}, 'raw_bytes': 24,
            'raw_mtime_ns': 1000, 'initial_manifest_identity': 'manifest-hash',
            'raw_frame_decisions_identity': 'decisions-hash'}

    def test_exact_input_needs_no_path_alias(self):
        self.assertEqual(validate_cache_config(self.config, dict(self.config)), [])

    def test_same_file_alias_is_recorded(self):
        current = {**self.config, 'raw': 'junction-recording'}
        with patch.object(Path, 'samefile', return_value=True):
            aliases = validate_cache_config(self.config, current)
        self.assertEqual(len(aliases), 1)
        self.assertEqual(aliases[0]['field'], 'raw')
        self.assertTrue(aliases[0]['same_filesystem_object_verified'])

    def test_distinct_file_with_equal_size_is_rejected(self):
        with patch.object(Path, 'samefile', return_value=False):
            with self.assertRaisesRegex(ValueError, 'input changed: raw'):
                validate_cache_config(self.config, {**self.config, 'raw': 'copied-recording'})

    def test_unresolvable_alias_is_rejected(self):
        with patch.object(Path, 'samefile', side_effect=FileNotFoundError):
            with self.assertRaisesRegex(ValueError, 'input changed: raw'):
                validate_cache_config(self.config, {**self.config, 'raw': 'missing-recording'})

    def test_nonpath_input_changes_are_rejected_even_with_alias(self):
        for field in ('source_identity', 'raw_bytes', 'raw_mtime_ns',
                'initial_manifest_identity', 'raw_frame_decisions_identity'):
            with self.subTest(field=field), patch.object(Path, 'samefile', return_value=True):
                with self.assertRaisesRegex(ValueError, 'input changed: ' + field):
                    validate_cache_config(self.config,
                        {**self.config, 'raw': 'junction-recording', field: 'changed'})

    def test_config_field_loss_is_rejected(self):
        current = dict(self.config)
        del current['source_identity']
        with self.assertRaisesRegex(ValueError, 'fields changed'):
            validate_cache_config(self.config, current)


if __name__ == '__main__':
    unittest.main()
