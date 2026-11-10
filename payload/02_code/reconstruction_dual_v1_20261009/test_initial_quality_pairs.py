"""Exercise the actual accepted coarse graph and preselection pair identity contract."""

import importlib.util
import unittest

import numpy as np

from common import ROOT
from initial_quality_pairs import PAIR_BASE, quality_pairs

spec = importlib.util.spec_from_file_location('quality_pair_reference',
    ROOT / '02_code/gaussian_pipeline_optimization_v2_20261005/indexed_candidate_pairs.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


class InitialQualityPairTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{'image_name': 'object_' + str(index) + '.jpg', 'camera_frame': index,
            'camera_header_ns': index * 2_000_000_000,
            'C_W_m': [float(np.cos(index)), float(np.sin(index)), 0.]}
            for index in range(24)]
        self.ids = {row['image_name']: 100 - index for index, row in enumerate(self.rows)}

    def test_actual_coarse_graph_adds_verified_candidate_proposals_beyond_temporal_window(self):
        pairs, trace = quality_pairs(reference.candidate_pairs, self.rows[::-1], self.ids, set(), [])
        expected, _ = reference.candidate_pairs(self.rows, indexed=True)
        self.assertEqual({frozenset(pair) for pair in pairs}, {
            frozenset((self.rows[a]['image_name'], self.rows[b]['image_name'])) for a, b in expected})
        self.assertGreater(len(pairs), 0)
        self.assertTrue(all(self.ids[a] < self.ids[b] for a, b in pairs))
        self.assertEqual(trace['temporal_unattempted_pairs'], 0)
        self.assertFalse(trace['old_SfM_model_or_final_poses_used'])

    def test_native_pair_identity_prevents_repeat_matching_and_duplicate_temporal_orientation(self):
        first, second = self.rows[0]['image_name'], self.rows[1]['image_name']
        a, b = sorted((self.ids[first], self.ids[second]))
        pairs, _ = quality_pairs(reference.candidate_pairs, self.rows, self.ids,
            {PAIR_BASE * a + b}, [(first, second), (second, first)])
        self.assertNotIn(frozenset((first, second)), {frozenset(pair) for pair in pairs})
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_invalid_coarse_coordinates_and_duplicate_native_identities_fail_closed(self):
        self.rows[0]['C_W_m'] = [float('nan'), 0., 0.]
        with self.assertRaisesRegex(ValueError, 'finite metric camera positions'):
            quality_pairs(reference.candidate_pairs, self.rows, self.ids, set(), [])
        self.rows[0]['C_W_m'] = [0., 0., 0.]
        duplicated = {name: 1 for name in self.ids}
        with self.assertRaisesRegex(ValueError, 'distinct native image identities'):
            quality_pairs(reference.candidate_pairs, self.rows, duplicated, set(), [])


if __name__ == '__main__':
    unittest.main()
