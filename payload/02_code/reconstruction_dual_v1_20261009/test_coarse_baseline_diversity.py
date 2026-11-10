"""Verify proposal diversity, preserved native graphs and bounded offline ownership."""

import importlib.util
from types import SimpleNamespace
import unittest

import numpy as np

from common import ROOT
from coarse_baseline_diversity import (baseline_pairs, candidate_pair_extension, checked_centers,
    diverse_neighbors, witness_pairs)

spec = importlib.util.spec_from_file_location('baseline_diversity_original_graph',
    ROOT / '02_code/gaussian_pipeline_optimization_v2_20261005/indexed_candidate_pairs.py')
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
POLICY = {'position_cell_m': .1, 'translation_min_m': .035}


class CoarseBaselineDiversityTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{'image_name': 'view_' + str(number).zfill(2) + '.jpg', 'camera_frame': number,
            'camera_header_ns': number * 20_000_000,
            'C_W_m': [0. if number < 28 else .2 * (number - 27), 0., 0.]}
            for number in range(32)]

    def test_native_duplicate_neighbors_gain_actual_baseline_and_closed_triangle_proposals(self):
        before, _ = reference.candidate_pairs(self.rows, indexed=True)
        extra, trace = baseline_pairs(self.rows, POLICY)
        self.assertNotIn((18, 28), before)
        self.assertTrue({(18, 28), (18, 29), (28, 29)} <= extra)
        self.assertLessEqual(len(extra), 10 * len(self.rows))
        self.assertEqual(trace['maximum_distinct_anchor_cells'], 4)
        self.assertFalse(trace['class_dimensions_or_final_SfM_used'])

    def test_extension_preserves_every_original_edge_and_restores_the_callable(self):
        native = SimpleNamespace(candidate_pairs=reference.candidate_pairs)
        original = native.candidate_pairs
        before = np.asarray([row['C_W_m'] for row in self.rows]).copy()
        expected, _ = original(self.rows, indexed=True)
        with candidate_pair_extension(native, True, POLICY):
            actual, trace = native.candidate_pairs(self.rows, indexed=True)
            self.assertTrue(expected <= actual)
            self.assertGreater(trace['additional_unique_pairs'], 0)
            self.assertTrue(trace['original_indexed_graph_preserved'])
        self.assertIs(native.candidate_pairs, original)
        np.testing.assert_array_equal(before, [row['C_W_m'] for row in self.rows])

    def test_all_coincident_views_cannot_invent_a_metric_baseline(self):
        for row in self.rows:
            row['C_W_m'] = [0., 0., 0.]
        pairs, _ = baseline_pairs(self.rows, POLICY)
        self.assertEqual(pairs, set())

    def test_native_strongest_prefix_is_kept_and_unverified_neighbors_are_not_added(self):
        centers = checked_centers(self.rows, POLICY)
        kept = [row['image_name'] for row in self.rows[1:5]]
        allowed = kept + [self.rows[28]['image_name'], self.rows[29]['image_name']]
        selected = diverse_neighbors(self.rows[0]['image_name'], allowed, centers, POLICY, kept=kept)
        self.assertEqual(selected[:4], kept)
        self.assertEqual(set(selected), set(allowed))
        self.assertNotIn(self.rows[30]['image_name'], selected)

    def test_invalid_numeric_coordinates_and_duplicate_raw_identities_are_errors(self):
        self.rows[0]['C_W_m'] = [float('nan'), 0., 0.]
        with self.assertRaisesRegex(ValueError, 'finite metric camera positions'):
            baseline_pairs(self.rows, POLICY)
        self.rows[0]['C_W_m'] = [0., 0., 0.]
        self.rows[1]['image_name'] = self.rows[0]['image_name']
        with self.assertRaisesRegex(ValueError, 'unique images'):
            baseline_pairs(self.rows, POLICY)

    def test_disabled_extension_preserves_the_legacy_graph_without_reading_policy(self):
        native = SimpleNamespace(candidate_pairs=reference.candidate_pairs)
        original = native.candidate_pairs
        with candidate_pair_extension(native, False, {}):
            self.assertIs(native.candidate_pairs, original)
        self.assertIs(native.candidate_pairs, original)

    def test_native_failure_propagates_and_restores_the_original_callable(self):
        def failed(*args, **kwargs):
            raise RuntimeError('retained native failure')

        native = SimpleNamespace(candidate_pairs=failed)
        with self.assertRaisesRegex(RuntimeError, 'retained native failure'):
            with candidate_pair_extension(native, True, POLICY):
                native.candidate_pairs(self.rows, indexed=True)
        self.assertIs(native.candidate_pairs, failed)

    def test_fresh_SfM_graph_retains_all_three_verified_local_witness_edges(self):
        rows = self.rows[:3]
        for row in rows:
            row['initial_trackability'] = {'status': 'qualified_visual_trackability',
                'local_three_view_geometry': {'name': row['image_name'], 'status': 'qualified_three_view_geometry',
                    'source_witness_query_name': rows[0]['image_name'],
                    'source_witness_anchor_names': [rows[1]['image_name'], rows[2]['image_name']]}}
        native = SimpleNamespace(candidate_pairs=lambda rows, indexed: (set(), {}))
        with candidate_pair_extension(native, True, POLICY):
            pairs, trace = native.candidate_pairs(rows, indexed=True)
        self.assertEqual(pairs, {(0, 1), (0, 2), (1, 2)})
        self.assertEqual(trace['same_attempt_verified_witness_triangle_pairs'], 3)
        self.assertFalse(trace['private_local_poses_or_3D_points_imported_into_SfM'])

    def test_a_frozen_witness_with_a_missing_actual_camera_is_rejected(self):
        rows = self.rows[:3]
        for row in rows:
            row['initial_trackability'] = {'status': 'qualified_visual_trackability'}
        rows[0]['initial_trackability']['local_three_view_geometry'] = {
            'name': rows[0]['image_name'], 'status': 'qualified_three_view_geometry',
            'source_witness_query_name': rows[0]['image_name'],
            'source_witness_anchor_names': [rows[1]['image_name'], 'missing.jpg']}
        with self.assertRaisesRegex(ValueError, 'lost a nominated camera'):
            witness_pairs(rows)

    def test_final_graph_carries_a_native_connection_even_when_it_is_not_a_local_triangle(self):
        rows = self.rows[:4]
        for key, row in enumerate(rows):
            row['initial_connectivity_parent_image_name'] = rows[key - 1]['image_name'] if key else None
        native = SimpleNamespace(candidate_pairs=lambda rows, indexed: (set(), {}))
        with candidate_pair_extension(native, True, POLICY):
            pairs, trace = native.candidate_pairs(rows, indexed=True)
        self.assertTrue({(0, 1), (1, 2), (2, 3)} <= pairs)
        self.assertEqual(trace['same_attempt_verified_connected_spanning_pairs'], 3)

    def test_invalid_final_connection_witness_does_not_become_an_SfM_proposal(self):
        rows = self.rows[:4]
        native = SimpleNamespace(candidate_pairs=lambda rows, indexed: (set(), {}))
        rows[0]['initial_connectivity_parent_image_name'] = None
        with candidate_pair_extension(native, True, POLICY):
            with self.assertRaisesRegex(ValueError, 'incomplete camera roles'):
                native.candidate_pairs(rows, indexed=True)
        for key, row in enumerate(rows):
            row['initial_connectivity_parent_image_name'] = rows[key - 1]['image_name'] if key else None
        rows[1]['initial_connectivity_parent_image_name'] = rows[2]['image_name']
        with candidate_pair_extension(native, True, POLICY):
            with self.assertRaisesRegex(ValueError, 'cycle or disconnected'):
                native.candidate_pairs(rows, indexed=True)


if __name__ == '__main__':
    unittest.main()
