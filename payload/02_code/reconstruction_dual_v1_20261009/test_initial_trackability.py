"""Exercise multi-view feature support and pre-selection eligibility boundaries."""

import copy
import importlib
import sqlite3
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from initial_trackability import PAIR_BASE, cap_by_acquisition_time, evaluate_trackability, temporal_pairs, trackability_selection
from run_initial_safe import load_native
from front_end import initial_budget


class InitialTrackabilityTests(unittest.TestCase):
    def test_cap_spreads_a_nonuniform_shortlist_and_preserves_both_ends(self):
        times = [10, 11, 12, 13, 29, 49, 70, 80, 90]
        rows = [{'camera_header_ns': stamp, 'decision': 'selected_initial_sfm_pending',
                 'selection_reason': 'coarse_quality'} for stamp in times]
        chosen = cap_by_acquisition_time(list(reversed(rows)), 5)
        self.assertEqual([row['camera_header_ns'] for row in chosen], [10, 29, 49, 70, 90])
        self.assertEqual([row['camera_header_ns'] for row in rows
                          if row['decision'] == 'eligible_not_selected'], [11, 12, 13, 80])
        self.assertTrue(all('selection_reason' not in row for row in rows if row not in chosen))

    def test_cap_can_reuse_nearby_times_when_most_frames_follow_a_gap(self):
        rows = [{'camera_header_ns': stamp, 'decision': 'selected_initial_sfm_pending'}
                for stamp in [0, 1000, 1001, 1002, 1003, 1004]]
        chosen = cap_by_acquisition_time(rows, 5)
        self.assertEqual(len({row['camera_header_ns'] for row in chosen}), 5)
        self.assertEqual([chosen[0]['camera_header_ns'], chosen[-1]['camera_header_ns']], [0, 1004])
        self.assertIs(cap_by_acquisition_time(chosen, 5), chosen)

    def database(self, counts=(40, 40, 40)):
        connection = sqlite3.connect(':memory:')
        self.addCleanup(connection.close)
        connection.execute('CREATE TABLE images(image_id INTEGER, name TEXT)')
        connection.execute('CREATE TABLE keypoints(image_id INTEGER, rows INTEGER)')
        connection.execute('CREATE TABLE two_view_geometries(pair_id INTEGER, rows INTEGER, data BLOB)')
        for key, count in enumerate(counts, 1):
            connection.execute('INSERT INTO images VALUES (?,?)', (key, str(key) + '.jpg'))
            connection.execute('INSERT INTO keypoints VALUES (?,?)', (key, count))
        return connection

    def edge(self, connection, first, second, count=30, shift_first=0, shift_second=0):
        matches = np.column_stack([np.arange(count) + shift_first, np.arange(count) + shift_second]).astype(np.uint32)
        connection.execute('INSERT INTO two_view_geometries VALUES (?,?,?)',
            (PAIR_BASE * first + second, count, matches.tobytes()))

    def test_exact_native_minimum_requires_distinct_features_in_three_views(self):
        connection = self.database()
        for a, b in ((1, 2), (1, 3), (2, 3)):
            self.edge(connection, a, b)
        result = evaluate_trackability(connection, ['1.jpg', '2.jpg', '3.jpg'], 30, 2)
        self.assertTrue(all(row['status'] == 'qualified_visual_trackability' for row in result.values()))
        self.assertEqual([row['supported_distinct_feature_count'] for row in result.values()], [30] * 3)

    def test_one_neighbor_cannot_count_twice(self):
        connection = self.database((40, 40))
        self.edge(connection, 1, 2)
        self.edge(connection, 1, 2)
        result = evaluate_trackability(connection, ['1.jpg', '2.jpg'], 30, 2)
        self.assertTrue(all(row['status'].startswith('abstain') for row in result.values()))

    def test_total_matches_do_not_replace_same_feature_support(self):
        connection = self.database((80, 80, 80))
        self.edge(connection, 1, 2, shift_first=0)
        self.edge(connection, 1, 3, shift_first=30)
        self.edge(connection, 2, 3)
        result = evaluate_trackability(connection, ['1.jpg', '2.jpg', '3.jpg'], 30, 2)
        self.assertTrue(all(row['status'].startswith('abstain') for row in result.values()))

    def test_failed_peer_cannot_lend_support_to_its_neighbors(self):
        connection = self.database()
        self.edge(connection, 1, 2)
        self.edge(connection, 1, 3)
        result = evaluate_trackability(connection, ['1.jpg', '2.jpg', '3.jpg'], 30, 2)
        self.assertTrue(all(row['supported_distinct_feature_count'] == 0 for row in result.values()))

    def test_unnominated_neighbor_is_not_evidence_for_final_shortlist(self):
        connection = self.database()
        for a, b in ((1, 2), (1, 3), (2, 3)):
            self.edge(connection, a, b)
        result = evaluate_trackability(connection, ['1.jpg', '2.jpg'], 30, 2)
        self.assertTrue(all(row['status'].startswith('abstain') for row in result.values()))

    def test_corrupt_feature_reference_is_rejected(self):
        connection = self.database((20, 40))
        self.edge(connection, 1, 2)
        with self.assertRaisesRegex(ValueError, 'outside'):
            evaluate_trackability(connection, ['1.jpg', '2.jpg'], 30, 2)

    def test_conflicting_cycle_cannot_masquerade_as_same_feature_support(self):
        connection = self.database()
        self.edge(connection, 1, 2)
        self.edge(connection, 1, 3)
        indices = np.arange(30, dtype=np.uint32)
        matches = np.column_stack([indices, (indices + 1) % 30]).astype(np.uint32)
        connection.execute('INSERT INTO two_view_geometries VALUES (?,?,?)',
            (PAIR_BASE * 2 + 3, 30, matches.tobytes()))
        result = evaluate_trackability(connection, ['1.jpg', '2.jpg', '3.jpg'], 30, 2)
        self.assertTrue(all(row['status'].startswith('abstain') for row in result.values()))
        self.assertTrue(all(row['supported_distinct_feature_count'] == 0 for row in result.values()))
        self.assertEqual(result['1.jpg']['feature_track_consistency']['ambiguous_components_excluded'], 1)

    def test_ambiguous_component_does_not_erase_independent_good_tracks(self):
        connection = self.database()
        for a, b in ((1, 2), (1, 3), (2, 3)):
            self.edge(connection, a, b)
        for first, second, a, b in ((1, 2, 30, 30), (1, 3, 30, 30), (2, 3, 30, 31)):
            matches = np.asarray([[a, b]], np.uint32)
            connection.execute('INSERT INTO two_view_geometries VALUES (?,?,?)',
                (PAIR_BASE * first + second, 1, matches.tobytes()))
        result = evaluate_trackability(connection, ['1.jpg', '2.jpg', '3.jpg'], 30, 2)
        self.assertTrue(all(row['status'] == 'qualified_visual_trackability' for row in result.values()))
        self.assertEqual([row['supported_distinct_feature_count'] for row in result.values()], [30] * 3)

    def test_component_results_are_independent_of_edge_insertion_order(self):
        left, right = self.database(), self.database()
        for connection, pairs in ((left, ((1, 2), (1, 3), (2, 3))),
                                  (right, ((2, 3), (1, 3), (1, 2)))):
            for a, b in pairs:
                self.edge(connection, a, b)
        self.assertEqual(evaluate_trackability(left, ['1.jpg', '2.jpg', '3.jpg'], 30, 2),
                         evaluate_trackability(right, ['3.jpg', '2.jpg', '1.jpg'], 30, 2))

    def test_empty_match_graph_abstains_for_every_image(self):
        result = evaluate_trackability(self.database(), ['1.jpg', '2.jpg', '3.jpg'], 30, 2)
        self.assertTrue(all(row['supported_distinct_feature_count'] == 0 for row in result.values()))

    def test_self_image_edge_is_not_an_additional_view(self):
        connection = self.database()
        self.edge(connection, 1, 1)
        with self.assertRaisesRegex(ValueError, 'distinct feature views'):
            evaluate_trackability(connection, ['1.jpg', '2.jpg', '3.jpg'], 30, 2)

    def test_temporal_window_boundary_and_already_attempted_pairs(self):
        rows = [{'camera_frame': key, 'camera_header_ns': stamp, 'image_name': str(key) + '.jpg'}
            for key, stamp in [(1, 0), (2, 800_000_000), (3, 800_000_001)]]
        ids = {'1.jpg': 3, '2.jpg': 1, '3.jpg': 2}
        self.assertEqual(temporal_pairs(rows[::-1], ids, set()), [('1.jpg', '2.jpg'), ('2.jpg', '3.jpg')])
        self.assertEqual(temporal_pairs(rows, ids, {PAIR_BASE * 1 + 3}), [('2.jpg', '3.jpg')])

    def test_duplicate_acquisition_timestamp_cannot_supply_two_views(self):
        rows = [{'camera_frame': key, 'camera_header_ns': 100, 'image_name': str(key) + '.jpg'} for key in (1, 2)]
        with self.assertRaisesRegex(ValueError, 'distinct original'):
            temporal_pairs(rows, {'1.jpg': 1, '2.jpg': 2}, set())

    def test_extra_gate_preserves_original_numeric_quality_and_coarse_pose(self):
        native = load_native()
        policy = dict(native.baseline.POLICY)
        rows = [{'camera_frame': key, 'camera_header_ns': key * 500_000_000, 'image_name': str(key) + '.jpg',
            'coarse_pose_valid': True, 'excluded_role': None, 'C_W_m': [key, 0., 0.], 'R_WC': np.eye(3).tolist(),
            'quaternion_WC_xyzw': [0., 0., 0., 1.], 'quality': {'sharpness': 100., 'clipped_fraction': 0.,
                'mean_brightness': 100., 'corner_count_half_resolution': 100,
                'occupied_grid_cells_4x4': 8, 'quality_score': 5.}} for key in range(6)]
        preserved = copy.deepcopy(rows)
        old, _ = native.baseline.choose(copy.deepcopy(rows), policy, 10.)
        equivalent, _ = native.selection_engine.choose_batched(copy.deepcopy(rows), policy, 10., native.baseline)
        self.assertEqual([row['image_name'] for row in old], [row['image_name'] for row in equivalent])
        selected, _ = native.selection_engine.choose_batched(rows, policy, 10., native.baseline,
            additional_quality_gate=lambda row: row['camera_frame'] != 2)
        self.assertNotIn(2, [row['camera_frame'] for row in selected])
        self.assertEqual(rows[2]['decision'], 'rejected_initial_visual_trackability')
        for before, after in zip(preserved, rows):
            for field in ('quality', 'C_W_m', 'R_WC', 'quaternion_WC_xyzw', 'coarse_pose_valid', 'excluded_role'):
                self.assertEqual(before[field], after[field])

    def test_selector_restored_when_precheck_fails(self):
        native = load_native()
        spatial = importlib.import_module('spatial_novelty')
        entry = importlib.import_module('run_initial_pipeline')
        original = native.selection_engine.choose_batched
        original_context = spatial.spatial_novelty_execution
        with patch('initial_trackability.select', side_effect=RuntimeError('precheck failed')):
            with self.assertRaisesRegex(RuntimeError, 'precheck failed'):
                with trackability_selection(native, SimpleNamespace(mode='optimized')):
                    with spatial.spatial_novelty_execution(True):
                        native.selection_engine.choose_batched([], {}, 1., native.baseline)
        self.assertIs(native.selection_engine.choose_batched, original)
        self.assertIs(entry.choose_batched, original)
        self.assertIs(spatial.spatial_novelty_execution, original_context)

    def test_actual_spatial_optimizer_compiles_before_extra_quality_gate(self):
        native = load_native()
        spatial = importlib.import_module('spatial_novelty')
        entry = importlib.import_module('run_initial_pipeline')
        original = native.selection_engine.choose_batched
        rows = [{'camera_frame': key, 'camera_header_ns': key * 50_000_000, 'image_name': str(key) + '.jpg',
            'coarse_pose_valid': True, 'excluded_role': None, 'C_W_m': [key * .05, 0., 0.],
            'R_WC': np.eye(3).tolist(), 'quaternion_WC_xyzw': [0., 0., 0., 1.],
            'quality': {'sharpness': 100., 'clipped_fraction': 0., 'mean_brightness': 100.,
                'corner_count_half_resolution': 100, 'occupied_grid_cells_4x4': 8, 'quality_score': 5.}}
            for key in range(30)]
        policy = dict(native.baseline.POLICY)
        expected, _ = original(copy.deepcopy(rows), policy, 10., native.baseline,
            additional_quality_gate=lambda row: row['camera_frame'] != 2)

        def precheck(native, args, selector, incoming, declared_policy, threshold, baseline):
            return selector(incoming, declared_policy, threshold, baseline,
                additional_quality_gate=lambda row: row['camera_frame'] != 2)

        with patch('initial_trackability.select', side_effect=precheck) as checker:
            with trackability_selection(native, SimpleNamespace(mode='optimized')):
                with spatial.spatial_novelty_execution(True) as trace:
                    self.assertIs(native.selection_engine.choose_batched, entry.choose_batched)
                    actual, _ = native.selection_engine.choose_batched(copy.deepcopy(rows), policy, 10., native.baseline)
        checker.assert_called_once()
        self.assertEqual([row['image_name'] for row in expected], [row['image_name'] for row in actual])
        self.assertGreater(trace['hash_instances'], 0)
        self.assertGreater(trace['hash_queries'], 0)
        self.assertTrue(trace['aliases_restored'])
        self.assertIs(native.selection_engine.choose_batched, original)
        self.assertIs(entry.choose_batched, original)

    def test_initial_outer_budget_covers_both_preserved_and_new_quality_work(self):
        command = ['python', 'owned', '--maximum-seconds', '1200', '--', 'python', 'initial', '--mode', 'optimized']
        updated = initial_budget(command)
        self.assertEqual(updated[3], '3000')
        self.assertEqual(updated[:3] + updated[4:], command[:3] + command[4:])
        self.assertEqual(command[3], '1200')

    def test_initial_budget_rejects_ambiguous_or_changed_ownership_contract(self):
        for command in ([], ['--maximum-seconds', '1199'],
                ['--maximum-seconds', '1200', '--maximum-seconds', '1200']):
            with self.assertRaises(ValueError):
                initial_budget(command)


if __name__ == '__main__':
    unittest.main()
