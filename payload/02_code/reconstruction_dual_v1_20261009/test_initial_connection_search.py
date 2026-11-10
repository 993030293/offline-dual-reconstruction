"""Exercise bounded graph recovery; native camera geometry has separate regressions."""

from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
import sqlite3
import unittest

import numpy as np
import pycolmap

from initial_connectivity import PAIR_BASE, matching_components
from initial_connection_search import BATCH_PAIRS, ranked_pairs, recover_connections, spanning_parents

POLICY = {'position_cell_m': .1, 'translation_min_m': .035,
    'direction_cell_degrees': 10., 'bridge_gap_seconds': .8}


class InitialConnectionSearchTests(unittest.TestCase):
    def fixture(self, sizes):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        database = Path(directory.name) / 'database.db'
        rows, groups, start = [], [], 1
        with closing(sqlite3.connect(database)) as connection:
            connection.executescript('CREATE TABLE images(image_id INTEGER PRIMARY KEY,name TEXT);'
                'CREATE TABLE matches(pair_id INTEGER PRIMARY KEY);'
                'CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY,rows INTEGER);')
            for size in sizes:
                group = []
                for key in range(start, start + size):
                    name = str(key).zfill(3) + '.jpg'
                    connection.execute('INSERT INTO images VALUES (?,?)', (key, name))
                    rows.append({'image_name': name, 'C_W_m': [float(key), 0., 0.],
                        'R_WC': np.eye(3).tolist(), 'camera_header_ns': key * 100_000_000})
                    group.append(name)
                    if key > start:
                        self.insert(connection, key - 1, key, 30)
                groups.append(group)
                start += size
            connection.commit()
        return database, rows, groups

    @staticmethod
    def insert(connection, first, second, count):
        pair = PAIR_BASE * min(first, second) + max(first, second)
        connection.execute('INSERT INTO matches VALUES (?)', (pair,))
        if count:
            connection.execute('INSERT INTO two_view_geometries VALUES (?,?)', (pair, count))

    def writer(self, database, *, connect):
        calls = []

        def match(pairs, label):
            calls.append(list(pairs))
            with closing(sqlite3.connect(database)) as connection:
                for key, (first, second) in enumerate(pairs):
                    self.insert(connection, int(Path(first).stem), int(Path(second).stem),
                        15 if connect and key == 0 else 0)
                connection.commit()
        return calls, match

    def test_native_pair_gate_is_distinct_from_absolute_camera_PnP(self):
        self.assertEqual(pycolmap.GlobalPipelineOptions().min_num_matches, 15)
        self.assertEqual(pycolmap.IncrementalMapperOptions().abs_pose_min_num_inliers, 30)
        database, rows, groups = self.fixture([4, 4])
        with closing(sqlite3.connect(database)) as connection:
            self.insert(connection, 4, 5, 20)
            connection.commit()
            self.assertEqual(len(matching_components(connection, [r['image_name'] for r in rows], 30)), 2)
            self.assertEqual(len(matching_components(connection, [r['image_name'] for r in rows], 15)), 1)
            parents = spanning_parents(connection, [r['image_name'] for r in rows], 15)
        self.assertEqual(len(parents), 8)
        self.assertEqual(sum(parent is None for parent in parents.values()), 1)
        calls, writer = self.writer(database, connect=True)
        after, trace = recover_connections(database, rows, POLICY, 15, writer)
        self.assertEqual(len(after), 1)
        self.assertEqual(calls, [])
        self.assertEqual(trace['actual_requested_pairs'], 0)

    def test_real_connection_stops_before_all_cross_pairs_are_tried(self):
        database, rows, groups = self.fixture([20, 20])
        calls, writer = self.writer(database, connect=True)
        after, trace = recover_connections(database, rows, POLICY, 15, writer)
        self.assertEqual(len(after), 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), BATCH_PAIRS)
        self.assertEqual(trace['original_unattempted_cross_component_proposals'], 400)
        self.assertFalse(trace['all_cross_component_pairs_attempted'])
        self.assertTrue(trace['every_requested_batch_actually_attempted'])

    def test_no_connection_returns_explicit_bounded_abstention(self):
        database, rows, groups = self.fixture([32, 32])
        calls, writer = self.writer(database, connect=False)
        after, trace = recover_connections(database, rows, POLICY, 15, writer)
        self.assertEqual(after, groups)
        self.assertEqual(trace['actual_requested_pairs'], 8 * len(rows))
        self.assertLess(trace['actual_requested_pairs'], trace['original_unattempted_cross_component_proposals'])
        self.assertTrue(trace['unresolved_groups_are_abstentions_not_proven_unmatchable'])
        with closing(sqlite3.connect(database)) as connection:
            with self.assertRaisesRegex(ValueError, 'actual native connected tree'):
                spanning_parents(connection, [r['image_name'] for r in rows], 15)

    def test_a_success_exit_without_attempted_pairs_is_rejected(self):
        database, rows, groups = self.fixture([3, 3])
        with self.assertRaisesRegex(ValueError, 'unattempted'):
            recover_connections(database, rows, POLICY, 15, lambda pairs, label: None)

    def test_native_error_is_not_hidden_by_abstention(self):
        database, rows, groups = self.fixture([3, 3])
        def failed(pairs, label):
            raise RuntimeError('retained native matcher failure')
        with self.assertRaisesRegex(RuntimeError, 'retained native matcher failure'):
            recover_connections(database, rows, POLICY, 15, failed)

    def test_ranking_balances_components_and_does_not_retry_failed_pairs(self):
        database, rows, groups = self.fixture([28, 3, 3])
        with closing(sqlite3.connect(database)) as connection:
            self.insert(connection, 1, 29, 0)
            connection.commit()
            pairs = ranked_pairs(connection, groups, rows, POLICY)
            repeated = ranked_pairs(connection, groups, list(reversed(rows)), POLICY)
        membership = {name: index for index, group in enumerate(groups) for name in group}
        self.assertEqual(len({frozenset((membership[a], membership[b])) for a, b in pairs[:3]}), 3)
        self.assertEqual(pairs, repeated)
        self.assertNotIn(('001.jpg', '029.jpg'), pairs)
        self.assertEqual(len(pairs), len(set(pairs)))

    def test_unusable_coarse_metadata_is_not_silently_replaced(self):
        database, rows, groups = self.fixture([3, 3])
        rows[0]['R_WC'][0][0] = float('nan')
        with closing(sqlite3.connect(database)) as connection:
            with self.assertRaisesRegex(ValueError, 'proper coarse camera rotations'):
                ranked_pairs(connection, groups, rows, POLICY)
        rows[0]['R_WC'] = np.eye(3).tolist()
        rows[0]['image_name'] = rows[1]['image_name']
        with closing(sqlite3.connect(database)) as connection:
            with self.assertRaisesRegex(ValueError, 'actual complete current camera cohort'):
                ranked_pairs(connection, groups, rows, POLICY)


if __name__ == '__main__':
    unittest.main()
