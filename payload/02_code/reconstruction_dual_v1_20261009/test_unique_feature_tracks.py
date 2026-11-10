"""Compare complete compiled support with the preserved scalar oracle."""

import sqlite3
import unittest

import numpy as np

from unique_feature_tracks import PAIR_BASE, reference_unambiguous_support as reference
from unique_feature_tracks import unambiguous_support as candidate


class FastFeatureSupportTests(unittest.TestCase):
    def database(self, keys):
        connection = sqlite3.connect(':memory:')
        self.addCleanup(connection.close)
        connection.execute('CREATE TABLE two_view_geometries(pair_id INTEGER, rows INTEGER, data BLOB)')
        return connection, set(keys), {key: 40 for key in keys}

    @staticmethod
    def edge(connection, first, second, values):
        values = np.asarray(values, dtype=np.uint32).reshape(-1, 2)
        connection.execute('INSERT INTO two_view_geometries VALUES (?,?,?)',
            (PAIR_BASE * first + second, len(values), values.tobytes()))

    def equivalent(self, connection, keys, counts, minimum=3):
        expected = reference(connection, keys, counts, minimum)
        actual = candidate(connection, keys, counts, minimum)
        self.assertEqual(actual, expected)
        return actual

    def test_empty_and_unnominated_graphs(self):
        connection, keys, counts = self.database([1, 2, 3])
        self.equivalent(connection, keys, counts)
        self.edge(connection, 4, 5, [(200, 200)])
        self.equivalent(connection, keys, counts)

    def test_duplicate_edges_do_not_overflow_sparse_boolean_adjacency(self):
        connection, keys, counts = self.database([1, 2, 3])
        for first, second in ((1, 2), (1, 3), (2, 3)):
            self.edge(connection, first, second, [(0, 0)] * 512)
        support, trace = self.equivalent(connection, keys, counts)
        self.assertEqual(trace['unambiguous_multiview_components'], 1)
        self.assertEqual([len(value) for value in support.values()], [1, 1, 1])

    def test_conflicting_feature_cycle_abstains_but_keeps_independent_tracks(self):
        connection, keys, counts = self.database([1, 2, 3, 4])
        self.edge(connection, 1, 2, [(0, 0), (10, 10)])
        self.edge(connection, 1, 3, [(0, 0), (10, 10)])
        self.edge(connection, 2, 3, [(0, 0), (10, 11)])
        support, trace = self.equivalent(connection, keys, counts)
        self.assertEqual(trace['ambiguous_components_excluded'], 1)
        self.assertEqual([len(support[key]) for key in (1, 2, 3)], [1, 1, 1])
        self.assertEqual(support[4], {})

    def test_large_native_image_identifiers_preserve_point_and_neighbor_masks(self):
        keys = (PAIR_BASE - 4, PAIR_BASE - 3, PAIR_BASE - 2)
        connection, keys, counts = self.database(keys)
        ordered = sorted(keys)
        for first, second in ((ordered[0], ordered[1]), (ordered[0], ordered[2]), (ordered[1], ordered[2])):
            self.edge(connection, first, second, [(index, index) for index in range(30)])
        self.equivalent(connection, keys, counts)

    def test_corrupt_keypoint_index_and_self_view_are_not_hidden(self):
        for pair, expected in (((1, 2), 'outside'), ((1, 1), 'distinct feature views')):
            connection, keys, counts = self.database([1, 2, 3])
            self.edge(connection, *pair, [(40, 0)])
            for evaluator in (reference, candidate):
                with self.assertRaisesRegex(ValueError, expected):
                    evaluator(connection, keys, counts, 3)

    def test_corrupt_blob_shape_is_not_ignored(self):
        connection, keys, counts = self.database([1, 2, 3])
        connection.execute('INSERT INTO two_view_geometries VALUES (?,?,?)',
            (PAIR_BASE + 2, 3, np.asarray([[0, 0]], dtype=np.uint32).tobytes()))
        for evaluator in (reference, candidate):
            with self.assertRaises(ValueError):
                evaluator(connection, keys, counts, 3)

    def test_random_sparse_graphs_match_all_bits_counts_and_ambiguity(self):
        rng = np.random.default_rng(20261013)
        for trial in range(160):
            size = int(rng.integers(3, 15))
            connection, keys, counts = self.database(range(1, size + 1))
            edges = []
            for first in keys:
                for second in keys:
                    if first < second and rng.random() < .45:
                        values = rng.integers(0, 40, size=(int(rng.integers(1, 20)), 2), dtype=np.uint32)
                        edges.append((first, second, values))
            rng.shuffle(edges)
            for first, second, values in edges:
                self.edge(connection, first, second, values)
            active = {key for key in keys if rng.random() > .15}
            self.equivalent(connection, active, counts, int(rng.integers(2, 6)))

    def test_component_and_support_results_do_not_depend_on_insertion_order(self):
        edges = [(1, 2, [(0, 0), (1, 1)]), (1, 3, [(0, 0)]),
            (2, 3, [(0, 0)]), (2, 4, [(1, 0)]), (3, 4, [(0, 0)])]
        results = []
        for order in (edges, list(reversed(edges))):
            connection, keys, counts = self.database([1, 2, 3, 4])
            for first, second, values in order:
                self.edge(connection, first, second, values)
            results.append(self.equivalent(connection, keys, counts))
        self.assertEqual(results[0], results[1])


if __name__ == '__main__':
    unittest.main()
