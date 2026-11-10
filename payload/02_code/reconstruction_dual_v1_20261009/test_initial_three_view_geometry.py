"""Exercise native calibrated geometry and distinguish it from graph-only support."""

from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
import pycolmap

from initial_three_view_geometry import (PAIR_BASE, WINDOW_NS, evaluate_geometry, geometric_support,
    pair_matches, replay_geometry_proof)
from initial_trackability import evaluate_trackability


class InitialThreeViewGeometryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.database = self.root / 'database.db'
        options = pycolmap.SyntheticDatasetOptions()
        options.num_rigs, options.num_cameras_per_rig, options.num_frames_per_rig = 1, 1, 8
        options.num_points3D = 128
        with pycolmap.Database.open(self.database) as db:
            self.model = pycolmap.synthesize_dataset(options, db)
        self.camera = next(iter(self.model.cameras.values()))
        self.subjects = {image.name: (image.image_id, image.image_id * 40_000_000)
            for image in self.model.images.values()}
        self.name = min(self.subjects, key=lambda name: self.subjects[name][0])
        self.allowed = range(len(next(image for image in self.model.images.values() if image.name == self.name).points2D))
        self.selected = [{'image_name': name, 'camera_header_ns': stamp} for name, (_, stamp) in self.subjects.items()]

    def connection(self):
        return closing(sqlite3.connect('file:' + self.database.as_posix() + '?mode=ro', uri=True))

    def make_star(self):
        target = self.subjects[self.name][0]
        others = sorted(key for key, _ in self.subjects.values() if key != target)
        anchors = set(others[:2])
        with closing(sqlite3.connect(self.database)) as connection:
            for pair, in connection.execute('SELECT pair_id FROM two_view_geometries').fetchall():
                a, b = divmod(pair, PAIR_BASE)
                if ((target in (a, b) and (b if a == target else a) not in anchors)
                        or {a, b} == anchors):
                    connection.execute('UPDATE two_view_geometries SET rows=0,data=? WHERE pair_id=?', (b'', pair))
            connection.commit()

    def test_real_calibrated_three_view_geometry_passes_native_point_and_pose_gates(self):
        before = self.database.read_bytes()
        params = self.camera.params.copy()
        with self.connection() as connection:
            result = geometric_support(connection, self.name, self.subjects, self.camera, self.allowed)
        self.assertEqual(result['status'], 'qualified_three_view_geometry')
        self.assertGreaterEqual(result['qualified_distinct_features'], 30)
        self.assertEqual(result['minimum_features'], pycolmap.IncrementalMapperOptions().abs_pose_min_num_inliers)
        self.assertEqual(result['maximum_reprojection_px'], pycolmap.IncrementalTriangulatorOptions().complete_max_reproj_error)
        self.assertEqual(result['minimum_angle_degrees'], pycolmap.IncrementalTriangulatorOptions().min_angle)
        self.assertEqual(self.database.read_bytes(), before)
        np.testing.assert_array_equal(self.camera.params, params)
        self.assertNotIn('cam_from_world', result)

    def test_graph_supported_star_cannot_invent_a_closed_geometric_witness(self):
        self.make_star()
        with self.connection() as connection:
            graph = evaluate_trackability(connection, self.subjects, 30, 2, include_feature_ids=True)
            self.assertEqual(graph[self.name]['status'], 'qualified_visual_trackability')
            result = geometric_support(connection, self.name, self.subjects, self.camera,
                graph[self.name]['supported_feature_indices'])
        self.assertEqual(result['status'], 'abstain_three_view_geometry')
        self.assertEqual(result['qualified_distinct_features'], 0)
        self.assertEqual(len(result['attempts']), 1)
        self.assertEqual(result['attempts'][0]['closed_three_view_features'], 0)

    def test_ambiguous_or_unqualified_feature_ids_cannot_restore_support(self):
        with self.connection() as connection:
            result = geometric_support(connection, self.name, self.subjects, self.camera, range(29))
        self.assertEqual(result['status'], 'abstain_three_view_geometry')
        self.assertEqual(result['qualified_distinct_features'], 0)

    def test_verified_witnesses_can_span_time_without_using_final_frame_poses(self):
        target_id, target_stamp = self.subjects[self.name]
        remote = {name: (key, stamp if key == target_id else target_stamp + (index + 2) * WINDOW_NS)
            for index, (name, (key, stamp)) in enumerate(self.subjects.items())}
        with self.connection() as connection:
            result = geometric_support(connection, self.name, remote, self.camera, self.allowed)
        self.assertEqual(result['status'], 'qualified_three_view_geometry')
        self.assertGreaterEqual(result['qualified_distinct_features'], 30)
        self.assertFalse(result['temporal_limit_applied_to_geometric_witness'])

    def test_adapter_rejects_graph_false_positive_before_selection_freeze(self):
        self.make_star()
        with self.connection() as connection:
            decisions = evaluate_trackability(connection, self.subjects, 30, 2, include_feature_ids=True)
            result = evaluate_geometry(connection, self.selected, decisions, self.camera)
        self.assertEqual(decisions[self.name]['status'], 'abstain_local_three_view_geometry')
        self.assertEqual(result[self.name]['qualified_distinct_features'], 0)
        self.assertEqual(len(self.selected), 8)

    def test_actual_keypoint_identity_corruption_is_an_error(self):
        with self.connection() as connection:
            with self.assertRaisesRegex(ValueError, 'outside the actual query keypoints'):
                geometric_support(connection, self.name, self.subjects, self.camera, [2**31])
        others = sorted(key for key, _ in self.subjects.values() if key != self.subjects[self.name][0])
        a, b = sorted((self.subjects[self.name][0], others[0]))
        matches = np.array([[2**31, 0]], np.uint32)
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute('UPDATE two_view_geometries SET rows=1,data=? WHERE pair_id=?',
                (matches.tobytes(), PAIR_BASE * a + b))
            connection.commit()
        with self.connection() as connection:
            with self.assertRaisesRegex(ValueError, 'outside its actual keypoint arrays'):
                geometric_support(connection, self.name, self.subjects, self.camera, self.allowed)

    def test_same_attempt_cache_requires_the_actual_geometry_and_current_witnesses(self):
        cache, first_trace, second_trace = {}, {}, {}
        with self.connection() as connection:
            first = evaluate_trackability(connection, self.subjects, 30, 2, include_feature_ids=True)
            before = evaluate_geometry(connection, self.selected, first, self.camera, cache=cache, trace=first_trace)
            second = evaluate_trackability(connection, self.subjects, 30, 2, include_feature_ids=True)
            after = evaluate_geometry(connection, self.selected, second, self.camera, cache=cache, trace=second_trace)
        self.assertEqual(before, after)
        self.assertGreater(second_trace['same_attempt_positive_proof_cache_hits'], 0)
        self.make_star()
        trace = {}
        with self.connection() as connection:
            changed = evaluate_trackability(connection, self.subjects, 30, 2, include_feature_ids=True)
            result = evaluate_geometry(connection, self.selected, changed, self.camera, cache=cache, trace=trace)
        self.assertEqual(result[self.name]['status'], 'abstain_three_view_geometry')
        self.assertEqual(changed[self.name]['status'], 'abstain_local_three_view_geometry')

    def test_cached_geometric_witness_can_be_replayed_from_actual_pixels(self):
        with self.connection() as connection:
            result = geometric_support(connection, self.name, self.subjects, self.camera, self.allowed)
            replay = geometric_support(connection, self.name, self.subjects, self.camera,
                result['witness_input_feature_indices'], witness_pair=result['witness_anchor_names'])
        self.assertEqual(replay['status'], 'qualified_three_view_geometry')
        self.assertEqual(replay['qualified_feature_indices'], result['qualified_feature_indices'])
        self.assertEqual(replay['qualified_distinct_features'], result['qualified_distinct_features'])

    def test_failed_fast_four_neighbors_searches_additional_actual_verified_witnesses(self):
        target = self.subjects[self.name][0]
        with self.connection() as connection:
            neighbors = [(name, key, len(pair_matches(connection, target, key)))
                for name, (key, _) in self.subjects.items() if name != self.name]
        neighbors.sort(key=lambda row: (-row[2], row[0]))
        fast = {key for _, key, count in neighbors[:4] if count >= 30}
        self.assertEqual(len(fast), 4)
        with closing(sqlite3.connect(self.database)) as connection:
            for pair, in connection.execute('SELECT pair_id FROM two_view_geometries').fetchall():
                a, b = divmod(pair, PAIR_BASE)
                if {a, b} <= fast:
                    connection.execute('UPDATE two_view_geometries SET rows=0,data=? WHERE pair_id=?', (b'', pair))
            connection.commit()
        with self.connection() as connection:
            result = geometric_support(connection, self.name, self.subjects, self.camera, self.allowed)
        self.assertEqual(result['status'], 'qualified_three_view_geometry')
        self.assertGreater(len(result['attempts']), 6)
        self.assertTrue(all(row['closed_three_view_features'] == 0 for row in result['attempts'][:6]))
        self.assertGreaterEqual(result['qualified_distinct_features'], 30)

    def test_coarse_duplicate_positions_cannot_veto_the_original_valid_visual_search(self):
        selected = [dict(row, C_W_m=[0., 0., 0.]) for row in self.selected]
        with self.connection() as connection:
            original_decisions = evaluate_trackability(connection, self.subjects, 30, 2, include_feature_ids=True)
            original = evaluate_geometry(connection, selected, original_decisions, self.camera)
            diverse_decisions = evaluate_trackability(connection, self.subjects, 30, 2, include_feature_ids=True)
            diverse = evaluate_geometry(connection, selected, diverse_decisions, self.camera,
                coarse_policy={'position_cell_m': .1, 'translation_min_m': .035})
        self.assertEqual(original, diverse)


class NativeBaselineDiversityGeometryTests(unittest.TestCase):
    def test_repeated_views_keep_original_search_and_then_use_a_true_baseline(self):
        from itertools import combinations
        from coarse_baseline_diversity import checked_centers

        camera = pycolmap.Camera(model='OPENCV', width=1024, height=768,
            params=[500., 500., 512., 384., 0., 0., 0., 0.])
        xyz = np.random.default_rng(29).uniform([-.8, -.6, 3.], [.8, .6, 4.5], (96, 3))
        centers = {'duplicate_' + str(number) + '.jpg': np.zeros(3) for number in range(9)}
        centers.update(moving_a=np.array([.5, 0., 0.]), moving_b=np.array([1., 0., 0.]))
        subjects = {name: (key, key * 20_000_000) for key, name in enumerate(centers, 1)}
        selected = [{'image_name': name, 'camera_header_ns': stamp, 'C_W_m': centers[name].tolist()}
            for name, (_, stamp) in subjects.items()]
        policy = {'position_cell_m': .1, 'translation_min_m': .035}
        matches = np.column_stack((np.arange(96), np.arange(96))).astype(np.uint32)
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'database.db'
            with closing(sqlite3.connect(database)) as connection:
                connection.executescript('CREATE TABLE images(image_id INTEGER PRIMARY KEY,name TEXT);'
                    'CREATE TABLE keypoints(image_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB);'
                    'CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY,rows INTEGER,data BLOB);')
                for name, (key, _) in subjects.items():
                    pixels = camera.img_from_cam(xyz - centers[name]).astype(np.float32)
                    connection.execute('INSERT INTO images VALUES (?,?)', (key, name))
                    connection.execute('INSERT INTO keypoints VALUES (?,?,?,?)', (key, 96, 2, pixels.tobytes()))
                for first, second in combinations(range(1, len(subjects) + 1), 2):
                    connection.execute('INSERT INTO two_view_geometries VALUES (?,?,?)',
                        (PAIR_BASE * first + second, 96, matches.tobytes()))
                connection.commit()
            with closing(sqlite3.connect('file:' + database.as_posix() + '?mode=ro', uri=True)) as connection:
                old = geometric_support(connection, 'duplicate_0.jpg', subjects, camera, range(96))
                new = geometric_support(connection, 'duplicate_0.jpg', subjects, camera, range(96),
                    coarse_centers=checked_centers(selected, policy), coarse_policy=policy)
                self.assertEqual(old['status'], 'abstain_three_view_geometry')
                self.assertEqual(new['status'], 'qualified_three_view_geometry')
                self.assertEqual(new['qualified_distinct_features'], 96)
                self.assertGreater(len(new['attempts']), 28)
                self.assertLessEqual(new['maximum_geometry_pair_attempts'], 50)
                self.assertTrue(new['original_strongest_eight_search_preserved_before_baseline_extension'])
                decisions = evaluate_trackability(connection, subjects, 30, 2, include_feature_ids=True)
                allowed = {name: row['supported_feature_indices'] for name, row in decisions.items()}
                proofs = evaluate_geometry(connection, selected, decisions, camera, coarse_policy=policy)
                self.assertTrue(all(row['status'] == 'qualified_visual_trackability' for row in decisions.values()))
                for name, proof in proofs.items():
                    replay = replay_geometry_proof(connection, name, proof, subjects, camera, allowed)
                    self.assertEqual(replay['qualified_feature_indices'], proof['qualified_feature_indices'])


class ThreeViewCameraRoleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.database = Path(temporary.name) / 'database.db'
        self.camera = pycolmap.Camera(model='OPENCV', width=1024, height=768,
            params=[500., 500., 512., 384., 0., 0., 0., 0.])
        xyz = np.random.default_rng(29).uniform([-.8, -.6, 3.], [.8, .6, 4.5], (96, 3))
        centers = {'a.jpg': np.array([0., 0., 0.]), 'b.jpg': np.array([.5, 0., 0.]),
            'c.jpg': np.array([.001, 0., 0.])}
        self.subjects = {name: (key, key * 20_000_000) for key, name in enumerate(centers, 1)}
        self.selected = [{'image_name': name, 'camera_header_ns': stamp}
            for name, (_, stamp) in self.subjects.items()]
        matches = np.column_stack((np.arange(96), np.arange(96))).astype(np.uint32)
        with closing(sqlite3.connect(self.database)) as connection:
            connection.executescript('CREATE TABLE images(image_id INTEGER PRIMARY KEY,name TEXT);'
                'CREATE TABLE keypoints(image_id INTEGER PRIMARY KEY,rows INTEGER,cols INTEGER,data BLOB);'
                'CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY,rows INTEGER,data BLOB);')
            for name, (key, _) in self.subjects.items():
                pixels = self.camera.img_from_cam(xyz - centers[name]).astype(np.float32)
                connection.execute('INSERT INTO images VALUES (?,?)', (key, name))
                connection.execute('INSERT INTO keypoints VALUES (?,?,?,?)', (key, 96, 2, pixels.tobytes()))
            for first, second in ((1, 2), (1, 3), (2, 3)):
                connection.execute('INSERT INTO two_view_geometries VALUES (?,?,?)',
                    (PAIR_BASE * first + second, 96, matches.tobytes()))
            connection.commit()

    def connection(self):
        return closing(sqlite3.connect('file:' + self.database.as_posix() + '?mode=ro', uri=True))

    def decisions(self, connection):
        return evaluate_trackability(connection, self.subjects, 30, 2, include_feature_ids=True)

    def test_seed_camera_is_verified_without_requiring_a_degenerate_other_seed_pair(self):
        before = self.database.read_bytes()
        with self.connection() as connection:
            direct = geometric_support(connection, 'b.jpg', self.subjects, self.camera, range(96))
            decisions, trace = self.decisions(connection), {}
            results = evaluate_geometry(connection, self.selected, decisions, self.camera, trace=trace)
        self.assertEqual(direct['status'], 'abstain_three_view_geometry')
        self.assertTrue(all(row['status'] == 'qualified_visual_trackability' for row in decisions.values()))
        self.assertTrue(all(row['qualified_distinct_features'] == 96 for row in results.values()))
        self.assertIn(results['b.jpg']['verified_camera_role'], ('first_seed_camera', 'second_seed_camera'))
        self.assertGreater(trace['actual_native_camera_role_rechecks'], 0)
        self.assertEqual(self.database.read_bytes(), before)

    def test_seed_role_cannot_copy_another_views_positive_decision_without_its_own_support(self):
        with self.connection() as connection:
            decisions = self.decisions(connection)
            decisions['b.jpg']['supported_feature_indices'] = list(range(29))
            results = evaluate_geometry(connection, self.selected, decisions, self.camera)
        self.assertEqual(results['c.jpg']['status'], 'qualified_three_view_geometry')
        self.assertEqual(decisions['b.jpg']['status'], 'abstain_local_three_view_geometry')
        self.assertEqual(results['b.jpg']['qualified_distinct_features'], 0)

    def test_a_missing_real_third_pair_cannot_be_replaced_by_camera_role_bookkeeping(self):
        with closing(sqlite3.connect(self.database)) as connection:
            connection.execute('UPDATE two_view_geometries SET rows=0,data=? WHERE pair_id=?',
                (b'', PAIR_BASE + 2))
            connection.commit()
        with self.connection() as connection:
            decisions = self.decisions(connection)
            results = evaluate_geometry(connection, self.selected, decisions, self.camera)
        self.assertFalse(any(row['status'] == 'qualified_three_view_geometry' for row in results.values()))

    def test_all_camera_roles_replay_the_canonical_actual_pixel_witness(self):
        with self.connection() as connection:
            decisions = self.decisions(connection)
            allowed = {name: row['supported_feature_indices'] for name, row in decisions.items()}
            results = evaluate_geometry(connection, self.selected, decisions, self.camera)
            for name, proof in results.items():
                replay = replay_geometry_proof(connection, name, proof, self.subjects, self.camera, allowed)
                self.assertEqual(replay['qualified_feature_indices'], proof['qualified_feature_indices'])
                self.assertEqual(replay['verified_camera_role'], proof['verified_camera_role'])
                self.assertEqual(replay['status'], 'qualified_three_view_geometry')

    def test_seed_cache_requires_current_canonical_and_target_feature_eligibility(self):
        cache, trace = {}, {}
        with self.connection() as connection:
            initial = self.decisions(connection)
            evaluate_geometry(connection, self.selected, initial, self.camera, cache=cache)
            repeated = self.decisions(connection)
            evaluate_geometry(connection, self.selected, repeated, self.camera, cache=cache, trace=trace)
            self.assertEqual(trace['same_attempt_positive_proof_cache_hits'], 3)
            changed = self.decisions(connection)
            changed['b.jpg']['supported_feature_indices'] = list(range(29))
            result = evaluate_geometry(connection, self.selected, changed, self.camera, cache=cache)
        self.assertEqual(result['b.jpg']['status'], 'abstain_three_view_geometry')
        with self.connection() as connection:
            allowed = {name: list(range(96)) for name in self.subjects}
            allowed[cache['b.jpg']['result']['source_witness_query_name']] = list(range(29))
            with self.assertRaisesRegex(ValueError, 'lost its actual canonical'):
                replay_geometry_proof(connection, 'b.jpg', cache['b.jpg']['result'],
                    self.subjects, self.camera, allowed)

    def test_corrupted_seed_feature_identity_is_an_error(self):
        with self.connection() as connection:
            with self.assertRaisesRegex(ValueError, 'outside the actual verified camera keypoints'):
                geometric_support(connection, 'c.jpg', self.subjects, self.camera, range(96),
                    witness_pair=['a.jpg', 'b.jpg'], verified_camera='b.jpg', verified_allowed_features=[2**31])


if __name__ == '__main__':
    unittest.main()
