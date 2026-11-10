"""Keep isolated image groups from qualifying through local matches alone."""

import sqlite3
from contextlib import closing
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from initial_connectivity import PAIR_BASE, matching_components, component_recovery_pairs, main_component
from initial_trackability import select
from common import ROOT, read_json


class InitialConnectivityTests(unittest.TestCase):
    def setUp(self):
        self.connection = sqlite3.connect(':memory:')
        self.addCleanup(self.connection.close)
        self.connection.executescript('CREATE TABLE images(image_id INTEGER PRIMARY KEY,name TEXT);'
            'CREATE TABLE matches(pair_id INTEGER PRIMARY KEY);'
            'CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY,rows INTEGER);')
        self.connection.executemany('INSERT INTO images VALUES (?,?)',
            [(index, str(index) + '.jpg') for index in range(1, 9)])

    def edge(self, first, second, count=30):
        pair = PAIR_BASE * min(first, second) + max(first, second)
        self.connection.execute('INSERT INTO matches VALUES (?)', (pair,))
        self.connection.execute('INSERT INTO two_view_geometries VALUES (?,?)', (pair, count))

    def test_locally_strong_island_still_requires_a_real_connection(self):
        for first, second in ((1, 2), (1, 3), (1, 4), (2, 3), (3, 4), (5, 6), (5, 7), (6, 7)):
            self.edge(first, second)
        self.edge(4, 5, 29)
        names = [str(index) + '.jpg' for index in range(1, 8)]
        groups = matching_components(self.connection, names, 30)
        self.assertEqual(groups, [names[:4], names[4:]])
        self.assertEqual(main_component(groups), set(names[:4]))
        pairs = component_recovery_pairs(self.connection, groups)
        self.assertEqual(len(pairs), 11)
        self.assertNotIn(('4.jpg', '5.jpg'), pairs)
        self.connection.execute('UPDATE two_view_geometries SET rows=30 WHERE pair_id=?', (PAIR_BASE * 4 + 5,))
        self.assertEqual(matching_components(self.connection, names, 30), [names])

    def test_unnominated_images_cannot_connect_two_scenes(self):
        for first, second in ((1, 2), (3, 4), (2, 8), (4, 8)):
            self.edge(first, second)
        groups = matching_components(self.connection, ['1.jpg', '2.jpg', '3.jpg', '4.jpg'], 30)
        self.assertEqual(groups, [['1.jpg', '2.jpg'], ['3.jpg', '4.jpg']])
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            main_component(groups)

    def test_failed_attempts_are_not_silently_retried(self):
        self.connection.execute('INSERT INTO matches VALUES (?)', (PAIR_BASE + 2,))
        self.assertEqual(component_recovery_pairs(self.connection, [['1.jpg'], ['2.jpg']]), [])

    def test_a_merged_group_does_not_prove_every_requested_pair_was_attempted(self):
        self.edge(1, 2)
        self.edge(3, 4)
        before = matching_components(self.connection, ['1.jpg', '2.jpg', '3.jpg', '4.jpg'], 30)
        self.edge(1, 3)
        self.assertEqual(len(matching_components(self.connection, ['1.jpg', '2.jpg', '3.jpg', '4.jpg'], 30)), 1)
        self.assertEqual(len(component_recovery_pairs(self.connection, before)), 3)

    def test_invalid_or_overlapping_roles_are_rejected(self):
        with self.assertRaises(ValueError):
            matching_components(self.connection, ['missing.jpg'], 30)
        for groups in ([], [[]], [['1.jpg'], ['1.jpg']], [['absent.jpg']]):
            with self.assertRaises(ValueError):
                component_recovery_pairs(self.connection, groups)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            main_component([['1.jpg', '2.jpg'], ['2.jpg']])

    def test_connected_input_needs_no_additional_matching(self):
        for first, second in ((1, 2), (2, 3)):
            self.edge(first, second)
        groups = matching_components(self.connection, ['3.jpg', '1.jpg', '2.jpg'], 30)
        self.assertEqual(groups, [['1.jpg', '2.jpg', '3.jpg']])
        self.assertEqual(component_recovery_pairs(self.connection, groups), [])

    def test_selection_refills_before_manifest_without_using_a_final_model(self):
        with TemporaryDirectory() as directory:
            output = Path(directory)
            raw = output / 'fixture.mcap'
            raw.write_bytes(b'numeric selection fixture')
            info = {'camera_matrix': [100, 0, 8, 0, 100, 8, 0, 0, 1],
                'distortion': [0] * 5, 'distortion_model': 'plumb_bob', 'frame_id': 'camera_optical',
                'width': 16, 'height': 16}
            (output / 'RAW_SENSOR_CONTRACT.json').write_text(json.dumps({'camera_info': info}))
            rows = [{'image_name': str(index) + '.jpg', 'camera_frame': index,
                'camera_header_ns': index * 100, 'coarse_pose_valid': True,
                'quality': {'quality_score': 1}, 'C_W_m': [index, 0, 0]}
                for index in range(11)]
            for row in rows:
                row['R_WC'] = np.eye(3).tolist()

            def choose(incoming, policy, sharpness, baseline, additional_quality_gate):
                eligible = [row for row in incoming if additional_quality_gate(row)]
                return eligible[:policy['maximum_exported_frames']], eligible

            def selected_messages(topic, wanted):
                for index in wanted:
                    yield index, {'payload': (index * 100, 'camera_optical', np.zeros((16, 16, 3), np.uint8))}

            native = SimpleNamespace(ROOT=ROOT, McapIndex=lambda path:
                SimpleNamespace(iter_selected_messages=selected_messages), decode_bayer=lambda payload: payload,
                demosaic_rggb8_to_bgr=lambda pixels: pixels,
                candidate_pairs=lambda selected, indexed: (set(), {'isolated_connectivity_fixture': True}))

            def execute(command, folder, label, budget, gpu, **keywords):
                database = Path(command[command.index('--database_path') + 1])
                with closing(sqlite3.connect(database)) as connection:
                    if 'feature_extractor' in command:
                        connection.executescript('CREATE TABLE IF NOT EXISTS images(image_id INTEGER PRIMARY KEY,name TEXT);'
                            'CREATE TABLE IF NOT EXISTS keypoints(image_id INTEGER PRIMARY KEY,rows INTEGER);'
                            'CREATE TABLE IF NOT EXISTS cameras(camera_id INTEGER PRIMARY KEY);'
                            'CREATE TABLE IF NOT EXISTS matches(pair_id INTEGER PRIMARY KEY);'
                            'CREATE TABLE IF NOT EXISTS two_view_geometries(pair_id INTEGER PRIMARY KEY,rows INTEGER,data BLOB);')
                        connection.execute('INSERT OR IGNORE INTO cameras VALUES (1)')
                        for name in Path(command[command.index('--image_list_path') + 1]).read_text().splitlines():
                            key = int(Path(name).stem) + 1
                            connection.execute('INSERT INTO images VALUES (?,?)', (key, name))
                            connection.execute('INSERT INTO keypoints VALUES (?,30)', (key,))
                    else:
                        for line in Path(command[command.index('--match_list_path') + 1]).read_text().splitlines():
                            first, second = sorted(int(Path(name).stem) + 1 for name in line.split())
                            pair = PAIR_BASE * first + second
                            connection.execute('INSERT INTO matches VALUES (?)', (pair,))
                            same = (6 <= first <= 8) == (6 <= second <= 8)
                            if same:
                                matches = np.column_stack([np.arange(30), np.arange(30)]).astype(np.uint32)
                                connection.execute('INSERT INTO two_view_geometries VALUES (?,30,?)', (pair, matches.tobytes()))
                    connection.commit()
                return 0.0

            # This fixture isolates connectivity; real geometry has separate native regressions.
            with patch('initial_trackability.run_native', side_effect=execute), \
                    patch('initial_trackability.evaluate_geometry', return_value={}):
                selected, eligible = select(native, SimpleNamespace(raw=raw, output=output), choose, rows,
                    {'maximum_exported_frames': 8, 'maximum_wall_seconds': 10, 'jpeg_quality': 95,
                        'position_cell_m': .1, 'translation_min_m': .035,
                        'direction_cell_degrees': 10., 'bridge_gap_seconds': .8}, 1,
                    SimpleNamespace(CAMERA_TOPIC='/fixture'))
            self.assertEqual({row['camera_frame'] for row in selected}, {0, 1, 2, 3, 4, 8, 9, 10})
            receipt = read_json(output / 'INITIAL_TRACKABILITY_RECEIPT.json')
            self.assertEqual(receipt['checked_candidates'], 11)
            self.assertEqual(receipt['passes'], 2)
            self.assertEqual(set(receipt['rejected_names']), {'5.jpg', '6.jpg', '7.jpg'})
            self.assertEqual(set(receipt['rejected_reasons'].values()), {'abstain_disconnected_visual_component'})
            self.assertFalse((output / 'INITIAL_FRAME_MANIFEST.json').exists())
            self.assertFalse(receipt['SfM_model_or_final_poses_used'])
            self.assertTrue(receipt['final_shortlist_connected_by_actual_native_matches'])
            self.assertTrue(all(row['quality'] == {'quality_score': 1} for row in rows))


if __name__ == '__main__':
    unittest.main()
