"""Exercise paired image-space rejection, dependency cascades, and unchanged limits."""

import ast
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pycolmap

from common import ROOT
from destination_camera_admission import registered, stabilize

PROJECT = ROOT
sys.path.insert(0, str(PROJECT / '05_packages/e26_video_to_target_m3_nollm_v1r23_20260927_installed_candidate/code/validated_fresh_runtime'))
from metric_admission import camera_geometry_metrics

tree = ast.parse((PROJECT / '02_code/gaussian_pipeline_optimization_v1_20261005/export_admitted_map.py').read_text(encoding='utf-8'))
function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'filter_points')
scope = {'np': np}
exec(compile(ast.Module(body=[function], type_ignores=[]), '<original_point_filter_for_regression>', 'exec'), scope)
original_filter_points = scope['filter_points']


class DestinationAdmissionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        options = pycolmap.SyntheticDatasetOptions()
        options.num_rigs, options.num_cameras_per_rig, options.num_frames_per_rig = 1, 1, 40
        options.num_points3D = 200
        self.source = pycolmap.synthesize_dataset(options)
        self.destination = pycolmap.Reconstruction(self.source)
        self.names = sorted(registered(self.source))
        self.camera_policy = {'minimum_triangulated_points': 30, 'grid_rows': 4, 'grid_columns': 4,
            'minimum_occupied_grid_cells': 4, 'minimum_bbox_area_fraction': .01,
            'minimum_retained_cameras': 32, 'minimum_retained_fraction_of_sim3_inliers': .9}
        self.point_policy = {'maximum_point_reprojection_error_px': 8., 'minimum_point_track_length': 3,
                            'minimum_retained_sparse_points': 10}

    def reject_destination(self, name):
        image = registered(self.destination)[name]
        camera = self.destination.cameras[image.camera_id]
        for point in image.points2D:
            point.xy = np.array([camera.width * .1, camera.height * .1])

    def run_pair(self, filter_function=original_filter_points, full_count=40):
        return stabilize(self.source, self.destination, self.point_policy, self.camera_policy,
            np.full(3, -1000.), np.full(3, 1000.), filter_function, camera_geometry_metrics,
            full_count, self.output)

    def test_normal_pair_does_not_claim_recovery(self):
        supports, excluded, removed, result = self.run_pair()
        self.assertFalse(result['used'])
        self.assertTrue(result['original_final_support_guard_passed'])
        self.assertEqual((len(supports[0]), len(excluded), len(removed)), (40, 0, 0))
        self.assertFalse((self.output / 'ORIGINAL_DESTINATION_SUPPORT_REJECTION.json').exists())

    def test_destination_only_rejection_preserves_full_denominator_and_original_coordinates(self):
        name = self.names[0]
        image = registered(self.source)[name]
        self.assertTrue(camera_geometry_metrics(self.source, image, self.camera_policy)['passed'])
        self.reject_destination(name)
        support, excluded, removed, result = self.run_pair()
        self.assertEqual(result['full_registered_denominator'], 40)
        self.assertEqual(result['retained_cameras'], 39)
        self.assertEqual(set(excluded), {name})
        self.assertTrue(excluded[name]['source']['passed'])
        self.assertFalse(excluded[name]['destination']['passed'])
        self.assertTrue(all(row['passed'] for side in support for row in side.values()))
        self.assertTrue((self.output / 'source_before_destination_admission/images.bin').is_file())
        self.assertTrue((self.output / 'pinhole_before_destination_admission/images.bin').is_file())
        self.assertFalse(result['training_admission_granted'])

    def test_point_track_cascade_is_rechecked_in_both_spaces(self):
        for model in (self.source, self.destination):
            images = registered(model)
            vulnerable = images[self.names[1]]
            kept = [point.point3D_id for point in vulnerable.points2D if point.has_point3D()][:30]
            for point_id in list(model.points3D):
                point = model.points3D[point_id]
                entries = [(int(entry.image_id), int(entry.point2D_idx)) for entry in point.track.elements]
                for image_id, point_index in entries:
                    if image_id == vulnerable.image_id and point_id not in kept:
                        model.delete_observation(image_id, point_index)
            fragile = model.points3D[kept[0]]
            allowed = {images[name].image_id for name in self.names[:3]}
            entries = [(int(entry.image_id), int(entry.point2D_idx)) for entry in fragile.track.elements]
            for image_id, point_index in entries:
                if image_id not in allowed:
                    model.delete_observation(image_id, point_index)
        self.reject_destination(self.names[0])
        _, excluded, _, result = self.run_pair()
        self.assertEqual(set(excluded), set(self.names[:2]))
        self.assertEqual(len(result['rounds']), 2)
        self.assertEqual(result['retained_cameras'], 38)
        self.assertEqual(set(self.source.points3D), set(self.destination.points3D))

    def test_full_map_retention_is_not_relaxed(self):
        for name in self.names[:5]:
            self.reject_destination(name)
        with self.assertRaisesRegex(ValueError, 'retention'):
            self.run_pair()

    def test_point_bound_is_not_relaxed(self):
        self.point_policy['minimum_retained_sparse_points'] = 201
        with self.assertRaisesRegex(ValueError, 'point bounds'):
            self.run_pair()

    def test_coordinate_error_cannot_be_hidden_by_rejecting_the_view(self):
        self.reject_destination(self.names[0])
        key = next(iter(self.destination.points3D))
        self.destination.points3D[key].xyz += np.array([.01, 0., 0.])
        with self.assertRaisesRegex(ValueError, 'coordinates'):
            self.run_pair()

    def test_filter_cannot_modify_coordinates_in_both_models(self):
        self.reject_destination(self.names[0])

        def broken_filter(model, *args):
            result = original_filter_points(model, *args)
            key = next(iter(model.points3D))
            model.points3D[key].xyz += np.array([.01, 0., 0.])
            self.destination.points3D[key].xyz += np.array([.01, 0., 0.])
            return result

        with self.assertRaisesRegex(ValueError, 'three-dimensional point'):
            self.run_pair(broken_filter)

    def test_missing_pair_is_not_silently_admitted(self):
        self.destination.deregister_frame(registered(self.destination)[self.names[0]].frame_id)
        with self.assertRaisesRegex(ValueError, 'equal camera'):
            self.run_pair()


if __name__ == '__main__':
    unittest.main()
