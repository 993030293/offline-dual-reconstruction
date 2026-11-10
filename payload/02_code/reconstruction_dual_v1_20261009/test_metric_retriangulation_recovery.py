"""Check metric retriangulation accounting and bounded geometry guards."""

from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pycolmap

import metric_retriangulation_recovery as recovery
from run_native_supported import metric_retriangulation_inputs


def fixture(frames=4, points=20):
    options = pycolmap.SyntheticDatasetOptions()
    options.num_rigs = 1
    options.num_cameras_per_rig = 1
    options.num_frames_per_rig = frames
    options.num_points3D = points
    options.camera_model_id = pycolmap.CameraModelId.OPENCV
    options.camera_params = [1280.0, 1270.0, 512.0, 384.0, 0.05, 0.001, 0.0003, -0.0002]
    return pycolmap.synthesize_dataset(options)


class MetricRetriangulationRecoveryTest(unittest.TestCase):
    def test_center_replacement_preserves_rotations_and_reads_back_exact_centers(self):
        model = fixture()
        rotations = {
            image.name: image.cam_from_world().rotation.matrix().copy()
            for image in model.images.values()
        }
        rows = {
            image.name: {
                "image_name": image.name,
                "C_W_m": image.projection_center() + np.array([1.0, -2.0, 0.5]),
            }
            for image in model.images.values()
        }
        result = recovery._replace_supported_centers(model, rows)
        self.assertLess(result["maximum"], 1e-9)
        for image in model.images.values():
            np.testing.assert_allclose(image.projection_center(), rows[image.name]["C_W_m"], atol=1e-9, rtol=0)
            np.testing.assert_array_equal(image.cam_from_world().rotation.matrix(), rotations[image.name])

    def test_factor_accounting_uses_only_active_position_priors(self):
        solution = {"num_residuals": 200 + 3 * 7, "num_residual_blocks": 100 + 7}
        self.assertTrue(recovery._factor_count_verified(solution, 200, 7))
        self.assertFalse(recovery._factor_count_verified(solution, 200, 8))

    def test_excessive_invalid_track_pruning_is_rejected_before_mutation(self):
        model = fixture(points=20)
        point_id = next(iter(model.points3D))
        audit = {"invalid": [{"point3D_id": int(point_id)}]}
        points_before = model.num_points3D()
        with self.assertRaisesRegex(ValueError, "bounded recovery fractions"):
            recovery._prune_invalid_tracks(model, audit)
        self.assertEqual(model.num_points3D(), points_before)
        self.assertIn(point_id, model.points3D)

    def test_bounded_invalid_track_pruning_deletes_only_evidenced_track(self):
        model = fixture(points=20)
        point_id = next(iter(model.points3D))
        track_length = model.points3D[point_id].track.length()
        audit = {"invalid": [{"point3D_id": int(point_id)}]}
        with patch.object(recovery, "MAXIMUM_PRUNED_POINT_FRACTION", 0.1), patch.object(
            recovery, "MAXIMUM_PRUNED_OBSERVATION_FRACTION", 0.1
        ):
            result = recovery._prune_invalid_tracks(model, audit)
        self.assertEqual(result["removed_points"], 1)
        self.assertEqual(result["removed_observations"], track_length)
        self.assertNotIn(point_id, model.points3D)

    def test_adapter_prefers_final_point_rebuilding_database(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = Path(directory)
            final = backend / "all_point_rebuilding"
            earlier = backend / "native_visual_recovery"
            images = backend / "images"
            final.mkdir()
            earlier.mkdir()
            images.mkdir()
            (final / "database.db").write_bytes(b"final")
            (earlier / "database.db").write_bytes(b"earlier")
            (final / "RESULT.json").write_text(
                json.dumps({"database": str(final / "database.db"), "images": str(images)})
            )
            (earlier / "RESULT.json").write_text(
                json.dumps({"database": str(earlier / "database.db"), "images": str(images)})
            )
            database, selected_images, receipt = metric_retriangulation_inputs(backend, {})
            self.assertTrue(database.samefile(final / "database.db"))
            self.assertTrue(selected_images.samefile(images))
            self.assertTrue(receipt.samefile(final / "RESULT.json"))

    def test_native_database_copy_can_change_without_mutating_its_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.db"
            owned = Path(directory) / "owned.db"
            with closing(sqlite3.connect(source)) as connection:
                connection.execute("CREATE TABLE images(name TEXT NOT NULL)")
                connection.execute("INSERT INTO images VALUES ('frame.jpg')")
                connection.commit()
            source_digest, names = recovery._copy_database(source, owned)
            with closing(sqlite3.connect(owned)) as connection:
                connection.execute("INSERT INTO images VALUES ('native_write.jpg')")
                connection.commit()
            self.assertEqual(recovery._file_digest(source), source_digest)
            self.assertEqual(names, {"frame.jpg"})
            self.assertNotEqual(recovery._file_digest(owned), source_digest)


if __name__ == "__main__":
    unittest.main()
