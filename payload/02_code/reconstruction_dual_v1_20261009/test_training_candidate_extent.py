"""Check metric extent scope for retained nontraining camera witnesses."""

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_full_raw_ab_20261008"))

from joint_camera_admission import adapt, training_candidate_extent


def image(name, center, supported):
    return SimpleNamespace(
        name=name,
        has_pose=True,
        supported=supported,
        projection_center=lambda: np.asarray(center, dtype=np.float64),
    )


def support(_model, value, _policy):
    return {"passed": value.supported}


class TrainingCandidateExtentTests(unittest.TestCase):
    def setUp(self):
        self.camera_policy = {
            "minimum_retained_cameras": 2,
            "minimum_retained_fraction_of_sim3_inliers": 0.5,
        }
        self.point_policy = {"maximum_camera_radius_m": 20.0}

    def test_far_nontraining_witness_is_preserved_outside_training_aabb(self):
        model = SimpleNamespace(images={
            1: image("a.jpg", [1.0, 0.0, 0.0], True),
            2: image("b.jpg", [2.0, 0.0, 0.0], True),
            3: image("witness.jpg", [100.0, 0.0, 0.0], False),
        })
        with tempfile.TemporaryDirectory() as directory:
            centers = training_candidate_extent(
                model,
                self.camera_policy,
                self.point_policy,
                support,
                Path(directory),
            )
            self.assertEqual(centers.shape, (2, 3))
            receipt = json.loads(
                (Path(directory) / "PRE_FILTER_TRAINING_EXTENT.json").read_text()
            )
            self.assertEqual(receipt["nontraining_witness_names"], ["witness.jpg"])
            self.assertEqual(receipt["full_map_maximum_camera_radius_m"], 100.0)
            self.assertEqual(receipt["training_candidate_maximum_camera_radius_m"], 2.0)
            self.assertTrue(receipt["thresholds_unchanged"])

    def test_far_supported_camera_remains_fatal(self):
        model = SimpleNamespace(images={
            1: image("a.jpg", [1.0, 0.0, 0.0], True),
            2: image("bad.jpg", [100.0, 0.0, 0.0], True),
        })
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "geometry-supported training candidate"):
                training_candidate_extent(
                    model,
                    self.camera_policy,
                    self.point_policy,
                    support,
                    Path(directory),
                )

    def test_export_adapter_compiles_against_the_packaged_source(self):
        sys.path.insert(
            0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005")
        )
        import export_admitted_map

        original = adapt(export_admitted_map)
        try:
            self.assertIsNot(export_admitted_map.main, original)
            self.assertEqual(
                export_admitted_map.main.__code__.co_filename,
                "<explicit_joint_camera_point_admission_export>",
            )
        finally:
            export_admitted_map.main = original


if __name__ == "__main__":
    unittest.main()
