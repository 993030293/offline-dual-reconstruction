"""Check fail-closed selection of the component metric visual reference."""

from pathlib import Path
import sys
import unittest

import pycolmap


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))

import run_metric_refinement as metric


def fixture():
    options = pycolmap.SyntheticDatasetOptions()
    options.num_rigs = 1
    options.num_cameras_per_rig = 1
    options.num_frames_per_rig = 8
    options.num_points3D = 80
    options.camera_model_id = pycolmap.CameraModelId.OPENCV
    options.camera_params = [800.0, 795.0, 320.0, 240.0, 0.01, 0.0, 0.0, 0.0]
    return pycolmap.synthesize_dataset(options)


class ComponentReferenceTest(unittest.TestCase):
    def test_rejected_visual_ba_falls_back_to_verified_input_geometry(self):
        model = fixture()
        identity = metric.observation_identity(model)
        validity = metric.projection_validity(model)
        results = {
            "visual_metric_aligned": {
                "status": "failed_projection_validation",
                "solver": {"usable": True, "termination_type": "NO_CONVERGENCE"},
            }
        }
        reference, evidence = metric.select_component_gauge_reference(
            model, results, validity, identity, 2 * len(identity)
        )
        self.assertEqual(evidence["source_role"], metric.VERIFIED_ALIGNED_INPUT_ROLE)
        self.assertFalse(evidence["visual_BA_geometry_authority"])
        self.assertEqual(metric.observation_identity(reference), identity)
        self.assertEqual(metric.projection_validity(reference), validity)

    def test_invalid_input_reference_remains_fatal(self):
        model = fixture()
        identity = metric.observation_identity(model)
        results = {
            "visual_metric_aligned": {
                "status": "failed_projection_validation",
                "solver": {"usable": True},
            }
        }
        with self.assertRaisesRegex(ValueError, "projection-valid aligned input"):
            metric.select_component_gauge_reference(
                model,
                results,
                {"valid": False, "observation_count": len(identity), "invalid_count": 1, "invalid": [{}]},
                identity,
                2 * len(identity),
            )

    def test_changed_observation_identity_remains_fatal(self):
        model = fixture()
        identity = metric.observation_identity(model)
        results = {
            "visual_metric_aligned": {
                "status": "failed_projection_validation",
                "solver": {"usable": True},
            }
        }
        with self.assertRaisesRegex(ValueError, "observation identity changed"):
            metric.select_component_gauge_reference(
                model,
                results,
                metric.projection_validity(model),
                identity[:-1],
                2 * len(identity),
            )


if __name__ == "__main__":
    unittest.main()
