"""Check fail-closed visual-component metric gauge recovery."""

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pycolmap

import metric_component_alignment_recovery as recovery


def fixture(frames=40, points=400):
    options = pycolmap.SyntheticDatasetOptions()
    options.num_rigs = 1
    options.num_cameras_per_rig = 1
    options.num_frames_per_rig = frames
    options.num_points3D = points
    options.camera_model_id = pycolmap.CameraModelId.OPENCV
    options.camera_params = [
        1280.0,
        1270.0,
        512.0,
        384.0,
        0.05,
        0.001,
        0.0003,
        -0.0002,
    ]
    return pycolmap.synthesize_dataset(options)


def serializable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serializable(item) for item in value]
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serializable(value), indent=2), encoding="utf-8")


def observation_identity(model):
    return [
        (int(image.image_id), index, int(point.point3D_id))
        for image in sorted(model.images.values(), key=lambda value: value.image_id)
        for index, point in enumerate(image.points2D)
        if point.has_point3D()
    ]


HELPERS = SimpleNamespace(serializable=serializable)
CHECKS = SimpleNamespace(
    observation_identity=observation_identity,
    atomic_json=atomic_json,
)


def test_policy(minimum_frames=30):
    return (
        Path("synthetic_policy.json"),
        {
            "maximum_camera_radius_m": 100.0,
            "minimum_retained_sparse_points": 1,
        },
        {"minimum_retained_fraction_of_sim3_inliers": 0.9},
        {"minimum_supported_frames": minimum_frames},
    )


class TrackComponentTest(unittest.TestCase):
    def test_native_edge_threshold_separates_weak_subgraph(self):
        weights = np.zeros((5, 5), dtype=np.uint32)
        weights[0, 1] = weights[1, 0] = 40
        weights[1, 2] = weights[2, 1] = 5
        weights[2, 3] = weights[3, 2] = 35
        weights[3, 4] = weights[4, 3] = 35
        components = recovery._connected_components(weights, 30)
        self.assertEqual(components, [[2, 3, 4], [0, 1]])

    def test_component_transform_preserves_pixel_reprojection(self):
        model = fixture()
        names = sorted(image.name for image in model.images.values())
        point_ids = list(model.points3D)
        before, invalid_before = recovery._reprojection_errors(model)
        transform = pycolmap.Sim3d(
            0.7,
            pycolmap.Rotation3d(np.array([0.0, 0.0, 0.1, 0.99498743710662])),
            np.array([2.0, -1.0, 0.5]),
        )
        recovery._transform_component(model, names, point_ids, transform)
        after, invalid_after = recovery._reprojection_errors(model)
        self.assertEqual(invalid_before, 0)
        self.assertEqual(invalid_after, 0)
        np.testing.assert_allclose(after, before, atol=1e-6, rtol=0)


class ComponentRecoveryTest(unittest.TestCase):
    def test_supported_component_recovers_metric_gauge_without_new_observations(self):
        model = fixture()
        target = pycolmap.Reconstruction(model)
        target.transform(
            pycolmap.Sim3d(
                0.8,
                pycolmap.Rotation3d(
                    np.array([0.0, 0.0, 0.05, 0.998749217771909])
                ),
                np.array([1.0, -0.5, 0.25]),
            )
        )
        rows = [
            {
                "image_id": image.image_id,
                "image_name": image.name,
                "C_W_m": target.images[image.image_id].projection_center(),
            }
            for image in model.images.values()
        ]
        identity = observation_identity(model)
        with tempfile.TemporaryDirectory() as directory, patch.object(
            recovery, "_load_policy", return_value=test_policy()
        ):
            recovered, result = recovery.recover_metric_geometry(
                model,
                rows,
                [],
                Path(directory) / "recovery",
                HELPERS,
                CHECKS,
            )
            self.assertEqual(result["status"], "completed_component_metric_guards")
            self.assertEqual(result["retained_registered_cameras"], 40)
            self.assertEqual(result["new_observations_invented"], False)
            self.assertTrue(result["source_geometry_unchanged"])
            self.assertTrue(result["retained_observation_identity_preserved"])
            self.assertLess(result["maximum_reprojection_error_delta_px"], 1e-6)
            self.assertEqual(recovered.compute_num_observations(), model.compute_num_observations())
            self.assertEqual(identity, observation_identity(recovered))
            self.assertEqual(identity, observation_identity(model))
            self.assertTrue((Path(result["model"]) / "images.bin").is_file())

    def test_component_below_existing_minimum_prior_count_is_rejected(self):
        model = fixture(frames=20)
        rows = [
            {
                "image_id": image.image_id,
                "image_name": image.name,
                "C_W_m": image.projection_center(),
            }
            for image in model.images.values()
        ]
        with tempfile.TemporaryDirectory() as directory, patch.object(
            recovery, "_load_policy", return_value=test_policy(minimum_frames=30)
        ):
            with self.assertRaisesRegex(ValueError, "No visual-track component"):
                recovery.recover_metric_geometry(
                    model,
                    rows,
                    [],
                    Path(directory) / "rejected",
                    HELPERS,
                    CHECKS,
                )
            self.assertTrue((Path(directory) / "rejected" / "FAILURE.json").is_file())

    def test_weak_component_camera_is_retained_as_nontraining_witness(self):
        model = fixture()
        witness = max(model.images.values(), key=lambda image: image.image_id)
        for index in reversed(range(len(witness.points2D))):
            if witness.points2D[index].has_point3D():
                model.delete_observation(witness.image_id, index)
        rotation = witness.cam_from_world().rotation.matrix()
        far_center = np.array([200.0, 0.0, 0.0])
        witness.frame.set_cam_from_world(
            witness.camera_id,
            pycolmap.Rigid3d(
                pycolmap.Rotation3d(rotation), -rotation @ far_center
            ),
        )
        rows = [
            {
                "image_id": image.image_id,
                "image_name": image.name,
                "C_W_m": image.projection_center(),
            }
            for image in model.images.values()
        ]
        with tempfile.TemporaryDirectory() as directory, patch.object(
            recovery, "_load_policy", return_value=test_policy()
        ):
            recovered, result = recovery.recover_metric_geometry(
                model,
                rows,
                [],
                Path(directory) / "witness",
                HELPERS,
                CHECKS,
            )
            self.assertEqual(recovered.num_reg_images(), model.num_reg_images())
            self.assertEqual(result["retained_registered_cameras"], 40)
            self.assertEqual(result["metric_gauged_registered_cameras"], 39)
            self.assertEqual(result["dropped_camera_names"], [])
            self.assertEqual(result["metric_gauge_excluded_camera_names"], [witness.name])
            self.assertEqual(result["nontraining_witness_observations"][witness.name], 0)
            self.assertGreater(result["full_map_camera_radius_m"]["maximum"], 100.0)
            self.assertLess(result["camera_radius_m"]["maximum"], 100.0)
            self.assertTrue(result["full_registered_denominator_preserved"])

    def test_verified_aligned_input_role_is_recorded_without_claiming_visual_ba(self):
        model = fixture()
        rows = [
            {
                "image_id": image.image_id,
                "image_name": image.name,
                "C_W_m": image.projection_center(),
            }
            for image in model.images.values()
        ]
        with tempfile.TemporaryDirectory() as directory, patch.object(
            recovery, "_load_policy", return_value=test_policy()
        ):
            _, result = recovery.recover_metric_geometry(
                model,
                rows,
                [],
                Path(directory) / "verified_input",
                HELPERS,
                CHECKS,
                source_role="verified_aligned_input_before_metric_BA",
            )
            self.assertEqual(
                result["source_visual_geometry_authority"],
                "verified_aligned_input_before_metric_BA",
            )
            self.assertFalse(result["source_visual_BA_required_and_retained_as_authority"])
            self.assertTrue(result["verified_aligned_input_required_and_retained_as_authority"])


if __name__ == "__main__":
    unittest.main()
