"""Check observation metrics for registered images without sparse points."""

from types import SimpleNamespace
import unittest

from run_prior_ba import observation_metrics


class ObservationMetricsTest(unittest.TestCase):
    def test_registered_zero_observation_image_is_reported_without_projection(self):
        image = SimpleNamespace(
            image_id=1,
            name="zero_observation.jpg",
            has_pose=True,
            points2D=[],
        )
        model = SimpleNamespace(
            images={1: image},
            points3D={},
            cameras={},
            num_reg_images=lambda: 1,
            num_points3D=lambda: 0,
        )
        result = observation_metrics(model)
        self.assertEqual(result["registered_images"], 1)
        self.assertEqual(result["observations"], 0)
        self.assertEqual(result["reprojection_px"], {"count": 0})
        self.assertEqual(result["per_image"][0]["observations"], 0)
        self.assertEqual(result["per_image"][0]["count"], 0)


if __name__ == "__main__":
    unittest.main()
