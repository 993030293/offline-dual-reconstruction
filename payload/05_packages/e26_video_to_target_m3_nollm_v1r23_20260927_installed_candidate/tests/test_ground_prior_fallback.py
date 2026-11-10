"""Reject a narrow lower artifact while recovering a supported ground plane."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ground" / "code"))
import ground_target_extraction as ground


def grid_patch(count: int, low: float, high: float, height) -> tuple[np.ndarray, np.ndarray]:
    xy = np.linspace(low, high, count)
    x, y = np.meshgrid(xy, xy, indexing="ij")
    z = np.broadcast_to(height(x), x.shape)
    vertices = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    faces = []
    for row in range(count - 1):
        for col in range(count - 1):
            first = row * count + col
            faces.extend(((first, first + count, first + 1), (first + count, first + count + 1, first + 1)))
    return vertices, np.asarray(faces, dtype=np.int64)


class GroundPriorFallbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        patches = (
            grid_patch(61, 0.25, 0.55, lambda x: np.full_like(x, -0.14)),
            grid_patch(41, 0.10, 0.90, lambda x: np.zeros_like(x)),
            grid_patch(81, 0.10, 0.90, lambda x: 0.40 + x),
        )
        vertex_chunks = []
        triangle_chunks = []
        offset = 0
        for vertices, triangles in patches:
            vertex_chunks.append(vertices)
            triangle_chunks.append(triangles + offset)
            offset += len(vertices)
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.concatenate(vertex_chunks)),
            o3d.utility.Vector3iVector(np.concatenate(triangle_chunks).astype(np.int32)),
        )
        mesh.compute_triangle_normals()
        cls.vertices = np.asarray(mesh.vertices)
        cls.triangles = np.asarray(mesh.triangles)
        cls.normals = np.asarray(mesh.triangle_normals)
        config = ground.read_json(ROOT / "ground" / "config.json")
        cls.policy = ground.load_policies(config)[0]
        cls.receipt = {
            "roi_world_lio_m": {"minimum": [0.0, 0.0, -0.2], "maximum": [1.0, 1.0, 1.3]},
            "roi_derivation": {
                "metric_sparse_vehicle_support": {
                    "ground_z_m": 0.0,
                    "ground_support_count": 5000,
                }
            },
        }

    def test_sparse_prior_recovers_wide_ground_below_tilted_object(self) -> None:
        fit = ground.fit_ground_plane(
            self.vertices, self.triangles, self.normals, self.receipt, self.policy
        )
        self.assertEqual(fit["candidate_selection_mode"], "sparse_ground_prior_window")
        self.assertGreaterEqual(fit["inlier_fraction"], self.policy.minimum_accepted_inlier_fraction)
        self.assertTrue(all(value >= 0.6 for value in fit["inlier_roi_span_fraction"]))
        self.assertLess(abs(fit["ground_z_at_roi_center_m"]), 0.02)

    def test_without_sparse_support_the_narrow_artifact_is_rejected(self) -> None:
        no_prior = {**self.receipt, "roi_derivation": {}}
        with self.assertRaises(RuntimeError):
            ground.fit_ground_plane(
                self.vertices, self.triangles, self.normals, no_prior, self.policy
            )


if __name__ == "__main__":
    unittest.main()
