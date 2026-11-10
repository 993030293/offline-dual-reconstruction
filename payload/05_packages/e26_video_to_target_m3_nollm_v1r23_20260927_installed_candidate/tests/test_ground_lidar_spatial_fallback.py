"""Keep the mesh span gate unless independent registered LiDAR proves support."""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import open3d as o3d


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ground" / "code"))
import ground_target_extraction as ground


class GroundLidarSpatialFallbackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        axis = np.linspace(0.2, 0.75, 61)
        x, y = np.meshgrid(axis, axis, indexing="ij")
        vertices = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        faces = []
        for row in range(60):
            for col in range(60):
                first = row * 61 + col
                faces.extend(((first, first + 61, first + 1), (first + 61, first + 62, first + 1)))
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32)),
        )
        mesh.compute_triangle_normals()
        cls.vertices = np.asarray(mesh.vertices)
        cls.triangles = np.asarray(mesh.triangles)
        cls.normals = np.asarray(mesh.triangle_normals)
        cls.policy = ground.load_policies(ground.read_json(ROOT / "ground" / "config.json"))[0]
        cls.receipt = {
            "roi_world_lio_m": {"minimum": [0.0, 0.0, -0.2], "maximum": [1.0, 1.0, 1.0]},
            "roi_derivation": {"metric_sparse_vehicle_support": {
                "ground_z_m": 0.0, "ground_support_count": 5000,
            }},
        }

    def test_narrow_mesh_without_independent_mcap_still_fails(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "does not span enough"):
            ground.fit_ground_plane(
                self.vertices, self.triangles, self.normals, self.receipt, self.policy
            )

    def test_narrow_mesh_accepted_only_with_registered_lidar_proof(self) -> None:
        evidence = {"status": "accepted", "roi_span_fraction": [0.9, 0.8]}
        with patch(
            "registered_lidar_ground_support.validate_registered_lidar_support",
            return_value=evidence,
        ) as validate:
            fit = ground.fit_ground_plane(
                self.vertices, self.triangles, self.normals, self.receipt,
                self.policy, raw_mcap=Path("current_raw.mcap"),
            )
        self.assertEqual(fit["candidate_selection_mode"], "sparse_ground_prior_window")
        self.assertIs(fit["registered_lidar_spatial_support"], evidence)
        self.assertEqual(validate.call_count, 1)

    def test_insufficient_independent_support_remains_failed_closed(self) -> None:
        with patch(
            "registered_lidar_ground_support.validate_registered_lidar_support",
            side_effect=RuntimeError("Registered LiDAR ground support is insufficient"),
        ):
            with self.assertRaisesRegex(RuntimeError, "insufficient"):
                ground.fit_ground_plane(
                    self.vertices, self.triangles, self.normals, self.receipt,
                    self.policy, raw_mcap=Path("current_raw.mcap"),
                )

    def test_short_mesh_axis_requires_raw_lidar_extent(self) -> None:
        x, y = np.meshgrid(np.linspace(0.2, 0.6, 61),
                           np.linspace(0.1, 0.8, 61), indexing="ij")
        vertices = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        faces = []
        for row in range(60):
            for col in range(60):
                first = row * 61 + col
                faces.extend(((first, first + 61, first + 1),
                              (first + 61, first + 62, first + 1)))
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32)),
        )
        mesh.compute_triangle_normals()
        triangles = np.asarray(mesh.triangles)
        normals = np.asarray(mesh.triangle_normals)
        with self.assertRaisesRegex(RuntimeError, "lacks horizontal support"):
            ground.fit_ground_plane(
                vertices, triangles, normals, self.receipt, self.policy
            )
        with patch(
            "registered_lidar_ground_support.validate_registered_lidar_support",
            return_value={"status": "accepted"},
        ) as validate:
            fit = ground.fit_ground_plane(
                vertices, triangles, normals, self.receipt, self.policy,
                raw_mcap=Path("current_raw.mcap"),
            )
        self.assertFalse(fit["mesh_absolute_extent_gate_passed"])
        self.assertEqual(fit["registered_lidar_spatial_support"]["status"], "accepted")
        self.assertEqual(validate.call_count, 1)

    def test_lidar_corroborates_mesh_fraction_between_m3_and_s08_gates(self) -> None:
        vertex_chunks = []
        face_chunks = []
        offset = 0
        for count, height in ((51, -0.04), (45, 0.0), (40, 0.04)):
            axis = np.linspace(0.2, 0.75, count)
            x, y = np.meshgrid(axis, axis, indexing="ij")
            vertices = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, height)))
            faces = []
            for row in range(count - 1):
                for col in range(count - 1):
                    first = row * count + col + offset
                    faces.extend(((first, first + count, first + 1),
                                  (first + count, first + count + 1, first + 1)))
            vertex_chunks.append(vertices)
            face_chunks.append(np.asarray(faces, dtype=np.int32))
            offset += len(vertices)
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.concatenate(vertex_chunks)),
            o3d.utility.Vector3iVector(np.concatenate(face_chunks)),
        )
        mesh.compute_triangle_normals()
        policy = replace(self.policy, low_centroid_quantile=0.95,
                         minimum_accepted_inlier_fraction=0.50)
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        normals = np.asarray(mesh.triangle_normals)
        with self.assertRaisesRegex(RuntimeError, "not dominant"):
            ground.fit_ground_plane(vertices, triangles, normals, self.receipt, policy)
        with patch(
            "registered_lidar_ground_support.validate_registered_lidar_support",
            return_value={"status": "accepted"},
        ) as validate:
            fit = ground.fit_ground_plane(
                vertices, triangles, normals, self.receipt, policy,
                raw_mcap=Path("current_raw.mcap"),
                corroborated_minimum_inlier_fraction=0.35,
            )
        self.assertGreaterEqual(fit["inlier_fraction"], 0.35)
        self.assertLess(fit["inlier_fraction"], 0.50)
        self.assertEqual(fit["inlier_fraction_acceptance"], "registered_lidar_corroborated")
        self.assertEqual(validate.call_count, 1)


if __name__ == "__main__":
    unittest.main()
