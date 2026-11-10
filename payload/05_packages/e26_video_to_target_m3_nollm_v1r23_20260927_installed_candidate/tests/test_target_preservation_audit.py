"""Check that the full-run gate catches target surfaces removed by M3."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import open3d as o3d

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from audit_target_preservation import audit


def grid_mesh(columns: int, rows: int) -> o3d.geometry.TriangleMesh:
    x, y = np.meshgrid(np.arange(columns) * 0.02, np.arange(rows) * 0.02, indexing="ij")
    points = np.column_stack((x.ravel(), y.ravel(), np.full(x.size, 0.5)))
    cells = np.arange(columns * rows).reshape(columns, rows)
    a = cells[:-1, :-1].ravel()
    b = cells[1:, :-1].ravel()
    c = cells[:-1, 1:].ravel()
    d = cells[1:, 1:].ravel()
    faces = np.vstack((np.column_stack((a, b, c)), np.column_stack((b, d, c))))
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(faces.astype(np.int32))
    return mesh


class TargetPreservationAuditTests(unittest.TestCase):
    def test_deleted_half_of_observed_target_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = grid_mesh(50, 30)
            target = grid_mesh(25, 30)
            raw_path = root / "raw.ply"
            target_path = root / "target.ply"
            self.assertTrue(o3d.io.write_triangle_mesh(str(raw_path), raw))
            self.assertTrue(o3d.io.write_triangle_mesh(str(target_path), target))
            sparse = root / "points3D.txt"
            sparse.write_text(
                "".join(
                    f"{index + 1} {x:.6f} {y:.6f} {z:.6f}\n"
                    for index, (x, y, z) in enumerate(np.asarray(raw.vertices))
                ),
                encoding="utf-8",
            )
            receipt = root / "tsdf.json"
            receipt.write_text(json.dumps({
                "roi_world_lio_m": {"minimum": [-0.1, -0.1, -0.1], "maximum": [1.1, 0.7, 1.0]},
                "roi_derivation": {"spatial_sparse_floor_plane": {
                    "normal": [0.0, 0.0, 1.0], "offset_m": 0.0,
                }},
            }), encoding="utf-8")
            passed = audit(sparse, raw_path, raw_path, receipt)
            failed = audit(sparse, raw_path, target_path, receipt)
            self.assertEqual(passed["status"], "pass")
            self.assertEqual(failed["status"], "failed_quality_gate")
            self.assertLess(failed["retained_raw_support_fraction"], 0.8)


if __name__ == "__main__":
    unittest.main()
