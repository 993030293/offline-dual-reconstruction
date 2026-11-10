"""Regression checks for choosing a coherent S07 surface without scene labels."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "code" / "run_reliable_lidar_tsdf.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("run_reliable_lidar_tsdf", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def candidate(area_fraction: float, component_count: int, support: float, triangles: int) -> dict:
    return {
        "dominant_component_area_fraction": area_fraction,
        "component_count": component_count,
        "sparse_support_fraction_within_5cm": support,
        "triangle_count": triangles,
    }


class SurfaceCandidateSelectionTest(unittest.TestCase):
    def test_broad_surface_restores_supported_box_faces(self) -> None:
        selected, audit = MODULE.select_surface_candidate(
            candidate(0.7804, 10870, 0.9865, 1958279),
            candidate(0.9966, 295, 0.9725, 915406),
            candidate(0.9937, 418, 0.9784, 1000372),
        )
        self.assertEqual(selected, "pgsr_depth_broad")
        self.assertTrue(audit["fused_fragmented"])
        self.assertTrue(audit["visual_coherent"])
        self.assertTrue(audit["visual_supported"])
        self.assertTrue(audit["broad_adds_supported_surface"])

    def test_fragmented_broad_surface_keeps_strict_candidate(self) -> None:
        selected, audit = MODULE.select_surface_candidate(
            candidate(0.8554, 19645, 0.9911, 1800000),
            candidate(0.9917, 813, 0.9741, 1161377),
            candidate(0.9479, 2642, 0.9857, 1390269),
        )
        self.assertEqual(selected, "pgsr_depth_only")
        self.assertFalse(audit["broad_coherent"])

    def test_incoherent_visual_candidate_keeps_fused_surface(self) -> None:
        selected, audit = MODULE.select_surface_candidate(
            candidate(0.8021, 24289, 0.98, 2000000),
            candidate(0.8759, 7682, 0.97, 1500000),
            candidate(0.91, 4000, 0.98, 1700000),
        )
        self.assertEqual(selected, "registered_lidar_fused")
        self.assertFalse(audit["visual_coherent"])

    def test_clean_but_unsupported_visual_candidate_is_not_promoted(self) -> None:
        selected, audit = MODULE.select_surface_candidate(
            candidate(0.80, 12000, 0.99, 2000000),
            candidate(0.995, 300, 0.90, 1000000),
            candidate(0.993, 350, 0.91, 1120000),
        )
        self.assertEqual(selected, "registered_lidar_fused")
        self.assertFalse(audit["visual_supported"])

    def test_broad_candidate_cannot_bypass_failed_strict_candidate(self) -> None:
        selected, audit = MODULE.select_surface_candidate(
            candidate(0.80, 12000, 0.99, 2000000),
            candidate(0.975, 1100, 0.97, 1000000),
            candidate(0.991, 600, 0.98, 1120000),
        )
        self.assertEqual(selected, "registered_lidar_fused")
        self.assertTrue(audit["broad_coherent"])
        self.assertFalse(audit["visual_coherent"])


if __name__ == "__main__":
    unittest.main()
