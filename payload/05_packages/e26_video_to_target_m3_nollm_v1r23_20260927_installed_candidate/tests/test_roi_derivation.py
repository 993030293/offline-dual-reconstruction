"""Regression checks for target ROI support selection."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "code" / "derive_vehicle_roi.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("derive_vehicle_roi", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def grid(x_values, y_values, z_values) -> np.ndarray:
    return np.stack(
        np.meshgrid(x_values, y_values, z_values, indexing="ij"), axis=-1
    ).reshape(-1, 3)


class RoiDerivationTest(unittest.TestCase):
    def test_near_equal_height_modes_use_target_region_spatial_support(self) -> None:
        def mode(count: int, size: int, height: float) -> np.ndarray:
            index = np.arange(count) % (size * size)
            x = ((index // size) - size / 2) * 0.05
            y = ((index % size) - size / 2) * 0.05
            return np.column_stack((x, y, np.full(count, height)))

        broad_lower = mode(970, 16, -0.731)
        dense_upper = mode(1000, 8, -0.611)
        selected, audit = MODULE.select_ground_height_mode(
            np.vstack((broad_lower, dense_upper)), np.zeros(2), 2.5
        )
        self.assertAlmostEqual(selected, -0.73)
        self.assertEqual(len(audit["candidates"]), 2)
        self.assertGreater(
            audit["candidates"][0]["inner_occupied_cell_count"],
            audit["candidates"][1]["inner_occupied_cell_count"],
        )

    def test_clear_height_mode_keeps_count_winner(self) -> None:
        def mode(count: int, size: int, height: float) -> np.ndarray:
            index = np.arange(count) % (size * size)
            x = ((index // size) - size / 2) * 0.05
            y = ((index % size) - size / 2) * 0.05
            return np.column_stack((x, y, np.full(count, height)))

        broad_lower = mode(800, 16, -0.731)
        dense_upper = mode(1000, 8, -0.611)
        selected, audit = MODULE.select_ground_height_mode(
            np.vstack((broad_lower, dense_upper)), np.zeros(2), 2.5
        )
        self.assertAlmostEqual(selected, -0.61)
        self.assertEqual(len(audit["candidates"]), 1)

    def test_dense_target_excludes_separate_overhead_support(self) -> None:
        target = grid(
            np.arange(-0.25, 0.26, 0.05),
            np.arange(-0.25, 0.26, 0.05),
            np.arange(0.20, 0.66, 0.05),
        )
        overhead = grid(
            np.arange(3.00, 3.26, 0.05),
            np.arange(2.00, 2.26, 0.05),
            np.arange(1.30, 1.46, 0.05),
        )
        selected, receipt = MODULE.select_roi_support(
            np.vstack((target, overhead)), focus_xy=np.zeros(2), median_depth=2.0
        )
        self.assertEqual(receipt["mode"], "dominant_adaptive_3d_component")
        self.assertEqual(len(selected), len(target))
        self.assertLess(float(selected[:, 0].max()), 1.0)
        self.assertGreater(receipt["component"]["selected_fraction"], 0.75)

    def test_ambiguous_components_fail_closed(self) -> None:
        first = grid(
            np.arange(-0.25, 0.26, 0.05),
            np.arange(-0.25, 0.26, 0.05),
            np.arange(0.20, 0.66, 0.05),
        )
        second = first + np.array([3.0, 2.0, 0.0])
        with self.assertRaisesRegex(RuntimeError, "No dominant non-ground component"):
            MODULE.select_roi_support(
                np.vstack((first, second)), focus_xy=np.zeros(2), median_depth=2.0
            )

    def test_valid_broad_roi_keeps_all_support(self) -> None:
        support = grid(
            np.arange(-1.0, 1.01, 0.10),
            np.arange(-1.0, 1.01, 0.10),
            np.arange(0.20, 0.61, 0.10),
        )
        selected, receipt = MODULE.select_roi_support(
            support, focus_xy=np.zeros(2), median_depth=4.0
        )
        self.assertEqual(receipt["mode"], "all_non_ground_quantiles")
        np.testing.assert_array_equal(selected, support)

    def test_small_distant_target_is_not_rejected_by_camera_depth_ratio(self) -> None:
        support = grid(
            np.arange(-0.30, 0.31, 0.03),
            np.arange(-0.30, 0.31, 0.03),
            np.arange(0.20, 0.51, 0.05),
        )
        selected, receipt = MODULE.select_roi_support(
            support, focus_xy=np.zeros(2), median_depth=10.0
        )
        np.testing.assert_array_equal(selected, support)
        self.assertLess(max(receipt["selected_horizontal_extent_m"]), 0.25 * 10.0)

    def test_large_near_target_is_not_rejected_by_camera_depth_ratio(self) -> None:
        support = grid(
            np.arange(-2.0, 2.01, 0.20),
            np.arange(-1.0, 1.01, 0.20),
            np.arange(0.20, 1.01, 0.20),
        )
        selected, receipt = MODULE.select_roi_support(
            support, focus_xy=np.zeros(2), median_depth=2.0
        )
        np.testing.assert_array_equal(selected, support)
        self.assertGreater(max(receipt["selected_horizontal_extent_m"]), 1.25 * 2.0)

    def test_adaptive_neighborhood_tracks_point_spacing(self) -> None:
        support = grid(
            np.arange(-0.5, 0.51, 0.05),
            np.arange(-0.5, 0.51, 0.05),
            np.arange(0.20, 0.61, 0.10),
        )
        _, small = MODULE.select_roi_support(
            support, focus_xy=np.zeros(2), median_depth=2.0
        )
        _, large = MODULE.select_roi_support(
            3.0 * support, focus_xy=np.zeros(2), median_depth=6.0
        )
        self.assertAlmostEqual(
            large["component"]["epsilon_m"], 3.0 * small["component"]["epsilon_m"]
        )

    def test_dominant_background_away_from_focus_fails_closed(self) -> None:
        background = grid(
            np.arange(2.0, 2.51, 0.05),
            np.arange(2.0, 2.51, 0.05),
            np.arange(0.20, 0.81, 0.05),
        )
        with self.assertRaisesRegex(RuntimeError, "does not contain the optical focus"):
            MODULE.select_roi_support(
                background, focus_xy=np.zeros(2), median_depth=2.0
            )

    def test_low_clearance_clutter_uses_stable_higher_core(self) -> None:
        target = grid(
            np.arange(-0.50, 0.51, 0.05),
            np.arange(-0.50, 0.51, 0.05),
            np.arange(0.30, 0.71, 0.05),
        )
        centers = [(-1.5, -1.5), (-1.5, 1.5), (1.5, -1.5), (1.5, 1.5), (0, 1.6)]
        low_clutter = [
            grid(
                np.arange(x - 0.14, x + 0.15, 0.04),
                np.arange(y - 0.14, y + 0.15, 0.04),
                np.arange(0.09, 0.111, 0.005),
            )
            for x, y in centers
        ]
        selected, _, _, drift, _, coverage, admission = (
            MODULE.select_height_consistent_roi_support(
                np.vstack((target, *low_clutter)),
                focus_xy=np.zeros(2),
                ground_z=0.0,
                search_radius=2.5,
                median_depth=2.0,
            )
        )
        self.assertEqual(len(selected), len(target))
        self.assertEqual(
            admission["mode"], "majority_component_with_stable_higher_clearance_core"
        )
        self.assertLess(drift, 0.20)
        self.assertGreater(coverage, 0.98)

    def test_low_box_extension_may_shrink_at_higher_clearance(self) -> None:
        main = grid(
            np.arange(-0.50, 0.51, 0.05),
            np.arange(-0.40, 0.41, 0.05),
            np.arange(0.14, 0.71, 0.05),
        )
        low_side_box = grid(
            np.arange(0.75, 1.16, 0.05),
            np.arange(-0.20, 0.21, 0.05),
            np.arange(0.20, 0.46, 0.05),
        )
        low_bridge = grid(
            np.arange(0.45, 0.81, 0.05),
            np.arange(-0.20, 0.21, 0.05),
            np.arange(0.09, 0.171, 0.04),
        )
        selected, _, _, drift, core_drift, coverage, admission = (
            MODULE.select_height_consistent_roi_support(
                np.vstack((main, low_side_box, low_bridge)),
                focus_xy=np.zeros(2),
                ground_z=0.0,
                search_radius=2.5,
                median_depth=2.0,
            )
        )
        self.assertGreater(float(selected[:, 0].max()), 1.1)
        self.assertGreater(drift, 0.20)
        self.assertGreater(core_drift, 0.20)
        self.assertEqual(coverage, 1.0)
        self.assertLess(admission["all_height_bound_expansion_m"], 0.01)

    def test_near_equal_low_components_still_fail_closed(self) -> None:
        target = grid(
            np.arange(-0.50, 0.51, 0.05),
            np.arange(-0.50, 0.51, 0.05),
            np.arange(0.30, 0.71, 0.05),
        )
        competing_low_component = grid(
            np.arange(1.0, 2.01, 0.05),
            np.arange(-0.50, 0.51, 0.05),
            np.arange(0.09, 0.111, 0.005),
        )
        with self.assertRaisesRegex(RuntimeError, "remains ambiguous"):
            MODULE.select_height_consistent_roi_support(
                np.vstack((target, competing_low_component)),
                focus_xy=np.zeros(2),
                ground_z=0.0,
                search_radius=2.5,
                median_depth=2.0,
            )


if __name__ == "__main__":
    unittest.main()
