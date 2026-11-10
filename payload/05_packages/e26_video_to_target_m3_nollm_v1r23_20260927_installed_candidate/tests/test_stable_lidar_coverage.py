"""Check sparse-frame admission without inventing LiDAR anchors."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "code" / "extract_stable_registered_lidar.py"
SPEC = importlib.util.spec_from_file_location("extract_stable_registered_lidar", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def rows(counts: list[int]) -> list[dict]:
    return [
        {
            "image_name": f"frame_{index:05d}.jpg",
            "stable_anchor_point_count": count,
            "cropped_non_ground_point_count": count,
        }
        for index, count in enumerate(counts)
    ]


class StableLidarCoverageTest(unittest.TestCase):
    def test_one_isolated_sparse_frame_is_explicitly_recorded(self) -> None:
        counts = [100] * 180
        counts[61] = 2
        coverage = MODULE.assess_anchor_coverage(rows(counts))
        self.assertEqual(coverage["status"], "isolated_sparse_frames")
        self.assertEqual(coverage["sparse_frame_count"], 1)
        self.assertEqual(coverage["sparse_frames"][0]["frame_index"], 61)
        self.assertEqual(coverage["minimum_frame_anchor_count"], 2)

    def test_dense_sequence_preserves_strong_status(self) -> None:
        coverage = MODULE.assess_anchor_coverage(rows([100] * 180))
        self.assertEqual(coverage["status"], "strong_all_frames")
        self.assertEqual(coverage["sparse_frame_count"], 0)

    def test_clustered_or_missing_support_fails_closed(self) -> None:
        adjacent = [100] * 180
        adjacent[61:63] = [2, 3]
        absent = [100] * 180
        absent[61] = 0
        boundary = [100] * 180
        boundary[0] = 2
        for name, counts in (("adjacent", adjacent), ("absent", absent), ("boundary", boundary)):
            with self.subTest(case=name):
                with self.assertRaisesRegex(RuntimeError, "unsupported sparse frame"):
                    MODULE.assess_anchor_coverage(rows(counts))

    def test_many_sparse_frames_or_low_total_fails_closed(self) -> None:
        many = [100] * 180
        for index in (20, 60, 100, 140):
            many[index] = 2
        with self.assertRaisesRegex(RuntimeError, "too many sparse frames"):
            MODULE.assess_anchor_coverage(rows(many))
        with self.assertRaisesRegex(RuntimeError, "total="):
            MODULE.assess_anchor_coverage(rows([20] * 180))


if __name__ == "__main__":
    unittest.main()
