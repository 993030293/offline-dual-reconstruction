"""Check LiDAR alignment when metric admission excludes selected images."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "code" / "run_reliable_lidar_tsdf.py"
SPEC = importlib.util.spec_from_file_location("run_reliable_lidar_tsdf", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load {MODULE_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LidarViewAlignmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.manifest = Path(self.directory.name) / "registered_image_manifest.json"
        self.selected = [f"frame_{index:02d}.jpg" for index in range(4)]
        self.manifest.write_text(
            json.dumps([{"image_name": name} for name in self.selected]), encoding="utf-8"
        )
        manifest_hash = MODULE.sha256_file(self.manifest)
        self.lidar = {
            "image_manifest": str(self.manifest),
            "image_manifest_sha256": manifest_hash,
            "image_count": 4,
            "per_frame": [{"image_name": name} for name in self.selected],
        }
        self.metric = {
            "source_registered_manifest_sha256": manifest_hash,
            "registered_camera_count": 3,
            "images": [{"name": name} for name in self.selected[:3]],
        }
        self.views = [Path(name).stem for name in self.selected[:3]]

    def test_selected_anchor_excluded_from_training_is_recorded(self) -> None:
        self.lidar["image_manifest"] = r"C:\original_run\registered_image_manifest.json"
        result = MODULE.assess_lidar_view_alignment(self.lidar, self.metric, self.views, self.manifest)
        self.assertEqual(result["status"], "complete_training_subset")
        self.assertEqual(result["unused_anchor_image_names"], [self.selected[3]])
        self.assertEqual(result["trained_view_count"], 3)

    def test_missing_selected_anchor_fails(self) -> None:
        self.lidar["per_frame"].pop(0)
        self.lidar["image_count"] = 3
        with self.assertRaisesRegex(RuntimeError, "do not exactly cover selected images"):
            MODULE.assess_lidar_view_alignment(self.lidar, self.metric, self.views, self.manifest)

    def test_training_view_without_matching_anchor_fails(self) -> None:
        self.views[0] = "foreign_frame"
        with self.assertRaisesRegex(RuntimeError, "Loaded PGSR views disagree"):
            MODULE.assess_lidar_view_alignment(self.lidar, self.metric, self.views, self.manifest)

    def test_manifest_identity_mismatch_fails(self) -> None:
        self.metric["source_registered_manifest_sha256"] = "0" * 64
        with self.assertRaisesRegex(RuntimeError, "do not share the selected image manifest"):
            MODULE.assess_lidar_view_alignment(self.lidar, self.metric, self.views, self.manifest)


if __name__ == "__main__":
    unittest.main()
