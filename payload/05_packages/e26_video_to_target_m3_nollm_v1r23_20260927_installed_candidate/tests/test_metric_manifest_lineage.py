"""The Sim3 manifest may omit only frames outside recorded metric time support."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


RUNTIME = Path(__file__).resolve().parents[1] / "code" / "validated_fresh_runtime"
sys.path.insert(0, str(RUNTIME))

from metric_manifest_lineage import sha256_file, verify_metric_manifest_lineage


class MetricManifestLineageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.registered_path = root / "registered.json"
        self.metric_bound_path = root / "metric_bound.json"
        self.receipt_path = root / "support.json"
        self.registered = [
            {"image_name": "early.jpg", "camera_header_ns": 90, "image_sha256": "early"},
            {"image_name": "middle.jpg", "camera_header_ns": 110, "image_sha256": "middle"},
            {"image_name": "late.jpg", "camera_header_ns": 120, "image_sha256": "late"},
        ]
        self.metric_bound = self.registered[1:]
        self.receipt = {
            "schema_version": "e26_metric_time_support_filter_v2",
            "status": "complete",
            "policy_sha256": "policy",
            "input_registered_count": 3,
            "metric_bound_count": 2,
            "metric_bound_fraction": 2 / 3,
            "visual_only_count": 1,
            "minimum_supported_frames": 2,
            "minimum_supported_fraction": 0.5,
            "odometry_start_ns": 100,
            "odometry_end_ns": 130,
            "excluded": [{"image_name": "early.jpg", "camera_header_ns": 90}],
        }
        self.write_fixture()

    def write_fixture(self) -> None:
        for path, value in (
            (self.registered_path, self.registered),
            (self.metric_bound_path, self.metric_bound),
            (self.receipt_path, self.receipt),
        ):
            path.write_text(json.dumps(value), encoding="utf-8")

    def verify(self, manifest_hash: str | None = None):
        return verify_metric_manifest_lineage(
            self.registered_path,
            self.metric_bound_path,
            self.receipt_path,
            manifest_hash or sha256_file(self.metric_bound_path),
            "policy",
        )

    def test_accepts_exact_time_supported_subset(self) -> None:
        registered, metric_bound = self.verify()
        self.assertEqual(registered, self.registered)
        self.assertEqual(metric_bound, self.metric_bound)

    def test_rejects_wrong_sim3_input_hash(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Sim3 input lineage"):
            self.verify("wrong")

    def test_rejects_subset_with_changed_image_identity_even_if_hash_matches(self) -> None:
        self.metric_bound[0] = {**self.metric_bound[0], "image_sha256": "changed"}
        self.write_fixture()
        with self.assertRaisesRegex(RuntimeError, "time-supported subset"):
            self.verify()

    def test_rejects_incorrect_time_filter_even_if_hash_matches(self) -> None:
        self.metric_bound = self.registered[:2]
        self.write_fixture()
        with self.assertRaisesRegex(RuntimeError, "time-supported subset"):
            self.verify()

    def test_rejects_forged_support_receipt(self) -> None:
        self.receipt["excluded"] = []
        self.write_fixture()
        with self.assertRaisesRegex(RuntimeError, "does not describe"):
            self.verify()

    def test_all_public_s06_wrappers_supply_both_lineage_artifacts(self) -> None:
        package = RUNTIME.parents[1]
        for name in (
            "RUN_LEGACY_PGSR_S00_S08.ps1",
            "RUN_RELIABLE_LIDAR_S00_S08.ps1",
            "RUN_TESLA_RELIABLE_S00_S08.ps1",
        ):
            with self.subTest(wrapper=name):
                source = (package / name).read_text(encoding="utf-8-sig")
                self.assertIn("prepare_metric_pgsr_input_inliers.py", source)
                self.assertIn("--metric-bound-manifest", source)
                self.assertIn("--metric-time-support-receipt", source)


if __name__ == "__main__":
    unittest.main()
