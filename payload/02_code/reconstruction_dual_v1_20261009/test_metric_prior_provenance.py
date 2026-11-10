"""Check that metric receipts describe the actual bound coarse source rows."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gaussian_pipeline_optimization_v1_20261005"))
from coarse_prior_provenance import (CURRENT_ALIGNMENT_SOURCE, LEGACY_ALIGNMENT_SOURCE,
                                    summarize_used_priors, unique_coarse_lookup, validate_alignment_source)


class MetricPriorProvenanceTests(unittest.TestCase):
    def summarize(self, rows, names):
        return summarize_used_priors(unique_coarse_lookup(rows), [{"image_name": name} for name in names])

    def test_fresh_generated_positions_are_not_recorded_odometry(self):
        actual = self.summarize([{"image_name": "a", "source_role": "fresh_raw_LiDAR_IMU_coarse_estimate",
            "pose_producer": "raw_IMU_deskew_and_local_map_translation_ICP", "coarse_source_run_root": "current"}], ["a"])
        self.assertEqual(actual["source_role_counts"], {"fresh_raw_LiDAR_IMU_coarse_estimate": 1})
        self.assertEqual(actual["coarse_pose_producer"], "raw_IMU_deskew_and_local_map_translation_ICP")
        self.assertFalse(actual["GT_used"])

    def test_recorded_unknown_producer_remains_unknown(self):
        actual = self.summarize([{"image_name": "a", "source_role": "recorded_raw_odometry_coarse_estimate",
                                 "pose_producer": "unconfirmed", "time_offset_ns": 0}], ["a"])
        self.assertEqual(actual["source_role_counts"], {"recorded_raw_odometry_coarse_estimate": 1})
        self.assertEqual(actual["coarse_pose_producer"], "unconfirmed")
        self.assertFalse(actual["time_offset_verified"])

    def test_unbound_candidates_do_not_change_actual_source(self):
        actual = self.summarize([{"image_name": "a", "pose_producer": "used"},
                                 {"image_name": "unused", "pose_producer": "other"}], ["a"])
        self.assertEqual(actual["pose_producer_counts"], {"used": 1})
        self.assertEqual(actual["supported_position_prior_count"], 1)

    def test_mixed_sources_are_reported_without_invention(self):
        actual = self.summarize([{"image_name": "a", "source_role": "recorded", "pose_producer": "first"},
                                 {"image_name": "b", "source_role": "fresh", "pose_producer": "second"}], ["a", "b"])
        self.assertEqual(actual["coarse_pose_producer"], "mixed")
        self.assertEqual(actual["source_role_counts"], {"fresh": 1, "recorded": 1})

    def test_absent_or_empty_labels_are_unconfirmed(self):
        actual = self.summarize([{"image_name": "a", "pose_producer": " "}], ["a"])
        self.assertEqual(actual["coarse_pose_producer"], "unconfirmed")
        self.assertEqual(actual["source_role_counts"], {"unconfirmed": 1})

    def test_duplicate_input_or_bound_names_fail(self):
        with self.assertRaises(ValueError):
            unique_coarse_lookup([{"image_name": "a"}, {"image_name": "a"}])
        with self.assertRaises(ValueError):
            self.summarize([{"image_name": "a"}], ["a", "a"])

    def test_missing_source_or_empty_binding_fails(self):
        for names in (["absent"], []):
            with self.subTest(names=names), self.assertRaises(ValueError):
                self.summarize([{"image_name": "a"}], names)

    def test_current_and_historical_alignment_sources_are_compatible(self):
        provenance = self.summarize([{"image_name": "a"}], ["a"])
        validate_alignment_source({"source": CURRENT_ALIGNMENT_SOURCE, "old_alignment_reused": False,
                                   "coarse_prior_provenance": provenance})
        validate_alignment_source({"source": LEGACY_ALIGNMENT_SOURCE, "old_alignment_reused": False})

    def test_false_source_inconsistent_counts_or_reuse_fails(self):
        provenance = self.summarize([{"image_name": "a"}], ["a"])
        bad_count = {**provenance, "supported_position_prior_count": 2}
        for row in ({"source": "GT", "old_alignment_reused": False},
                    {"source": CURRENT_ALIGNMENT_SOURCE, "old_alignment_reused": False},
                    {"source": CURRENT_ALIGNMENT_SOURCE, "old_alignment_reused": False, "coarse_prior_provenance": bad_count},
                    {"source": LEGACY_ALIGNMENT_SOURCE, "old_alignment_reused": True}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                validate_alignment_source(row)


if __name__ == "__main__":
    unittest.main()
