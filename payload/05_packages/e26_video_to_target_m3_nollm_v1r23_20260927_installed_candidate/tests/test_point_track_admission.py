"""A weaker two-view track is used only when the stronger gate lacks enough points."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


RUNTIME = Path(__file__).resolve().parents[1] / "code" / "validated_fresh_runtime"
sys.path.insert(0, str(RUNTIME))

from point_track_admission import choose_track_length


class PointTrackAdmissionTest(unittest.TestCase):
    def test_three_view_support_wins_when_sufficient(self) -> None:
        attempts = [{"minimum_track_length": 3, "retained_points": 33114}]
        self.assertEqual(choose_track_length(attempts, 10000), 3)

    def test_two_view_support_recovers_small_but_valid_model(self) -> None:
        attempts = [
            {"minimum_track_length": 3, "retained_points": 7040},
            {"minimum_track_length": 2, "retained_points": 10664},
        ]
        self.assertEqual(choose_track_length(attempts, 10000), 2)

    def test_insufficient_two_view_support_still_fails_closed(self) -> None:
        attempts = [
            {"minimum_track_length": 3, "retained_points": 7040},
            {"minimum_track_length": 2, "retained_points": 9999},
        ]
        self.assertIsNone(choose_track_length(attempts, 10000))

    def test_weaker_filter_must_not_reduce_point_count(self) -> None:
        attempts = [
            {"minimum_track_length": 3, "retained_points": 7040},
            {"minimum_track_length": 2, "retained_points": 7000},
        ]
        with self.assertRaisesRegex(RuntimeError, "not ordered"):
            choose_track_length(attempts, 10000)


if __name__ == "__main__":
    unittest.main()
