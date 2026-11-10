"""Regression cases for serialization ownership after checkpoint continuation."""

import unittest

from ply_serialization import validate_save_schedule


def events(*steps):
    return [{"path": f"output/model/point_cloud/iteration_{step}/point_cloud.ply"} for step in steps]


class SaveScheduleTests(unittest.TestCase):
    def test_resume_after_earlier_saves(self):
        schedule = [2000, 7000, 12000]
        actual = events(12000)
        self.assertNotEqual(len(actual), len(schedule))
        result = validate_save_schedule(actual, schedule, 8000)
        self.assertEqual(result["observed_process_iterations"], [12000])

    def test_continuous_full_schedule(self):
        self.assertEqual(validate_save_schedule(events(2000, 7000, 12000),
            [2000, 7000, 12000], 0)["expected_process_iterations"], [2000, 7000, 12000])

    def test_resume_at_save_boundary(self):
        self.assertEqual(validate_save_schedule(events(7000, 12000),
            [2000, 7000, 12000], 2000)["expected_process_iterations"], [7000, 12000])

    def test_missing_duplicate_or_earlier_save_is_rejected(self):
        for actual in ([], events(12000, 12000), events(7000, 12000), events(7000)):
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                validate_save_schedule(actual, [2000, 7000, 12000], 8000)

    def test_short_continuation(self):
        self.assertEqual(validate_save_schedule(events(1600), [1600], 900)
                         ["expected_process_iterations"], [1600])

    def test_invalid_event_directory_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_save_schedule([{"path": "output/model/final/point_cloud.ply"}], [1600], 900)


if __name__ == "__main__":
    unittest.main()
