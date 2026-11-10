"""Choose the strongest observed point-track support that meets the point-count gate."""

from __future__ import annotations


def choose_track_length(attempts: list[dict], minimum_points: int) -> int | None:
    if minimum_points <= 0 or not attempts:
        raise ValueError("Point-count gate and track attempts must be positive")
    previous_length = None
    previous_count = None
    for attempt in attempts:
        track_length = int(attempt["minimum_track_length"])
        retained_points = int(attempt["retained_points"])
        if track_length < 2 or retained_points < 0:
            raise ValueError("Invalid point-track attempt")
        if previous_length is not None and (
            track_length >= previous_length or retained_points < previous_count
        ):
            raise RuntimeError("Point-track attempts are not ordered by decreasing support")
        if retained_points >= minimum_points:
            return track_length
        previous_length = track_length
        previous_count = retained_points
    return None
