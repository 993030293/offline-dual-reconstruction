"""Inspect an MCAP and choose a deterministic supported runtime profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ODOMETRY_TOPIC = "/localization/pose/odom"
STATIC_TRANSFORM_TOPIC = "/tf_static"
REGISTERED_LIDAR_TOPIC = "/localization/lidar/points_registered_full"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcap", required=True, type=Path)
    args = parser.parse_args()

    from rosbags.highlevel import AnyReader

    mcap = args.mcap.resolve(strict=True)
    with AnyReader([mcap]) as reader:
        topics = [connection.topic for connection in reader.connections]
    counts = {
        "embedded_odometry": topics.count(ODOMETRY_TOPIC),
        "static_transform": topics.count(STATIC_TRANSFORM_TOPIC),
        "registered_lidar": topics.count(REGISTERED_LIDAR_TOPIC),
    }
    if counts["static_transform"] != 1:
        raise RuntimeError(
            f"expected exactly one {STATIC_TRANSFORM_TOPIC} connection, "
            f"found {counts['static_transform']}"
        )
    profile = (
        "ReliableLidar"
        if counts["embedded_odometry"] == 1 and counts["registered_lidar"] == 1
        else "LegacyPgsr"
    )
    result = {
        "schema_version": "e26_unattended_profile_routing_v1",
        "status": "complete",
        "mcap": str(mcap),
        "topic_connection_counts": counts,
        "selected_profile": profile,
        "language_model_used": False,
        "visual_language_model_used": False,
        "manual_visual_decision_used": False,
        "routing_rule": (
            "ReliableLidar requires one embedded odometry connection and one registered LiDAR "
            "connection; all other supported inputs use LegacyPgsr."
        ),
    }
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
