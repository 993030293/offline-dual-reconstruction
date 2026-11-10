"""Detect whether an MCAP contains the embedded metric odometry contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ODOMETRY_TOPIC = "/localization/pose/odom"
TF_STATIC_TOPIC = "/tf_static"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcap", required=True, type=Path)
    args = parser.parse_args()

    from rosbags.highlevel import AnyReader

    mcap = args.mcap.resolve(strict=True)
    with AnyReader([mcap]) as reader:
        topics = [connection.topic for connection in reader.connections]
    odometry_count = topics.count(ODOMETRY_TOPIC)
    static_count = topics.count(TF_STATIC_TOPIC)
    if static_count != 1:
        raise RuntimeError(
            f"expected exactly one {TF_STATIC_TOPIC} connection, found {static_count}"
        )
    if odometry_count not in (0, 1):
        raise RuntimeError(
            f"expected zero or one {ODOMETRY_TOPIC} connection, found {odometry_count}"
        )
    result = {
        "schema_version": "e26_mcap_metric_source_detection_v1",
        "status": "complete",
        "mcap": str(mcap),
        "embedded_odometry_connection_count": odometry_count,
        "tf_static_connection_count": static_count,
        "metric_source": "embedded_odometry" if odometry_count == 1 else "raw_lidar_imu_fallback",
        "language_model_used": False,
    }
    print(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
