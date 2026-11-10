"""Strict trajectory readers for FAST-LIO2/FAST-LIVO2 and TUM products."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .contracts import normalize_quaternion_xyzw


ODOMETRY_POSITION_FIELDS = tuple(
    f"field.pose.pose.position.{axis}" for axis in "xyz"
)
ODOMETRY_QUATERNION_FIELDS = tuple(
    f"field.pose.pose.orientation.{axis}" for axis in "xyzw"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Trajectory:
    """A strictly increasing sequence of ``T_target_source`` poses."""

    timestamps_ns: np.ndarray
    poses_xyz_xyzw: np.ndarray
    source_frame: str
    target_frame: str
    time_domain: str
    clock_source: str
    timestamp_field: str
    source_path: str
    source_sha256: str
    raw_sample_count: int
    duplicate_timestamp_count: int = 0

    def __post_init__(self) -> None:
        timestamps = np.asarray(self.timestamps_ns, dtype=np.int64)
        poses = np.asarray(self.poses_xyz_xyzw, dtype=np.float64)
        if timestamps.ndim != 1 or poses.shape != (len(timestamps), 7):
            raise ValueError("trajectory requires timestamps[N] and poses[N,7]")
        if len(timestamps) < 2 or not np.all(np.diff(timestamps) > 0):
            raise ValueError("trajectory timestamps must be strictly increasing")
        if not np.all(np.isfinite(poses)):
            raise ValueError("trajectory poses must be finite")
        normalized = poses.copy()
        normalized[:, 3:] = np.stack(
            [normalize_quaternion_xyzw(row) for row in poses[:, 3:]]
        )
        if not self.source_frame or not self.target_frame:
            raise ValueError("trajectory frame names cannot be empty")
        if not self.time_domain or not self.clock_source:
            raise ValueError("trajectory time domain and clock source are required")
        if len(self.source_sha256) != 64:
            raise ValueError("trajectory source_sha256 must be a SHA256 digest")
        if self.raw_sample_count < len(timestamps):
            raise ValueError("raw_sample_count cannot be smaller than sample count")
        if self.duplicate_timestamp_count != self.raw_sample_count - len(timestamps):
            raise ValueError("duplicate timestamp count is inconsistent")
        object.__setattr__(self, "timestamps_ns", timestamps.copy())
        object.__setattr__(self, "poses_xyz_xyzw", normalized)

    @property
    def sample_count(self) -> int:
        return int(len(self.timestamps_ns))

    @property
    def start_ns(self) -> int:
        return int(self.timestamps_ns[0])

    @property
    def end_ns(self) -> int:
        return int(self.timestamps_ns[-1])

    def to_binding_json(self) -> dict[str, object]:
        return {
            "transform_direction": "T_target_source",
            "source_frame": self.source_frame,
            "target_frame": self.target_frame,
            "translation_unit": "m",
            "quaternion_order": "xyzw",
            "time_unit": "int64_nanoseconds",
            "time_domain": self.time_domain,
            "clock_source": self.clock_source,
            "timestamp_field": self.timestamp_field,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "raw_sample_count": int(self.raw_sample_count),
            "sample_count": self.sample_count,
            "duplicate_timestamp_count": int(self.duplicate_timestamp_count),
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
        }


def _parse_int64_timestamp(value: str, *, field: str) -> int:
    text = str(value).strip()
    if not text:
        raise ValueError(f"empty trajectory timestamp in {field}")
    try:
        if any(character in text.lower() for character in (".", "e")):
            seconds = float(text)
            if not np.isfinite(seconds):
                raise ValueError
            return int(round(seconds * 1_000_000_000))
        return int(text)
    except ValueError as error:
        raise ValueError(f"invalid trajectory timestamp in {field}: {text}") from error


def load_odometry_csv(
    path: Path,
    *,
    time_domain: str,
    clock_source: str,
    timestamp_field: str = "field.header.stamp",
    expected_target_frame: str | None = None,
    expected_source_frame: str | None = None,
    duplicate_policy: str = "last",
) -> Trajectory:
    """Load a ROS odometry CSV and retain the final visual callback at a stamp."""

    resolved = path.resolve(strict=True)
    if duplicate_policy not in {"first", "last", "reject"}:
        raise ValueError("duplicate_policy must be first, last, or reject")
    samples: dict[int, tuple[np.ndarray, str, str]] = {}
    raw_timestamps: list[int] = []
    previous_timestamp: int | None = None
    with resolved.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            timestamp_field,
            "field.header.frame_id",
            "field.child_frame_id",
            *ODOMETRY_POSITION_FIELDS,
            *ODOMETRY_QUATERNION_FIELDS,
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"odometry CSV is missing fields: {sorted(missing)}")
        for ordinal, row in enumerate(reader):
            timestamp = _parse_int64_timestamp(row[timestamp_field], field=timestamp_field)
            if previous_timestamp is not None and timestamp < previous_timestamp:
                raise ValueError(
                    f"odometry timestamp decreases at raw row {ordinal}: "
                    f"{timestamp} < {previous_timestamp}"
                )
            previous_timestamp = timestamp
            raw_timestamps.append(timestamp)
            pose = np.asarray(
                [float(row[field]) for field in ODOMETRY_POSITION_FIELDS]
                + [float(row[field]) for field in ODOMETRY_QUATERNION_FIELDS],
                dtype=np.float64,
            )
            if not np.all(np.isfinite(pose)):
                raise ValueError(f"non-finite odometry pose at raw row {ordinal}")
            target_frame = str(row["field.header.frame_id"]).strip()
            source_frame = str(row["field.child_frame_id"]).strip()
            if timestamp in samples and duplicate_policy == "reject":
                raise ValueError(f"duplicate odometry timestamp: {timestamp}")
            if timestamp not in samples or duplicate_policy == "last":
                samples[timestamp] = (pose, target_frame, source_frame)
    if len(samples) < 2:
        raise ValueError("odometry CSV contains fewer than two unique poses")
    target_frames = {sample[1] for sample in samples.values()}
    source_frames = {sample[2] for sample in samples.values()}
    if len(target_frames) != 1 or len(source_frames) != 1:
        raise ValueError("odometry frame IDs change within the trajectory")
    target_frame = next(iter(target_frames))
    source_frame = next(iter(source_frames))
    if expected_target_frame is not None and target_frame != expected_target_frame:
        raise ValueError(
            f"unexpected target frame: {target_frame} != {expected_target_frame}"
        )
    if expected_source_frame is not None and source_frame != expected_source_frame:
        raise ValueError(
            f"unexpected source frame: {source_frame} != {expected_source_frame}"
        )
    timestamps = np.asarray(sorted(samples), dtype=np.int64)
    poses = np.stack([samples[int(timestamp)][0] for timestamp in timestamps])
    return Trajectory(
        timestamps_ns=timestamps,
        poses_xyz_xyzw=poses,
        source_frame=source_frame,
        target_frame=target_frame,
        time_domain=time_domain,
        clock_source=clock_source,
        timestamp_field=timestamp_field,
        source_path=str(resolved),
        source_sha256=sha256_file(resolved),
        raw_sample_count=len(raw_timestamps),
        duplicate_timestamp_count=len(raw_timestamps) - len(timestamps),
    )


def load_tum_trajectory(
    path: Path,
    *,
    source_frame: str,
    target_frame: str,
    time_domain: str,
    clock_source: str,
) -> Trajectory:
    resolved = path.resolve(strict=True)
    rows: list[list[float]] = []
    with resolved.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            fields = text.split()
            if len(fields) != 8:
                raise ValueError(f"TUM row {line_number} must contain 8 columns")
            row = [float(value) for value in fields]
            if not np.all(np.isfinite(row)):
                raise ValueError(f"TUM row {line_number} contains non-finite values")
            rows.append(row)
    if len(rows) < 2:
        raise ValueError("TUM trajectory contains fewer than two poses")
    array = np.asarray(rows, dtype=np.float64)
    timestamps = np.rint(array[:, 0] * 1_000_000_000).astype(np.int64)
    if not np.all(np.diff(timestamps) > 0):
        raise ValueError("TUM timestamps must be strictly increasing")
    return Trajectory(
        timestamps_ns=timestamps,
        poses_xyz_xyzw=array[:, 1:8],
        source_frame=source_frame,
        target_frame=target_frame,
        time_domain=time_domain,
        clock_source=clock_source,
        timestamp_field="tum_seconds",
        source_path=str(resolved),
        source_sha256=sha256_file(resolved),
        raw_sample_count=len(rows),
    )
