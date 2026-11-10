#!/usr/bin/env python
"""Read-only audit for the project's uncompressed ROS 2 MCAP recording."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import BinaryIO, Iterable

import numpy as np
import yaml


MAGIC = b"\x89MCAP0\r\n"
OP_SCHEMA = 0x03
OP_CHANNEL = 0x04
OP_MESSAGE = 0x05
OP_CHUNK = 0x06
OP_MESSAGE_INDEX = 0x07
OP_CHUNK_INDEX = 0x08


def _u16(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<H", data, offset)[0], offset + 2


def _u32(data: bytes, offset: int) -> tuple[int, int]:
    return struct.unpack_from("<I", data, offset)[0], offset + 4


def _string(data: bytes, offset: int) -> tuple[str, int]:
    size, offset = _u32(data, offset)
    end = offset + size
    return data[offset:end].decode("utf-8", "replace"), end


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise EOFError(f"Expected {size} bytes, received {len(data)}")
    return data


@dataclass(frozen=True)
class Channel:
    channel_id: int
    schema_id: int
    topic: str
    encoding: str


@dataclass(frozen=True)
class ChunkIndex:
    message_start_ns: int
    message_end_ns: int
    chunk_start: int
    chunk_length: int
    message_index_offsets: dict[int, int]
    compression: str
    compressed_size: int
    uncompressed_size: int


class Cdr:
    def __init__(self, data: bytes):
        if len(data) < 4:
            raise ValueError("CDR payload is too short")
        self.data = data
        self.base = 4
        self.offset = 4

    def align(self, size: int) -> None:
        self.offset += (-(self.offset - self.base)) % size

    def u8(self) -> int:
        value = self.data[self.offset]
        self.offset += 1
        return value

    def i32(self) -> int:
        self.align(4)
        value = struct.unpack_from("<i", self.data, self.offset)[0]
        self.offset += 4
        return value

    def u32(self) -> int:
        self.align(4)
        value = struct.unpack_from("<I", self.data, self.offset)[0]
        self.offset += 4
        return value

    def f64(self) -> float:
        self.align(8)
        value = struct.unpack_from("<d", self.data, self.offset)[0]
        self.offset += 8
        return value

    def string(self) -> str:
        size = self.u32()
        end = self.offset + size
        raw = self.data[self.offset:end]
        self.offset = end
        if raw.endswith(b"\0"):
            raw = raw[:-1]
        return raw.decode("utf-8", "replace")

    def header(self) -> tuple[int, int, str]:
        return self.i32(), self.u32(), self.string()


class McapIndex:
    """Minimal indexed reader. It never opens the input file for writing."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        self.size = self.path.stat().st_size
        self.summary_start = 0
        self.summary_offset_start = 0
        self.channels: dict[int, Channel] = {}
        self.schemas: dict[int, dict[str, object]] = {}
        self.chunks: list[ChunkIndex] = []
        self.locations: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self._records_start: dict[int, int] = {}
        self._load()

    def _load(self) -> None:
        with self.path.open("rb") as stream:
            if _read_exact(stream, 8) != MAGIC:
                raise ValueError("Invalid MCAP header magic")
            stream.seek(-8, os.SEEK_END)
            if _read_exact(stream, 8) != MAGIC:
                raise ValueError("Invalid MCAP trailing magic")
            stream.seek(-37, os.SEEK_END)
            header = _read_exact(stream, 9)
            if header[0] != 0x02 or struct.unpack("<Q", header[1:])[0] != 20:
                raise ValueError("MCAP Footer is missing or malformed")
            footer = _read_exact(stream, 20)
            self.summary_start, self.summary_offset_start, _ = struct.unpack(
                "<QQI", footer
            )
            self._load_summary(stream)
            self._load_message_indexes(stream)

    def _load_summary(self, stream: BinaryIO) -> None:
        stream.seek(self.summary_start)
        while stream.tell() < self.summary_offset_start:
            record_start = stream.tell()
            header = _read_exact(stream, 9)
            opcode = header[0]
            length = struct.unpack("<Q", header[1:])[0]
            body = _read_exact(stream, length)
            if opcode == OP_SCHEMA:
                offset = 0
                schema_id, offset = _u16(body, offset)
                name, offset = _string(body, offset)
                encoding, offset = _string(body, offset)
                data_size, offset = _u32(body, offset)
                self.schemas[schema_id] = {
                    "name": name,
                    "encoding": encoding,
                    "definition_bytes": data_size,
                }
            elif opcode == OP_CHANNEL:
                offset = 0
                channel_id, offset = _u16(body, offset)
                schema_id, offset = _u16(body, offset)
                topic, offset = _string(body, offset)
                encoding, offset = _string(body, offset)
                self.channels[channel_id] = Channel(
                    channel_id, schema_id, topic, encoding
                )
            elif opcode == OP_CHUNK_INDEX:
                self.chunks.append(self._parse_chunk_index(body))
            if stream.tell() <= record_start:
                raise RuntimeError("Summary parser did not advance")

    @staticmethod
    def _parse_chunk_index(body: bytes) -> ChunkIndex:
        offset = 0
        start, end, chunk_start, chunk_length = struct.unpack_from(
            "<QQQQ", body, offset
        )
        offset += 32
        map_size, offset = _u32(body, offset)
        map_end = offset + map_size
        message_index_offsets: dict[int, int] = {}
        while offset < map_end:
            channel_id, offset = _u16(body, offset)
            index_offset = struct.unpack_from("<Q", body, offset)[0]
            offset += 8
            message_index_offsets[channel_id] = index_offset
        offset += 8  # message_index_length
        compression, offset = _string(body, offset)
        compressed_size, uncompressed_size = struct.unpack_from(
            "<QQ", body, offset
        )
        return ChunkIndex(
            start,
            end,
            chunk_start,
            chunk_length,
            message_index_offsets,
            compression,
            compressed_size,
            uncompressed_size,
        )

    def _load_message_indexes(self, stream: BinaryIO) -> None:
        for chunk_number, chunk in enumerate(self.chunks):
            if chunk.compression:
                raise NotImplementedError(
                    f"Compressed chunk is not supported: {chunk.compression}"
                )
            for expected_channel, index_offset in chunk.message_index_offsets.items():
                stream.seek(index_offset)
                header = _read_exact(stream, 9)
                if header[0] != OP_MESSAGE_INDEX:
                    raise ValueError(f"Invalid MessageIndex at {index_offset}")
                body = _read_exact(stream, struct.unpack("<Q", header[1:])[0])
                channel_id, offset = _u16(body, 0)
                if channel_id != expected_channel:
                    raise ValueError("MessageIndex channel mismatch")
                records_size, offset = _u32(body, offset)
                end = offset + records_size
                while offset < end:
                    timestamp, relative_offset = struct.unpack_from(
                        "<QQ", body, offset
                    )
                    offset += 16
                    self.locations[channel_id].append(
                        (chunk_number, relative_offset)
                    )

    def _chunk_records_start(self, stream: BinaryIO, chunk_number: int) -> int:
        cached = self._records_start.get(chunk_number)
        if cached is not None:
            return cached
        chunk = self.chunks[chunk_number]
        stream.seek(chunk.chunk_start)
        header = _read_exact(stream, 9)
        if header[0] != OP_CHUNK:
            raise ValueError(f"Invalid Chunk at {chunk.chunk_start}")
        _read_exact(stream, 28)
        compression_size = struct.unpack("<I", _read_exact(stream, 4))[0]
        compression = _read_exact(stream, compression_size).decode()
        if compression:
            raise NotImplementedError(f"Compressed chunk: {compression}")
        _read_exact(stream, 8)  # records byte length
        self._records_start[chunk_number] = stream.tell()
        return stream.tell()

    def message_count(self, topic: str) -> int:
        channel_id = self.channel_id(topic)
        return len(self.locations[channel_id])

    def channel_id(self, topic: str) -> int:
        for channel_id, channel in self.channels.items():
            if channel.topic == topic:
                return channel_id
        raise KeyError(topic)

    def iter_messages(
        self, topic: str, *, prefix_bytes: int | None = None
    ) -> Iterable[dict[str, object]]:
        channel_id = self.channel_id(topic)
        with self.path.open("rb") as stream:
            for chunk_number, relative_offset in self.locations[channel_id]:
                yield self._read_message(
                    stream,
                    channel_id,
                    chunk_number,
                    relative_offset,
                    prefix_bytes,
                )

    def iter_selected_messages(
        self,
        topic: str,
        message_numbers: Iterable[int],
        *,
        prefix_bytes: int | None = None,
    ) -> Iterable[tuple[int, dict[str, object]]]:
        channel_id = self.channel_id(topic)
        locations = self.locations[channel_id]
        selected = sorted(set(message_numbers))
        with self.path.open("rb") as stream:
            for number in selected:
                if number < 0 or number >= len(locations):
                    raise IndexError(number)
                chunk_number, relative_offset = locations[number]
                yield number, self._read_message(
                    stream,
                    channel_id,
                    chunk_number,
                    relative_offset,
                    prefix_bytes,
                )

    def _read_message(
        self,
        stream: BinaryIO,
        channel_id: int,
        chunk_number: int,
        relative_offset: int,
        prefix_bytes: int | None,
    ) -> dict[str, object]:
        absolute = self._chunk_records_start(stream, chunk_number) + relative_offset
        stream.seek(absolute)
        header = _read_exact(stream, 9)
        if header[0] != OP_MESSAGE:
            raise ValueError(f"Invalid Message at {absolute}")
        body_length = struct.unpack("<Q", header[1:])[0]
        read_length = body_length
        if prefix_bytes is not None:
            read_length = min(body_length, 22 + prefix_bytes)
        body = _read_exact(stream, read_length)
        actual_channel = struct.unpack_from("<H", body, 0)[0]
        if actual_channel != channel_id:
            raise ValueError("Message channel mismatch")
        sequence = struct.unpack_from("<I", body, 2)[0]
        log_ns, publish_ns = struct.unpack_from("<QQ", body, 6)
        return {
            "sequence": sequence,
            "log_ns": log_ns,
            "publish_ns": publish_ns,
            "payload": body[22:],
            "body_length": body_length,
        }


def iso_utc(ns: int) -> str:
    return datetime.fromtimestamp(ns / 1e9, timezone.utc).isoformat()


def sentinel_hash(path: Path, block_size: int = 1024 * 1024) -> dict[str, object]:
    size = path.stat().st_size
    first = hashlib.sha256()
    last = hashlib.sha256()
    with path.open("rb") as stream:
        first.update(stream.read(min(block_size, size)))
        if size > block_size:
            stream.seek(max(0, size - block_size))
        last.update(stream.read(min(block_size, size)))
    return {
        "block_size": block_size,
        "first_sha256": first.hexdigest(),
        "last_sha256": last.hexdigest(),
    }


def timing_summary(log_ns: list[int], header_ns: list[int]) -> dict[str, object]:
    if not log_ns:
        return {"count": 0}
    log = np.asarray(log_ns, dtype=np.int64)
    result: dict[str, object] = {
        "count": int(log.size),
        "log_start_ns": int(log[0]),
        "log_end_ns": int(log[-1]),
        "log_start_utc": iso_utc(int(log[0])),
        "log_end_utc": iso_utc(int(log[-1])),
    }
    if log.size > 1:
        delta = np.diff(log).astype(np.float64) / 1e9
        med = float(np.median(delta))
        result.update(
            {
                "median_period_ms": med * 1000,
                "rate_hz": 1.0 / med if med > 0 else None,
                "p95_period_ms": float(np.percentile(delta, 95)) * 1000,
                "max_period_ms": float(delta.max()) * 1000,
                "nonpositive_intervals": int(np.count_nonzero(delta <= 0)),
                "gaps_over_1_5x_median": int(np.count_nonzero(delta > 1.5 * med)),
            }
        )
    if header_ns:
        header = np.asarray(header_ns, dtype=np.int64)
        result.update(
            {
                "header_start_ns": int(header[0]),
                "header_end_ns": int(header[-1]),
                "header_nonpositive_intervals": int(
                    np.count_nonzero(np.diff(header) <= 0)
                ),
            }
        )
    return result


def decode_header(payload: bytes) -> tuple[int, int, str]:
    cdr = Cdr(payload)
    sec, nanosec, frame = cdr.header()
    return sec * 1_000_000_000 + nanosec, cdr.offset, frame


def decode_image_header(payload: bytes) -> dict[str, object]:
    cdr = Cdr(payload)
    sec, nanosec, frame = cdr.header()
    height = cdr.u32()
    width = cdr.u32()
    encoding = cdr.string()
    is_bigendian = cdr.u8()
    step = cdr.u32()
    data_length = cdr.u32()
    return {
        "header_ns": sec * 1_000_000_000 + nanosec,
        "frame_id": frame,
        "height": height,
        "width": width,
        "encoding": encoding,
        "is_bigendian": is_bigendian,
        "step": step,
        "data_length": data_length,
    }


def decode_imu(payload: bytes) -> dict[str, object]:
    cdr = Cdr(payload)
    sec, nanosec, frame = cdr.header()
    orientation = [cdr.f64() for _ in range(4)]
    orientation_covariance = [cdr.f64() for _ in range(9)]
    angular_velocity = [cdr.f64() for _ in range(3)]
    angular_velocity_covariance = [cdr.f64() for _ in range(9)]
    linear_acceleration = [cdr.f64() for _ in range(3)]
    linear_acceleration_covariance = [cdr.f64() for _ in range(9)]
    return {
        "header_ns": sec * 1_000_000_000 + nanosec,
        "frame_id": frame,
        "orientation": orientation,
        "orientation_covariance": orientation_covariance,
        "angular_velocity": angular_velocity,
        "angular_velocity_covariance": angular_velocity_covariance,
        "linear_acceleration": linear_acceleration,
        "linear_acceleration_covariance": linear_acceleration_covariance,
    }


def decode_diagnostics(payload: bytes) -> dict[str, object]:
    cdr = Cdr(payload)
    sec, nanosec, frame = cdr.header()
    status_count = cdr.u32()
    statuses = []
    for _ in range(status_count):
        level = cdr.u8()
        name = cdr.string()
        message = cdr.string()
        hardware_id = cdr.string()
        value_count = cdr.u32()
        values = {}
        for _ in range(value_count):
            key = cdr.string()
            value = cdr.string()
            values[key] = value
        statuses.append(
            {
                "level": level,
                "name": name,
                "message": message,
                "hardware_id": hardware_id,
                "values": values,
            }
        )
    return {
        "header_ns": sec * 1_000_000_000 + nanosec,
        "frame_id": frame,
        "statuses": statuses,
    }


def decode_camera_info(payload: bytes) -> dict[str, object]:
    cdr = Cdr(payload)
    sec, nanosec, frame = cdr.header()
    height = cdr.u32()
    width = cdr.u32()
    distortion_model = cdr.string()
    d_count = cdr.u32()
    distortion = [cdr.f64() for _ in range(d_count)]
    camera_matrix = [cdr.f64() for _ in range(9)]
    return {
        "header_ns": sec * 1_000_000_000 + nanosec,
        "frame_id": frame,
        "height": height,
        "width": width,
        "distortion_model": distortion_model,
        "distortion": distortion,
        "camera_matrix": camera_matrix,
    }


def decode_tf_static(payload: bytes) -> list[dict[str, object]]:
    cdr = Cdr(payload)
    count = cdr.u32()
    transforms = []
    for _ in range(count):
        sec, nanosec, parent = cdr.header()
        child = cdr.string()
        translation = [cdr.f64() for _ in range(3)]
        rotation_xyzw = [cdr.f64() for _ in range(4)]
        transforms.append(
            {
                "header_ns": sec * 1_000_000_000 + nanosec,
                "parent": parent,
                "child": child,
                "translation": translation,
                "rotation_xyzw": rotation_xyzw,
            }
        )
    return transforms


def decode_pointcloud(payload: bytes) -> dict[str, object]:
    cdr = Cdr(payload)
    sec, nanosec, frame = cdr.header()
    height = cdr.u32()
    width = cdr.u32()
    field_count = cdr.u32()
    fields = []
    for _ in range(field_count):
        fields.append(
            {
                "name": cdr.string(),
                "offset": cdr.u32(),
                "datatype": cdr.u8(),
                "count": cdr.u32(),
            }
        )
    is_bigendian = cdr.u8()
    point_step = cdr.u32()
    row_step = cdr.u32()
    data_length = cdr.u32()
    raw = memoryview(payload)[cdr.offset : cdr.offset + data_length]
    offsets = {item["name"]: item["offset"] for item in fields}
    count = height * width
    result: dict[str, object] = {
        "header_ns": sec * 1_000_000_000 + nanosec,
        "frame_id": frame,
        "height": height,
        "width": width,
        "point_count": count,
        "fields": fields,
        "is_bigendian": is_bigendian,
        "point_step": point_step,
        "row_step": row_step,
        "data_length": data_length,
    }
    if data_length >= count * point_step and count:
        xyz = np.ndarray(
            shape=(count, 3),
            dtype="<f4",
            buffer=raw,
            offset=0,
            strides=(point_step, 4),
        )
        valid = np.isfinite(xyz).all(axis=1)
        result["valid_xyz"] = int(np.count_nonzero(valid))
        if np.any(valid):
            ranges = np.linalg.norm(xyz[valid], axis=1)
            result["range_median_m"] = float(np.median(ranges))
            result["range_p95_m"] = float(np.percentile(ranges, 95))
            result["range_max_m"] = float(np.max(ranges))
        if "time_offset_ns" in offsets:
            time_offsets = np.ndarray(
                shape=(count,),
                dtype="<u4",
                buffer=raw,
                offset=offsets["time_offset_ns"],
                strides=(point_step,),
            )
            result["time_offset_min_ns"] = int(time_offsets.min())
            result["time_offset_max_ns"] = int(time_offsets.max())
            result["time_span_ms"] = float(
                (time_offsets.max() - time_offsets.min()) / 1e6
            )
    return result


def numeric_axis_summary(values: np.ndarray) -> dict[str, object]:
    return {
        "mean": [float(value) for value in values.mean(axis=0)],
        "std": [float(value) for value in values.std(axis=0)],
        "min": [float(value) for value in values.min(axis=0)],
        "max": [float(value) for value in values.max(axis=0)],
        "norm_median": float(np.median(np.linalg.norm(values, axis=1))),
        "norm_p95": float(np.percentile(np.linalg.norm(values, axis=1), 95)),
    }


def sample_indices(count: int, maximum: int) -> set[int]:
    if count <= maximum:
        return set(range(count))
    return set(np.linspace(0, count - 1, maximum, dtype=int).tolist())


def audit(index: McapIndex, metadata: dict[str, object], lidar_samples: int) -> dict:
    result: dict[str, object] = {
        "source": {
            "mcap_path": str(index.path),
            "size_bytes": index.size,
            "mtime_ns": index.path.stat().st_mtime_ns,
            "sentinel_hash": sentinel_hash(index.path),
            "header_magic_valid": True,
            "trailing_magic_valid": True,
            "summary_start": index.summary_start,
            "summary_offset_start": index.summary_offset_start,
            "chunk_count": len(index.chunks),
            "all_chunks_uncompressed": all(not chunk.compression for chunk in index.chunks),
        },
        "metadata": metadata,
        "schemas": index.schemas,
        "channels": {
            channel.topic: {
                "channel_id": channel.channel_id,
                "schema_id": channel.schema_id,
                "encoding": channel.encoding,
                "indexed_messages": len(index.locations[channel.channel_id]),
            }
            for channel in index.channels.values()
        },
        "topics": {},
    }

    image_logs: list[int] = []
    image_headers: list[int] = []
    image_properties: dict[str, object] | None = None
    for message in index.iter_messages(
        "/sensing/camera/image_raw", prefix_bytes=512
    ):
        decoded = decode_image_header(message["payload"])
        image_logs.append(int(message["log_ns"]))
        image_headers.append(int(decoded["header_ns"]))
        if image_properties is None:
            image_properties = {
                key: value for key, value in decoded.items() if key != "header_ns"
            }
    result["topics"]["/sensing/camera/image_raw"] = {
        **timing_summary(image_logs, image_headers),
        "properties": image_properties,
    }

    imu_logs: list[int] = []
    imu_headers: list[int] = []
    gyroscope: list[list[float]] = []
    acceleration: list[list[float]] = []
    imu_frame = None
    orientation_covariance = None
    for message in index.iter_messages("/sensing/lidar/imu"):
        decoded = decode_imu(message["payload"])
        imu_logs.append(int(message["log_ns"]))
        imu_headers.append(int(decoded["header_ns"]))
        gyroscope.append(decoded["angular_velocity"])
        acceleration.append(decoded["linear_acceleration"])
        imu_frame = decoded["frame_id"]
        orientation_covariance = decoded["orientation_covariance"]
    gyro_array = np.asarray(gyroscope, dtype=np.float64)
    accel_array = np.asarray(acceleration, dtype=np.float64)
    result["topics"]["/sensing/lidar/imu"] = {
        **timing_summary(imu_logs, imu_headers),
        "frame_id": imu_frame,
        "orientation_covariance_first": orientation_covariance,
        "angular_velocity": numeric_axis_summary(gyro_array),
        "linear_acceleration": numeric_axis_summary(accel_array),
    }

    lidar_count = index.message_count("/sensing/lidar/points_raw")
    selected = sample_indices(lidar_count, lidar_samples)
    lidar_logs: list[int] = []
    lidar_headers: list[int] = []
    samples = []
    for number, message in enumerate(index.iter_messages("/sensing/lidar/points_raw")):
        if number not in selected:
            header_ns, _, _ = decode_header(message["payload"])
            lidar_logs.append(int(message["log_ns"]))
            lidar_headers.append(header_ns)
            continue
        decoded = decode_pointcloud(message["payload"])
        lidar_logs.append(int(message["log_ns"]))
        lidar_headers.append(int(decoded["header_ns"]))
        decoded["frame_number"] = number
        samples.append(decoded)
    result["topics"]["/sensing/lidar/points_raw"] = {
        **timing_summary(lidar_logs, lidar_headers),
        "sample_count": len(samples),
        "samples": samples,
    }

    for topic in ("/clock", "/diagnostics", "/tf_static", "/sensing/camera/camera_info"):
        logs = []
        for message in index.iter_messages(topic, prefix_bytes=4096):
            logs.append(int(message["log_ns"]))
        result["topics"][topic] = timing_summary(logs, [])

    diagnostics: dict[str, dict[str, object]] = {}
    for message in index.iter_messages("/diagnostics"):
        decoded = decode_diagnostics(message["payload"])
        for status in decoded["statuses"]:
            key = f"{status['name']}|{status['hardware_id']}"
            entry = diagnostics.setdefault(
                key,
                {
                    "name": status["name"],
                    "hardware_id": status["hardware_id"],
                    "levels": set(),
                    "messages": set(),
                    "observed_values": defaultdict(set),
                    "samples": 0,
                },
            )
            entry["samples"] += 1
            entry["levels"].add(status["level"])
            entry["messages"].add(status["message"])
            for value_key, value in status["values"].items():
                if len(entry["observed_values"][value_key]) < 20:
                    entry["observed_values"][value_key].add(value)
    result["diagnostics"] = []
    for entry in diagnostics.values():
        result["diagnostics"].append(
            {
                "name": entry["name"],
                "hardware_id": entry["hardware_id"],
                "samples": entry["samples"],
                "levels": sorted(entry["levels"]),
                "messages": sorted(entry["messages"]),
                "observed_values": {
                    key: sorted(values)
                    for key, values in entry["observed_values"].items()
                },
            }
        )

    first_camera_info = next(
        index.iter_messages("/sensing/camera/camera_info")
    )
    result["camera_info"] = decode_camera_info(first_camera_info["payload"])
    transforms = []
    for message in index.iter_messages("/tf_static"):
        transforms.extend(decode_tf_static(message["payload"]))
    result["tf_static"] = transforms

    return result


def markdown_report(data: dict) -> str:
    source = data["source"]
    topics = data["topics"]
    lines = [
        "# MCAP 数据审计报告",
        "",
        f"- 文件：`{source['mcap_path']}`",
        f"- 大小：{source['size_bytes']} bytes",
        f"- Chunk 数：{source['chunk_count']}",
        f"- 全部 Chunk 未压缩：{source['all_chunks_uncompressed']}",
        f"- 首块 SHA-256：`{source['sentinel_hash']['first_sha256']}`",
        f"- 末块 SHA-256：`{source['sentinel_hash']['last_sha256']}`",
        "",
        "## 话题时序",
        "",
        "| 话题 | 消息数 | 估计频率 | 中位周期 | 最大周期 | 非正间隔 | 大间隔 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for topic, summary in topics.items():
        rate = summary.get("rate_hz")
        lines.append(
            f"| `{topic}` | {summary.get('count', 0)} | "
            f"{rate:.3f} Hz | " if isinstance(rate, float) else
            f"| `{topic}` | {summary.get('count', 0)} | — | "
        )
        lines[-1] += (
            f"{summary.get('median_period_ms', math.nan):.3f} ms | "
            f"{summary.get('max_period_ms', math.nan):.3f} ms | "
            f"{summary.get('nonpositive_intervals', 0)} | "
            f"{summary.get('gaps_over_1_5x_median', 0)} |"
        )

    imu = topics["/sensing/lidar/imu"]
    lidar = topics["/sensing/lidar/points_raw"]
    camera = topics["/sensing/camera/image_raw"]
    spans = [sample.get("time_span_ms") for sample in lidar["samples"]]
    points = [sample.get("point_count") for sample in lidar["samples"]]
    lines.extend(
        [
            "",
            "## 传感器摘要",
            "",
            f"- Camera：{camera['properties']}",
            f"- IMU 角速度均值：{imu['angular_velocity']['mean']}",
            f"- IMU 角速度标准差：{imu['angular_velocity']['std']}",
            f"- IMU 加速度均值：{imu['linear_acceleration']['mean']}",
            f"- IMU 加速度标准差：{imu['linear_acceleration']['std']}",
            f"- LiDAR 抽样帧数：{lidar['sample_count']}",
            f"- LiDAR 抽样点数范围：{min(points)}–{max(points)}",
            f"- LiDAR 点内时间跨度范围：{min(spans):.3f}–{max(spans):.3f} ms",
            "",
            "## 解释边界",
            "",
            "- 本报告是数据完整性和内部时序审计，不是 2 cm 绝对精度证明。",
            "- 原始 MCAP 仅以只读方式打开；报告写入独立输出目录。",
            "- IMU 全段统计包含运动，不能直接当作静态零偏结果。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lidar-samples", type=int, default=64)
    args = parser.parse_args()

    bag_dir = args.bag_dir.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir == bag_dir or bag_dir in output_dir.parents:
        raise SystemExit("Refusing to write output inside the original bag directory")
    mcap_files = list(bag_dir.glob("*.mcap"))
    if len(mcap_files) != 1:
        raise SystemExit(f"Expected exactly one MCAP file, found {len(mcap_files)}")

    metadata_path = bag_dir / "metadata.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    index = McapIndex(mcap_files[0])
    data = audit(index, metadata, args.lidar_samples)

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "data_audit.json"
    md_path = output_dir / "data_audit.md"
    json_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    md_path.write_text(markdown_report(data), encoding="utf-8")
    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
