"""Decode a video, image directory, or ROS MCAP into sequential JPEG frames."""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from common import atomic_write_json, require_new_directory, sha256_file


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def encode_frame(frame: np.ndarray, quality: int) -> bytes:
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return encoded.tobytes()


def video_frames(path: Path) -> Iterator[tuple[int, int, np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_ns = int(capture.get(cv2.CAP_PROP_POS_MSEC) * 1_000_000)
            yield index, timestamp_ns, frame
            index += 1
    finally:
        capture.release()


def directory_frames(path: Path) -> Iterator[tuple[int, int, np.ndarray]]:
    sources = sorted(
        item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )
    if not sources:
        raise ValueError(f"no supported images found in {path}")
    for index, source in enumerate(sources):
        frame = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"cannot decode image: {source}")
        yield index, index, frame


def decode_ros_image(message: object) -> np.ndarray:
    width = int(message.width)
    height = int(message.height)
    step = int(message.step)
    encoding = str(message.encoding).lower()
    raw = np.frombuffer(message.data, dtype=np.uint8).reshape(height, step)
    if encoding == "bayer_rggb8":
        return cv2.cvtColor(raw[:, :width], cv2.COLOR_BayerRGGB2BGR)
    if encoding == "bayer_bggr8":
        return cv2.cvtColor(raw[:, :width], cv2.COLOR_BayerBG2BGR)
    if encoding == "mono8":
        return cv2.cvtColor(raw[:, :width], cv2.COLOR_GRAY2BGR)
    if encoding in {"bgr8", "rgb8"}:
        image = raw[:, : width * 3].reshape(height, width, 3)
        return image if encoding == "bgr8" else cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    raise ValueError(f"unsupported ROS image encoding: {message.encoding}")


def mcap_frames(path: Path, topic: str) -> Iterator[tuple[int, int, np.ndarray]]:
    try:
        from rosbags.highlevel import AnyReader
    except ImportError as error:
        raise RuntimeError(
            "MCAP decoding requires the rosbags package in the active Python environment"
        ) from error
    with AnyReader([path]) as reader:
        connections = [connection for connection in reader.connections if connection.topic == topic]
        if not connections:
            raise ValueError(f"topic not found in MCAP: {topic}")
        for index, (connection, log_ns, data) in enumerate(
            reader.messages(connections=connections)
        ):
            message = reader.deserialize(data, connection.msgtype)
            frame = decode_ros_image(message)
            stamp = getattr(getattr(message, "header", None), "stamp", None)
            if stamp is None:
                timestamp_ns = int(log_ns)
            else:
                timestamp_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
            yield index, timestamp_ns, frame


def detect_kind(path: Path) -> str:
    if path.is_dir():
        return "image-dir"
    if path.suffix.lower() == ".mcap":
        return "mcap"
    return "video"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kind", choices=("auto", "video", "image-dir", "mcap"), default="auto")
    parser.add_argument("--topic", default="/sensing/camera/image_raw")
    parser.add_argument("--jpeg-quality", type=int, default=97)
    parser.add_argument("--expected-frames", type=int)
    args = parser.parse_args()

    source = args.input.resolve(strict=True)
    kind = detect_kind(source) if args.kind == "auto" else args.kind
    output = require_new_directory(args.output)
    images = output / "images"
    images.mkdir()
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("jpeg quality must be between 1 and 100")

    if kind == "video":
        iterator = video_frames(source)
    elif kind == "image-dir":
        iterator = directory_frames(source)
    elif kind == "mcap":
        iterator = mcap_frames(source, args.topic)
    else:
        raise AssertionError(kind)

    started = time.monotonic()
    rows = []
    for index, timestamp_ns, frame in iterator:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"unexpected frame shape at index {index}: {frame.shape}")
        payload = encode_frame(frame, args.jpeg_quality)
        name = f"image_{index:05d}.jpg"
        target = images / name
        target.write_bytes(payload)
        rows.append(
            {
                "image_index": index,
                "timestamp_ns": int(timestamp_ns),
                "image_file": name,
                "image_sha256": sha256_file(target),
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
            }
        )
    if not rows:
        raise RuntimeError("input produced zero frames")
    if args.expected_frames is not None and len(rows) != args.expected_frames:
        raise RuntimeError(f"frame count {len(rows)} != expected {args.expected_frames}")
    manifest = {
        "schema_version": "s08_pipeline_image_manifest_v1",
        "status": "complete_machine_decoding",
        "source": str(source),
        "source_kind": kind,
        "language_model_used": False,
        "frame_count": len(rows),
        "jpeg_quality": args.jpeg_quality,
        "runtime_seconds": time.monotonic() - started,
        "records": rows,
    }
    atomic_write_json(output / "image_manifest.json", manifest)
    print(output / "image_manifest.json")


if __name__ == "__main__":
    main()
