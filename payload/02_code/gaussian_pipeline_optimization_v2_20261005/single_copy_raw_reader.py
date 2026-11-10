"""Read native MCAP payload bytes directly without copying a full message body slice."""

from contextlib import contextmanager
import importlib
from pathlib import Path
import struct
import sys


def iter_single_copy_messages(index, topic, message_numbers, *, prefix_bytes=None):
    if prefix_bytes is not None and (type(prefix_bytes) is not int or prefix_bytes < 0):
        raise ValueError("The native payload prefix must be a nonnegative integer")
    channel = index.channel_id(topic)
    locations = index.locations[channel]
    numbers = sorted(set(message_numbers))
    if any(type(number) is not int or not 0 <= number < len(locations) for number in numbers):
        raise IndexError("A selected raw camera message is outside its actual indexed topic")
    start_stat = index.path.stat()
    identity = start_stat.st_size, start_stat.st_mtime_ns
    if identity[0] != index.size:
        raise ValueError("The raw input size changed after native index construction")

    def exact(stream, length):
        value = stream.read(length)
        if len(value) != length:
            raise EOFError("The native raw payload ended before its frozen record boundary")
        return value

    with index.path.open("rb") as stream:
        for number in numbers:
            chunk_number, relative = locations[number]
            chunk = index.chunks[chunk_number]
            if chunk.compression:
                raise NotImplementedError("The exact single-copy reader requires native uncompressed chunks")
            records = index._chunk_records_start(stream, chunk_number)
            absolute, end = records+relative, records+chunk.uncompressed_size
            if records < 0 or end > index.size or relative < 0 or absolute+31 > end:
                raise ValueError("The camera message lies outside its native chunk")
            stream.seek(absolute)
            opcode, length = struct.unpack("<BQ", exact(stream, 9))
            if opcode != 5 or length < 22 or absolute+9+length > end or length-22 > 16*1024**2:
                raise ValueError("The native camera opcode or body bounds changed")
            actual_channel, sequence, log_ns, publish_ns = struct.unpack("<HIQQ", exact(stream, 22))
            if actual_channel != channel:
                raise ValueError("A camera payload no longer belongs to its indexed topic")
            payload_length = length-22
            if prefix_bytes is not None:
                payload_length = min(payload_length, prefix_bytes)
            # The payload owns its immutable bytes; there is no retained whole-body buffer.
            payload = exact(stream, payload_length)
            yield number, {"sequence": sequence, "log_ns": log_ns, "publish_ns": publish_ns,
                "body_length": length, "payload": payload}
    final = index.path.stat()
    if identity != (final.st_size, final.st_mtime_ns):
        raise ValueError("The raw source changed during exact single-copy reading")


@contextmanager
def single_copy_sensor_reader(enabled=True):
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root/"02_code/box2_initial_pose_quality_v2_20261002"))
    baseline = importlib.import_module("initial_selection")
    sys.path.insert(0, str(baseline.TOOLS))
    from mcap_audit import McapIndex
    original = McapIndex.iter_selected_messages
    trace = {"enabled": enabled, "camera_iterator_calls": 0, "payload_bytes_without_body_slice_copy": 0,
        "read_calls": 0, "mapped_input": False, "native_message_indices_and_order_preserved": True}

    def selected(index, topic, message_numbers, *, prefix_bytes=None):
        trace["camera_iterator_calls"] += 1
        iterator = iter_single_copy_messages(index, topic, message_numbers, prefix_bytes=prefix_bytes)
        try:
            for number, row in iterator:
                trace["payload_bytes_without_body_slice_copy"] += len(row["payload"])
                trace["read_calls"] += 1
                yield number, row
        finally:
            iterator.close()

    if enabled:
        McapIndex.iter_selected_messages = selected
    try:
        yield trace
    finally:
        McapIndex.iter_selected_messages = original
