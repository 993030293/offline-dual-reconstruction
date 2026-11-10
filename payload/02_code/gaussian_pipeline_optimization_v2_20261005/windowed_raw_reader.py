"""Bound read-only MCAP mappings without retaining the entire raw sequence in RSS."""

from contextlib import contextmanager
import mmap
import struct


def iter_windowed_messages(index, topic, message_numbers, *, prefix_bytes=None):
    if prefix_bytes is not None and (type(prefix_bytes) is not int or prefix_bytes < 0):
        raise ValueError("Prefix length must be a nonnegative integer")
    channel = index.channel_id(topic)
    locations = index.locations[channel]
    selected = sorted(set(message_numbers))
    if any(type(number) is not int or number < 0 or number >= len(locations) for number in selected):
        raise IndexError("The selected raw message is outside its indexed topic")
    original_stat = index.path.stat()
    identity = original_stat.st_size, original_stat.st_mtime_ns
    if original_stat.st_size != index.size:
        raise ValueError("Raw source size changed after index construction")
    window_limit = 64*1024**2
    mapped, mapped_begin, mapped_end = None, 0, 0
    with index.path.open("rb") as stream:
        def ensure(begin, length):
            nonlocal mapped, mapped_begin, mapped_end
            if begin < 0 or length < 0 or begin+length > index.size or length > window_limit:
                raise ValueError("A raw message exceeds its bounded window or file limits")
            if mapped is not None and mapped_begin <= begin and begin+length <= mapped_end:
                return
            if mapped is not None:
                mapped.close()
            mapped_begin = begin//mmap.ALLOCATIONGRANULARITY*mmap.ALLOCATIONGRANULARITY
            size = min(index.size-mapped_begin, max(window_limit, begin-mapped_begin+length))
            if size > window_limit+mmap.ALLOCATIONGRANULARITY:
                raise ValueError("The raw mapping exceeded its explicit allocation bound")
            mapped = mmap.mmap(stream.fileno(), size, access=mmap.ACCESS_READ, offset=mapped_begin)
            mapped_end = mapped_begin+size

        try:
            for number in selected:
                chunk_number, relative = locations[number]
                chunk = index.chunks[chunk_number]
                if chunk.compression:
                    raise NotImplementedError("The sensor map supports only uncompressed source chunks")
                records_begin = index._chunk_records_start(stream, chunk_number)
                records_end = records_begin+chunk.uncompressed_size
                absolute = records_begin+relative
                if records_begin < 0 or records_end > index.size or relative < 0 or absolute+9 > records_end:
                    raise ValueError("The indexed record lies outside its source chunk")
                ensure(absolute, 9)
                opcode, body_length = struct.unpack_from("<BQ", mapped, absolute-mapped_begin)
                if opcode != 5 or body_length < 22 or absolute+9+body_length > records_end:
                    raise ValueError("Raw opcode, message length or record boundary changed")
                ensure(absolute, 9+body_length)
                cid, sequence, log_ns, publish_ns = struct.unpack_from("<HIQQ", mapped, absolute-mapped_begin+9)
                if cid != channel:
                    raise ValueError("The raw record channel differs from its topic index")
                length = body_length-22
                if prefix_bytes is not None:
                    length = min(length, prefix_bytes)
                begin = absolute-mapped_begin+31
                yield number, {"sequence": sequence, "log_ns": log_ns, "publish_ns": publish_ns,
                    "payload": mapped[begin:begin+length], "body_length": body_length}
        finally:
            if mapped is not None:
                mapped.close()
    stat = index.path.stat()
    if (stat.st_size, stat.st_mtime_ns) != identity:
        raise ValueError("The raw source changed during bounded mapped reading")


@contextmanager
def windowed_sensor_reader(enabled=True):
    import mapped_raw_reader
    original = mapped_raw_reader.iter_mapped_messages
    if enabled:
        mapped_raw_reader.iter_mapped_messages = iter_windowed_messages
    try:
        yield
    finally:
        mapped_raw_reader.iter_mapped_messages = original
