"""Read indexed uncompressed MCAP payloads through an explicitly read-only map."""

import mmap
import struct


def iter_mapped_messages(index, topic, message_numbers, *, prefix_bytes=None):
    """Yield independent bytes so pending workers cannot retain a closed mapping."""
    if prefix_bytes is not None and (not isinstance(prefix_bytes, int) or prefix_bytes < 0):
        raise ValueError("Prefix length must be a nonnegative integer")
    channel_id = index.channel_id(topic)
    locations = index.locations[channel_id]
    selected = sorted(set(message_numbers))
    if any(not isinstance(number, int) or number < 0 or number >= len(locations) for number in selected):
        raise IndexError("Selected message is outside the indexed topic")
    original_stat = index.path.stat()
    original_identity = (original_stat.st_size, original_stat.st_mtime_ns)
    if original_stat.st_size != index.size:
        raise ValueError("Raw size changed after index construction")
    with index.path.open("rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            for number in selected:
                chunk_number, relative_offset = locations[number]
                chunk = index.chunks[chunk_number]
                if chunk.compression:
                    raise NotImplementedError("Memory-mapped input requires uncompressed chunks")
                records_start = index._chunk_records_start(stream, chunk_number)
                records_end = records_start + chunk.uncompressed_size
                absolute = records_start + relative_offset
                if records_start < 0 or records_end > len(mapped) or relative_offset < 0 or absolute + 9 > records_end:
                    raise ValueError("Message index or chunk bounds are invalid")
                opcode, body_length = struct.unpack_from("<BQ", mapped, absolute)
                if opcode != 0x05 or body_length < 22 or absolute + 9 + body_length > records_end:
                    raise ValueError("Message opcode, body length, or record boundary is invalid")
                actual_channel, sequence, log_ns, publish_ns = struct.unpack_from("<HIQQ", mapped, absolute + 9)
                if actual_channel != channel_id:
                    raise ValueError("Message channel differs from its index")
                payload_length = body_length - 22
                if prefix_bytes is not None:
                    payload_length = min(payload_length, prefix_bytes)
                begin = absolute + 9 + 22
                yield number, {"sequence": sequence, "log_ns": log_ns, "publish_ns": publish_ns,
                               "payload": mapped[begin:begin + payload_length], "body_length": body_length}
    final_stat = index.path.stat()
    if (final_stat.st_size, final_stat.st_mtime_ns) != original_identity:
        raise ValueError("Raw file changed during the mapped read")
