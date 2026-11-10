"""Decode exactly one complete lossless descriptor payload within its bound."""

import zlib

MAXIMUM_RAW_BYTES = 13 * 12000 * 128


def encode_payload(raw, level=1):
    if type(raw) is not bytes or not 0 < len(raw) <= MAXIMUM_RAW_BYTES or level not in (1, 3):
        raise ValueError("The private lossless request size or codec level is unsupported")
    return zlib.compress(raw, level)


def decode_payload(encoded, expected):
    if (type(encoded) is not bytes or type(expected) is not int or not 0 < expected <= MAXIMUM_RAW_BYTES
            or not 0 < len(encoded) <= expected + 65536):
        raise ValueError("The complete encoded request or output length exceeds its bound")
    decoder = zlib.decompressobj()
    raw = decoder.decompress(encoded, expected + 1)
    if len(raw) != expected or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("The compressed request is truncated, concatenated, or has a wrong complete length")
    return raw
