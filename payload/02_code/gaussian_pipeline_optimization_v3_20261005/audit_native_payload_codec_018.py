"""Keep the complete source-bound capture audit and compare the native codec."""

import inspect
from pathlib import Path

import audit_sift_payload_codec_018 as preserved
from bounded_zstd_payload import ZstdPayloadCodec, MAXIMUM_RAW_BYTES


def main():
    source = inspect.getsource(preserved.main)
    edits = [
        ('for level in (1, 3):', 'for level in (1,):'),
        ('len(rows) == 6', 'len(rows) == 3'),
        ('"complete48_original_requests_all_six_passes_exact"', '"complete48_original_requests_all_three_passes_exact"'),
        ('CODE / "bounded_zlib_payload.py"', 'CODE / "bounded_zstd_payload.py", CODE / "audit_sift_payload_codec_018.py", CODE / "zstd_checkpoint.py", ROOT / "05_envs/colmap-4.1.1-cuda/bin/zstd.dll"'),
        ('"zlib_runtime": zlib.ZLIB_RUNTIME_VERSION', '"native_zstd_version": _codec.version, "native_zstd_library": _codec.library_name'),
    ]
    for before, after in edits:
        expected = 2 if before == 'for level in (1, 3):' else 1
        if source.count(before) != expected:
            raise ValueError("The original complete codec audit boundary changed: " + before)
        source = source.replace(before, after)
    namespace = dict(preserved.main.__globals__)
    codec = ZstdPayloadCodec()
    namespace.update(OUT=preserved.OUT.with_name("b0"), __file__=str(Path(__file__).resolve()),
                     encode_payload=codec.encode, decode_payload=codec.decode,
                     MAXIMUM_RAW_BYTES=MAXIMUM_RAW_BYTES, _codec=codec)
    exec(compile(source, str(Path(__file__)), "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
