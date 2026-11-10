"""Reuse exact raw calibration bodies while decoding every new message header."""

from contextlib import contextmanager
import inspect
import math
import time

import run_effective_initial


class CalibrationBodyCache:
    def __init__(self, native, cdr_factory, enabled, capacity=16, maximum_payload_bytes=64*1024):
        started = time.perf_counter()
        if type(enabled) is not bool or not 1 <= capacity <= 16 or not 4 <= maximum_payload_bytes <= 64*1024:
            raise ValueError("Calibration reuse requires explicit bounded admission")
        source = inspect.getsource(native)
        for anchor in ("sec, nanosec, frame = cdr.header()", "camera_matrix = [cdr.f64() for _ in range(9)]",
                       "distortion = [cdr.f64() for _ in range(d_count)]"):
            if source.count(anchor) != 1:
                raise ValueError("The pinned native CameraInfo decoder changed")
        self.native, self.cdr_factory, self.enabled = native, cdr_factory, enabled
        self.capacity, self.maximum_payload_bytes = capacity, maximum_payload_bytes
        self.entries = {}
        self.trace = {"enabled": enabled, "calls": 0, "hits": 0, "native_calls": 0, "uncacheable": 0,
            "maximum_entries": capacity, "maximum_payload_bytes": maximum_payload_bytes,
            "cache_key_bytes": 0, "setup_seconds": 0., "call_seconds": 0.,
            "new_header_timestamp_and_frame_decoded_every_call": True,
            "complete_body_bytes_CDR_prefix_and_alignment_compared": True,
            "caller_returns_do_not_alias_private_cache": True}
        self.trace["setup_seconds"] = time.perf_counter()-started

    @staticmethod
    def clone(info):
        return {**info, "distortion": info["distortion"].copy(), "camera_matrix": info["camera_matrix"].copy()}

    @staticmethod
    def cacheable(info):
        keys = {"header_ns", "frame_id", "height", "width", "distortion_model", "distortion", "camera_matrix"}
        return (set(info) == keys and type(info["header_ns"]) is int and type(info["height"]) is int
            and type(info["width"]) is int and isinstance(info["frame_id"], str) and isinstance(info["distortion_model"], str)
            and type(info["distortion"]) is list and type(info["camera_matrix"]) is list and len(info["camera_matrix"]) == 9
            and all(type(x) is float and math.isfinite(x) for x in info["distortion"]+info["camera_matrix"]))

    def __call__(self, payload):
        started = time.perf_counter()
        self.trace["calls"] += 1
        key, header = None, None
        if self.enabled and type(payload) is bytes and 4 <= len(payload) <= self.maximum_payload_bytes:
            cdr = self.cdr_factory(payload)
            sec, ns, frame = cdr.header()
            header = sec*1000000000+ns, frame
            # Body alignment is relative to the native four-byte CDR base.
            key = payload[:4], (cdr.offset-4) % 8, payload[cdr.offset:]
        if key is not None and key in self.entries:
            result = self.clone(self.entries[key])
            result["header_ns"], result["frame_id"] = header
            self.trace["hits"] += 1
        else:
            result = self.native(payload)
            self.trace["native_calls"] += 1
            if key is not None and self.cacheable(result) and len(self.entries) < self.capacity:
                self.entries[key] = self.clone(result)
                self.trace["cache_key_bytes"] += len(key[0])+len(key[2])
            elif self.enabled:
                self.trace["uncacheable"] += 1
        self.trace["call_seconds"] += time.perf_counter()-started
        return result


@contextmanager
def calibration_body_execution(enabled=True):
    composition_started = time.perf_counter()
    original = run_effective_initial.combined_raw_execution
    factory = inspect.getsource(original)
    anchor = "    namespace.update(_effective_pairs=pairs, _effective_exports=exports)"
    after = '''    def camera_decoder(native_decoder, cdr_factory):
        cache = _calibration_cache(native_decoder, cdr_factory, _calibration_enabled)
        trace["calibration_body"] = cache.trace
        return cache
    before = '    from mcap_audit import McapIndex, Cdr, decode_camera_info, decode_tf_static'
    if source.count(before) != 1:
        raise ValueError("The original raw CameraInfo import changed")
    source = source.replace(before, before+'\\n    decode_camera_info = _camera_decoder(decode_camera_info, Cdr)')
    namespace.update(_effective_pairs=pairs, _effective_exports=exports, _camera_decoder=camera_decoder)'''
    if factory.count(anchor) != 1:
        raise ValueError("The accepted raw route changed its single source composition")
    namespace = dict(run_effective_initial.__dict__)
    namespace.update(_calibration_cache=CalibrationBodyCache, _calibration_enabled=enabled)
    exec(compile(factory.replace(anchor, after), "<accepted_raw_route_with_exact_calibration_body_reuse>", "exec"), namespace)
    run_effective_initial.combined_raw_execution = namespace["combined_raw_execution"]
    try:
        yield {"composition_seconds": time.perf_counter()-composition_started}
    finally:
        run_effective_initial.combined_raw_execution = original
