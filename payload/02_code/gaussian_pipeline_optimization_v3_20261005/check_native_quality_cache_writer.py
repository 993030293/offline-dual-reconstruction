"""Check original cache bytes, audit events, failures and owned native lifetimes."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
import threading
import time

import psutil

import profile_quality_corner_cost as reference
import native_quality_cache_writer as native


def record(key, scores):
    canonical = json.dumps(scores, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {"key": key, "quality": scores, "quality_sha256": hashlib.sha256(canonical).hexdigest()}


def error_signature(function):
    try:
        function()
    except BaseException as error:
        return [type(error).__name__, getattr(error, "errno", None), getattr(error, "winerror", None),
                str(error) if not isinstance(error, OSError) else None]
    raise ValueError("An expected original failure was not raised")


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "library"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(native.BASE)
    output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(native.__file__), Path(native.__file__).with_suffix(".cpp"),
             reference.V1 / "quality_engine.py", args.library.resolve(strict=True)]
    sources = [reference.data_identity(path) for path in paths]
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for position, path in enumerate(paths[:-1]):
        (snapshot / (str(position) + "_" + path.name)).write_bytes(path.read_bytes())
    process = psutil.Process()
    reference.atomic_json(output / "PROCESS_IDENTITY.json", {"pid": process.pid, "creation_time": process.create_time()})
    cache_type = reference.quality_engine.ScoreCache
    original_write, original_dump, original_fsync = cache_type.write, json.dump, os.fsync
    local = threading.local()
    blocked, release = threading.Event(), threading.Event()
    checks = {}
    state = {"status": "running", "accepted_count_contribution": 0, "checks": checks, "traces": []}

    class RenameDenied(RuntimeError):
        pass

    def observe(event, arguments):
        if event not in ("open", "tempfile.mkstemp", "os.rename"):
            return
        events = getattr(local, "events", None)
        if events is not None and any(isinstance(value, (str, bytes)) and str(output) in str(value) for value in arguments):
            events.append([event, list(arguments)])
        if event == "os.rename" and getattr(local, "deny", False):
            raise RenameDenied("The original rename audit denied publication")
        if event == "os.rename" and "owner_hold" in str(arguments[0]):
            blocked.set()
            if not release.wait(10):
                raise TimeoutError("The native ownership fixture was not released")

    sys.addaudithook(observe)

    def cache(folder):
        return cache_type(output / folder, {"fixture": "original-six-field-cache-v1"})

    def key(number):
        return hashlib.sha256(str(number).encode()).hexdigest()

    scores = dict(zip(reference.quality_engine.FIELDS, [123.25, .125, 127.5, 800, 16, 6.5]))

    def normalized(events, directory):
        result = []
        for event, arguments in events:
            values = [re.sub(r"\.[a-z0-9_]+\.tmp$", ".TOKEN.tmp", value.replace(str(directory), "CACHE"))
                      if isinstance(value, str) else value for value in arguments]
            result.append([event, values])
        return result

    try:
        left, right = cache("generic/current"), cache("generic/native")
        local.events = []
        left.write(key(0), scores)
        original_events = local.events
        local.events = []
        with native.native_quality_cache_execution(args.library) as trace:
            right.write(key(0), scores)
        native_events, local.events = local.events, None
        state["audit_events"] = {"current": original_events, "native": native_events}
        checks["original_open_mkstemp_rename_audit_order_and_arguments_exact"] = normalized(original_events, left.directory) == normalized(native_events, right.directory)
        state["traces"].append(trace)
        generic_cases = []
        with native.native_quality_cache_execution(args.library) as trace:
            for number in range(48):
                variant = dict(scores)
                variant["sharpness"] = (0., -0., 5e-324, 1e300, 1.7976931348623157e308, 1.25)[number % 6]
                variant["corner_count_half_resolution"] = (0, 1, 800, 2**63, 2**64 - 1, -1)[number % 6]
                if number % 2:
                    variant = dict(reversed(list(variant.items())))
                identity = key(number + 1)
                original_write(left, identity, variant)
                right.write(identity, variant)
                wanted, actual = left.read(identity), right.read(identity)
                exact = (left.directory / (identity + ".json")).read_bytes() == (right.directory / (identity + ".json")).read_bytes()
                exact = exact and wanted == actual and all(struct.pack("d", float(wanted[field])) == struct.pack("d", float(actual[field])) for field in variant)
                generic_cases.append(exact)
        state["traces"].append(trace)
        checks["all48_extreme_float_integer_key_order_cache_bytes_checksum_read_bits_exact"] = all(generic_cases) and trace["engine"]["native_calls"] == 48
        unicode_cache = cache("Unicode_物体_测试/native")
        with native.native_quality_cache_execution(args.library) as trace:
            unicode_cache.write(key(101), scores)
        state["traces"].append(trace)
        checks["actual_Unicode_path_write_and_original_checksum_read_exact"] = unicode_cache.read(key(101)) == scores and trace["engine"]["native_calls"] == 1
        variants = [{**scores, "sharpness": True}, {**scores, "corner_count_half_resolution": 2**1100},
                    {**scores, "extra": 1}, dict(scores)]
        with native.native_quality_cache_execution(args.library) as trace:
            for number, variant in enumerate(variants):
                identity = "short-key" if number == 3 else key(110 + number)
                original_write(left, identity, variant)
                right.write(identity, variant)
                if (left.directory / (identity + ".json")).read_bytes() != (right.directory / (identity + ".json")).read_bytes():
                    raise ValueError("An unsupported original cache-write fallback changed its bytes")
        state["traces"].append(trace)
        checks["four_unsupported_value_schema_key_routes_use_original_write_exact"] = trace["engine"]["fallback_calls"] == 4 and not trace["engine"]["native_calls"]
        collision = []
        original_names = tempfile._get_candidate_names
        try:
            for mode in ("current", "native"):
                fixture = cache("collision/" + mode)
                identity = key(120)
                retained = fixture.directory / (identity + ".collision.tmp")
                retained.write_bytes(b"protected original collision bytes")
                tempfile._get_candidate_names = lambda: iter(("collision", "available"))
                if mode == "native":
                    with native.native_quality_cache_execution(args.library) as trace:
                        fixture.write(identity, scores)
                    state["traces"].append(trace)
                    collision.append(trace["engine"]["create_attempts"] == 2)
                else:
                    original_write(fixture, identity, scores)
                collision.append(retained.read_bytes() == b"protected original collision bytes" and fixture.read(identity) == scores)
        finally:
            tempfile._get_candidate_names = original_names
        checks["actual_name_collision_retries_without_overwriting_prior_temporary"] = all(collision)
        missing = []
        for mode in ("current", "native"):
            fixture = cache("missing/" + mode)
            fixture.directory /= "not-created/nested"
            if mode == "native":
                with native.native_quality_cache_execution(args.library) as trace:
                    missing.append(error_signature(lambda: fixture.write(key(130), scores)))
                state["traces"].append(trace)
            else:
                missing.append(error_signature(lambda: original_write(fixture, key(130), scores)))
        checks["actual_missing_directory_type_errno_winerror_preserved"] = missing[0] == missing[1] and missing[0][0] == "FileNotFoundError"
        state["missing_directory_errors"] = missing
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
        kernel.CreateFileW.restype = ctypes.c_void_p
        kernel.CloseHandle.argtypes, kernel.CloseHandle.restype = [ctypes.c_void_p], ctypes.c_int
        locked, recovered = [], []
        for mode in ("current", "native"):
            fixture, identity = cache("locked/" + mode), key(140)
            target = fixture.directory / (identity + ".json")
            target.write_bytes(b"protected existing target")
            handle = kernel.CreateFileW(str(target), 0x80000000, 3, None, 3, 128, None)
            if handle == ctypes.c_void_p(-1).value:
                raise ctypes.WinError(ctypes.get_last_error())
            try:
                if mode == "native":
                    with native.native_quality_cache_execution(args.library) as trace:
                        locked.append(error_signature(lambda: fixture.write(identity, scores)))
                    state["traces"].append(trace)
                else:
                    locked.append(error_signature(lambda: original_write(fixture, identity, scores)))
                recovered.append(target.read_bytes() == b"protected existing target" and len(list(fixture.directory.glob("*.tmp"))) == 1)
            finally:
                if not kernel.CloseHandle(handle):
                    raise ctypes.WinError(ctypes.get_last_error())
            original_write(fixture, identity, scores)
            recovered.append(fixture.read(identity) == scores)
        checks["actual_locked_target_error_retained_temporary_target_and_recovery_exact"] = locked[0] == locked[1] and all(recovered)
        denied = []
        for mode in ("current", "native"):
            fixture, identity = cache("denied/" + mode), key(150)
            local.deny = True
            try:
                if mode == "native":
                    with native.native_quality_cache_execution(args.library) as trace:
                        denied.append(error_signature(lambda: fixture.write(identity, scores)))
                    state["traces"].append(trace)
                else:
                    denied.append(error_signature(lambda: original_write(fixture, identity, scores)))
            finally:
                local.deny = False
            if (fixture.directory / (identity + ".json")).exists() or len(list(fixture.directory.glob("*.tmp"))) != 1:
                raise ValueError("An audit-denied cache was published or lost its original temporary")
        checks["real_rename_audit_rejection_exception_and_unpublished_temporary_preserved"] = denied[0] == denied[1]
        fault_results = []
        for stage, winerror in ((1, 112), (2, 6)):
            identity = key(160 + stage)
            fixture = cache("fault_" + str(stage) + "/current")
            body = json.dumps(record(identity, scores), allow_nan=False).encode()
            try:
                if stage == 1:
                    def fail_dump(value, stream, **keywords):
                        stream.write(body[:len(body)//2].decode())
                        stream.flush()
                        raise OSError(28, os.strerror(28))
                    json.dump = fail_dump
                else:
                    def fail_fsync(descriptor):
                        raise OSError(9, os.strerror(9))
                    os.fsync = fail_fsync
                expected_error = error_signature(lambda: original_write(fixture, identity, scores))
            finally:
                json.dump, os.fsync = original_dump, original_fsync
            expected_temporary = next(fixture.directory.glob("*.tmp")).read_bytes()
            fixture = cache("fault_" + str(stage) + "/native")
            engine = native.NativeQualityCacheWriter(args.library)
            function = engine.dll.quality_cache_publish_fault
            function.argtypes = engine.function.argtypes + [ctypes.c_uint32, ctypes.c_uint32]
            function.restype = ctypes.c_int
            engine.function = lambda *parameters: function(*parameters, stage, winerror)
            try:
                actual_error = error_signature(lambda: engine.publish_record(fixture, identity, record(identity, scores)))
                actual_temporary = next(fixture.directory.glob("*.tmp")).read_bytes()
                details = engine.receipt()
                native_result = engine.state()["output"].tolist()
            finally:
                engine.close()
                state["traces"].append({"engine": engine.receipt()})
            exact = expected_error == actual_error and expected_temporary == actual_temporary and not (fixture.directory / (identity + ".json")).exists()
            fault_results.append({"stage": stage, "injected_WinError": winerror, "current": expected_error, "native": actual_error,
                                  "same_partial_or_full_temporary_bytes": exact, "native_receipt": details, "native_result": native_result})
        state["injected_failures_not_real_disk_full_or_power_loss"] = fault_results
        checks["injected_write_and_flush_failure_error_bytes_no_publication_and_handle_close_exact"] = all(item["same_partial_or_full_temporary_bytes"] and item["native_result"][0] == item["injected_WinError"] and item["native_receipt"]["closed_handles"] == 1 and not item["native_receipt"]["actual_flushes"] for item in fault_results)
        concurrent = []
        state["concurrent_outcomes"] = []
        for same_key in (False, True):
            for mode in ("current", "native"):
                fixture = cache("concurrent_" + str(same_key) + "/" + mode)
                def work(number):
                    identity = key(180 if same_key else 180 + number)
                    try:
                        fixture.write(identity, scores)
                        return {"identity": identity, "success": True}
                    except PermissionError as error:
                        return {"identity": identity, "success": False,
                                "error": [type(error).__name__, error.errno, error.winerror, None]}
                if mode == "native":
                    with native.native_quality_cache_execution(args.library) as trace:
                        with ThreadPoolExecutor(max_workers=4) as pool:
                            outcomes = list(pool.map(work, range(32)))
                    state["traces"].append(trace)
                    concurrent.append(trace["engine"]["native_calls"] == sum(item["success"] for item in outcomes) and trace["engine"]["actual_flushes"] == 32 and trace["engine"]["closed_handles"] == 32)
                else:
                    with ThreadPoolExecutor(max_workers=4) as pool:
                        outcomes = list(pool.map(work, range(32)))
                failures = [item for item in outcomes if not item["success"]]
                temporaries = list(fixture.directory.glob("*.tmp"))
                valid_errors = all(item["error"] == locked[0] for item in failures) if same_key else not failures
                concurrent.append(valid_errors and len(outcomes) == 32 and any(item["success"] for item in outcomes)
                    and len(temporaries) == len(failures) and all(fixture.read(item["identity"]) == scores for item in outcomes)
                    and all(path.read_bytes() == json.dumps(record(key(180), scores), allow_nan=False).encode() for path in temporaries))
                state["concurrent_outcomes"].append({"same_key": same_key, "mode": mode, "outcomes": outcomes,
                                                     "retained_full_temporary_count": len(temporaries)})
        checks["actual_four_workers_unique_and_same_key_durable_atomic_producers_exact"] = all(concurrent)
        held_engine = native.NativeQualityCacheWriter(args.library)
        fixture, identity = cache("owner_hold/native"), key(210)
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(held_engine.publish_record, fixture, identity, record(identity, scores))
                try:
                    if not blocked.wait(5):
                        raise TimeoutError("The real active native caller was not reached")
                    failure = error_signature(held_engine.close)
                    retained = not held_engine.closed and held_engine.active_calls == 1
                finally:
                    release.set()
                future.result(timeout=10)
        finally:
            held_engine.close()
        checks["actual_live_native_caller_prevents_unload_then_drains_closes_exact"] = retained and failure[0] == "RuntimeError" and held_engine.receipt()["DLL_unloaded"] and fixture.read(identity) == scores
        engine = native.NativeQualityCacheWriter(args.library)
        callback = native.AUDIT_RENAME(lambda: 1)
        invalid = []
        try:
            for length, count in ((0, 5), (16385, 5), (1, 4)):
                import numpy as np
                values = np.empty(5, np.uint64)
                invalid.append(engine.function(str(output / "invalid.tmp"), str(output / "invalid.json"), ctypes.c_char_p(b"x"), length, callback, values.ctypes.data, count) == 8)
        finally:
            engine.close()
        checks["actual_native_abi_length_and_result_budget_reject_without_files"] = all(invalid) and not (output / "invalid.tmp").exists() and not (output / "invalid.json").exists()
        old_abi = native.ABI_VERSION
        try:
            native.ABI_VERSION = 2
            mismatch = error_signature(lambda: native.NativeQualityCacheWriter(args.library))
        finally:
            native.ABI_VERSION = old_abi
        kernel.GetModuleHandleW.argtypes, kernel.GetModuleHandleW.restype = [ctypes.c_wchar_p], ctypes.c_void_p
        checks["actual_wrong_expected_abi_rejected_and_DLL_not_retained"] = mismatch[0] == "ValueError" and not kernel.GetModuleHandleW(str(args.library.resolve()))
        state["traces"].append({"engine": held_engine.receipt()})
        checks["all_owned_contexts_actual_handle_flush_close_and_original_methods_restored"] = all(item["engine"]["DLL_unloaded"] and not item["engine"]["active_calls"] and not item["engine"]["retained_output_bytes"] for item in state["traces"]) and cache_type.write is original_write and json.dump is original_dump and os.fsync is original_fsync
        checks["all_original_frozen_sources_unchanged"] = sources == [reference.data_identity(path) for path in paths]
        if not all(checks.values()):
            raise ValueError("An original cache interface, error or ownership contract failed: " + str([name for name, passed in checks.items() if not passed]))
        state.update(status="original_cache_generic_error_concurrency_and_lifecycle_passed_complete128_pending", sources=sources,
                     node_seconds=time.perf_counter() - started, real_disk_full_power_loss_full7338_training_not_run=True)
        reference.atomic_json(output / "RESULT.json", state)
        print({"status": state["status"], "checks": len(checks), "accepted_count_contribution": 0}, flush=True)
    except BaseException as error:
        state.update(status="failed", sources=sources, error_type=type(error).__name__, error=str(error), node_seconds=time.perf_counter()-started)
        reference.atomic_json(output / "FAILURE.json", state)
        raise
    finally:
        local.events, local.deny = None, False
        release.set()
        reference.atomic_json(output / "RESOURCE_CLOSE.json", {"original_cache_write_restored": cache_type.write is original_write,
            "original_dump_restored": json.dump is original_dump, "original_fsync_restored": os.fsync is original_fsync,
            "closed_fault_fixture_count": len(state.get("injected_failures_not_real_disk_full_or_power_loss", [])), "counted": False})


if __name__ == "__main__":
    main()
