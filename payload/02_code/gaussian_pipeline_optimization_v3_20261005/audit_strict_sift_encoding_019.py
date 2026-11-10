"""Check strict CPU encoding on complete captured payloads and API boundaries."""

import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import time
import warnings

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).parent
OUT = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup019/cpu0"
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from strict_sift_encoding import StrictSIFTEncoder, receipt, BUILD


def array_identity(value):
    return {"dtype": str(value.dtype), "shape": list(value.shape), "strides": list(value.strides),
            "owns_data": bool(value.flags.owndata), "writable": bool(value.flags.writeable),
            "bytes": hashlib.sha256(value.tobytes()).hexdigest()}


def attempt(encoder, value):
    before = array_identity(value)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = encoder(value)
            output = {"status": "completed", "output": array_identity(result)}
        except (ValueError, FloatingPointError) as error:
            output = {"status": "failed", "type": type(error).__name__, "message": str(error)}
    output["warnings"] = [{"type": item.category.__name__, "message": str(item.message)} for item in caught]
    output["input_unchanged"] = before == array_identity(value)
    return output


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), *[CODE / name for name in ("strict_sift_encoding.py", "native_sift_encoding.cpp",
        "build_native_sift_encoding.py", "build_native_vector_gray_stats.py", "integer_sift_matching.py")],
        BUILD / "native_sift_encoding.dll"]
    source_ids = [data_identity(path) for path in sources]
    snapshot = OUT / "source_snapshot"
    snapshot.mkdir()
    for number, path in enumerate(sources):
        shutil.copyfile(path, snapshot / (f"{number:02d}_" + path.name))
    capture_path = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/rc001_requests/REQUESTS.json"
    capture = json.loads(capture_path.read_text(encoding="utf-8"))
    inputs = [capture_path, *[Path(row["identity"]["path"]) for row in capture["blobs"].values()]]
    input_ids = [data_identity(path) for path in inputs]
    descriptors = {}
    for key, row in capture["blobs"].items():
        path = Path(row["identity"]["path"])
        raw = path.read_bytes()
        if len(raw) != row["rows"]*128 or hashlib.sha256(raw).hexdigest() != key:
            raise ValueError("An original complete captured descriptor changed")
        descriptors[key] = np.frombuffer(raw, np.uint8).reshape(row["rows"], 128).astype(np.float32)/512
    if len(capture["records"]) != 48:
        raise ValueError("The original complete48 request denominator changed")
    generic = []
    base = np.arange(31*128, dtype=np.uint32).reshape(31, 128).astype(np.uint8).astype(np.float32)/512
    class OwnArray(np.ndarray):
        pass
    cases = [("C", base), ("F", np.asfortranarray(base)), ("negative_row", base[::-1]),
             ("negative_column", base[:, ::-1]), ("strided", np.repeat(base, 2, axis=1)[:, ::2]),
             ("subclass", base.view(OwnArray)), ("empty", np.empty((0, 128), np.float32)),
             ("float64", base.astype(np.float64)), ("wrong_shape", base[:, :-1]),
             ("signed_zero", np.full((1, 128), -0., np.float32)),
             ("readonly", base.copy()), ("noninteger", np.full((1, 128), .001, np.float32)),
             ("nan", np.full((1, 128), np.nan, np.float32)), ("infinity", np.full((1, 128), np.inf, np.float32)),
             ("negative", np.full((1, 128), -.1, np.float32)),
             ("overflow", np.full((1, 128), np.finfo(np.float32).max, np.float32)),
             ("subnormal", np.full((1, 128), np.nextafter(np.float32(0), np.float32(1)), np.float32))]
    cases[10][1].flags.writeable = False
    for name, value in cases:
        a, b = StrictSIFTEncoder(False), StrictSIFTEncoder(True)
        left, right = attempt(a, value), attempt(b, value)
        generic.append({"name": name, "exact": left == right, "raw": left, "native": right,
                        "traces": [receipt(a), receipt(b)]})
    for error_mode in ("raise", "ignore", "warn"):
        value = cases[15][1]
        with np.errstate(over=error_mode):
            a, b = StrictSIFTEncoder(False), StrictSIFTEncoder(True)
            left, right = attempt(a, value), attempt(b, value)
        generic.append({"name": "overflow_mode_"+error_mode, "exact": left == right,
                        "raw": left, "native": right, "traces": [receipt(a), receipt(b)]})
    trials = []
    for trial in range(3):
        rows = {}
        for enabled in ((False, True) if trial != 1 else (True, False)):
            encoder = StrictSIFTEncoder(enabled)
            encoded_hashes, valid = [], True
            before = time.perf_counter()
            for record in capture["records"]:
                arrays = [descriptors[key] for key in [record["query"], *record["trains"]]]
                payload = b"".join(encoder(value).tobytes() for value in arrays)
                expected = b"".join(Path(capture["blobs"][key]["identity"]["path"]).read_bytes()
                                    for key in [record["query"], *record["trains"]])
                valid = valid and payload == expected
                encoded_hashes.append(hashlib.sha256(payload).hexdigest())
            elapsed = time.perf_counter()-before
            rows["native" if enabled else "raw"] = {"seconds_including_complete_expected_IO_and_hash": elapsed,
                "all48_original_payloads_exact": valid, "hashes": encoded_hashes, "trace": receipt(encoder)}
        trials.append(rows)
    checks = {
        "all_twenty_generic_API_errors_layouts_warnings_exact": len(generic) == 20 and all(row["exact"] for row in generic),
        "all_original48_payloads_three_full_pairs_exact": all(row[mode]["all48_original_payloads_exact"] for row in trials for mode in ("raw", "native")),
        "all_actual_native_outputs_single_array_and_baseline_six": all(len(call["actual_successful_numpy_outputs"]) == (1 if mode == "native" else 6)
            and call["route"] == ("native_single_scan" if mode == "native" else "original")
            for row in trials for mode in ("raw", "native") for call in row[mode]["trace"]["calls"]),
        "all_source_and_compiled_library_identities_unchanged": source_ids == [data_identity(path) for path in sources],
        "all_original_capture_inputs_unchanged": input_ids == [data_identity(path) for path in inputs],
        "no_Torch_or_CUDA_loaded": "torch" not in sys.modules,
    }
    result = {"status": "strict_CPU_component_passed_pending_full_pipeline0" if all(checks.values()) else "failed_verification",
              "checks": checks, "sources": source_ids, "inputs": input_ids, "generic": generic, "trials": trials,
              "median_seconds": {mode: statistics.median(row[mode]["seconds_including_complete_expected_IO_and_hash"] for row in trials) for mode in ("raw", "native")},
              "median_actual_array_output_bytes": {mode: statistics.median(row[mode]["trace"]["actual_recorded_successful_output_bytes"] for row in trials) for mode in ("raw", "native")},
              "complete_component_seconds": time.perf_counter()-started, "new_optimization_count": 0}
    atomic_json(OUT / "RESULT.json", result)
    print(json.dumps({key: result[key] for key in ("status", "checks", "median_seconds", "median_actual_array_output_bytes", "complete_component_seconds")}), flush=True)
    if not all(checks.values()):
        raise AssertionError("An original strict encoding, warning, layout or measured-output check failed")


if __name__ == "__main__":
    main()
