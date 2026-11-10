"""Preserve real third-batch prefixes through caller OSError and native SIGINT."""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import pickle
import random
import shutil
import signal
import sys
import time

import numpy as np

import localize_codec_requests as entry
from codec_sift_matching import CodecObservedWorker, read_server_trace
from check_codec_sift_protocol import vector_identity
from runtime import atomic_json, data_identity


def rng_identity():
    return {"python": hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
            "numpy": hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest()}


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wire-mode", choices=("raw", "codec"), required=True)
    parser.add_argument("--kind", choices=("error", "cancel"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(entry.OUT.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    function, composed = entry.compose(args.wire_mode == "codec")
    namespace = function.__globals__
    source_paths = [Path(__file__), Path(entry.__file__), *[entry.preserved.CODE / name for name in (
        "codec_sift_matching.py", "sift_payload_input_worker.py", "bounded_zstd_payload.py",
        "codec_owned_matching_worker.py", "codec_owned_json_write.py")]]
    sources = [data_identity(path) for path in source_paths]
    for number, path in enumerate(source_paths):
        shutil.copyfile(path, args.output.parent / (args.output.name + f"_SOURCE_{number:02d}_" + path.name))
    atomic_json(args.output.parent / (args.output.name + "_DRIVER_CONTRACT.json"), {
        "sources": sources, "wire_mode": args.wire_mode, "kind": args.kind,
        "registered_completed_native_batches_before_failure": 3, "new_optimization_count": 0})
    random.seed(170003)
    np.random.seed(170003)
    before_rng = rng_identity()
    originals = {"init": CodecObservedWorker.__init__, "batch": CodecObservedWorker.batch,
                 "own_batch": CodecObservedWorker.__dict__.get("batch"),
                 "signal": signal.getsignal(signal.SIGINT), "argv": sys.argv,
                 "contexts": {name: namespace[name] for name in (
                     "native_sift_process", "array_match_geometry", "owned_atomic_json")}}
    if not callable(originals["signal"]):
        raise ValueError("The original native SIGINT handler must be callable")
    workers, prefixes, signals, contexts = [], [], [], {}
    caught = None

    def observe_init(worker, *positional, **keywords):
        originals["init"](worker, *positional, **keywords)
        workers.append(worker)

    def observe_batch(worker, query, trains):
        values = originals["batch"](worker, query, trains)
        prefixes.append({"inputs": [vector_identity(value) for value in [query, *trains]],
                         "outputs": [[vector_identity(part) for part in value] for value in values]})
        if len(prefixes) == 3:
            atomic_json(args.output / "THIRD_NATIVE_BATCH_PREFIX.json", {"rows": prefixes,
                "caller_rng": rng_identity(), "three_actual_completed_worker_calls": len(worker.receipt["calls"])})
            if args.kind == "error":
                raise OSError("owned_complete_codec_caller_failure")
            signal.raise_signal(signal.SIGINT)
            raise AssertionError("The real native SIGINT was not delivered")
        return values

    def receive(signum, frame):
        signals.append({"signum": signum, "original_handler": originals["signal"].__name__,
                        "trigger": "signal.raise_signal(SIGINT)"})
        return originals["signal"](signum, frame)

    def observe_context(name):
        @contextmanager
        def wrapped(*positional, **keywords):
            with originals["contexts"][name](*positional, **keywords) as trace:
                contexts[name] = trace
                yield trace
        return wrapped

    try:
        CodecObservedWorker.__init__ = observe_init
        CodecObservedWorker.batch = observe_batch
        signal.signal(signal.SIGINT, receive)
        for name in originals["contexts"]:
            namespace[name] = observe_context(name)
        sys.argv = [str(Path(entry.__file__)), "--mode", "process", "--output", str(args.output)]
        try:
            function()
        except (OSError, KeyboardInterrupt) as error:
            expected = OSError if args.kind == "error" else KeyboardInterrupt
            if type(error) is not expected or (args.kind == "error" and str(error) != "owned_complete_codec_caller_failure"):
                raise
            caught = {"type": type(error).__name__, "message": str(error)}
    finally:
        CodecObservedWorker.__init__ = originals["init"]
        if originals["own_batch"] is None:
            del CodecObservedWorker.batch
        else:
            CodecObservedWorker.batch = originals["own_batch"]
        signal.signal(signal.SIGINT, originals["signal"])
        sys.argv = originals["argv"]
        namespace.update(originals["contexts"])
        resource = json.loads(workers[0].resource.read_text(encoding="utf-8")) if workers else {}
        decoder = read_server_trace(workers[0].log.name) if workers and args.wire_mode == "codec" else None
        checks = {
            "original_registered_caller_error_propagated": caught is not None,
            "three_actual_native_batches_scientific_prefix_retained": len(prefixes) == 3 and len(workers) == 1
                and len(workers[0].receipt["calls"]) == 3,
            "real_native_SIGINT_received_or_exact_OSError": (len(signals) == 1 and signals[0]["signum"] == signal.SIGINT)
                if args.kind == "cancel" else signals == [],
            "original_matching_stop_wait_closed": len(workers) == 1 and workers[0].stopped
                and workers[0].receipt["returncode"] == 0 and resource.get("closed", False),
            "original_native_sift_pool_joined": contexts.get("native_sift_process", {}).get("broker", {}).get("closed", False),
            "all_original_private_context_aliases_restored": all(trace.get("binding_restored", False) for trace in contexts.values())
                and len(contexts) == 3 and contexts["owned_atomic_json"]["closed"],
            "all_fault_hook_signal_and_argv_bindings_restored": CodecObservedWorker.__init__ is originals["init"]
                and CodecObservedWorker.batch is originals["batch"] and sys.argv is originals["argv"]
                and signal.getsignal(signal.SIGINT) is originals["signal"]
                and all(namespace[name] is value for name, value in originals["contexts"].items()),
            "no_partial_entry_published_as_complete": not (args.output / "RESULT.json").exists(),
            "original_full_worker_and_argv_aliases_restored": all(json.loads(
                (args.output / "BINDING_RESTORATION.json").read_text(encoding="utf-8"))[name]
                for name in ("worker_restored", "argv_restored")),
            "all_source_identities_unchanged": sources == [data_identity(path) for path in source_paths],
            "native_decoder_alias_restored_or_original_raw": decoder is None or decoder["stdin_alias_restored"],
        }
        result = {"status": "actual_expected_failure_preserved_pending_prefix_pair_review0" if all(checks.values()) else "failed_verification",
                  "checks": checks, "sources": sources, "wire_mode": args.wire_mode, "kind": args.kind,
                  "error": caught, "signals": signals, "prefix": prefixes, "before_rng": before_rng,
                  "after_rng": rng_identity(), "contexts": contexts,
                  "worker": workers[0].receipt if workers else None, "resource": resource, "decoder": decoder,
                  "extra_actual_written_input_bytes": workers[0].receipt["actual_written_input_bytes"] if workers else 0,
                  "complete_seconds": time.perf_counter()-started, "new_optimization_count": 0}
        atomic_json(args.output / "FAULT_RESULT.json", result)
    if not all(checks.values()):
        raise AssertionError("An actual full-prefix caller error or native SIGINT guard failed")
    raise SystemExit(86)


if __name__ == "__main__":
    main()
