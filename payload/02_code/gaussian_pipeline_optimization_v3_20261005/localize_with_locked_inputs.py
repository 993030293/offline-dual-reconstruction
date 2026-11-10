"""Measure unchanged current localization with explicit immutable input snapshots."""

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
import runtime
from runtime import atomic_json, data_identity
import localize_integer_sift
from locked_input_identity import locked_input_identity


def replace(old, new):
    for module in list(sys.modules.values()):
        if module is not None and getattr(module, "data_identity", None) is old:
            setattr(module, "data_identity", new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-mode", choices=["baseline", "locked"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    inputs = argparse.ArgumentParser(add_help=False)
    for name in ("export", "database", "initial-frames", "reference-run"):
        inputs.add_argument("--"+name, type=Path, required=True)
    frozen, _ = inputs.parse_known_args(remaining)
    roots = [ROOT/"02_code", frozen.export, frozen.database, frozen.initial_frames, frozen.reference_run/"00_contract",
        frozen.reference_run/"05_evaluation/C_augmented_212/H_LOCALIZATION.json"]
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(__file__).with_name("locked_input_identity.py"), Path(localize_integer_sift.__file__)]
    identity = [data_identity(p) for p in sources]
    atomic_json(args.output/"SNAPSHOT_PROTOCOL.json", {"sources": identity, "mode": args.snapshot_mode,
        "allowed_immutable_roots": list(map(str, roots)), "original_all24_and48_guards": True,
        "one_complete_native_hash_per_held_file": True, "digest_phase_minimum_reduction": .1,
        "whole_entry_minimum_reduction": .05, "all_lock_probe_verify_and_close_costs_included": True,
        "original_format_flags_thresholds_and_source_identity_unchanged": True, "counted": False})
    previous_argv, calls = sys.argv, []
    snapshot_trace, lifecycle_seconds = None, 0.
    original = runtime.data_identity
    try:
        started = time.perf_counter()
        with locked_input_identity(roots, args.snapshot_mode == "locked") as snapshot:
            current = runtime.data_identity
            lifecycle_seconds += time.perf_counter()-started

            def measured(path):
                before = time.perf_counter()
                value = current(path)
                calls.append({"path": str(Path(path)), "bytes": value["bytes"], "seconds": time.perf_counter()-before})
                return value

            replace(current, measured)
            try:
                sys.argv = [str(Path(localize_integer_sift.__file__)), "--matching-mode", "integer",
                    "--output", str(args.output/"entry"), *remaining]
                localize_integer_sift.main()
            finally:
                replace(measured, current)
            snapshot_trace = snapshot.receipt() if snapshot is not None else None
            closing = time.perf_counter()
        lifecycle_seconds += time.perf_counter()-closing
        if runtime.data_identity is not original or identity != [data_identity(p) for p in sources]:
            raise ValueError("Snapshot source identity or complete function restoration failed")
        atomic_json(args.output/"SNAPSHOT_TRACE.json", {"status": "actual_unchanged_localization_complete",
            "mode": args.snapshot_mode, "calls": calls, "snapshot": snapshot_trace,
            "identity_phase_seconds": sum(r["seconds"] for r in calls)+lifecycle_seconds,
            "lock_context_probe_verify_and_close_seconds": lifecycle_seconds,
            "no_held_handles_after_context": snapshot is None or not snapshot.rows,
            "sources_unchanged": True, "counted": False})
    finally:
        sys.argv = previous_argv


if __name__ == "__main__":
    main()
