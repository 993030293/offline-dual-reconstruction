"""Check exact private export source binding and normal/cancel restoration."""

import argparse
import json
from pathlib import Path
import time

from profile_sparse_ply_oracle import ROOT, native
from segmented_export_execution import segmented_export_execution
from runtime import atomic_json, data_identity


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.resolve().relative_to(ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/sg001")
    args.output.mkdir(parents=True, exist_ok=False)
    source_paths = [Path(__file__), Path(__file__).with_name("segmented_export_execution.py"),
                    Path(__file__).with_name("segmented_camera_support.py"), Path(native.__file__)]
    sources = [data_identity(path) for path in source_paths]
    original = native.main
    checks = {}
    for enabled in (False, True):
        for exception in (None, KeyboardInterrupt, SystemExit):
            caught = None
            try:
                with segmented_export_execution(native, args.output/"unused_states", enabled) as trace:
                    if native.main is original:
                        raise AssertionError("The private exact original export source was not bound")
                    if exception is not None:
                        raise exception("owned binding interruption")
            except BaseException as error:
                caught = error
            if exception is None:
                if caught is not None:
                    raise caught
            elif type(caught) is not exception or str(caught) != "owned binding interruption":
                raise AssertionError("A private export binding changed its original cancellation")
            if native.main is not original or not trace["original_main_restored"] or trace["stages"]:
                raise AssertionError("The private source binding leaked or executed an unrequested export")
            checks[f"enabled{enabled}_{exception.__name__ if exception else 'normal'}_exact_restore"] = True
    checks["all_source_identities_unchanged"] = sources == [data_identity(path) for path in source_paths]
    result = {"status": "private_segmented_export_source_bindings_passed", "checks": checks,
              "sources": sources, "complete_seconds": time.perf_counter()-started,
              "full_export_error_recovery_or_effect_claim": False, "new_optimization_count": 0}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps({"status": result["status"], "checks": checks}))


if __name__ == "__main__":
    main()
