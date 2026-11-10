"""Repair only the recorded SIFT option representation for one complete process trial."""

import argparse
import json
from pathlib import Path
import sys
import time

import localization_variant
import localization_contract_adapter as adapter
from atomic_json_windows import atomic_json
from query_feature_cache import option_identity
from runtime import data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    original = adapter.CompatibleNativeSiftProcess.bind
    prior_argv = sys.argv
    started = time.perf_counter()
    sources = [Path(__file__), Path(adapter.__file__), Path(localization_variant.__file__)]
    identities = [data_identity(path) for path in sources]

    def bind(owner, native):
        proxy = original(owner, native)
        owner.trace["frozen_complete_options"] = option_identity(owner.frozen_options)
        return proxy

    adapter.CompatibleNativeSiftProcess.bind = bind
    record = {"status": "repair_in_progress", "sources": identities,
        "only_trace_options_encoded_to_registered_enum_identity": True,
        "actual_worker_option_dictionary_unchanged": True, "algorithm_thresholds_unchanged": True}
    try:
        if args.self_check:
            import pycolmap
            options = adapter.native_process.options()
            options.num_threads = 4
            rejected = False
            try:
                json.dumps(options.todict(), allow_nan=False)
            except TypeError:
                rejected = True
            if not rejected:
                raise ValueError("The original enum recording failure was not reproduced")
            owner = adapter.CompatibleNativeSiftProcess([args.side / "initial/evaluation_images" /
                next((args.side / "initial/evaluation_images").glob("*.jpg")).name])
            native = pycolmap.Sift(options, pycolmap.Device.cpu)
            owner.bind(native)
            json.dumps(owner.trace, allow_nan=False)
            if option_identity(owner.frozen_options) != option_identity(native.options.todict()):
                raise ValueError("The actual worker dictionary changed during receipt repair")
            owner.close()
            record.update(status="native_enum_failure_reproduced_and_trace_repair_verified",
                all_complete_options_preserved=True, broker_closed=owner.closed)
            print(json.dumps(record))
        else:
            sys.argv = [str(Path(localization_variant.__file__)), "--side", str(args.side),
                "--output", str(args.output), "--mechanism", "process", "--compat-adapter"]
            localization_variant.main()
            if identities != [data_identity(path) for path in sources]:
                raise ValueError("The explicit receipt adapter source changed during its trial")
            record["status"] = "complete_process_trial_with_explicit_trace_receipt_repair"
    except BaseException as error:
        record.update(status="failed_process_trial_or_receipt_repair", error=repr(error))
        raise
    finally:
        adapter.CompatibleNativeSiftProcess.bind = original
        sys.argv = prior_argv
        record.update(binding_restored=adapter.CompatibleNativeSiftProcess.bind is original,
            elapsed_seconds=time.perf_counter()-started)
        if not args.self_check and args.output.is_dir():
            atomic_json(args.output / "JSON_RECEIPT_ADAPTER.json", record)


if __name__ == "__main__":
    main()
