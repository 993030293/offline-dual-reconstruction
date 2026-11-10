"""Reuse the original third-batch fault protocol with actual CPU-output costs."""

import inspect
from pathlib import Path

import fault_codec_localization as preserved
import localize_strict_encoding as actual
from strict_sift_encoding import receipt
from metric_branch_localization_003 import ObservedWorker


def main():
    source = inspect.getsource(preserved.main)
    edits = [
        ('"codec_sift_matching.py", "sift_payload_input_worker.py", "bounded_zstd_payload.py",\n        "codec_owned_matching_worker.py", "codec_owned_json_write.py"',
         '"strict_sift_encoding.py", "native_sift_encoding.cpp", "integer_sift_matching.py",\n        "strict_encoding_node.py", "strict_encoding_owned_json_write.py"'),
        ('decoder = read_server_trace(workers[0].log.name) if workers and args.wire_mode == "codec" else None',
         'decoder = None'),
        ('"extra_actual_written_input_bytes": workers[0].receipt["actual_written_input_bytes"] if workers else 0,',
         '"extra_actual_array_output_bytes": encoder_receipt(namespace["ENCODER_INSTANCE"])["actual_recorded_successful_output_bytes"],\n'
         '                  "encoding": encoder_receipt(namespace["ENCODER_INSTANCE"]),'),
        ('for name in ("worker_restored", "argv_restored")),',
         'for name in ("worker_restored", "argv_restored", "bytes_exact_restored")),')
    ]
    for before, after in edits:
        if source.count(before) != 1:
            raise ValueError("The preserved native third-batch fault boundary changed: " + before[:80])
        source = source.replace(before, after)
    namespace = dict(preserved.main.__globals__)
    namespace.update(entry=actual, CodecObservedWorker=ObservedWorker,
                     encoder_receipt=receipt, __file__=str(Path(__file__).resolve()))
    exec(compile(source, str(Path(__file__)), "exec"), namespace)
    namespace["main"]()


if __name__ == "__main__":
    main()
