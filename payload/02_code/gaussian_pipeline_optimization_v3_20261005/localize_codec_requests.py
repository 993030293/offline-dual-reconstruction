"""Run the full incumbent fusion and optionally encode only native requests."""

import argparse
import inspect
from pathlib import Path
import sys
import time

STARTED = time.perf_counter()
import localize_geometry_sift_fusion as preserved
from codec_owned_json_write import owned_atomic_json, COMPOSED_SOURCE
from codec_sift_matching import CodecObservedWorker, read_server_trace

OUT = preserved.ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/oup018/R170"


def compose(enabled):
    source = inspect.getsource(preserved.main)
    edits = [
        ('worker = ObservedWorker(log)', 'worker = CodecObservedWorker(log, enabled=WIRE_ENABLED)'),
        ('paths = [Path(__file__), CODE/"geometry_sift_fusion_node.py",',
         'paths = [Path(__file__), CODE/"localize_geometry_sift_fusion.py",\n'
         '        *[CODE/name for name in ("codec_localization_node.py", "codec_owned_json_write.py",\n'
         '            "codec_sift_matching.py", "sift_payload_input_worker.py", "bounded_zstd_payload.py",\n'
         '            "codec_owned_matching_worker.py", "fused_owned_matching_worker.py",\n'
         '            "sift_process_node_config.py", "run_sift_process_campaign_005.py")],\n'
         '        ROOT/"05_envs/colmap-4.1.1-cuda/bin/zstd.dll", CODE/"geometry_sift_fusion_node.py",'),
        ('<= 32*1024**3', '<= 16*1024**3'),
        ('        atomic_json(args.output/"RESULT.json",',
         '        wire = workers[0].receipt["wire"]\n'
         '        bodies = [row for row in wire if row["command"] == "batch"]\n'
         '        server = read_server_trace(workers[0].log.name) if WIRE_ENABLED else None\n'
         '        checks["complete50_actual_pipe_write_receipts"] = (len(wire) == 50 and len(bodies) == 48\n'
         '            and all(row["status"] == "completed" for row in wire)\n'
         '            and sum(row["actual_written_input_bytes"] for row in wire)\n'
         '                == workers[0].receipt["actual_written_input_bytes"])\n'
         '        checks["all48_complete_decoder_payloads_exact_or_original_raw"] = (not WIRE_ENABLED or (\n'
         '            server["stdin_alias_restored"] and len(server["rows"]) == 48\n'
         '            and all(a["raw_sha256"] == b["raw_sha256"] and a["raw_bytes"] == b["raw_payload_bytes"]\n'
         '                and a["wire_payload_bytes"] == b["transmitted_payload_bytes"]\n'
         '                for a, b in zip(server["rows"], bodies))))\n'
         '        atomic_json(args.output/"RESULT.json",'),
        ('"mode": args.mode, "checks": checks, "localization":',
         '"wire_mode": "codec" if WIRE_ENABLED else "raw", "decoder": server,\n'
         '            "actual_written_input_bytes": workers[0].receipt["actual_written_input_bytes"],\n'
         '            "mode": args.mode, "checks": checks, "localization":'),
    ]
    for before, after in edits:
        if source.count(before) != 1:
            raise ValueError("The incumbent full localization source boundary changed: " + before[:90])
        source = source.replace(before, after)
    namespace = dict(preserved.main.__globals__)
    namespace.update(OWNED=OUT, __file__=str(Path(__file__).resolve()), STARTED=STARTED,
        WIRE_ENABLED=enabled, CodecObservedWorker=CodecObservedWorker, read_server_trace=read_server_trace,
        owned_atomic_json=owned_atomic_json, COMPOSED_SOURCE=COMPOSED_SOURCE)
    exec(compile(source, str(Path(__file__)), "exec"), namespace)
    return namespace["main"], source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wire-mode", choices=("raw", "codec"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = sys.argv
    try:
        sys.argv = [str(Path(__file__)), "--mode", "process", "--output", str(args.output)]
        function, _ = compose(args.wire_mode == "codec")
        function()
    finally:
        sys.argv = original


if __name__ == "__main__":
    main()
