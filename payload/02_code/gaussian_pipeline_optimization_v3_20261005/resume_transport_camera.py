"""Combine complete transport and private loading without changing training identity."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"),
               str(ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005")]
from runtime import atomic_json, data_identity
from zstd_checkpoint import verified_expand
from private_checkpoint_load import private_checkpoint_load
import train_camera_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport-mode", choices=("native", "zstd_mapped"), required=True)
    parser.add_argument("--native-checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-archive", type=Path, required=True)
    parser.add_argument("--transport-output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if "--resume-file" in remaining or remaining.count("--output") != 1:
        raise ValueError("The combined route must own exactly one complete native restore input")
    args.transport_output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("zstd_checkpoint.py"), Path(__file__).with_name("private_checkpoint_load.py")]
    sources = [data_identity(path) for path in paths]
    before = time.perf_counter()
    native = data_identity(args.native_checkpoint)
    native_receipt = json.loads(args.native_checkpoint.with_suffix(".json").read_text())
    archive_receipt = json.loads(args.checkpoint_archive.with_suffix(args.checkpoint_archive.suffix+".json").read_text())
    if (native_receipt.get("status") != "complete" or native_receipt["checkpoint"]["sha256"] != native["sha256"]
            or archive_receipt["native_payload"] != native or archive_receipt["native_receipt"] != native_receipt):
        raise ValueError("The native and compressed routes do not restore the same complete saved prefix")
    archive = data_identity(args.checkpoint_archive)
    owned = args.transport_output/"native.pth"
    expansion = None
    if args.transport_mode == "native":
        shutil.copyfile(args.native_checkpoint, owned)
    else:
        expansion = verified_expand(args.checkpoint_archive, owned)
    identity = data_identity(owned)
    if identity["sha256"] != native["sha256"] or identity["bytes"] != native["bytes"]:
        raise ValueError("The complete owned restore input changed its original native bytes")
    atomic_json(owned.with_suffix(".json"), {**native_receipt, "checkpoint": identity})
    record = {"status": "verified_combined_restore_pending", "transport_mode": args.transport_mode,
        "original": native, "archive": archive, "owned": identity, "expansion": expansion, "sources": sources,
        "complete_prepare_and_verify_seconds": time.perf_counter()-before,
        "expanded_copy_preserved_for_private_mapping_lifetime": True, "new_optimization_count": 0}
    atomic_json(args.transport_output/"TRANSPORT.json", record)
    previous = sys.argv
    try:
        with private_checkpoint_load(owned, enabled=args.transport_mode == "zstd_mapped") as trace:
            sys.argv = [str(Path(train_camera_execution.__file__)), *remaining, "--resume-file", str(owned)]
            train_camera_execution.main()
        output = Path(remaining[remaining.index("--output")+1])
        state = json.loads((output/"STATE.json").read_text())
        restore = json.loads((output/"RESTORE_CHECK.json").read_text())
        if state["status"] != "completed" or not all(restore["checks"].values()) or len(trace["calls"]) != 2:
            raise ValueError("The combined continuation or complete native restore failed")
        if sources != [data_identity(p) for p in paths] or data_identity(args.native_checkpoint) != native or data_identity(owned) != identity or data_identity(args.checkpoint_archive) != archive:
            raise ValueError("A combined source, original, archive or private native copy changed")
        record.update(status="combined_actual_continuation_complete_quality_pending", restored_state=restore,
            loader=trace, output=str(output), all_sources_and_payloads_unchanged=True)
    except BaseException as error:
        record.update(status="failed_owned_verified_payload_preserved", error=repr(error))
        raise
    finally:
        sys.argv = previous
        atomic_json(args.transport_output/"TRANSPORT.json", record)


if __name__ == "__main__":
    main()
