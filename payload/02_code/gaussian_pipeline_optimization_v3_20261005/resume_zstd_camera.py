"""Expand a verified new codec artifact and invoke the unchanged complete-state training entry."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import atomic_json, data_identity
from zstd_checkpoint import verified_expand
import train_camera_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-archive", type=Path, required=True)
    parser.add_argument("--transport-output", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    if "--resume-file" in remaining or remaining.count("--output") != 1:
        raise ValueError("The new transport must own exactly one native resume argument")
    args.transport_output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), Path(__file__).with_name("zstd_checkpoint.py")]
    sources = [data_identity(path) for path in paths]
    manifest = json.loads(args.checkpoint_archive.with_suffix(args.checkpoint_archive.suffix+".json").read_text())
    if not manifest.get("native_receipt") or manifest["native_receipt"].get("status") != "complete":
        raise ValueError("A codec artifact must contain a real complete native-state receipt")
    expanded = args.transport_output/"expanded.pth"
    started = time.perf_counter()
    trace = verified_expand(args.checkpoint_archive, expanded)
    atomic_json(expanded.with_suffix(".json"), {**manifest["native_receipt"], "checkpoint": trace["expanded"]})
    receipt = {"status": "verified_native_checkpoint_expanded", "source_archive": data_identity(args.checkpoint_archive),
        "native_payload": manifest["native_payload"], "expansion": trace, "sources": sources,
        "whole_verify_expand_seconds": time.perf_counter()-started, "native_algorithm_identity_unchanged": True,
        "expanded_copy_removed": False, "counted": False}
    atomic_json(args.transport_output/"TRANSPORT.json", receipt)
    original_argv = sys.argv
    try:
        sys.argv = [str(Path(train_camera_execution.__file__)), *remaining, "--resume-file", str(expanded)]
        train_camera_execution.main()
        output = Path(remaining[remaining.index("--output")+1])
        state = json.loads((output/"STATE.json").read_text())
        restore = json.loads((output/"RESTORE_CHECK.json").read_text())
        if state["status"] != "completed" or not all(restore["checks"].values()):
            raise ValueError("The actual native continuation lost complete state")
        if sources != [data_identity(path) for path in paths] or data_identity(expanded) != trace["expanded"]:
            raise ValueError("A transport implementation or owned expansion changed during recovery")
        if expanded.parent.resolve() != args.transport_output.resolve():
            raise ValueError("The verified temporary state is outside the owned recovery directory")
        expanded.unlink()
        expanded.with_suffix(".json").unlink()
        receipt.update(status="native_recovery_complete_quality_pending", restored_state=restore,
            expanded_copy_removed=True, output=str(output))
    except BaseException as error:
        receipt.update(status="failed_verified_expansion_preserved", error=repr(error))
        raise
    finally:
        sys.argv = original_argv
        atomic_json(args.transport_output/"TRANSPORT.json", receipt)


if __name__ == "__main__":
    main()
