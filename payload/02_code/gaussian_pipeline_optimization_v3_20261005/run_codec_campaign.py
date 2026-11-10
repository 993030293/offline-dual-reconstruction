"""Compare existing gzip and bounded zstd on three actual complete native states."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT/"02_code/gaussian_pipeline_optimization_v2_20261005"
sys.path[:0] = [str(V1), str(V2)]
from runtime import atomic_json, data_identity
import checkpoint_archive
import zstd_checkpoint


def exact(a, b):
    import torch
    if type(a) is not type(b):
        return False
    if isinstance(a, torch.Tensor):
        return a.shape == b.shape and a.dtype == b.dtype and a.stride() == b.stride() and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return list(a) == list(b) and all(exact(a[key], b[key]) for key in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
    return bool(a == b)


def gzip_expand(path, target):
    stream, trace = checkpoint_archive.verified_checkpoint_stream(path)
    before = time.perf_counter()
    try:
        with target.open("xb") as output:
            shutil.copyfileobj(stream, output, length=8*1024**2)
    finally:
        stream.close()
    trace["verification_and_expansion_seconds"] = trace["verification_seconds"]+time.perf_counter()-before
    return trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.checkpoint) != 3 or len(set(args.checkpoint)) != 3:
        raise ValueError("Exactly three distinct complete real checkpoint inputs are required")
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(zstd_checkpoint.__file__), Path(checkpoint_archive.__file__)]
    identities = [data_identity(path) for path in sources+args.checkpoint]
    atomic_json(args.output/"PROTOCOL.json", {"sources_inputs": identities,
        "incumbent": "accepted_Q76_gzip_level1_and_verified_sequential_expand",
        "candidate": "existing_system_zstd_8MiB_independent_blocks_level3", "trials_per_input": 3,
        "minimum_whole_three_source_compression_reduction": .1, "minimum_decode_reduction": .1,
        "maximum_archive_bytes_ratio_to_gzip": 1.1, "include_new_manifest_bytes": True,
        "complete_native_bytes_tensors_optimizer_RNG_order_layout_and_remaining_queue_exact": True,
        "first_library_prepare_hash_copy_and_expansion_costs_recorded": True,
        "real_native_continuation_required_before_promotion": True, "counted": False})
    import torch
    trials, checks = [], {}
    for index, source in enumerate(args.checkpoint):
        oracle = torch.load(source, map_location="cpu", weights_only=False)
        for trial in range(3):
            output = args.output/f"input{index}_trial{trial}"
            output.mkdir()
            rows = {}
            for mode in (["gzip", "zstd"] if trial % 2 == 0 else ["zstd", "gzip"]):
                archive, expanded = output/(mode+".archive"), output/(mode+".pth")
                before = time.perf_counter()
                receipt = (checkpoint_archive if mode == "gzip" else zstd_checkpoint).encode_checkpoint(source, archive)
                compression_wall = time.perf_counter()-before
                before = time.perf_counter()
                trace = gzip_expand(archive, expanded) if mode == "gzip" else zstd_checkpoint.verified_expand(archive, expanded)
                expand_wall = time.perf_counter()-before
                payload = torch.load(expanded, map_location="cpu", weights_only=False)
                equal = exact(oracle, payload)
                identity = data_identity(expanded)
                native = data_identity(source)
                bytes_exact = identity["sha256"] == native["sha256"] and identity["bytes"] == native["bytes"]
                rows[mode] = {"whole_encode_seconds": compression_wall, "whole_verify_expand_seconds": expand_wall,
                    "codec_seconds": receipt["compression_seconds"], "transport_bytes_including_manifest": archive.stat().st_size+archive.with_suffix(".archive.json").stat().st_size,
                    "native_complete_payload_exact": equal, "native_bytes_exact": bytes_exact, "expand": trace}
                checks[f"input{index}_trial{trial}_{mode}_complete_exact"] = equal and bytes_exact
                del payload
            row = {"input": index, "trial": trial, "source": data_identity(source), "rows": rows}
            trials.append(row)
            atomic_json(args.output/"STATE.json", {"status": "running", "trials": trials, "checks": checks})
            print(json.dumps({"input": index, "trial": trial, "rows": {mode: {key: value for key, value in item.items() if key != "expand"} for mode, item in rows.items()}}), flush=True)
            if not all(checks.values()):
                raise ValueError("A native complete payload or original byte/layout guard failed")
        del oracle
    by_input = []
    for index in range(3):
        selected = [row for row in trials if row["input"] == index]
        medians = {mode: {key: statistics.median(row["rows"][mode][key] for row in selected)
            for key in ("whole_encode_seconds", "whole_verify_expand_seconds", "transport_bytes_including_manifest")}
            for mode in ("gzip", "zstd")}
        by_input.append({"input": index, "median": medians})
    sums = {mode: {key: sum(row["median"][mode][key] for row in by_input)
        for key in ("whole_encode_seconds", "whole_verify_expand_seconds", "transport_bytes_including_manifest")}
        for mode in ("gzip", "zstd")}
    checks["whole_three_source_compression_reduces10percent"] = sums["zstd"]["whole_encode_seconds"] <= .9*sums["gzip"]["whole_encode_seconds"]
    checks["each_stored_transport_within110percent_gzip"] = all(row["median"]["zstd"]["transport_bytes_including_manifest"] <= 1.1*row["median"]["gzip"]["transport_bytes_including_manifest"] for row in by_input)
    checks["native_expand_is_not_slower"] = sums["zstd"]["whole_verify_expand_seconds"] <= sums["gzip"]["whole_verify_expand_seconds"]
    archive = args.output/"input0_trial0/zstd.archive"
    native_manifest = json.loads(archive.with_suffix(".archive.json").read_text())
    for name in ("payload_sha", "bound", "decoder", "truncation", "record_bound", "trailing", "archive_tamper"):
        folder = args.output/("invalid_"+name)
        folder.mkdir()
        target = folder/"state.archive"
        raw = bytearray(archive.read_bytes())
        manifest = json.loads(archive.with_suffix(".archive.json").read_text())
        if name == "payload_sha":
            manifest["native_payload"]["sha256"] = "0"*64
        elif name == "bound":
            manifest["native_payload"]["bytes"] = 1
        elif name == "decoder":
            manifest["decoder_source"]["sha256"] = "0"*64
        elif name == "truncation":
            del raw[-1:]
        elif name == "record_bound":
            raw[len(zstd_checkpoint.MAGIC):len(zstd_checkpoint.MAGIC)+8] = zstd_checkpoint.RECORD.pack(zstd_checkpoint.BLOCK+1, 1)
        elif name == "trailing":
            raw.extend(b"x")
        else:
            raw[len(raw)//2] ^= 1
        target.write_bytes(raw)
        manifest["archive"]["path"] = str(target)
        if name in ("truncation", "record_bound", "trailing"):
            manifest["archive"] = data_identity(target)
        atomic_json(target.with_suffix(".archive.json"), manifest)
        try:
            zstd_checkpoint.verified_expand(target, folder/"expanded.pth")
            checks[name+"_rejected"] = False
        except ValueError:
            checks[name+"_rejected"] = True
    checks["sources_inputs_unchanged"] = identities == [data_identity(path) for path in sources+args.checkpoint]
    result = {"status": "codec_stage_passed_native_continuation_pending" if all(checks.values()) else "failed_verification",
        "checks": checks, "by_input": by_input, "sum_source_medians": sums, "trials": trials,
        "system_zstd_version": zstd_checkpoint.Zstd().version, "environment_modified": False, "counted": False}
    atomic_json(args.output/"RESULT.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "trials"}), flush=True)
    if not all(checks.values()):
        raise ValueError("A real complete checkpoint codec resource or integrity guard failed")


if __name__ == "__main__":
    main()
