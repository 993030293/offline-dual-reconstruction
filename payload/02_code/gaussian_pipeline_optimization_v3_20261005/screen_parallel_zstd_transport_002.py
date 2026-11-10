"""Screen exact ordered block scheduling on complete real transport inputs."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from run_codec_campaign import exact
from check_generic_checkpoint_transport import tensor_schema
from check_parallel_zstd_pipeline import pipeline_trace_valid
import parallel_zstd_checkpoint
import zstd_checkpoint


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scope = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/parallel_zstd_transport_001"
    args.output.resolve().relative_to(scope.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    old = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/codec_algorithms_001/PROTOCOL.json"
    expected = json.loads(old.read_text())["sources_inputs"][-3:]
    inputs = [Path(x["path"]) for x in expected]
    source_paths = [Path(__file__), Path(parallel_zstd_checkpoint.__file__), Path(zstd_checkpoint.__file__),
        Path(__file__).with_name("run_codec_campaign.py"), Path(__file__).with_name("check_generic_checkpoint_transport.py"),
        Path(__file__).with_name("check_parallel_zstd_pipeline.py")]
    sources = [data_identity(p) for p in source_paths]
    identities = [data_identity(p) for p in inputs]
    if identities != expected:
        raise ValueError("The three original native state revisions changed")
    process = psutil.Process()
    atomic_json(args.output / "CONTRACT.json", {
        "sources": sources, "inputs": identities,
        "owner": dict(pid=process.pid, creation_time=process.create_time()),
        "screen_only_not_formal_or_recovery": True, "minimum_complete_encoding_stage_reduction": .05,
        "whole_complete_transport_no_increase": True, "archive_body_bytes_exact": True,
        "workers": 2, "capacity": 4, "new_optimization_count": 0,
    })
    for ordinal, path in enumerate(source_paths):
        shutil.copyfile(path, args.output / f"SOURCE_{ordinal:02d}_{path.name}")
    import torch
    checks, rows = {}, []
    for index, source in enumerate(inputs):
        oracle = torch.load(source, map_location="cpu", weights_only=False)
        schemas = tensor_schema(oracle)
        modes = ["serial", "pipeline"] if index % 2 == 0 else ["pipeline", "serial"]
        sides = {}
        for mode in modes:
            before = time.perf_counter()
            folder = args.output / f"input{index}_{mode}"
            folder.mkdir()
            codec = zstd_checkpoint if mode == "serial" else parallel_zstd_checkpoint
            archive, received, expanded = folder / "encoded.bin", folder / "received.bin", folder / "expanded.pth"
            encode_started = time.perf_counter()
            encode = codec.encode_checkpoint(source, archive)
            encode_seconds = time.perf_counter() - encode_started
            shutil.copyfile(archive, received)
            shipped = data_identity(received)
            if any(shipped[k] != encode["archive"][k] for k in ["bytes", "sha256"]):
                raise ValueError("The complete archive changed during local transfer")
            atomic_json(received.with_suffix(".bin.json"), dict(encode, archive=shipped))
            decode = codec.verified_expand(received, expanded)
            payload = torch.load(expanded, map_location="cpu", weights_only=False)
            equal = exact(oracle, payload) and tensor_schema(payload) == schemas
            state_identity = data_identity(expanded)
            equal &= all(state_identity[k] == identities[index][k] for k in ["bytes", "sha256"])
            whole = time.perf_counter() - before
            checks[f"input{index}_{mode}_complete_bytes_values_layout_alias"] = equal
            if mode == "pipeline":
                trace = encode["ordered_pipeline"]
                checks[f"input{index}_actual_bounded_parallel_write_and_join"] = pipeline_trace_valid(trace)
            sides[mode] = {
                "complete_encode_seconds": encode_seconds,
                "complete_pack_local_copy_verify_expand_native_load_seconds": whole,
                "archive_and_manifest_bytes": archive.stat().st_size + archive.with_suffix(".bin.json").stat().st_size,
                "encode": encode, "expand": decode,
            }
            del payload
            if not equal:
                raise ValueError("A complete native byte, value or alias guard failed")
        del oracle
        checks[f"input{index}_full_archive_body_exact"] = all(sides["serial"]["encode"]["archive"][k] == sides["pipeline"]["encode"]["archive"][k] for k in ["bytes", "sha256"])
        rows.append(dict(input=index, modes=modes, sides=sides))
        atomic_json(args.output / "STATE.json", dict(status="screening", rows=rows, checks=checks))
        print(json.dumps({"input": index, "phase": {m: sides[m]["complete_encode_seconds"] for m in modes},
            "whole": {m: sides[m]["complete_pack_local_copy_verify_expand_native_load_seconds"] for m in modes}}), flush=True)
    sums = {mode: {key: sum(r["sides"][mode][key] for r in rows) for key in ["complete_encode_seconds", "complete_pack_local_copy_verify_expand_native_load_seconds"]} for mode in ["serial", "pipeline"]}
    checks["all_current_sources_and_native_inputs_unchanged"] = sources == [data_identity(p) for p in source_paths] and identities == [data_identity(p) for p in inputs]
    checks["CPU_only_without_CUDA_initialization"] = not torch.cuda.is_initialized()
    checks["complete_three_source_encoding_at_least5percent"] = sums["pipeline"]["complete_encode_seconds"] <= .95 * sums["serial"]["complete_encode_seconds"]
    checks["complete_three_source_whole_transport_no_increase"] = sums["pipeline"]["complete_pack_local_copy_verify_expand_native_load_seconds"] <= sums["serial"]["complete_pack_local_copy_verify_expand_native_load_seconds"]
    passed = all(checks.values())
    result = {
        "status": "screen_positive_formal_generic_recovery_pending" if passed else "screen_negative_no_promotion",
        "checks": checks, "complete_three_source_sums_seconds": sums,
        "encoding_reduction": 1 - sums["pipeline"]["complete_encode_seconds"] / sums["serial"]["complete_encode_seconds"],
        "whole_reduction": 1 - sums["pipeline"]["complete_pack_local_copy_verify_expand_native_load_seconds"] / sums["serial"]["complete_pack_local_copy_verify_expand_native_load_seconds"],
        "rows": rows, "sources": sources, "inputs": identities,
        "screen_envelope_seconds": time.perf_counter() - started,
        "next_action": "generic_then_formal_and_actual_native_recovery" if passed else "close_negative_count0",
        "new_optimization_count": 0, "network_transfer_new_training_GT_preview": "not_run",
    }
    atomic_json(args.output / "RESULT.json", result)
    print(json.dumps({k: result[k] for k in ["status", "checks", "complete_three_source_sums_seconds", "encoding_reduction", "whole_reduction"]}), flush=True)


if __name__ == "__main__":
    main()
