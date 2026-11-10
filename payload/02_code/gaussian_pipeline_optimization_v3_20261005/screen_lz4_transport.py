"""Screen complete local transport against accepted Zstd before promotion."""

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
import lz4_checkpoint
import zstd_checkpoint


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scope = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/lz4_transport_001"
    args.output.resolve().relative_to(scope.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    old = ROOT / "04_outputs/gaussian_pipeline_optimization_v3_20261005/codec_algorithms_001/PROTOCOL.json"
    inputs = [Path(x["path"]) for x in json.loads(old.read_text())["sources_inputs"][-3:]]
    source_paths = [Path(__file__), Path(lz4_checkpoint.__file__), Path(zstd_checkpoint.__file__),
        Path(__file__).with_name("run_codec_campaign.py"), Path(__file__).with_name("check_generic_checkpoint_transport.py")]
    sources = [data_identity(p) for p in source_paths]
    identities = [data_identity(p) for p in inputs]
    process = psutil.Process()
    atomic_json(args.output / "CONTRACT.json", {
        "sources": sources, "inputs": identities,
        "owner": dict(pid=process.pid, creation_time=process.create_time()),
        "screen_only_not_formal_or_recovery": True, "maximum_archive_ratio_to_zstd": 1.20,
        "minimum_complete_three_source_stage_reduction": .05, "new_optimization_count": 0,
    })
    for ordinal, path in enumerate(source_paths):
        shutil.copyfile(path, args.output / f"SOURCE_{ordinal:02d}_{path.name}")
    import torch
    checks, rows = {}, []
    for index, source in enumerate(inputs):
        oracle = torch.load(source, map_location="cpu", weights_only=False)
        schemas = tensor_schema(oracle)
        modes = ["zstd", "lz4"] if index % 2 == 0 else ["lz4", "zstd"]
        sides = {}
        for mode in modes:
            before = time.perf_counter()
            folder = args.output / f"input{index}_{mode}"
            folder.mkdir()
            codec = zstd_checkpoint if mode == "zstd" else lz4_checkpoint
            archive, received, expanded = folder / "encoded.bin", folder / "received.bin", folder / "expanded.pth"
            encode = codec.encode_checkpoint(source, archive)
            shutil.copyfile(archive, received)
            shipped = data_identity(received)
            if any(shipped[k] != encode["archive"][k] for k in ["bytes", "sha256"]):
                raise ValueError("A complete archive changed during local transfer")
            received_manifest = dict(encode, archive=shipped)
            atomic_json(received.with_suffix(".bin.json"), received_manifest)
            decode = codec.verified_expand(received, expanded)
            payload = torch.load(expanded, map_location="cpu", weights_only=False)
            equal = exact(oracle, payload) and tensor_schema(payload) == schemas
            state_identity = data_identity(expanded)
            equal &= all(state_identity[k] == identities[index][k] for k in ["bytes", "sha256"])
            whole = time.perf_counter() - before
            checks[f"input{index}_{mode}_complete_bytes_values_layout_alias"] = equal
            sides[mode] = {
                "complete_pack_local_copy_verify_expand_native_load_seconds": whole,
                "archive_and_manifest_bytes": archive.stat().st_size + archive.with_suffix(".bin.json").stat().st_size,
                "encode": encode, "expand": decode,
            }
            del payload
            if not equal:
                raise ValueError("A complete native byte, value or alias guard failed")
        del oracle
        rows.append(dict(input=index, modes=modes, sides=sides))
        atomic_json(args.output / "STATE.json", dict(status="screening", rows=rows, checks=checks))
        print(json.dumps({"input": index, "costs": {m: sides[m]["complete_pack_local_copy_verify_expand_native_load_seconds"] for m in modes},
            "archive_ratio": sides["lz4"]["archive_and_manifest_bytes"] / sides["zstd"]["archive_and_manifest_bytes"]}), flush=True)
    sums = {mode: sum(r["sides"][mode]["complete_pack_local_copy_verify_expand_native_load_seconds"] for r in rows) for mode in ["zstd", "lz4"]}
    ratios = [r["sides"]["lz4"]["archive_and_manifest_bytes"] / r["sides"]["zstd"]["archive_and_manifest_bytes"] for r in rows]
    checks["all_current_sources_and_native_inputs_unchanged"] = sources == [data_identity(p) for p in source_paths] and identities == [data_identity(p) for p in inputs]
    checks["CPU_only_without_CUDA_initialization"] = not torch.cuda.is_initialized()
    checks["all_archives_within_preregistered1_20_storage_ratio"] = all(r <= 1.20 for r in ratios)
    checks["complete_three_source_diagnostic_stage_at_least5percent"] = sums["lz4"] <= .95 * sums["zstd"]
    passed = all(checks.values())
    result = {
        "status": "screen_positive_formal_generic_recovery_pending" if passed else "screen_negative_no_promotion",
        "checks": checks, "complete_three_source_stage_seconds": sums,
        "complete_three_source_stage_reduction": 1 - sums["lz4"] / sums["zstd"],
        "archive_ratios_to_zstd": ratios, "rows": rows, "sources": sources, "inputs": identities,
        "screen_envelope_seconds": time.perf_counter() - started,
        "next_action": "generic_then_formal_and_actual_native_recovery" if passed else "close_negative_count0",
        "new_optimization_count": 0, "network_transfer_new_training_GT_preview": "not_run",
    }
    atomic_json(args.output / "RESULT.json", result)
    print(json.dumps({k: result[k] for k in ["status", "checks", "complete_three_source_stage_seconds", "complete_three_source_stage_reduction", "archive_ratios_to_zstd"]}), flush=True)


if __name__ == "__main__":
    main()
