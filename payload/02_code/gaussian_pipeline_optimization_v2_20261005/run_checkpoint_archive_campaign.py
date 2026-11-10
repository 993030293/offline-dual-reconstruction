"""Compare complete actual checkpoint payloads and lossless archive costs over three sources."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"02_code/gaussian_pipeline_optimization_v1_20261005"))
from runtime import atomic_json, data_identity
from checkpoint_archive import encode_checkpoint, load_checkpoint_archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.checkpoint) != 3 or len(set(map(str, args.checkpoint))) != 3:
        raise ValueError("The archive campaign requires three distinct complete real checkpoints")
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    paths = [Path(__file__), Path(__file__).with_name("checkpoint_archive.py")]
    identities = [data_identity(path) for path in paths]
    state = {"status": "running", "sources": identities, "rows": [], "checks": {},
        "minimum_archive_bytes_reduction": .1, "maximum_source_checkpoint_bytes": 4*1024**3,
        "maximum_seconds": 300, "maximum_RSS_GiB": 8, "cold_compression_and_read_costs_recorded": True,
        "native_complete_payload_values_types_layout_and_RNG_exact": True, "original_checkpoints_preserved": True,
        "native_integration_pending": True, "counted": False}
    atomic_json(args.output/"PROTOCOL.json", state)

    def exact(a, b):
        if type(a) is not type(b):
            return False
        if isinstance(a, torch.Tensor):
            return a.shape == b.shape and a.dtype == b.dtype and a.stride() == b.stride() and torch.equal(a, b)
        if isinstance(a, np.ndarray):
            return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
        if isinstance(a, dict):
            return list(a) == list(b) and all(exact(a[key], b[key]) for key in a)
        if isinstance(a, (list, tuple)):
            return len(a) == len(b) and all(exact(x, y) for x, y in zip(a, b))
        return bool(a == b)

    try:
        for index, source in enumerate(args.checkpoint):
            archive = args.output/f"trial{index}.ptgz"
            receipt = encode_checkpoint(source, archive)
            before = time.perf_counter()
            native = torch.load(source, map_location="cpu", weights_only=False)
            native_read = time.perf_counter()-before
            payload, trace = load_checkpoint_archive(archive, map_location="cpu", weights_only=False)
            preserved = exact(native, payload)
            row = {"source": data_identity(source), "archive": data_identity(archive), "complete_payload_exact": preserved,
                "compression_seconds": receipt["compression_seconds"], "native_read_seconds": native_read, "archive_read": trace,
                "payload_and_sidecar_bytes": archive.stat().st_size+archive.with_suffix(".ptgz.json").stat().st_size,
                "reduction_including_new_sidecar": 1-(archive.stat().st_size+archive.with_suffix(".ptgz.json").stat().st_size)/source.stat().st_size,
                "native_model_optimizer_densification_exposure_RNG_and_sampler_preserved": preserved}
            state["rows"].append(row)
            atomic_json(args.output/"STATE.json", state)
            print(json.dumps(row), flush=True)
            if not preserved:
                raise AssertionError("A lossless complete native checkpoint changed")
            del native, payload
        tests = {}
        source = args.output/"trial0.ptgz"
        manifest = source.with_suffix(".ptgz.json")
        original = json.loads(manifest.read_text())
        for name in ("payload_sha", "byte_bound", "decoder_sha"):
            folder = args.output/name
            folder.mkdir()
            target = folder/"invalid.ptgz"
            shutil.copyfile(source, target)
            record = json.loads(manifest.read_text())
            record["archive"]["path"] = str(target)
            if name == "payload_sha":
                record["native_payload"]["sha256"] = "0"*64
            elif name == "byte_bound":
                record["native_payload"]["bytes"] = 1
            else:
                record["decoder_source"]["sha256"] = "0"*64
            atomic_json(target.with_suffix(".ptgz.json"), record)
            try:
                load_checkpoint_archive(target, map_location="cpu", weights_only=False)
                tests[name+"_rejected"] = False
            except ValueError:
                tests[name+"_rejected"] = True
        tamper = args.output/"tampered.ptgz"
        data = bytearray(source.read_bytes())
        data[len(data)//2] ^= 1
        tamper.write_bytes(data)
        original["archive"]["path"] = str(tamper)
        atomic_json(tamper.with_suffix(".ptgz.json"), original)
        try:
            load_checkpoint_archive(tamper, map_location="cpu", weights_only=False)
            tests["archive_tamper_rejected"] = False
        except ValueError:
            tests["archive_tamper_rejected"] = True
        checks = {"three_actual_full_payload_oracles_exact": len(state["rows"]) == 3 and all(row["complete_payload_exact"] for row in state["rows"]),
            "all_three_disk_reductions_at_least10pct_including_manifest": all(row["reduction_including_new_sidecar"] >= .1 for row in state["rows"]),
            "all_malformed_lifecycle_cases_rejected": all(tests.values()), "sources_unchanged": identities == [data_identity(path) for path in paths]}
        state.update(checks=checks, lifecycle=tests, status="disk_and_payload_passed_native_resume_pending" if all(checks.values()) else "failed_verification")
        atomic_json(args.output/"RESULT.json", state)
        if not all(checks.values()):
            raise AssertionError("A checkpoint archive resource or integrity protection failed")
    except BaseException as error:
        state.update(status="failed", error=repr(error))
        atomic_json(args.output/"STATE.json", state)
        raise
    atomic_json(args.output/"STATE.json", state)


if __name__ == "__main__":
    main()
