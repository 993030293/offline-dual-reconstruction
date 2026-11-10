"""Check lossless complete checkpoint transports on a new full-training checkpoint."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "02_code/gaussian_pipeline_optimization_v1_20261005"
V2 = ROOT / "02_code/gaussian_pipeline_optimization_v2_20261005"
V3 = ROOT / "02_code/gaussian_pipeline_optimization_v3_20261005"
sys.path[:0] = [str(V1), str(V2), str(V3)]
from runtime import atomic_json, data_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    import numpy as np
    import checkpoint_archive
    import zstd_checkpoint
    torch.set_num_threads(1)
    identity = data_identity(args.checkpoint)
    receipt = json.loads(args.checkpoint.with_suffix(".json").read_text())
    if receipt["status"] != "complete" or receipt["iteration"] != 6000:
        raise ValueError("A complete current intermediate training checkpoint is required")
    sources = [Path(__file__), Path(checkpoint_archive.__file__), Path(zstd_checkpoint.__file__)]
    source_ids = [data_identity(path) for path in sources]
    started = time.perf_counter()
    before = time.perf_counter()
    original = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    original_read_seconds = time.perf_counter()-before
    rows = []

    def equal(a, b):
        if type(a) is not type(b):
            return False
        if isinstance(a, torch.Tensor):
            return a.dtype == b.dtype and torch.equal(a, b)
        if isinstance(a, np.ndarray):
            return a.dtype == b.dtype and np.array_equal(a, b)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(equal(value, b[key]) for key, value in a.items())
        if isinstance(a, (tuple, list)):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        return a == b

    try:
        for trial in range(3):
            order = ("native", "gzip", "zstd") if trial != 1 else ("zstd", "gzip", "native")
            for label in order:
                folder = args.output / (str(trial)+"_"+label)
                folder.mkdir()
                before = time.perf_counter()
                if label == "native":
                    path = folder / "checkpoint.pth"
                    shutil.copyfile(args.checkpoint, path)
                    compressed = {"native_payload": identity, "archive": data_identity(path)}
                    copied = time.perf_counter()
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                    verified = {"complete_native_bytes_exact": path.read_bytes() == args.checkpoint.read_bytes()}
                elif label == "gzip":
                    path = folder / "checkpoint.pth.gz"
                    compressed = checkpoint_archive.encode_checkpoint(args.checkpoint, path)
                    copied = time.perf_counter()
                    payload, verified = checkpoint_archive.load_checkpoint_archive(path, map_location="cpu", weights_only=False)
                else:
                    path = folder / "checkpoint.pth.zst"
                    compressed = zstd_checkpoint.encode_checkpoint(args.checkpoint, path)
                    copied = time.perf_counter()
                    expanded = folder / "expanded.pth"
                    verified = zstd_checkpoint.verified_expand(path, expanded)
                    payload = torch.load(expanded, map_location="cpu", weights_only=False)
                duration = time.perf_counter()-before
                row = {"trial": trial, "label": label, "archive_bytes": path.stat().st_size,
                    "compression_or_copy_seconds": copied-before, "verification_and_read_seconds": duration-(copied-before),
                    "complete_write_verify_read_seconds": duration, "complete_state_literal_equality": bool(equal(original, payload)),
                    "archive_receipt": compressed, "verification": verified}
                rows.append(row)
                atomic_json(args.output / "STATE.json", {"status": "running", "rows": rows})
                if not row["complete_state_literal_equality"]:
                    raise ValueError("The transport changed a complete native checkpoint state")
                del payload
        atomic_json(args.output / "RESULT.json", {"status": "complete_current_checkpoint_transport_outcomes_recorded",
            "rows": rows, "native_source": identity, "source_ids": source_ids,
            "original_complete_read_seconds": original_read_seconds, "elapsed_seconds": time.perf_counter()-started,
            "source_checkpoint_unchanged": identity == data_identity(args.checkpoint),
            "source_code_unchanged": source_ids == [data_identity(path) for path in sources],
            "full_state_and_rng_preserved": all(row["complete_state_literal_equality"] for row in rows),
            "compression_storage_is_not_training_or_recovery_speed_claim": True})
    except BaseException as error:
        atomic_json(args.output / "FAILURE.json", {"status": "failed_current_checkpoint_transport_audit",
            "rows": rows, "error": repr(error), "elapsed_seconds": time.perf_counter()-started})
        raise


if __name__ == "__main__":
    main()
