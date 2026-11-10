"""Initialize an immutable raw-to-S08 run contract and verify the raw MCAP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--mcap", required=True, type=Path)
    parser.add_argument("--expected-mcap-sha256", required=True)
    parser.add_argument("--candidate-count", type=int, default=480)
    parser.add_argument("--image-count", type=int, default=180)
    args = parser.parse_args()

    run = args.run_root.resolve()
    mcap = args.mcap.resolve(strict=True)
    if run.exists():
        raise FileExistsError(f"Run root already exists: {run}")
    if args.candidate_count < args.image_count or args.image_count < 30:
        raise ValueError("Frame counts must satisfy candidate_count >= image_count >= 30")
    expected = args.expected_mcap_sha256.lower()
    actual = sha256_file(mcap)
    if actual != expected:
        raise RuntimeError(f"Raw MCAP SHA256 mismatch: expected {expected}, observed {actual}")
    run.mkdir(parents=True)
    write_json_new(
        run / "00_contract" / "raw_identity_receipt.json",
        {
            "schema_version": "e26_raw_identity_receipt_v1",
            "status": "verified",
            "path": str(mcap),
            "bytes": mcap.stat().st_size,
            "sha256": actual,
        },
    )
    write_json_new(
        run / "00_contract" / "run_contract.json",
        {
            "schema_version": "e26_video_to_s08_nollm_fresh_run_contract_v2",
            "status": "frozen_before_derived_outputs",
            "scope": "S00 through S08 inclusive",
            "terminal_artifact": "offline point-cloud preview after ground removal and before component filtering",
            "candidate_count": args.candidate_count,
            "selected_image_count": args.image_count,
            "frame_selection": "color-independent temporal bins with sharpness ranking",
            "language_model_used": False,
            "semantic_model_used": False,
            "sam2_used": False,
            "object_mask_used": False,
            "manual_visual_gate_used": False,
            "forbidden_stages": ["S09", "S10", "S11", "S12", "S13", "S14"],
            "raw_mcap": {
                "path": str(mcap),
                "bytes": mcap.stat().st_size,
                "sha256": actual,
            },
        },
    )
    print(json.dumps({"status": "initialized", "run_root": str(run), "mcap_sha256": actual}, indent=2))


if __name__ == "__main__":
    main()
