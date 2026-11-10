"""Replace the old S02 semantic review with machine-only input checks."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from common import atomic_write_json, image_rows, read_json, sha256_file


def sample_positions(count: int, maximum: int) -> list[int]:
    if count <= maximum:
        return list(range(count))
    return sorted(set(int(value) for value in np.linspace(0, count - 1, maximum)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-manifest", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-metric-frames", type=int, default=256)
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--cut-difference-threshold", type=float, default=35.0)
    args = parser.parse_args()

    started = time.monotonic()
    manifest_path = args.image_manifest.resolve(strict=True)
    root = args.image_root.resolve(strict=True)
    rows = image_rows(read_json(manifest_path))
    positions = sample_positions(len(rows), args.max_metric_frames)
    sample_set = set(positions)
    missing = []
    identity_failures = []
    dimensions = set()
    sampled = []
    previous_small = None
    previous_index = None
    for row in rows:
        index = int(row["image_index"])
        path = root / row["image_file"]
        if not path.is_file():
            missing.append(index)
            continue
        if args.verify_hashes and row.get("image_sha256"):
            actual = sha256_file(path)
            if actual != row["image_sha256"]:
                identity_failures.append(index)
        if index not in sample_set:
            continue
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            missing.append(index)
            continue
        dimensions.add((int(gray.shape[1]), int(gray.shape[0])))
        small = cv2.resize(gray, (128, 96), interpolation=cv2.INTER_AREA)
        difference = None
        if previous_small is not None:
            difference = float(cv2.absdiff(small, previous_small).mean())
        sampled.append(
            {
                "image_index": index,
                "mean_luma": float(small.mean()),
                "luma_stddev": float(small.std()),
                "laplacian_variance": float(cv2.Laplacian(small, cv2.CV_64F).var()),
                "difference_from_previous_sample": difference,
                "previous_sample_index": previous_index,
            }
        )
        previous_small = small
        previous_index = index

    cut_candidates = [
        row["image_index"]
        for row in sampled
        if row["difference_from_previous_sample"] is not None
        and row["difference_from_previous_sample"] >= args.cut_difference_threshold
    ]
    failures = []
    if missing:
        failures.append("missing_or_undecodable_frames")
    if identity_failures:
        failures.append("image_hash_mismatch")
    if len(dimensions) != 1:
        failures.append("sampled_dimensions_not_constant")
    receipt = {
        "schema_version": "s02_machine_frame_inventory_v1",
        "status": "failed_machine_input_integrity" if failures else "complete_machine_only_input_integrity",
        "language_model_used": False,
        "replaces": "legacy S02 agent visual review",
        "semantic_static_configuration": "not_assessed",
        "target_visibility": "not_assessed",
        "claim_boundary": (
            "This stage checks frame identity, readability, dimensions, and low-level temporal diagnostics. "
            "It does not infer scene semantics, static configuration, or target visibility."
        ),
        "image_manifest": str(manifest_path),
        "image_manifest_sha256": sha256_file(manifest_path),
        "frame_count": len(rows),
        "hashes_verified": args.verify_hashes,
        "sample_count": len(sampled),
        "sampled_dimensions": [list(value) for value in sorted(dimensions)],
        "cut_difference_threshold": args.cut_difference_threshold,
        "cut_candidates": cut_candidates,
        "missing_or_undecodable_indices": missing,
        "identity_failure_indices": identity_failures,
        "failures": failures,
        "runtime_seconds": time.monotonic() - started,
        "sample_metrics": sampled,
    }
    atomic_write_json(args.output, receipt)
    print(receipt["status"])
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
