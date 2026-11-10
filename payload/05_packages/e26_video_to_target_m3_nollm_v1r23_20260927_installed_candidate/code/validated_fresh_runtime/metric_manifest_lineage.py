"""Verify that a Sim3-bound image manifest is the time-supported registered subset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_metric_manifest_lineage(
    registered_path: Path,
    metric_bound_path: Path,
    support_receipt_path: Path,
    sim3_manifest_sha256: str,
    policy_sha256: str,
) -> tuple[list[dict], list[dict]]:
    registered = read_json(registered_path)
    metric_bound = read_json(metric_bound_path)
    receipt = read_json(support_receipt_path)
    if not isinstance(registered, list) or not isinstance(metric_bound, list):
        raise RuntimeError("Image manifests must contain arrays")
    if receipt.get("schema_version") != "e26_metric_time_support_filter_v2" or receipt.get("status") != "complete":
        raise RuntimeError("Metric time support receipt is incomplete or unsupported")
    if receipt.get("policy_sha256") != policy_sha256:
        raise RuntimeError("Metric time support uses a different alignment policy")
    if sha256_file(metric_bound_path) != sim3_manifest_sha256:
        raise RuntimeError("Metric-bound manifest does not match the Sim3 input lineage")
    names = [row["image_name"] for row in registered]
    if len(names) != len(set(names)):
        raise RuntimeError("Registered image manifest contains duplicate image names")
    start_ns = int(receipt["odometry_start_ns"])
    end_ns = int(receipt["odometry_end_ns"])
    if start_ns > end_ns:
        raise RuntimeError("Metric time support has an invalid timestamp range")
    expected = [
        row for row in registered
        if start_ns <= int(row["camera_header_ns"]) <= end_ns
    ]
    excluded = [
        {"image_name": row["image_name"], "camera_header_ns": int(row["camera_header_ns"])}
        for row in registered
        if not start_ns <= int(row["camera_header_ns"]) <= end_ns
    ]
    if metric_bound != expected:
        raise RuntimeError("Metric-bound manifest is not the registered time-supported subset")
    if (
        receipt.get("input_registered_count") != len(registered)
        or receipt.get("metric_bound_count") != len(expected)
        or receipt.get("visual_only_count") != len(excluded)
        or receipt.get("excluded") != excluded
    ):
        raise RuntimeError("Metric time support receipt does not describe the registered subset")
    fraction = len(expected) / len(registered) if registered else 0.0
    if (
        receipt.get("metric_bound_fraction") != fraction
        or len(expected) < int(receipt["minimum_supported_frames"])
        or fraction < float(receipt["minimum_supported_fraction"])
    ):
        raise RuntimeError("Metric time support gate does not match the registered subset")
    return registered, metric_bound
