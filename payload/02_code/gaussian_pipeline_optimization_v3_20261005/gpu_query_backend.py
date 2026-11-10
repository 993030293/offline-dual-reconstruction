"""Bind an explicit GPU feature provider without mislabelling CPU cache records."""

from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import time

import cv2
import hashlib
import numpy as np
import psutil

from gpu_query_node_config import CODE, read
from native_sift_process import native_sift_process, options
from query_feature_cache import file_identity, option_identity
import feature_localization


@contextmanager
def query_backend(paths, output, gpu):
    if not gpu:
        with native_sift_process(paths, enabled=True) as trace:
            trace["actual_backend"] = "original_CPU_two_processes"
            yield trace
        return
    started = time.perf_counter()
    output = Path(output)
    command = [sys.executable, "-B", str(CODE / "profile_gpu_query_sift.py"),
        "--mode", "gpu", "--output", str(output)]
    with output.with_suffix(".log").open("xb") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        owner = {"pid": child.pid, "creation_time": psutil.Process(child.pid).create_time()}
        try:
            returncode = child.wait(timeout=600)
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)
    if returncode:
        raise RuntimeError("The explicit complete GPU provider failed; CPU fallback is forbidden")
    result = read(output / "RESULT.json")
    if result["status"] != "screen_passed_complete_localization_quality_pending" or not all(result["checks"].values()):
        raise ValueError("The complete GPU feature provider guards did not pass")
    ordered = tuple(Path(p).resolve() for p in paths)
    if tuple(Path(p).resolve() for p in result["original_query_paths"]) != ordered:
        raise ValueError("GPU features must preserve the original complete query order")
    artifacts = {row["name"]: Path(row["feature_artifact"]) for row in result["rows"]}
    identities = {Path(row["path"]).resolve(): {k: row[k] for k in ("bytes", "sha256")}
        for row in result["inputs"]}
    gray_records = {row["original"]: row for row in result["gray_rows"]}
    original = feature_localization.FeatureBank.extra
    trace = {"actual_backend": "COLMAP_CUDA_CLI_SIFT_GPU", "setup_seconds": time.perf_counter() - started,
        "restore_seconds": 0., "binding_restored": False,
        "broker": {"delivered": 0, "closed": False, "actual_CPU_cache_bypassed": True,
            "GPU_profile": str(output / "RESULT.json"), "profile_owner": owner,
            "actual_CLI_owners": result["owners"], "records": []}}
    def extra(bank, path):
        began = time.perf_counter()
        path = Path(path).resolve()
        ordinal = trace["broker"]["delivered"]
        if ordinal >= len(ordered) or path != ordered[ordinal]:
            raise ValueError("A new query order or unsupported query cannot consume GPU artifacts")
        if (option_identity(bank.extractor.options.todict()) != result["original_CPU_options"]
                or bank.protocol["SIFT_max_features"] != 12000 or bank.protocol["SIFT_peak_threshold"] != .004
                or file_identity(path) != identities[path]):
            raise ValueError("The original query source and SIFT options changed")
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        row = gray_records[path.name]
        if gray is None or list(gray.shape) != row["shape"] or hashlib.sha256(gray.tobytes()).hexdigest() != row["gray_sha256"]:
            raise ValueError("The original caller grayscale differs from the GPU provider")
        with np.load(artifacts[path.name], allow_pickle=False) as saved:
            keypoints, descriptors = saved["keypoints"].copy(), saved["descriptors"].copy()
        bank.protocol["query_feature_backend"] = "COLMAP_CUDA_CLI_SIFT_GPU"
        trace["broker"]["delivered"] += 1
        trace["broker"]["records"].append({"ordinal": ordinal, "name": path.name,
            "features": len(keypoints), "seconds": time.perf_counter() - began,
            "actual_backend": "GPU", "CPU_provenance_cache_used": False})
        return keypoints, descriptors
    try:
        feature_localization.FeatureBank.extra = extra
        yield trace
    finally:
        began = time.perf_counter()
        feature_localization.FeatureBank.extra = original
        trace["broker"]["closed"] = True
        trace.update(binding_restored=feature_localization.FeatureBank.extra is original,
            restore_seconds=time.perf_counter() - began)
