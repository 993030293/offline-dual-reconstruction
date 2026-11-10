"""Start an owned immutable GPU query producer before heavy localization imports."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil


class EarlyGPUQueryProvider:
    def __init__(self, output, command=None):
        self.output = Path(output).resolve()
        if self.output.exists():
            raise FileExistsError("The cold GPU producer needs a new output directory")
        root = Path(__file__).resolve().parents[2]
        script = Path(__file__).with_name("profile_gpu_query_sift.py")
        command = command or [sys.executable, "-B", str(script), "--mode", "gpu", "--output", str(self.output)]
        self.stream = self.output.with_suffix(".log").open("xb")
        self.child = None
        self.result = None
        self.trace = {"actual_backend": "COLMAP_CUDA_CLI_SIFT_GPU", "setup_seconds": 0., "restore_seconds": 0.,
                      "binding_restored": False, "producer_start_monotonic": time.perf_counter(),
                      "root": str(root), "first_use_monotonic": None, "first_wait_seconds": 0.,
                      "broker": {"delivered": 0, "closed": False, "actual_CPU_cache_bypassed": True,
                                 "GPU_profile": str(self.output / "RESULT.json"), "records": []}}
        try:
            self.child = subprocess.Popen(command, stdout=self.stream, stderr=subprocess.STDOUT)
            self.trace["broker"]["profile_owner"] = {"pid": self.child.pid,
                "creation_time": psutil.Process(self.child.pid).create_time()}
        except BaseException:
            self.stream.close()
            raise
        self.trace["setup_seconds"] = time.perf_counter() - self.trace["producer_start_monotonic"]

    def drain(self):
        if self.child is None:
            return
        before = time.perf_counter()
        try:
            code = self.child.wait(timeout=600)
        finally:
            if self.child.poll() is None:
                self.child.terminate()
                try:
                    self.child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.child.kill()
                    self.child.wait(timeout=10)
            self.stream.close()
        self.trace.update(producer_returncode=code, producer_finished_observed_monotonic=time.perf_counter(),
                          last_drain_seconds=time.perf_counter() - before)
        return code

    def prepare(self, ordered):
        if self.result is not None:
            return
        before = time.perf_counter()
        self.trace["first_use_monotonic"] = before
        if self.drain():
            raise RuntimeError("The explicit complete GPU provider failed; CPU fallback is forbidden")
        result = json.loads((self.output / "RESULT.json").read_text(encoding="utf-8"))
        if result["status"] != "screen_passed_complete_localization_quality_pending" or not all(result["checks"].values()):
            raise ValueError("The complete GPU feature provider guards did not pass")
        if tuple(Path(path).resolve() for path in result["original_query_paths"]) != ordered:
            raise ValueError("GPU features must preserve the original complete query order")
        self.result = result
        self.trace["broker"]["actual_CLI_owners"] = result["owners"]
        self.trace["first_wait_seconds"] = time.perf_counter() - before

    def consume(self, bank, path, ordered):
        import cv2
        import numpy as np
        from query_feature_cache import file_identity, option_identity

        began = time.perf_counter()
        self.prepare(ordered)
        ordinal = self.trace["broker"]["delivered"]
        path = Path(path).resolve()
        if ordinal >= len(ordered) or path != ordered[ordinal]:
            raise ValueError("A new query order or unsupported query cannot consume GPU artifacts")
        result = self.result
        identities = {Path(row["path"]).resolve(): {key: row[key] for key in ("bytes", "sha256")} for row in result["inputs"]}
        if (option_identity(bank.extractor.options.todict()) != result["original_CPU_options"]
                or bank.protocol["SIFT_max_features"] != 12000 or bank.protocol["SIFT_peak_threshold"] != .004
                or file_identity(path) != identities[path]):
            raise ValueError("The original query source and SIFT options changed")
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        row = next(row for row in result["gray_rows"] if row["original"] == path.name)
        if gray is None or list(gray.shape) != row["shape"] or hashlib.sha256(gray.tobytes()).hexdigest() != row["gray_sha256"]:
            raise ValueError("The original caller grayscale differs from the GPU provider")
        artifact = next(Path(row["feature_artifact"]) for row in result["rows"] if row["name"] == path.name)
        with np.load(artifact, allow_pickle=False) as saved:
            keypoints, descriptors = saved["keypoints"].copy(), saved["descriptors"].copy()
        bank.protocol["query_feature_backend"] = "COLMAP_CUDA_CLI_SIFT_GPU"
        self.trace["broker"]["delivered"] += 1
        self.trace["broker"]["records"].append({"ordinal": ordinal, "name": path.name, "features": len(keypoints),
            "seconds": time.perf_counter() - began, "actual_backend": "GPU", "CPU_provenance_cache_used": False})
        return keypoints, descriptors


@contextmanager
def early_gpu_backend(paths, output, gpu, producer=None):
    if not gpu:
        from gpu_query_backend import query_backend
        with query_backend(paths, output, gpu=False) as trace:
            yield trace
        return
    import feature_localization
    if producer is None or producer.output != Path(output).resolve():
        raise ValueError("The original cold GPU producer must be started by its bootstrap owner")
    ordered = tuple(Path(path).resolve() for path in paths)
    original = feature_localization.FeatureBank.extra

    def extra(bank, path):
        return producer.consume(bank, path, ordered)

    failure = None
    try:
        feature_localization.FeatureBank.extra = extra
        yield producer.trace
    except BaseException as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        producer.trace["consumer_failure"] = failure
        raise
    finally:
        before = time.perf_counter()
        changed = feature_localization.FeatureBank.extra is not extra
        feature_localization.FeatureBank.extra = original
        code = producer.drain()
        producer.trace["broker"]["closed"] = True
        producer.trace.update(binding_restored=feature_localization.FeatureBank.extra is original,
                              restore_seconds=time.perf_counter() - before)
        if code and failure is None:
            raise RuntimeError("The explicit complete GPU provider failed; CPU fallback is forbidden")
        if changed:
            raise RuntimeError("Another writer changed the owned early query feature hook")
