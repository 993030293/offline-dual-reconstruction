"""Replace only worker feature transport while retaining the complete R90 matcher."""

from contextlib import contextmanager
import hashlib
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from descriptor_residency_plan import DescriptorResidencyPlan, MAXIMUM_CACHE_BYTES, compact_plan, shape_admitted
from integer_sift_matching import DistanceWorker, bytes_exact


class ResidentWorker(DistanceWorker):
    def __init__(self, log, cache_bytes=MAXIMUM_CACHE_BYTES):
        self.rpc_failed = False
        self.plan = DescriptorResidencyPlan(cache_bytes)
        source = Path(__file__).with_name("resident_sift_worker.py").resolve().as_posix()
        if source[1:3] != ":/" or source[0].lower() != "c":
            raise ValueError("The private resident worker must stay inside the existing C drive project")
        self.command = ["wsl", "-d", "E26-C2-LIV-20260907-134802", "--",
            "/opt/e26_thin_surface_v1/venv/bin/python", "/mnt/c/"+source[3:], "--cache-bytes", str(cache_bytes)]
        self.log = Path(log).open("xb")
        started = time.perf_counter()
        self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, bufsize=0)
        self.stopped = False
        self.receipt = {"command": self.command, "pid": self.process.pid, "calls": [], "counted": False}
        try:
            self.receipt["info"] = self.rpc({"command": "info"})
            self.receipt["info"].pop("payload")
            if self.receipt["info"]["cache_byte_budget"] != self.plan.limit:
                raise ValueError("The actual CUDA worker cache budget differs from the owned client")
        except BaseException:
            self.process.stdin.close()
            self.process.stdout.close()
            self.process.wait(timeout=30)
            self.log.close()
            raise
        self.receipt["startup_seconds"] = time.perf_counter()-started

    def rpc(self, record, payload=b""):
        try:
            return super().rpc(record, payload)
        except BaseException:
            self.rpc_failed = True
            raise

    def batch(self, query, trains):
        if self.stopped:
            raise RuntimeError("The private resident worker cannot be reused after close")
        started = time.perf_counter()
        q, values = bytes_exact(query), [bytes_exact(t) for t in trains]
        counts = [len(t) for t in values]
        if not shape_admitted(len(q), counts):
            raise ValueError("The unchanged numerical and resource admission excludes this request")
        binaries = [t.tobytes() for t in values]
        items = [[hashlib.sha256(data).hexdigest(), len(t)] for data, t in zip(binaries, values)]
        expected = self.plan.plan(items)
        wire_plan = compact_plan(expected)
        by_key = {key: data for (key, _), data in zip(items, binaries)}
        payload = q.tobytes()+b"".join(by_key[key] for key, _ in expected["missing"])
        reply = self.rpc({"command": "resident_batch", "query_rows": len(q), "train_items": items, "plan": wire_plan}, payload)
        data = reply.pop("payload")
        if (reply["status"] != "resident_batched" or reply["query_rows"] != len(q)
                or reply["train_counts"] != counts or len(data) != len(q)*16*len(trains)+sum(counts)*4
                or reply["plan"] != wire_plan or reply["wire_descriptor_bytes"] != len(payload)):
            raise ValueError("The original complete result identity or private residency plan changed")
        result, offset = [], 0
        for count in counts:
            forward = np.frombuffer(data, np.int32, count=len(q)*4, offset=offset).reshape(len(q), 4).copy()
            offset += len(q)*16
            reverse = np.frombuffer(data, np.int32, count=count, offset=offset).copy()
            offset += count*4
            distances = np.sqrt(forward[:, (1, 3)].astype(np.float32)/262144)
            result.append((forward, distances, reverse))
        self.receipt["calls"].append(dict(reply, total_prepare_IPC_kernel_seconds=time.perf_counter()-started))
        return result

    def match(self, query, train):
        return self.batch(query, [train])[0]

    def close(self):
        try:
            if self.rpc_failed and not self.stopped:
                active_error = sys.exception()
                started = time.perf_counter()
                self.process.stdin.close()
                self.process.stdout.close()
                self.receipt["returncode"] = self.process.wait(timeout=30)
                self.log.close()
                self.stopped = True
                self.receipt.update(shutdown_seconds=time.perf_counter()-started,
                    shutdown_after_explicit_RPC_failure=True)
                if active_error is None:
                    raise RuntimeError("The resident worker failed; no successful shutdown is claimed")
                return
            super().close()
        finally:
            self.plan.clear()
            self.receipt["client_resident_entries_after_close"] = len(self.plan.rows)


@contextmanager
def resident_sift_transport(enabled):
    import integer_sift_matching as native
    import generic_sift_matching as generic
    previous = native.DistanceWorker
    previous_generic = generic.GenericWorker
    started = time.perf_counter()
    trace = {"enabled": enabled, "adapter_setup_seconds": None, "adapter_restore_seconds": None}
    if enabled:
        native.DistanceWorker = ResidentWorker
        generic.GenericWorker = ResidentWorker
    trace["adapter_setup_seconds"] = time.perf_counter()-started
    try:
        yield trace
    finally:
        started = time.perf_counter()
        native.DistanceWorker = previous
        generic.GenericWorker = previous_generic
        trace["adapter_restore_seconds"] = time.perf_counter()-started
        trace["original_worker_factory_restored"] = native.DistanceWorker is previous and generic.GenericWorker is previous_generic
