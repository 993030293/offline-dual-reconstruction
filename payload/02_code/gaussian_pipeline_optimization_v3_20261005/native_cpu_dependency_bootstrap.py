"""Overlap one owned CPU dependency import with immutable input validation."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import importlib
import threading
import time


def import_native_dependencies():
    started = time.perf_counter()
    torch = importlib.import_module("torch")
    if torch.__version__ != "2.7.1+cu128" or torch.cuda.is_initialized():
        raise ValueError("The CPU-only native bootstrap requires the pinned fresh Torch process")
    importlib.import_module("numpy")
    importlib.import_module("torch._dynamo")
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU dependency import initialized a CUDA context")
    return {"thread_ident": threading.get_ident(), "actual_import_seconds": time.perf_counter() - started,
        "pinned_Torch_version": torch.__version__, "CUDA_context_initialized": False,
        "native_optimizer_and_training_formulas_unchanged": True}


@contextmanager
def native_cpu_dependency_bootstrap(enabled, importer=import_native_dependencies):
    if type(enabled) is not bool:
        raise TypeError("The native CPU dependency mode must be explicit")
    started = time.perf_counter()
    trace = {"enabled": enabled, "maximum_workers": 1, "native_import_calls": 0,
             "started_parent_thread_ident": threading.get_ident(), "new_optimization_count": 0}
    pool, future = None, None
    try:
        if enabled:
            pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="owned-CPU-dependency")
            future = pool.submit(importer)
            trace["native_import_calls"] = 1
        yield trace
    finally:
        wait_started = time.perf_counter()
        try:
            if future is not None:
                trace["actual_import"] = future.result(timeout=120)
        finally:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=False)
            trace.update(join_wall_seconds=time.perf_counter() - wait_started,
                owned_worker_joined=pool is None or all(not thread.is_alive() for thread in pool._threads),
                full_context_wall_seconds=time.perf_counter() - started)
            if not trace["owned_worker_joined"]:
                raise RuntimeError("The native CPU dependency owner did not join its worker")
